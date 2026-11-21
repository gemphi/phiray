//! How much light one lamp of a known power actually puts on a surface,
//! against Cycles' own numbers (`phiphx/tests/truth/blender_lamp_power.py`,
//! Blender 5.0.1, `rendering.md` §54).
//!
//! The scene is the simplest one that can settle a constant: a large plane
//! at z = 0 with a pure Lambertian albedo of 0.8, one lamp 2 m above it
//! pointing down, a black world, and an orthographic camera looking
//! straight down. The number compared is **scene-linear radiance** at the
//! centre — `Tracer::radiance`, before any view transform — so nothing
//! about tone mapping enters it.
//!
//! The physics these pin, for a point lamp of power `P` at distance `d`
//! over a Lambertian albedo `a`:
//!
//!   irradiance  E = P / (4π d²)       radiance  L = a E / π = a P / (4π² d²)
//!
//! which for `P = 1000`, `d = 2`, `a = 0.8` is 5.066 — and Cycles measures
//! 5.047. A spot's power is a point's, masked by its cone (Cycles gives the
//! same 5.049 inside a 60° cone), and a sun's "power" is irradiance
//! already, so `L = a E / π`.

mod power;
use power::{W, dark, overhead, plane};
use phiray::{Camera, Light, Mode, Ray, Scene, Tracer, Tri};
/// Blender's own radiance at the centre pixel (`blender_lamp_power.py`).
const POINT_1000W_AT_2M: f32 = 5.046_447;
const SPOT_1000W_AT_2M: f32 = 5.047_564;
const SUN_5W_AT_2M: f32 = 1.270_905;
/// A 100 W square area light, 1 m on a side, 2 m up. EEVEE converts a
/// lamp's radiant flux to radiance with `L = P / (π A)` over the full area
/// (`eevee_light.cc`'s `shape_radiance_get`), so the irradiance straight
/// below is `L A cos / d² = P cos / (π d²)` — the area cancels, which is
/// what "make illumination power constant" means. Cycles measures 1.862
/// against the 2.026 that limit gives, the difference being the source's
/// finite size at this distance.
const AREA_100W_1M_AT_2M: f32 = 1.861_637;
/// How close the two have to be. This test exists to catch a *constant*
/// being wrong — a factor of π, a missing 4, a squared distance — not to
/// certify the last percent. Two modelling differences are left inside it
/// and are the reason it is not tighter: the tracer carries a GGX lobe at
/// f0 = 0.04 over its Lambert where the reference's Principled puts almost
/// none into this geometry, and the two count an indirect bounce off a
/// 20 m plane slightly differently. Both are a few per cent; a factor of π
/// is 214 %.
const TOLERANCE: f32 = 0.15;
/// The mean linear radiance over the middle of the frame, where the lamp is
/// straight overhead and the geometry has nothing to say.
fn centre_radiance(gpu: &phiray::Gpu, cam: Camera) -> f32 {
    let mut tr = Tracer::from_scene(gpu, Scene::from_tris(&plane(), &dark())).expect("scene");
    tr.set_mode(Mode::Rendered);
    tr.set_samples(256);
    while !tr.sample(gpu, cam, W, W).1 {}
    let lin = tr.radiance(gpu);
    let (lo, hi) = (W / 2 - 2, W / 2 + 2);
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in lo..hi {
        for x in lo..hi {
            sum += lin[(y * W + x) as usize][0];
            n += 1.0;
        }
    }
    sum / n
}
#[test]
fn a_lamps_power_lands_on_a_surface_as_cycles_puts_it_there() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let at = [0.0, 0.0, 2.0];
    let down = [0.0, 0.0, -1.0];

    // A point lamp: 1000 W two metres up.
    let point = Light { pos: at, kind: 1, color: [1000.0; 3], size: 0.0, dir: down, ..Light::default() };
    let got = centre_radiance(&gpu, overhead(point, 1));
    let off = (got / POINT_1000W_AT_2M - 1.0).abs();
    // Tighter than TOLERANCE: 10 % here was a shading point 10 cm off the floor.
    assert!(off < 0.03, "a 1000 W point at 2 m over albedo 0.8: {got:.3}, Cycles {POINT_1000W_AT_2M:.3} ({:.0}% out)", off * 100.0);

    // A spot of the same power, well inside its cone: Blender masks a
    // point, it does not spread it, so the two agree.
    let cone = 30.0_f32.to_radians().cos();
    let spot = Light { pos: at, kind: 2, color: [1000.0; 3], size: 0.0, dir: down, cone, blend: 1e-8, right: [1.0, 0.0, 0.0], spot_scale: [1.0, 1.0], ..Light::default() };
    let got = centre_radiance(&gpu, overhead(spot, 1));
    let off = (got / SPOT_1000W_AT_2M - 1.0).abs();
    assert!(off < TOLERANCE, "a 1000 W spot inside its cone: {got:.3}, Cycles {SPOT_1000W_AT_2M:.3} ({:.0}% out)", off * 100.0);

    // A sun: its "power" is irradiance already, so distance does not enter.
    let sun = Light { pos: at, kind: 0, color: [5.0; 3], size: 0.0, dir: down, ..Light::default() };
    let got = centre_radiance(&gpu, overhead(sun, 1));
    let off = (got / SUN_5W_AT_2M - 1.0).abs();
    assert!(off < TOLERANCE, "a 5 W/m² sun: {got:.3}, Cycles {SUN_5W_AT_2M:.3} ({:.0}% out)", off * 100.0);

    // An area light, 1 m square. Its power does not depend on its size —
    // Blender keeps the illumination constant as the light grows — so the
    // kernel's job is the cosine and the inverse square, same as a point's.
    let area = Light { pos: at, kind: 3, color: [100.0; 3], size: 1.0, size_y: 1.0, dir: down, right: [1.0, 0.0, 0.0], ..Light::default() };
    // Tighter than the others: the disk form factor is exact for a disk
    // and very close for a square, so what is left here is only the lobe.
    let got = centre_radiance(&gpu, overhead(area, 1));
    let off = (got / AREA_100W_1M_AT_2M - 1.0).abs();
    assert!(off < 0.06, "a 100 W 1 m area light at 2 m: {got:.3}, Cycles {AREA_100W_1M_AT_2M:.3} ({:.0}% out)", off * 100.0);
    // It is a *shape* correction, not a scale: four times further away the
    // source is a point again and the far-field form takes over.
    let far = centre_radiance(&gpu, overhead(Light { pos: [0.0, 0.0, 8.0], ..area }, 1));
    let point_form = 0.8 * 100.0 / (std::f32::consts::PI * std::f32::consts::PI * 64.0);
    assert!((far / point_form - 1.0).abs() < 0.06, "at 8 m an area light is a point: {far:.4} against {point_form:.4}");
}

/// Cycles' specular lobe at its peak, for a 1000 W point lamp 2 m above a
/// black surface at `f0 = 0.04` — Base Color black, so the diffuse term is
/// zero and only the microfacet reflection answers. Lamp and camera are
/// both straight overhead, which puts the half-vector on the normal.
const SPEC: [(f32, f32); 4] = [(0.1, 142.403), (0.3, 7.450), (0.6, 0.4936), (1.0, 0.06865)];

/// The specular lobe, roughness by roughness. The diffuse terms above say
/// the light arriving is right; this says the surface sends the right
/// amount of it back into the camera, which is what the contact sheet's
/// broader, softer highlights were about.
#[test]
fn the_specular_lobe_matches_cycles_across_roughness() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let lamp = Light { pos: [0.0, 0.0, 2.0], kind: 1, color: [1000.0; 3], size: 0.0, dir: [0.0, 0.0, -1.0], ..Light::default() };
    for (rough, want) in SPEC {
        let n = [0.0, 0.0, 1.0];
        let q = [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0]];
        let black = [0.0, 0.0, 0.0, 1.0];
        let scene = vec![
            Tri::new([q[0], q[1], q[2]], [n; 3], black).with_pbr(rough, 0.0),
            Tri::new([q[0], q[2], q[3]], [n; 3], black).with_pbr(rough, 0.0),
        ];
        let mut tr = Tracer::from_scene(&gpu, Scene::from_tris(&scene, &dark())).expect("scene");
        tr.set_mode(Mode::Rendered);
        tr.set_samples(256);
        let cam = overhead(lamp, 1);
        while !tr.sample(&gpu, cam, W, W).1 {}
        let lin = tr.radiance(&gpu);
        let got = lin[((W / 2) * W + W / 2) as usize][0];
        let off = (got / want - 1.0).abs();
        assert!(off < TOLERANCE, "roughness {rough}: phiray {got:.4}, Cycles {want:.4} ({:.0}% out)", off * 100.0);
    }
}

/// The specular lobe's **shape**, which is what a highlight looks like.
/// The peak above says how much light the lobe sends back; this says how
/// wide it is, which is the difference between a crisp highlight and a
/// smeared one.
///
/// Cycles' profile across the middle row at roughness 0.3, as a fraction
/// of its own peak, over the eight pixels around centre — and its half-max
/// width in pixels of the 32-pixel, 2-metre frame.
const SPEC_PROFILE_R03: [f32; 4] = [0.550, 0.725, 0.892, 1.000];
const SPEC_HALF_WIDTH_R03: usize = 8;

#[test]
fn the_specular_lobes_shape_is_cycles_shape() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let lamp = Light { pos: [0.0, 0.0, 2.0], kind: 1, color: [1000.0; 3], size: 0.0, dir: [0.0, 0.0, -1.0], ..Light::default() };
    let n = [0.0, 0.0, 1.0];
    let q = [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0]];
    let black = [0.0, 0.0, 0.0, 1.0];
    let scene = vec![
        Tri::new([q[0], q[1], q[2]], [n; 3], black).with_pbr(0.3, 0.0),
        Tri::new([q[0], q[2], q[3]], [n; 3], black).with_pbr(0.3, 0.0),
    ];
    let mut tr = Tracer::from_scene(&gpu, Scene::from_tris(&scene, &dark())).expect("scene");
    tr.set_mode(Mode::Rendered);
    tr.set_samples(256);
    let cam = overhead(lamp, 1);
    while !tr.sample(&gpu, cam, W, W).1 {}
    let lin = tr.radiance(&gpu);
    let row: Vec<f32> = (0..W).map(|x| lin[((W / 2) * W + x) as usize][0]).collect();
    let peak = row.iter().cloned().fold(0.0, f32::max);
    let wide = row.iter().filter(|v| **v > peak / 2.0).count();
    assert!(wide.abs_diff(SPEC_HALF_WIDTH_R03) <= 1, "the lobe is as wide as Cycles': {wide} px against {SPEC_HALF_WIDTH_R03}");
    for (i, want) in SPEC_PROFILE_R03.iter().enumerate() {
        let got = row[(W as usize / 2) - 4 + i] / peak;
        assert!((got - want).abs() < 0.05, "the lobe's falloff at {} px from centre: {got:.3} against Cycles' {want:.3}", 4 - i);
    }
}

/// Light Paths ▸ Clamping: the 1000 W point lamp puts 5.05 on the plane,
/// and a Direct clamp of 2 caps every sample's direct light at 2 — so the
/// mean is 2, not 5.05 (`film_clamp_light`). An Indirect clamp leaves
/// direct light alone.
#[test]
fn a_direct_clamp_caps_the_lamp_and_an_indirect_one_does_not() {
    let Some(gpu) = Ray::gpu() else { return };
    let at = [0.0, 0.0, 2.0];
    let point = Light { pos: at, kind: 1, color: [1000.0; 3], size: 0.0, dir: [0.0, 0.0, -1.0], ..Light::default() };
    let centre = |direct: f32, indirect: f32| {
        let mut tr = Tracer::from_scene(&gpu, Scene::from_tris(&plane(), &dark())).expect("scene");
        tr.set_mode(Mode::Rendered);
        tr.set_samples(64);
        let cam = Camera { limit: [direct, indirect, 0.0, 0.0], ..overhead(point, 1) };
        while !tr.sample(&gpu, cam, W, W).1 {}
        let lin = tr.radiance(&gpu);
        lin[(W / 2 * W + W / 2) as usize][0]
    };
    let free = centre(0.0, 0.0);
    let capped = centre(2.0, 0.0);
    let indirect_only = centre(0.0, 2.0);
    assert!(free > 4.0, "unclamped {free}");
    assert!((capped - 2.0).abs() < 0.05, "direct clamp 2 -> {capped}");
    assert!((indirect_only - free).abs() / free < 0.05, "an indirect clamp leaves direct light: {indirect_only} vs {free}");
}

/// Denoise after Adaptive Sampling: a pixel that settled early has fewer
/// samples than the frame, and its mean is over ITS count. The filter took
/// the frame's count instead, so every settled pixel came out darker by
/// the share of the render it sat out — blotches over a whole 256-sample
/// frame. The denoised plane must stay as bright as the traced one.
#[test]
fn denoising_an_adaptive_render_keeps_its_brightness() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let point = Light { pos: [0.0, 0.0, 2.0], kind: 1, color: [100.0; 3], size: 0.0, dir: [0.0, 0.0, -1.0], ..Light::default() };
    let mut tr = Tracer::from_scene(&gpu, Scene::from_tris(&plane(), &dark())).expect("scene");
    tr.set_mode(Mode::Rendered);
    tr.set_samples(64);
    // A loose threshold, so the flat plane settles long before 64.
    tr.set_adaptive([0.5, 4.0, 4.0, 1.0]);
    let cam = overhead(point, 1);
    while !tr.sample(&gpu, cam, W, W).1 {}
    // What is shown, not the sums: the sums' scale cancels on a flat
    // plane, the picture's does not.
    let mean = |rgba: &[u8]| rgba.chunks_exact(4).map(|p| f32::from(p[0])).sum::<f32>() / (rgba.len() / 4) as f32;
    let traced = mean(&tr.pixels(&gpu));
    assert!(tr.unconverged(&gpu).is_some_and(|left| left < W * W), "some pixels settled early");
    tr.denoise(&gpu);
    let denoised = mean(&tr.pixels(&gpu));
    assert!(traced > 40.0 && traced < 230.0, "a mid-grey plane, not clipped: {traced:.1}");
    assert!((denoised / traced - 1.0).abs() < 0.05, "denoised {denoised:.3} against traced {traced:.3}");
}
