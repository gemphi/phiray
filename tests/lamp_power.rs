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

use phiray::{Atlas, Camera, LIGHTS, Light, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 32;
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
/// The same dome with two walls a metre high and a metre apart: the floor
/// between them sees only the strip of sky the trench leaves open, plus
/// whatever bounces off the walls. Cycles, by bounce depth:
///   1 -> 0.462   4 -> 0.584   12 -> 0.605   (open plane: 0.800)
/// This is the number that says whether indirect light is a real path or a
/// stand-in.
const TRENCH_B1: f32 = 0.462_051;
const TRENCH_B4: f32 = 0.583_955;
/// A uniform dome of radiance 1 over albedo 0.8. The law has no constant
/// in it at all: a dome of radiance `L` puts irradiance `πL` on a surface
/// (the hemisphere), and a Lambertian albedo `a` turns that back into
/// radiance `aE/π = aL`. Cycles measures 0.800 for `L = 1`, and
/// `(0.790, 0.000004, 0.797)` for a magenta dome — green exactly zero.
const WORLD_1_OVER_ALBEDO: f32 = 0.800_055;
const WORLD_HALF_OVER_ALBEDO: f32 = 0.400_027;

/// How close the two have to be. This test exists to catch a *constant*
/// being wrong — a factor of π, a missing 4, a squared distance — not to
/// certify the last percent. Two modelling differences are left inside it
/// and are the reason it is not tighter: the tracer carries a GGX lobe at
/// f0 = 0.04 over its Lambert where the reference's Principled puts almost
/// none into this geometry, and the two count an indirect bounce off a
/// 20 m plane slightly differently. Both are a few per cent; a factor of π
/// is 214 %.
const TOLERANCE: f32 = 0.15;

/// A **uniform environment image** of radiance `level`, with its sampling
/// table and prefiltered chain — the other way a world can arrive. The
/// answer it gives must be the flat colour's answer: a dome of radiance L
/// is a dome of radiance L however it is stored.
fn uniform_env(level: f32) -> (Atlas, u32, u32) {
    let (w, h) = (32u32, 16u32);
    let rgb = vec![level; (w * h * 3) as usize];
    let mut atlas = Atlas::default();
    let map = atlas.add_hdr(w, h, &rgb);
    let cdf = atlas.add_env_cdf(w, h, &rgb);
    atlas.add_env_lighting(w, h, &rgb);
    (atlas, map + 1, cdf + 1)
}

/// A black world: the lamp is the only light.
fn dark() -> Atlas {
    let (w, h) = (8u32, 4u32);
    let rgb = vec![0.0f32; (w * h * 3) as usize];
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    atlas.add_env_lighting(w, h, &rgb);
    atlas
}

/// Two walls a metre high and a metre apart, along Y — a trench whose
/// floor the camera looks straight down into.
fn trench() -> Vec<Tri> {
    let c = [0.8, 0.8, 0.8, 1.0];
    let mut out = plane();
    for x in [-0.5f32, 0.5] {
        let n = [if x < 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0];
        let q = [[x, -10.0, 0.0], [x, 10.0, 0.0], [x, 10.0, 1.0], [x, -10.0, 1.0]];
        out.push(Tri::new([q[0], q[1], q[2]], [n; 3], c).with_pbr(1.0, 0.0));
        out.push(Tri::new([q[0], q[2], q[3]], [n; 3], c).with_pbr(1.0, 0.0));
    }
    out
}

/// A 20 x 20 m plane at z = 0, albedo 0.8, as rough as a surface gets so
/// its lobe is Lambert and nothing else.
fn plane() -> Vec<Tri> {
    let n = [0.0, 0.0, 1.0];
    let q = [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0]];
    let c = [0.8, 0.8, 0.8, 1.0];
    vec![
        Tri::new([q[0], q[1], q[2]], [n; 3], c).with_pbr(1.0, 0.0),
        Tri::new([q[0], q[2], q[3]], [n; 3], c).with_pbr(1.0, 0.0),
    ]
}

/// Straight down on the origin, orthographic, 2 m across — the frame
/// `blender_lamp_power.py` sets up.
fn overhead(light: Light, count: u32) -> Camera {
    let mut lights = [Light::default(); LIGHTS];
    lights[0] = light;
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        back: [0.0, 0.0, 1.0],
        pivot: [0.0, 0.0, 0.0],
        dist: 10.0,
        units: 1.0,
        persp: 0,
        // The frame spans 2 m, as the reference's `ortho_scale = 2.0` does:
        // the kernel's extent is `1 / (aspect × scale)`. It stood at 1.0 —
        // a 1 m frame — which left every flat-field number here right (the
        // centre of a flat field is the centre either way) and every
        // *profile* twice as wide as Blender's in pixels (§58).
        scale: 0.5,
        aspect: 1.0,
        // A black sky and no environment: the lamp is all there is.
        sky: [0.0; 4],
        env: [0.0, 0.0, 0.0, 1.0],
        envmap: 0,
        env_cdf: 0,
        env_hidden: 0,
        preview: [0; 4],
        count,
        lights,
        ..Default::default()
    }
}

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

/// The same plane under a uniform world and no lamp at all.
fn world_radiance(gpu: &phiray::Gpu, level: f32) -> [f32; 3] {
    world_on(gpu, &plane(), level, 0)
}

/// The same, over whatever geometry, at `bounces` (0 = the kernel's own).
fn world_on(gpu: &phiray::Gpu, scene: &[Tri], level: f32, bounces: u32) -> [f32; 3] {
    world_with(gpu, scene, Camera { env: [level, level, level, 1.0], bounces, ..overhead(Light::default(), 0) })
}

/// The centre of `scene` under `cam`'s world, at 256 samples.
fn world_with(gpu: &phiray::Gpu, scene: &[Tri], cam: Camera) -> [f32; 3] {
    let mut tr = Tracer::from_scene(gpu, Scene::from_tris(scene, &dark())).expect("scene");
    tr.set_mode(Mode::Rendered);
    tr.set_samples(256);
    while !tr.sample(gpu, cam, W, W).1 {}
    let lin = tr.radiance(gpu);
    let (lo, hi) = (W / 2 - 2, W / 2 + 2);
    let mut sum = [0.0; 3];
    let mut n = 0.0;
    for y in lo..hi {
        for x in lo..hi {
            for k in 0..3 {
                sum[k] += lin[(y * W + x) as usize][k];
            }
            n += 1.0;
        }
    }
    sum.map(|c| c / n)
}

/// A uniform world is the one term with no constant to get wrong: whatever
/// radiance the dome has, a Lambertian surface sends back that times its
/// albedo. Blender measures exactly that; this says we do too, and that the
/// answer is linear in the dome's level.
#[test]
fn a_uniform_world_comes_back_as_the_albedo_times_its_radiance() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    // Tighter than the lamps' 15 %: this law has no constant in it at all,
    // so what is left is one indirect bounce the reference caps and we do
    // not — measured at 1.5 %.
    let full = world_radiance(&gpu, 1.0)[0];
    let off = (full / WORLD_1_OVER_ALBEDO - 1.0).abs();
    assert!(off < 0.05, "a dome of radiance 1 over albedo 0.8: {full:.3}, Cycles {WORLD_1_OVER_ALBEDO:.3} ({:.0}% out)", off * 100.0);
    let half = world_radiance(&gpu, 0.5)[0];
    let off = (half / WORLD_HALF_OVER_ALBEDO - 1.0).abs();
    assert!(off < 0.05, "half the dome is half the light: {half:.3}, Cycles {WORLD_HALF_OVER_ALBEDO:.3} ({:.0}% out)", off * 100.0);
    // And it is exactly linear — no constant hiding under it. A three-point
    // studio rig used to fire whenever a scene had no lamp and no
    // environment *image*, so a flat world came back as `0.83 L + 0.068`.
    assert!((full / half - 2.0).abs() < 0.02, "linear in the dome's level: {full:.3} against {half:.3}");
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
    assert!(off < TOLERANCE, "a 1000 W point at 2 m over albedo 0.8: {got:.3}, Cycles {POINT_1000W_AT_2M:.3} ({:.0}% out)", off * 100.0);

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

/// Indirect light in shadow. The open plane says the world's *level* is
/// right; this says whether what reaches an occluded surface is a real
/// bounce path. Cycles puts 0.462 on the floor of a one-metre trench at
/// one diffuse bounce, 0.584 at four — its default — and 0.605 at twelve,
/// against 0.800 in the open: a surface that can only see a strip of sky
/// gets a fraction of the dome, and the walls hand it a little back with
/// each bounce.
///
/// Four is the number to hit, because four is what Blender ships. One is
/// left loose on purpose: Cycles counts a "bounce" from the camera hit and
/// this path counts from the surface, so the two differ by one step at the
/// shallowest depth and agree once there is room to converge.
#[test]
fn a_shadowed_surface_gets_what_bounces_to_it() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let open = world_radiance(&gpu, 1.0)[0];
    let one = world_on(&gpu, &trench(), 1.0, 1)[0];
    let four = world_on(&gpu, &trench(), 1.0, 4)[0];
    assert!(one < open * 0.7, "a trench floor is much darker than open ground: {one:.3} against {open:.3}");
    // Truncating the path loses light, so it climbs with depth, as Cycles'
    // own numbers do (0.462 -> 0.584 -> 0.605).
    assert!(four > one * 1.1, "more bounces, more light: {one:.3} at one, {four:.3} at four");
    let off = (four / TRENCH_B4 - 1.0).abs();
    assert!(off < 0.05, "four bounces into a trench: {four:.3}, Cycles {TRENCH_B4:.3} ({:.0}% out)", off * 100.0);
    assert!(one > TRENCH_B1 * 0.75, "and one bounce is within a step of Cycles' one: {one:.3} against {TRENCH_B1:.3}");
}

/// Light Paths ▸ Max Bounces ▸ Total: 0 is direct light only, so the
/// trench floor keeps the strip of sky it sees and loses everything the
/// walls bounce to it; Total 12 with Diffuse 4 is the four-bounce number.
#[test]
fn total_bounces_zero_is_direct_light_only() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let lit = Camera { env: [1.0, 1.0, 1.0, 1.0], bounces: 4, ..overhead(Light::default(), 0) };
    let none = world_with(&gpu, &trench(), lit.with_depth(0, 8, 12, 4))[0];
    let four = world_with(&gpu, &trench(), lit.with_depth(12, 8, 12, 4))[0];
    assert!(none > 0.1, "the sky the floor sees still lights it: {none:.3}");
    assert!(none < four * 0.9, "no bounces, less light: {none:.3} against {four:.3} at four");
    let off = (four / TRENCH_B4 - 1.0).abs();
    assert!(off < 0.05, "Total 12 leaves Diffuse 4 as it was: {four:.3}, Cycles {TRENCH_B4:.3}");
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

/// A world stored as an **image** must light a surface exactly as the same
/// world stored as a colour does. The flat-colour path is pinned against
/// Cycles above (`a_uniform_world_…`, 0.812 for a dome of radiance 1 over
/// albedo 0.8, where Cycles gives 0.800); this says the image path agrees
/// with it.
///
/// They are different code: a colour is read straight out of `env_at`,
/// while an image is sampled, has a CDF for next-event estimation, a
/// prefiltered chain, MIS weights against the cosine ray, and a tail that
/// closes a truncated path. Any of those can carry a normalization error
/// that a flat colour would never show.
#[test]
fn a_world_stored_as_an_image_lights_as_the_same_world_stored_as_a_colour() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let flat = world_radiance(&gpu, 1.0)[0];
    let (atlas, map, cdf) = uniform_env(1.0);
    let cam = Camera { env: [1.0, 1.0, 1.0, 1.0], envmap: map, env_cdf: cdf, ..overhead(Light::default(), 0) };
    let mut tr = Tracer::from_scene(&gpu, Scene::from_tris(&plane(), &atlas)).expect("scene");
    tr.set_mode(Mode::Rendered);
    tr.set_samples(256);
    while !tr.sample(&gpu, cam, W, W).1 {}
    let lin = tr.radiance(&gpu);
    let (lo, hi) = (W / 2 - 2, W / 2 + 2);
    let mut sum = 0.0;
    let mut n = 0.0;
    for y in lo..hi {
        for x in lo..hi {
            sum += lin[(y * W + x) as usize][0];
            n += 1.0;
        }
    }
    let image = sum / n;
    eprintln!("flat colour {flat:.4}, image {image:.4}, Cycles {WORLD_1_OVER_ALBEDO:.4}");
    let off = (image / flat - 1.0).abs();
    assert!(off < 0.05, "an image world and a colour world light the same: {image:.4} against {flat:.4} ({:.0}% apart)", off * 100.0);
    let off = (image / WORLD_1_OVER_ALBEDO - 1.0).abs();
    assert!(off < 0.08, "and both against Cycles: {image:.4} against {WORLD_1_OVER_ALBEDO:.4} ({:.0}% out)", off * 100.0);
}

/// Cycles through a **non-uniform** environment image: a band of radiance
/// 4 above 50° elevation over 0.1 elsewhere, on the albedo-0.8 plane.
/// (`truth/blender_lamp_power.py` writes the same sky as `sky.exr`; the
/// test builds it from the same rule rather than reading the file, so the
/// comparison cannot drift on an image loader.)
const WORLD_BAND: f32 = 1.290_959;

/// The sky that number came from. phiray's rows run **zenith first** —
/// `env_at` maps `v = 0.5 − atan2(z, |xy|)/π`, so `v = 0` is straight up —
/// where Blender's image buffer starts at the nadir, so this is the
/// reference's rows reversed.
fn banded_env() -> (Atlas, u32, u32) {
    let (w, h) = (64u32, 32u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        let elev = std::f32::consts::FRAC_PI_2 - (y as f32 + 0.5) / h as f32 * std::f32::consts::PI;
        let v = if elev > 50.0_f32.to_radians() { 4.0 } else { 0.1 };
        for _ in 0..w {
            rgb.extend([v, v, v]);
        }
    }
    let mut atlas = Atlas::default();
    let map = atlas.add_hdr(w, h, &rgb);
    let cdf = atlas.add_env_cdf(w, h, &rgb);
    atlas.add_env_lighting(w, h, &rgb);
    (atlas, map + 1, cdf + 1)
}

/// An HDRI is what every real file lights with, and it is a different path
/// from a flat colour on our side: sampled through a table, prefiltered,
/// weighted against the cosine ray. A uniform image is the invariant test
/// above; this is the one that can actually catch a sampling error, because
/// the answer depends on *where* the light is in the sky.
#[test]
fn a_banded_sky_lights_the_plane_as_cycles_does() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let (atlas, map, cdf) = banded_env();
    let cam = Camera { env: [1.0, 1.0, 1.0, 1.0], envmap: map, env_cdf: cdf, ..overhead(Light::default(), 0) };
    let mut tr = Tracer::from_scene(&gpu, Scene::from_tris(&plane(), &atlas)).expect("scene");
    tr.set_mode(Mode::Rendered);
    tr.set_samples(256);
    while !tr.sample(&gpu, cam, W, W).1 {}
    let lin = tr.radiance(&gpu);
    let (lo, hi) = (W / 2 - 2, W / 2 + 2);
    let (mut sum, mut n) = (0.0, 0.0);
    for y in lo..hi {
        for x in lo..hi {
            sum += lin[(y * W + x) as usize][0];
            n += 1.0;
        }
    }
    let got = sum / n;
    let off = (got / WORLD_BAND - 1.0).abs();
    eprintln!("banded sky: phiray {got:.4}, Cycles {WORLD_BAND:.4} (x{:.3})", got / WORLD_BAND);
    assert!(off < 0.10, "a banded sky: {got:.4}, Cycles {WORLD_BAND:.4} ({:.0}% out)", off * 100.0);
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
