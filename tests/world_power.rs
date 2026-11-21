//! How much light the world puts on a surface, against Cycles' own numbers
//! (`truth/blender_lamp_power.py`): a uniform dome, the same dome as an
//! image, a banded sky, and a trench that shows indirect light is a real
//! path. The lamps' half is `lamp_power.rs`; the rig is `power/`.
mod power;
use power::{W, dark, overhead, plane};
use phiray::{Atlas, Camera, Light, Mode, Ray, Scene, Tracer, Tri};
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
