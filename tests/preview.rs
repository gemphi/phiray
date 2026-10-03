//! Material Preview is deterministic: one dispatch is the frame, the same
//! frame every time, smooth where the surface is smooth — and a texture is
//! read at the level the pixel's footprint asks for.

use phiray::{Atlas, Camera, Light, Mode, Ray, Tracer, Tri};

const W: u32 = 48;

/// A sky bright above and dark below, with Material Preview's lighting.
fn sky_atlas() -> Atlas {
    let (w, h) = (32u32, 16u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for _ in 0..w {
            rgb.extend(if y < h / 2 { [3.0, 3.0, 3.0] } else { [0.05, 0.05, 0.05] });
        }
    }
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    let lighting = atlas.add_env_lighting(w, h, &rgb);
    assert_eq!((lighting.irradiance, lighting.prefiltered), (2, 3));
    atlas
}

fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        pivot: [0.0, 0.0, 0.0],
        dist: 4.0,
        units: 1.0,
        persp: 1,
        scale: 1.0,
        aspect: 1.0,
        sky: [0.24, 0.24, 0.24, 0.0],
        sun: [0.0, -0.7, 0.7],
        env: [1.0, 1.0, 1.0, 1.0],
        envmap: 1,
        env_cdf: 2,
        env_hidden: 1,
        preview: [3, 4, phiray::trace::ibl::LEVELS, 0],
        ..Default::default()
    }
}

/// A card facing the camera, `half` wide, at `y` (further from the camera
/// as `y` grows), its texture `tex` tiled `tiles` times across.
fn card(half: f32, y: f32, tex: Option<(u32, f32)>) -> Vec<Tri> {
    let n = [0.0, -1.0, 0.0];
    let q = [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]];
    let color = [0.8, 0.8, 0.8, 1.0];
    let (a, b) = (Tri::new([q[0], q[1], q[2]], [n; 3], color), Tri::new([q[0], q[2], q[3]], [n; 3], color));
    match tex {
        Some((id, tiles)) => vec![
            a.with_texture(id, [[0.0, 0.0], [tiles, 0.0], [tiles, tiles]]),
            b.with_texture(id, [[0.0, 0.0], [tiles, tiles], [0.0, tiles]]),
        ],
        None => vec![a, b],
    }
}

fn render(mode: Mode, cam: Camera, scene: &[Tri], atlas: &Atlas) -> (Vec<u8>, u32) {
    let gpu = Ray::gpu().unwrap();
    let mut tracer = Tracer::new(&gpu, scene, atlas).unwrap();
    tracer.set_mode(mode);
    tracer.set_samples(8);
    let mut passes = 0;
    let (mut px, mut done) = tracer.render(&gpu, cam, W, W);
    passes += 1;
    while !done {
        (px, done) = tracer.render(&gpu, cam, W, W);
        passes += 1;
    }
    (px, passes)
}

fn luma(px: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * f32::from(px[i]) + 0.7152 * f32::from(px[i + 1]) + 0.0722 * f32::from(px[i + 2])
}

/// Mean absolute difference between horizontal neighbours over the card's middle.
fn roughness_of(px: &[u8]) -> f32 {
    let mut sum = 0.0;
    let mut n = 0;
    for y in W / 3..2 * W / 3 {
        for x in W / 3..2 * W / 3 - 1 {
            sum += (luma(px, x, y) - luma(px, x + 1, y)).abs();
            n += 1;
        }
    }
    sum / n as f32
}

#[test]
fn one_pass_is_the_frame_and_the_same_frame_every_time() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let scene = card(3.0, 2.0, None);
    let (first, passes) = render(Mode::Preview, camera(), &scene, &atlas);
    assert_eq!(passes, 1, "deterministic: no refinement");
    let (again, _) = render(Mode::Preview, camera(), &scene, &atlas);
    assert_eq!(first, again, "the same frame again");
    assert!(luma(&first, W / 2, W / 2) > 60.0, "the sky lights the card: {}", luma(&first, W / 2, W / 2));
}

#[test]
fn a_flat_surface_is_smooth_where_one_path_traced_sample_is_noisy() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let scene = card(3.0, 2.0, None);
    let (preview, _) = render(Mode::Preview, camera(), &scene, &atlas);
    let gpu = Ray::gpu().unwrap();
    let mut tracer = Tracer::new(&gpu, &scene, &atlas).unwrap();
    tracer.set_mode(Mode::Rendered);
    tracer.set_samples(64);
    let (traced, _) = tracer.render(&gpu, camera(), W, W);
    let (smooth, noisy) = (roughness_of(&preview), roughness_of(&traced));
    assert!(smooth < 1.0, "a flat card is flat: {smooth}");
    assert!(noisy > smooth * 4.0, "the first path-traced sample is the noisy one: {noisy} vs {smooth}");
}

#[test]
fn a_distant_checker_reads_as_grey_through_its_mip_levels() {
    if Ray::gpu().is_none() {
        return;
    }
    let mut atlas = sky_atlas();
    // A 64 × 64 checker of single texels.
    let mut rgba = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            let on = (x + y) % 2 == 0;
            rgba.extend(if on { [255, 255, 255, 255] } else { [0, 0, 0, 255] });
        }
    }
    let tex = atlas.add(64, 64, &rgba);
    // Tiled eight times over a card that fills the frame: 512 texels over
    // 48 pixels — read at its base level this would be a moiré of black
    // and white; at the level the footprint asks for it is grey.
    let scene = card(3.0, 2.0, Some((tex, 8.0)));
    let (px, _) = render(Mode::Preview, camera(), &scene, &atlas);
    let plain = render(Mode::Preview, camera(), &card(3.0, 2.0, None), &atlas).0;
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for y in W / 3..2 * W / 3 {
        for x in W / 3..2 * W / 3 {
            let ratio = luma(&px, x, y) / luma(&plain, x, y).max(1.0);
            lo = lo.min(ratio);
            hi = hi.max(ratio);
        }
    }
    assert!(hi - lo < 0.25, "no moiré: the checker is one grey across the card ({lo:.2}..{hi:.2})");
    assert!(lo > 0.2 && hi < 0.8, "grey, not black or white ({lo:.2}..{hi:.2})");
}

#[test]
fn a_lamp_casts_a_hard_shadow() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let mut scene = card(3.0, 2.0, None);
    // A small card between a lamp off to the right and the big one: its
    // shadow falls on the big card to the left of it, where the camera sees
    // the big card past the small one.
    scene.extend(card(0.3, 1.0, None));
    let mut lights = [Light::default(); phiray::LIGHTS];
    lights[0] = Light { pos: [1.5, -2.0, 0.0], kind: 1, color: [800.0, 800.0, 800.0], size: 0.1, dir: [0.0, 1.0, 0.0], cone: 0.0 };
    let cam = Camera { count: 1, lights, envmap: 0, env_cdf: 0, preview: [0; 4], env: [0.05, 0.05, 0.05, 1.0], ..camera() };
    let (px, _) = render(Mode::Preview, cam, &scene, &atlas);
    let lit = luma(&px, W / 2 + 16, W / 2);
    let shaded = luma(&px, W / 2 - 16, W / 2);
    // Without the blocker the same two spots are lit alike (the right a
    // little more, nearer the lamp).
    let open = render(Mode::Preview, cam, &card(3.0, 2.0, None), &atlas).0;
    assert!((luma(&open, W / 2 - 16, W / 2) - luma(&open, W / 2 + 16, W / 2)).abs() < 40.0);
    assert!(lit > shaded + 40.0, "lit {lit} shaded {shaded}");
}
