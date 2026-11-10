//! The scene the Material Preview tests look at: a grey card facing the
//! camera under a sky bright above and dark below (`preview.rs`,
//! `preview_lamp.rs`).
use phiray::{Atlas, Camera, Mode, Ray, Tracer, Tri};
pub const W: u32 = 48;
/// A sky bright above and dark below, with Material Preview's lighting.
pub fn sky_atlas() -> Atlas {
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
pub fn camera() -> Camera {
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
pub fn card(half: f32, y: f32, tex: Option<(u32, f32)>) -> Vec<Tri> {
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
pub fn render(mode: Mode, cam: Camera, scene: &[Tri], atlas: &Atlas) -> (Vec<u8>, u32) {
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
pub fn luma(px: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * f32::from(px[i]) + 0.7152 * f32::from(px[i + 1]) + 0.0722 * f32::from(px[i + 2])
}
