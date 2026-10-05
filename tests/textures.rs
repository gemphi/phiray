//! The texture nodes beyond Noise fBm and Voronoi F1 (`noise.wgsl`,
//! `voronoi.wgsl`, `pattern.wgsl`): each on a flat wall's base colour under
//! Material Preview, which settles in one pass. A Checker is its two
//! colours and nothing between; a Brick shows its mortar; a Linear
//! Gradient rises left to right; Magic is coloured; a Wave varies; every
//! Noise type and Voronoi feature varies and differs from the defaults;
//! F2 lies farther than F1.

use phiray::proc::*;
use phiray::{Atlas, Camera, Mode, Proc, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

fn sky_atlas() -> Atlas {
    let (w, h) = (32u32, 16u32);
    let rgb: Vec<f32> = (0..h).flat_map(|y| (0..w).flat_map(move |_| if y < h / 2 { [0.9; 3] } else { [0.05; 3] })).collect();
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    atlas.add_env_lighting(w, h, &rgb);
    atlas
}

fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        dist: 4.0,
        units: 1.0,
        persp: 1,
        scale: 1.0,
        aspect: 1.0,
        sky: [0.24, 0.24, 0.24, 0.0],
        sun: [0.0, -0.7, 0.7],
        env: [1.0; 4],
        envmap: 1,
        env_cdf: 2,
        env_hidden: 1,
        preview: [3, 4, phiray::trace::ibl::LEVELS, 0],
        ..Default::default()
    }
}

/// A wall filling the frame with `proc` on its base colour, at a scale
/// the frame shows (Generated runs 0 ..= 1 over the wall, the frame sees
/// about a twentieth of it).
fn frame(gpu: &phiray::Gpu, proc: Proc) -> Vec<u8> {
    let n = [0.0, -1.0, 0.0];
    let wall = Tri::new([[-9.0, 0.0, -9.0], [9.0, 0.0, -9.0], [0.0, 0.0, 12.0]], [n; 3], [1.0; 4]);
    shot(gpu, wall, proc, camera())
}

/// The same, a floor seen from above: a pattern in X and Y (Brick) needs
/// a surface across both — on the XZ wall its Generated Y is one value.
fn floor_frame(gpu: &phiray::Gpu, proc: Proc) -> Vec<u8> {
    let n = [0.0, 0.0, 1.0];
    let floor = Tri::new([[-9.0, -9.0, 0.0], [9.0, -9.0, 0.0], [0.0, 12.0, 0.0]], [n; 3], [1.0; 4]);
    shot(gpu, floor, proc, Camera { up: [0.0, 1.0, 0.0], back: [0.0, 0.0, 1.0], ..camera() })
}

fn shot(gpu: &phiray::Gpu, tri: Tri, proc: Proc, cam: Camera) -> Vec<u8> {
    let tri = tri.with_pbr(0.8, 0.0).with_procs(Some(proc), None);
    let mut tr = Tracer::from_scene(gpu, Scene::from_tris(&[tri], &sky_atlas())).expect("scene");
    tr.set_mode(Mode::Preview);
    tr.set_samples(1);
    let (mut px, mut done) = tr.render(gpu, cam, W, W);
    while !done {
        (px, done) = tr.render(gpu, cam, W, W);
    }
    px
}

fn luma(p: &[u8]) -> f32 {
    0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
}

fn spread(px: &[u8]) -> (f32, f32) {
    let l: Vec<f32> = px.chunks(4).map(luma).collect();
    let mean = l.iter().sum::<f32>() / l.len() as f32;
    (mean, (l.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l.len() as f32).sqrt())
}

#[test]
fn checker_brick_gradient_magic_and_wave() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    // Checker, red and blue: every pixel is one or the other.
    let checker = Proc { scale: 40.0, color1: [1.0, 0.0, 0.0], color2: [0.0, 0.0, 1.0], ..Proc::checker(1) }.colored();
    let px = frame(&gpu, checker);
    let (red, blue) = (px.chunks(4).filter(|p| p[0] > p[2] + 60).count(), px.chunks(4).filter(|p| p[2] > p[0] + 60).count());
    assert!(red > 100 && blue > 100 && red + blue > px.len() / 4 - 60, "two colours, nothing between but the cells' edges: {red} red, {blue} blue");
    // Brick on a floor: light bricks, black mortar between (0.03 wide, a
    // pixel or two at this scale — the frame sees about a world unit, a
    // twentieth of the floor), a minority of the frame. Black mortar under
    // the sky's specular reads about 38.
    let px = floor_frame(&gpu, Proc { scale: 20.0, smoothness: 0.03, ..Proc::brick(1) });
    let (dark, n) = (px.chunks(4).filter(|p| luma(p) < 80.0).count(), px.len() / 4);
    assert!(dark > n / 20 && dark < n / 2, "the mortar is a minority of dark pixels: {dark} of {n}");
    // Linear Gradient: Generated x, rising left to right along a row —
    // never falling, and by a few levels over the twentieth of the wall
    // the frame sees.
    let px = frame(&gpu, Proc::gradient(1));
    let row: Vec<f32> = px[24 * 4 * W as usize..25 * 4 * W as usize].chunks(4).map(luma).collect();
    assert!(row.windows(2).all(|w| w[1] >= w[0]) && row[W as usize - 1] > row[0] + 3.0, "rising: {row:?}");
    // Magic: coloured.
    let px = frame(&gpu, Proc { scale: 20.0, ..Proc::magic(1) });
    assert!(px.chunks(4).any(|p| (p[0] as i32 - p[2] as i32).abs() > 40), "Magic is coloured");
    // Wave: bands vary.
    let (_, dev) = spread(&frame(&gpu, Proc { scale: 10.0, ..Proc::wave(1) }));
    assert!(dev > 10.0, "Wave varies: {dev}");
}

#[test]
fn every_noise_type_and_voronoi_feature_varies() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let fbm = frame(&gpu, Proc { scale: 40.0, ..Proc::noise(1) });
    for variant in [0, 2, 3, 4] {
        let px = frame(&gpu, Proc { scale: 40.0, variant, offset: 0.5, gain: 2.0, ..Proc::noise(1) });
        let (_, dev) = spread(&px);
        let differs = px.iter().zip(&fbm).filter(|(a, b)| a.abs_diff(**b) > 8).count();
        assert!(dev > 3.0 && differs > 200, "noise type {variant}: deviation {dev}, {differs} channels unlike fBm");
    }
    let f1 = spread(&frame(&gpu, Proc { scale: 40.0, ..Proc::voronoi(1) }));
    let f2 = spread(&frame(&gpu, Proc { scale: 40.0, variant: 1, ..Proc::voronoi(1) }));
    assert!(f2.0 > f1.0 + 5.0, "F2 lies farther than F1: {f2:?} {f1:?}");
    for (variant, metric) in [(2, 0), (3, 0), (4, 0), (0, 1), (0, 2), (0, 3)] {
        let (_, dev) = spread(&frame(&gpu, Proc { scale: 40.0, variant, direction: metric, ..Proc::voronoi(1) }));
        assert!(dev > 3.0, "voronoi feature {variant} metric {metric}: deviation {dev}");
    }
    let (_, dev) = spread(&frame(&gpu, Proc { scale: 40.0, kind: PROC_VORONOI_RADIUS, variant: 4, ..Proc::voronoi(1) }));
    assert!(dev > 3.0, "N-Sphere Radius varies: {dev}");
}
