//! Workbench's Depth of Field (`workbench_effect_dof.cc`; `Camera::dof`):
//! a striped card standing out of a light wall, looked at through a
//! camera with an aperture. Focused on the card, its stripes stay crisp;
//! focused on the wall far behind it, they smear toward their mean — the
//! gather's blur blended in where the circle of confusion is wide — and
//! a wider aperture smears them more; the card's edge against the wall in
//! focus softens on the card's side alone, as Workbench's gather has it
//! (a pixel gathers only what reaches it: the sharp wall does not bleed
//! over the card, and the wall's own radius is nought). Off, the frame is
//! the plain one. Flat lighting keeps the greys what the stripes say.

use phiray::{Camera, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 64;

/// Perspective, looking along +y from `y = −4` at a frame four units
/// across at the card; Flat lighting.
fn camera() -> Camera {
    Camera { right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], pivot: [0.0, 0.0, 0.0], units: 1.0, persp: 1, scale: 0.25, aspect: 1.0, dist: 4.0, sky: [0.3, 0.3, 0.3, 0.0], env: [0.2, 0.22, 0.25, 1.0], shade: [2, 0, 0, 0], ..Default::default() }
}

/// Depth of Field on, focused `focus` units ahead of the eye: a 100 mm
/// lens at f/0.5 on a 36 mm sensor (`DofPass::init`'s words — wide open,
/// so a 64-pixel frame shows a circle of confusion of about 3 pixels),
/// the far plane at 100.
fn with_dof(focus: f32) -> Camera {
    let (lens, fstop, sensor) = (0.1_f32, 0.5_f32, 0.036_f32);
    let aperture = 0.5 * lens / fstop;
    let aperture_size = aperture * (lens / (focus - lens)).abs();
    Camera { dof: [aperture_size, -focus, 1.0 / sensor, -100.0], ..camera() }
}

/// A light wall at `y = 8` and a card at `y = 0` filling the left of the
/// frame, its right edge down the middle, in vertical stripes a quarter
/// unit (four pixels) wide, dark and mid grey by turns.
fn scene() -> Scene {
    let mut s = Scene::default();
    let mut quad = |q: [[f32; 3]; 4], grey: f32| {
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let t = Tri::new(tri, [[0.0, -1.0, 0.0]; 3], [grey, grey, grey, 1.0]);
            let m = s.material(Mat::of(&t));
            let v = tri.map(|p| s.vertex(p, [0.0, -1.0, 0.0], [0.0; 2]));
            s.face(v, m);
        }
    };
    quad([[-30.0, 8.0, -30.0], [30.0, 8.0, -30.0], [30.0, 8.0, 30.0], [-30.0, 8.0, 30.0]], 0.8);
    for k in 0..24 {
        let (a, b) = (-6.0 + 0.25 * k as f32, -6.0 + 0.25 * (k + 1) as f32);
        quad([[a, 0.0, -6.0], [b, 0.0, -6.0], [b, 0.0, 6.0], [a, 0.0, 6.0]], if k % 2 == 0 { 0.1 } else { 0.6 });
    }
    s
}

fn frame(gpu: &phiray::Gpu, cam: Camera) -> Vec<u8> {
    let mut t = Tracer::from_scene(gpu, scene()).expect("a scene");
    t.set_mode(Mode::Solid);
    let (mut px, mut done) = t.render(gpu, cam, W, W);
    while !done {
        (px, done) = t.render(gpu, cam, W, W);
    }
    px
}

fn grey(px: &[u8], x: u32, y: u32) -> i32 {
    let i = ((y * W + x) * 4) as usize;
    i32::from(px[i])
}

/// The stripes' contrast along the middle row: the greys' span over the
/// card well inside its edge.
fn contrast(px: &[u8]) -> i32 {
    let mid = W / 2;
    let greys: Vec<i32> = (2..26).map(|x| grey(px, x, mid)).collect();
    greys.iter().max().unwrap() - greys.iter().min().unwrap()
}

#[test]
fn stripes_out_of_focus_smear_and_in_focus_stay_crisp() {
    let Some(gpu) = Ray::gpu() else { return };
    let plain = frame(&gpu, camera());
    // The eye sits 4 units before the card (`pivot + back · dist`).
    let on_card = frame(&gpu, with_dof(4.0));
    let on_wall = frame(&gpu, with_dof(12.0));
    let wide = frame(&gpu, Camera { dof: [with_dof(12.0).dof[0] * 3.0, -12.0, 1.0 / 0.036, -100.0], ..camera() });
    let (crisp, focused, blurred, wider) = (contrast(&plain), contrast(&on_card), contrast(&on_wall), contrast(&wide));
    assert!(crisp > 100, "dark and mid grey stripes: {crisp}");
    assert!((focused - crisp).abs() <= 4, "focused on the card, its stripes are as crisp: {focused} vs {crisp}");
    assert!(blurred < crisp - 30, "focused on the wall, the stripes smear: {blurred} vs {crisp}");
    assert!(wider < blurred - 10, "a wider aperture smears them more: {wider} vs {blurred}");
    // The smear keeps the card's light in scene-linear terms — the blur
    // runs before the view transform — so six stripes of 0.1 and 0.6
    // average to 0.35: what the stripes lose, their neighbours gain.
    let mid = W / 2;
    let mean = (2..26).map(|x| phiray::Srgb::byte_to_linear(on_wall[((mid * W + x) * 4) as usize])).sum::<f32>() / 24.0;
    assert!((mean - 0.35).abs() <= 0.01, "{mean}");
    // The wall, in focus and uniform, is left as it was; the card's edge
    // against it softens on the card's side (the wall's radius is nought).
    assert_eq!(grey(&on_wall, mid + 2, mid), grey(&plain, mid + 2, mid));
    assert_eq!(grey(&on_wall, W - 5, mid), grey(&plain, W - 5, mid));
    assert!(grey(&on_wall, mid - 2, mid) != grey(&plain, mid - 2, mid) || grey(&on_wall, mid - 1, mid) != grey(&plain, mid - 1, mid));
}

#[test]
fn depth_of_field_off_changes_nothing() {
    let Some(gpu) = Ray::gpu() else { return };
    let plain = frame(&gpu, camera());
    let off = frame(&gpu, Camera { dof: [0.0, -4.0, 1.0 / 0.036, -100.0], ..camera() });
    assert_eq!(off, plain);
}

#[test]
#[ignore]
fn dump_rows() {
    let Some(gpu) = Ray::gpu() else { return };
    let mid = W / 2;
    for (name, cam) in [("plain", camera()), ("on_card", with_dof(4.0)), ("on_wall", with_dof(12.0)), ("wide", Camera { dof: [with_dof(12.0).dof[0] * 3.0, -12.0, 1.0 / 0.036, -100.0], ..camera() })] {
        let px = frame(&gpu, cam);
        let row: Vec<i32> = (0..40).map(|x| grey(&px, x, mid)).collect();
        eprintln!("{name}: {row:?}");
    }
}
