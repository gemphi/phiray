//! Workbench's Shadow (`workbench_shadow.cc`, `get_shadow`; `Camera::
//! shadow`, `::shadow_fade`): the one directional light's shadow over the
//! opaque surfaces — a block standing out of a wall casts onto the wall
//! beside it, the open wall keeps the terminator's factor, and the
//! intensity says how dark; off, nothing changes. Flat lighting keeps the
//! base colour uniform so the shadow is all that differs.

use phiray::{Camera, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

/// Orthographic, looking along +y at a frame four units across; Flat
/// lighting, half grey.
fn camera() -> Camera {
    Camera { right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], units: 1.0, scale: 0.25, aspect: 1.0, dist: 10.0, sky: [0.3, 0.3, 0.3, 0.0], env: [0.2, 0.22, 0.25, 1.0], shade: [2, 0, 0, 0], ..Default::default() }
}

/// Shadow on at `intensity`: the light travelling along (1, 1, 0) — from
/// the viewer's side, slanting to +x — with Blender's default terminator
/// (shift 0.1, focus 0 → 0.99991).
fn shadowed(intensity: f32) -> Camera {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    Camera { shadow: [s, s, 0.0, intensity], shadow_fade: [0.1, 1.0 - 0.0001 * 0.9, 0.0, 0.0], ..camera() }
}

/// A wall at `y = 0` facing the viewer with a one-unit block standing a
/// unit out of its middle, all one grey.
fn scene() -> Scene {
    let mut s = Scene::default();
    let mut quad = |q: [[f32; 3]; 4], n: [f32; 3]| {
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let t = Tri::new(tri, [n; 3], [0.5, 0.5, 0.5, 1.0]);
            let m = s.material(Mat::of(&t));
            let v = tri.map(|p| s.vertex(p, n, [0.0; 2]));
            s.face(v, m);
        }
    };
    quad([[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]], [0.0, -1.0, 0.0]);
    let (a, b) = (-0.5, 0.5);
    quad([[a, -1.0, a], [b, -1.0, a], [b, -1.0, b], [a, -1.0, b]], [0.0, -1.0, 0.0]);
    quad([[b, 0.0, a], [b, -1.0, a], [b, -1.0, b], [b, 0.0, b]], [1.0, 0.0, 0.0]);
    quad([[a, 0.0, a], [a, -1.0, a], [a, -1.0, b], [a, 0.0, b]], [-1.0, 0.0, 0.0]);
    quad([[a, 0.0, b], [b, 0.0, b], [b, -1.0, b], [a, -1.0, b]], [0.0, 0.0, 1.0]);
    quad([[a, 0.0, a], [b, 0.0, a], [b, -1.0, a], [a, -1.0, a]], [0.0, 0.0, -1.0]);
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

/// Display grey of scene-linear `v`.
fn srgb(v: f32) -> i32 {
    i32::from(phiray::Srgb::linear_to_byte(v))
}

#[test]
fn the_block_casts_onto_the_wall_and_the_terminator_shades_the_rest() {
    let Some(gpu) = Ray::gpu() else { return };
    let plain = frame(&gpu, camera());
    let half = frame(&gpu, shadowed(0.5));
    let mid = W / 2;
    let base = grey(&plain, 6, mid);
    assert_eq!(base, srgb(0.5), "Flat: half grey");
    // The wall faces the light at `−N · dir = 1/√2`: `smoothstep(0.1,
    // 0.99991, 0.707)` ≈ 0.747, so lit wall takes `0.747 · 0.5 + 0.5`.
    let t = (std::f32::consts::FRAC_1_SQRT_2 - 0.1) / (1.0 - 0.0001 * 0.9 - 0.1);
    let mix = t * t * (3.0 - 2.0 * t);
    let open = grey(&half, 6, mid);
    assert!((open - srgb(0.5 * (mix * 0.5 + 0.5))).abs() <= 1, "the open wall keeps the terminator's factor: {open}");
    // Light slanting to +x: the block (x in −½ ..= ½, a unit out) shadows
    // the wall at x in ½ ..= 1½ — pixel 36 is x = 1 — to `1 − intensity`.
    let shadow = grey(&half, 36, mid);
    assert!((shadow - srgb(0.25)).abs() <= 1, "in the block's shadow: {shadow}");
    // The other side of the block is lit like the open wall.
    assert_eq!(grey(&half, 12, mid), open);
    // The block's own face toward the viewer faces the light the same way.
    assert_eq!(grey(&half, mid, mid), open);
    // Full intensity: the shadow is black.
    let full = frame(&gpu, shadowed(1.0));
    assert_eq!(grey(&full, 36, mid), 0);
    assert!((grey(&full, 6, mid) - srgb(0.5 * mix)).abs() <= 1);
    // The shadow's extent: pixels 31 ..= 40 are x in 0.63 ..= 1.46, its
    // edges softened by the anti-aliasing jitter either side.
    assert!((31..=40).all(|x| grey(&half, x, mid) == shadow), "{:?}", (28..43).map(|x| grey(&half, x, mid)).collect::<Vec<_>>());
    assert_eq!(grey(&half, 43, mid), open);
}

#[test]
fn shadow_off_changes_nothing() {
    let Some(gpu) = Ray::gpu() else { return };
    let plain = frame(&gpu, camera());
    let off = frame(&gpu, shadowed(0.0));
    assert_eq!(off, plain);
}


/// An instance told to cast no shadow (`Scene::hide_shadow` — Viewport
/// Display ▸ Shadow off, `OB_DRAW_NO_SHADOW_CAST`): the wall behind it is
/// lit as the open wall, the block itself still shaded.
#[test]
fn an_instance_that_casts_no_shadow_leaves_the_wall_lit() {
    let Some(gpu) = Ray::gpu() else { return };
    // The block as its own instance, so it alone can be told.
    let mut s = Scene::default();
    let quad = |s: &mut Scene, q: [[f32; 3]; 4], n: [f32; 3]| {
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let t = Tri::new(tri, [n; 3], [0.5, 0.5, 0.5, 1.0]);
            let m = s.material(Mat::of(&t));
            let v = tri.map(|p| s.vertex(p, n, [0.0; 2]));
            s.face(v, m);
        }
    };
    quad(&mut s, [[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]], [0.0, -1.0, 0.0]);
    let block = s.instance_of([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [1.0; 3], phiray::trace::hash::ObjectHash(0));
    let (a, b) = (-0.5, 0.5);
    quad(&mut s, [[a, -1.0, a], [b, -1.0, a], [b, -1.0, b], [a, -1.0, b]], [0.0, -1.0, 0.0]);
    quad(&mut s, [[b, 0.0, a], [b, -1.0, a], [b, -1.0, b], [b, 0.0, b]], [1.0, 0.0, 0.0]);
    quad(&mut s, [[a, 0.0, a], [a, -1.0, a], [a, -1.0, b], [a, 0.0, b]], [-1.0, 0.0, 0.0]);
    s.hide_shadow(block);
    let mut t = Tracer::from_scene(&gpu, s).expect("a scene");
    t.set_mode(Mode::Solid);
    let cam = shadowed(0.5);
    let (mut px, mut done) = t.render(&gpu, cam, W, W);
    while !done {
        (px, done) = t.render(&gpu, cam, W, W);
    }
    let mid = W / 2;
    assert_eq!(grey(&px, 36, mid), grey(&px, 6, mid), "no shadow falls on the wall");
    assert_ne!(grey(&px, 36, mid), srgb(0.25));
}
