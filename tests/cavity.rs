//! Cavity ▸ World — Workbench's screen-space ambient occlusion
//! (`workbench_cavity.bsl.hh`'s `cavity_compute`, `Camera::cavity`,
//! `::ssao`): the wall beside a block standing out of it darkens (the
//! block's side faces occlude it), the block's rim brightens (an edge),
//! and the open wall far from anything is left as it was; off, nothing
//! changes. Flat lighting keeps the base colour uniform so the effect is
//! all that differs.

use phiray::{Camera, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

/// Orthographic, looking along +y at a frame four units across (a pixel is
/// a twelfth of a unit); Flat lighting, half grey.
fn camera() -> Camera {
    Camera { right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], units: 1.0, scale: 0.25, aspect: 1.0, dist: 10.0, sky: [0.3, 0.3, 0.3, 0.0], env: [0.2, 0.22, 0.25, 1.0], shade: [2, 0, 0, 0], ..Default::default() }
}

/// SSAO on: 16 samples a frame over 8 frames, half a unit wide, the
/// factors and attenuation 1.
fn ssao() -> Camera {
    Camera { cavity: [1.0, 1.0, 1.0, 0.5], ssao: [16, 8, 0, 0], ..camera() }
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

#[test]
fn the_wall_beside_the_block_darkens_and_its_rim_brightens() {
    let Some(gpu) = Ray::gpu() else { return };
    let plain = frame(&gpu, camera());
    let shaded = frame(&gpu, ssao());
    let mid = W / 2;
    // Flat: the plain frame is one grey over wall and block alike.
    let base = grey(&plain, 4, mid);
    assert_eq!(grey(&plain, mid, mid), base, "Flat lighting: the block is the wall's grey");
    assert_eq!(grey(&plain, mid + 7, mid), base);
    // The wall a pixel out from the block's side (the block spans pixels
    // 18 ..= 29) sits in a cavity.
    let corner = grey(&shaded, mid + 7, mid);
    assert!(corner < base - 10, "the cavity darkens the wall beside the block: {corner} vs {base}");
    // The block's face at its rim stands out from the wall: an edge. (The
    // last pixel inside, 29, straddles the edge under the anti-aliasing
    // jitter and mixes rim and cavity.)
    let rim = grey(&shaded, mid + 4, mid);
    assert!(rim > base + 5, "the rim brightens: {rim} vs {base}");
    // The open wall far from the block, within a frame's noise.
    let open = grey(&shaded, 4, mid);
    assert!((open - base).abs() <= 3, "the open wall is left alone: {open} vs {base}");
    // The frame is one of eight runs of the disk, all drawn: still stable
    // from frame to frame once converged.
    let again = frame(&gpu, ssao());
    assert_eq!(again, shaded, "deterministic");
}

#[test]
fn ssao_off_or_without_distance_changes_nothing() {
    let Some(gpu) = Ray::gpu() else { return };
    let plain = frame(&gpu, camera());
    let off = frame(&gpu, Camera { cavity: [1.0, 1.0, 1.0, 0.5], ssao: [0, 8, 0, 0], ..camera() });
    assert_eq!(off, plain, "no samples: the effects pass does not run");
    let near = frame(&gpu, Camera { cavity: [1.0, 1.0, 1.0, 0.0], ..ssao() });
    let mid = W / 2;
    assert_eq!(grey(&near, mid + 7, mid), grey(&plain, mid + 7, mid), "a zero distance samples the pixel itself: no occlusion");
}

