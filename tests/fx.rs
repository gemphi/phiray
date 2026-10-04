//! Workbench's Solid effects (`fx.wgsl`): the object outline draws
//! `object_outline_color` where a pixel's neighbours belong to another
//! object — along each object's edge and the background beside it — a
//! quarter per neighbour (`workbench_effect_outline_frag.glsl`), and the
//! curvature effect brightens a ridge and darkens a valley from the slope
//! of the normals (`curvature_compute`), never across an object's edge.
//! With both off the frame is the frame of before, pixel for pixel.

use phiray::{Camera, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 64;

fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        pivot: [0.0, 0.0, 0.0],
        units: 1.0,
        scale: 0.2,
        aspect: 1.0,
        persp: 0,
        dist: 10.0,
        sky: [0.3, 0.3, 0.3, 0.0],
        env: [0.2, 0.22, 0.25, 1.0],
        grid: 0.0,
        ..Default::default()
    }
}

/// Two square walls facing the viewer, each its own instance: a large grey
/// one and, in front of its right half, a smaller one. `bend` tilts the
/// large wall's vertex normals outward (+1, a ridge) or inward (−1, a
/// valley) left to right, so its surface reads as curved.
fn scene(bend: f32) -> Scene {
    let mut s = Scene::default();
    let n = [0.0, -1.0, 0.0];
    let mut wall = |s: &mut Scene, y: f32, half: f32, c: f32, world_x: f32, normals: [[f32; 3]; 4]| {
        s.instance([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [world_x, y, 0.0, 1.0]]);
        let q = [[-half, 0.0, -half], [half, 0.0, -half], [half, 0.0, half], [-half, 0.0, half]];
        for (tri, nn) in [([q[0], q[1], q[2]], [normals[0], normals[1], normals[2]]), ([q[0], q[2], q[3]], [normals[0], normals[2], normals[3]])] {
            let t = Tri::new(tri, nn, [c, c, c, 1.0]);
            let m = s.material(Mat::of(&t));
            let v = [(tri[0], nn[0]), (tri[1], nn[1]), (tri[2], nn[2])].map(|(p, nv)| s.vertex(p, nv, [0.0; 2]));
            s.face(v, m);
        }
    };
    let tilt = |x: f32| {
        let v = [x * bend * 0.6, -1.0, 0.0];
        let l = (v[0] * v[0] + 1.0).sqrt();
        [v[0] / l, v[1] / l, 0.0]
    };
    // Left vertices tilt one way, right vertices the other.
    wall(&mut s, 0.0, 2.0, 0.6, 0.0, [tilt(-1.0), tilt(1.0), tilt(1.0), tilt(-1.0)]);
    wall(&mut s, -1.0, 0.7, 0.6, 1.0, [n; 4]);
    s
}

fn frame(gpu: &phiray::Gpu, scene: Scene, cam: Camera) -> Vec<u8> {
    let mut t = Tracer::from_scene(gpu, scene).expect("a scene");
    t.set_mode(Mode::Solid);
    let (mut px, mut done) = t.render(gpu, cam, W, W);
    while !done {
        (px, done) = t.render(gpu, cam, W, W);
    }
    px
}

fn at(px: &[u8], x: u32, y: u32) -> u8 {
    px[((y * W + x) * 4) as usize]
}

#[test]
fn the_outline_runs_along_each_objects_edge_and_the_frame_is_otherwise_the_same() {
    let Some(gpu) = Ray::gpu() else { return };
    let plain = frame(&gpu, scene(0.0), camera());
    let outlined = frame(&gpu, scene(0.0), Camera { fx: [1.0, 0.0, 0.0, 1.0], outline: [0.0, 0.0, 0.0, 1.0], ..camera() });
    // 12.8 pixels a unit about column 32: the large wall's edges fall at
    // columns 6 and 58, the small wall's (x 0.3..1.7) at 36 and 54.
    // The line is a quarter to a half black per pixel, and the eight
    // jittered samples soften it over its neighbours, as Blender's do.
    let edge = (0..W).filter(|&x| at(&outlined, x, W / 2) + 8 < at(&plain, x, W / 2)).collect::<Vec<_>>();
    assert!(edge.len() >= 4 && edge.len() <= 16, "a dark line at each edge, a pixel or two wide: {edge:?}");
    for near in [6u32, 36, 54, 58] {
        assert!(edge.iter().any(|&x| x.abs_diff(near) <= 2), "an edge about column {near}: {edge:?}");
    }
    // Interiors, and the background away from any edge, are untouched.
    for (x, y) in [(45, W / 2), (14, W / 2), (W / 2, 20), (2, 2)] {
        assert_eq!(at(&outlined, x, y), at(&plain, x, y), "at {x},{y}");
    }
    // The large wall's edge against the sky is outlined on the sky's side
    // too (Blender's outline straddles the edge): the sky pixel at column
    // 5 or 4 is darker than the sky.
    assert!((3..=5).any(|x| at(&outlined, x, W / 2) + 5 < at(&plain, x, W / 2)), "the sky beside the wall darkens");
    // Outline off: identical.
    assert_eq!(frame(&gpu, scene(0.0), Camera { fx: [0.0, 0.0, 0.0, 1.0], ..camera() }), plain);
}

#[test]
fn curvature_brightens_a_ridge_and_darkens_a_valley_within_an_object() {
    let Some(gpu) = Ray::gpu() else { return };
    let factors = |ridge: f32, valley: f32| [0.0, 0.5 / ridge.powi(2).max(1e-4), 0.7 / valley.powi(2).max(1e-4), 1.0];
    let flat = frame(&gpu, scene(0.0), Camera { fx: factors(1.0, 1.0), ..camera() });
    let flat_off = frame(&gpu, scene(0.0), camera());
    assert_eq!(flat, flat_off, "a flat surface has no curvature: nothing changes");
    let ridge = frame(&gpu, scene(1.0), Camera { fx: factors(1.0, 1.0), ..camera() });
    let ridge_off = frame(&gpu, scene(1.0), camera());
    let valley = frame(&gpu, scene(-1.0), Camera { fx: factors(1.0, 1.0), ..camera() });
    let valley_off = frame(&gpu, scene(-1.0), camera());
    // Inside the large wall, left of the small one.
    let (x, y) = (14, W / 2);
    assert!(at(&ridge, x, y) > at(&ridge_off, x, y), "ridge: {} > {}", at(&ridge, x, y), at(&ridge_off, x, y));
    assert!(at(&valley, x, y) < at(&valley_off, x, y), "valley: {} < {}", at(&valley, x, y), at(&valley_off, x, y));
    // The small flat wall in front is untouched, and so is the sky.
    assert_eq!(at(&ridge, 40, y), at(&ridge_off, 40, y));
    assert_eq!(at(&ridge, 2, 2), at(&ridge_off, 2, 2));
    // A stronger ridge factor clamps sooner: no brighter than factor 1.
    let strong = frame(&gpu, scene(1.0), Camera { fx: factors(2.0, 1.0), ..camera() });
    assert!(at(&strong, x, y) >= at(&ridge, x, y), "{} vs {}", at(&strong, x, y), at(&ridge, x, y));
}

