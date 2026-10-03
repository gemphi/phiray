//! Solid's Lighting and Color as the Shading popover sets them
//! (`Camera::shade`, `::single`): Color ▸ Object shades each object with
//! its own colour, ▸ Random with a hue hashed from its name
//! (`hash::hue_of`), ▸ Single with the one colour, each at roughness 0.4
//! and no metal (`workbench_engine.cc`'s `get_material`); Lighting ▸ Flat
//! is the colour itself, unlit (`WORKBENCH_LIGHTING_FLAT`). The default
//! words draw the frame of before.

use phiray::trace::hash;
use phiray::{Atlas, Camera, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

fn camera() -> Camera {
    Camera { right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], units: 1.0, scale: 0.25, aspect: 1.0, dist: 10.0, sky: [0.3, 0.3, 0.3, 0.0], env: [0.2, 0.22, 0.25, 1.0], ..Default::default() }
}

/// Two grey walls side by side, "Left" red by object colour and "Right" blue.
fn scene() -> Scene {
    let mut s = Scene::default();
    let n = [0.0, -1.0, 0.0];
    for (x, name, color) in [(-1.0f32, "Left", [1.0, 0.0, 0.0]), (1.0, "Right", [0.0, 0.0, 1.0])] {
        s.instance_of([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [x, 0.0, 0.0, 1.0]], color, hash::object_hash(name));
        let q = [[-0.9, 0.0, -1.5], [0.9, 0.0, -1.5], [0.9, 0.0, 1.5], [-0.9, 0.0, 1.5]];
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let t = Tri::new(tri, [n; 3], [0.5, 0.5, 0.5, 1.0]);
            let m = s.material(Mat::of(&t));
            let v = tri.map(|p| s.vertex(p, n, [0.0; 2]));
            s.face(v, m);
        }
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

fn at(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

#[test]
fn color_object_random_and_single_shade_as_blender_does() {
    let Some(gpu) = Ray::gpu() else { return };
    let (l, r) = (W / 4, 3 * W / 4);
    let plain = frame(&gpu, camera());
    assert_eq!(at(&plain, l, W / 2), at(&plain, r, W / 2), "Material: both walls the same grey");
    let object = frame(&gpu, Camera { shade: [0, 4, 0, 0], ..camera() });
    let (pl, pr) = (at(&object, l, W / 2), at(&object, r, W / 2));
    assert!(pl[0] > pl[2] + 60 && pr[2] > pr[0] + 60, "Object: red left, blue right: {pl:?} {pr:?}");
    let random = frame(&gpu, Camera { shade: [0, 1, 0, 0], ..camera() });
    let (rl, rr) = (at(&random, l, W / 2), at(&random, r, W / 2));
    assert_ne!(rl, rr, "Random: the two names hash to two hues");
    // The hue the kernel drew is the hue `hash.rs` computes: the dominant
    // channel of saturation-0.5 HSV follows the hue sextant.
    let sextant = |name: &str| (hash::hue_of(hash::object_hash(name)) * 6.0).floor() as u32 % 6;
    let dominant = |c: [u8; 3]| (0..3).max_by_key(|&k| c[k]).unwrap();
    for (name, c) in [("Left", rl), ("Right", rr)] {
        let expect = match sextant(name) { 0 | 5 => 0, 1 | 2 => 1, _ => 2 };
        assert_eq!(dominant(c), expect, "{name}: hue sextant {} drew {c:?}", sextant(name));
    }
    let single = frame(&gpu, Camera { shade: [0, 2, 0, 0], single: [0.0, 0.6, 0.0, 1.0], ..camera() });
    let (sl, sr) = (at(&single, l, W / 2), at(&single, r, W / 2));
    assert_eq!(sl, sr, "Single: one colour");
    assert!(sl[1] > sl[0] + 60, "and it is the green asked: {sl:?}");
    assert_eq!(at(&single, 1, 1), at(&plain, 1, 1), "the sky is not a material");
}

#[test]
fn flat_lighting_is_the_colour_itself() {
    let Some(gpu) = Ray::gpu() else { return };
    let flat = frame(&gpu, Camera { shade: [2, 4, 0, 0], ..camera() });
    let (top, mid, low) = (at(&flat, W / 4, 8), at(&flat, W / 4, W / 2), at(&flat, W / 4, W - 8));
    assert_eq!(top, mid, "no shading across the wall");
    assert_eq!(mid, low);
    // Linear red 1.0 shows as 255, the rest 0 — the colour, untouched.
    assert_eq!(mid, [255, 0, 0], "{mid:?}");
    let lit = frame(&gpu, Camera { shade: [0, 4, 0, 0], ..camera() });
    assert_ne!(at(&lit, W / 4, W / 2), mid, "Studio lighting shades it");
}
