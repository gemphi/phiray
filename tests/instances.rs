//! Objects as instances: each mesh stays in its own space under a tree of
//! its own, and a matrix places it. Moving an object is a new matrix
//! (`Tracer::reinstance`), not a new scene — and draws what a scene baked
//! in world space draws.

use phiray::{Atlas, Camera, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

/// Looking down −Y, orthographic, no floor grid in shot.
fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        units: 1.0,
        scale: 0.1,
        aspect: 1.0,
        persp: 0,
        dist: 10.0,
        sky: [0.2, 0.22, 0.25, 0.0],
        env: [0.2, 0.22, 0.25, 1.0],
        ..Default::default()
    }
}

/// A 2 × 2 wall at the origin, facing the camera, in its own space.
fn wall(c: f32) -> [Tri; 2] {
    let q = [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let n = [0.0, -1.0, 0.0];
    [Tri::new([q[0], q[1], q[2]], [n; 3], [c, c, c, 1.0]), Tri::new([q[0], q[2], q[3]], [n; 3], [c, c, c, 1.0])]
}

/// A scale of `s`, a turn of `turn` radians about Y (the view axis), moved to `t`.
fn placed(s: f32, turn: f32, t: [f32; 3]) -> [[f32; 4]; 4] {
    let (sn, cs) = turn.sin_cos();
    [[s * cs, 0.0, -s * sn, 0.0], [0.0, s, 0.0, 0.0], [s * sn, 0.0, s * cs, 0.0], [t[0], t[1], t[2], 1.0]]
}

fn to_world(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r])
}

/// Two walls, the dim one at the left and the bright one placed by `right`:
/// as instances, and baked into world space.
fn scenes(right: [[f32; 4]; 4]) -> (Scene, Scene) {
    let worlds = [placed(1.0, 0.0, [-2.0, 0.0, 0.0]), right];
    let mut inst = Scene::default();
    let mut baked = Scene::default();
    for (world, c) in worlds.iter().zip([0.3, 0.8]) {
        inst.instance(*world);
        for tri in wall(c) {
            let m = inst.material(Mat::of(&tri));
            let v = [tri.a, tri.b, tri.c].map(|p| inst.vertex(p, tri.na, [0.0; 2]));
            inst.face(v, m);
            let w = [tri.a, tri.b, tri.c].map(|p| to_world(world, p));
            baked.add(&Tri::new(w, [tri.na; 3], tri.color));
        }
    }
    (inst, baked)
}

fn solid(gpu: &phiray::Gpu, scene: Scene) -> Tracer {
    let mut t = Tracer::from_scene(gpu, scene).expect("a scene");
    t.set_mode(Mode::Solid);
    t
}

/// Pixels that differ by more than a step between two images.
fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(p, q)| p.iter().zip(*q).any(|(x, y)| x.abs_diff(*y) > 2)).count()
}

#[test]
fn an_instanced_scene_draws_what_the_baked_one_draws() {
    let Some(gpu) = Ray::gpu() else { return };
    let (inst, baked) = scenes(placed(1.5, 0.4, [2.0, 0.5, 0.0]));
    let a = solid(&gpu, inst).render(&gpu, camera(), W, W).0;
    let b = solid(&gpu, baked).render(&gpu, camera(), W, W).0;
    let off = differing(&a, &b);
    assert!(off <= (W * W / 100) as usize, "{off} pixels differ (edges may round differently; the walls may not)");
    assert!(a.chunks_exact(4).any(|p| p[0] > 150), "the bright wall is in the frame");
}

#[test]
fn moving_an_instance_is_a_new_matrix_and_the_image_starts_over() {
    let Some(gpu) = Ray::gpu() else { return };
    let (inst, _) = scenes(placed(1.0, 0.0, [2.0, 0.0, 0.0]));
    let mut moved = solid(&gpu, inst);
    let before = moved.render(&gpu, camera(), W, W).0;
    let far = placed(0.5, 1.2, [1.0, 0.0, 2.0]);
    assert!(moved.reinstance(&gpu, &[placed(1.0, 0.0, [-2.0, 0.0, 0.0]), far]));
    let (sample, _) = moved.sample(&gpu, camera(), W, W);
    assert_eq!(sample, 0, "a moved object: the image starts over");
    let after = moved.pixels(&gpu);
    assert_ne!(before, after, "the bright wall moved");
    let fresh = solid(&gpu, scenes(far).1).render(&gpu, camera(), W, W).0;
    let off = differing(&after, &fresh);
    assert!(off <= (W * W / 100) as usize, "{off} pixels differ from a scene baked at the new place");
    assert!(!moved.reinstance(&gpu, &[far]), "one matrix for two objects: build again");
}

#[test]
fn a_new_tree_after_a_move_draws_the_same_and_keeps_its_samples() {
    let Some(gpu) = Ray::gpu() else { return };
    let (inst, _) = scenes(placed(1.0, 0.0, [2.0, 0.0, 0.0]));
    let mut t = Tracer::from_scene(&gpu, inst).expect("a scene");
    assert!(t.reinstance(&gpu, &[placed(1.0, 0.0, [-2.0, 0.0, 0.0]), placed(1.0, 0.0, [0.0, 0.0, 2.0])]));
    (0..3).for_each(|_| _ = t.sample(&gpu, camera(), W, W));
    let kept = t.pixels(&gpu);
    assert!(t.retree(&gpu));
    assert_eq!(t.sample(&gpu, camera(), W, W).0, 3, "the fourth sample, not the first again");
    assert_ne!(kept, t.pixels(&gpu), "and it refined");
    t.set_mode(Mode::Solid);
    let drawn = t.render(&gpu, camera(), W, W).0;
    let fresh = solid(&gpu, scenes(placed(1.0, 0.0, [0.0, 0.0, 2.0])).1).render(&gpu, camera(), W, W).0;
    assert!(differing(&drawn, &fresh) <= (W * W / 100) as usize);
    // An unplaced scene draws as it did before instances: one at the identity.
    let plain = Tracer::new(&gpu, &wall(0.8), &Atlas::default()).expect("a scene");
    assert_eq!(plain.extent.round(), 3.0, "the wall's diagonal");
}
