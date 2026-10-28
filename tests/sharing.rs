//! Shared geometry (roadmap §4.4, Step 1): a mesh placed by many instances
//! goes up once — its faces, vertices and tree — and draws everywhere the
//! instances stand, as Cycles draws many `Object`s over one `Geometry`.
//! The image is the one a scene of as many separate copies draws, in the
//! rasterized prepass and traced alike, and the instance count is no
//! longer capped at the 65,536 the face word used to hold.

use phiray::{Camera, Mat, Mode, Ray, Scene, Tracer, Tri};
use phiray::trace::hash::ObjectHash;

const W: u32 = 64;

/// Looking down −Y, orthographic, the grid of walls filling the frame.
fn camera(scale: f32) -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        units: 1.0,
        scale,
        aspect: 1.0,
        persp: 0,
        dist: 10.0,
        sky: [0.2, 0.22, 0.25, 0.0],
        env: [0.2, 0.22, 0.25, 1.0],
        ..Default::default()
    }
}

/// A small wall facing the camera, in its own space: two triangles, its
/// corner normals leaning so the lighting shows the placement.
fn wall() -> [Tri; 2] {
    let q = [[-0.4, 0.0, -0.4], [0.4, 0.0, -0.4], [0.4, 0.0, 0.4], [-0.4, 0.0, 0.4]];
    let n = [0.3, -1.0, 0.2];
    [Tri::new([q[0], q[1], q[2]], [n; 3], [0.8, 0.6, 0.4, 1.0]), Tri::new([q[0], q[2], q[3]], [n; 3], [0.8, 0.6, 0.4, 1.0])]
}

/// Instance `k` of `n` on a square grid in the XZ plane, each turned a
/// little about the view axis and scaled by its place.
fn placed(k: usize, n: usize) -> [[f32; 4]; 4] {
    let side = (n as f32).sqrt().ceil() as usize;
    let (x, z) = ((k % side) as f32 - side as f32 / 2.0, (k / side) as f32 - side as f32 / 2.0);
    let turn = k as f32 * 0.37;
    let s = 0.7 + 0.3 * ((k % 7) as f32 / 7.0);
    let (sn, cs) = turn.sin_cos();
    [[s * cs, 0.0, -s * sn, 0.0], [0.0, s, 0.0, 0.0], [s * sn, 0.0, s * cs, 0.0], [x, 0.0, z, 1.0]]
}

/// The wall's faces added to the mesh being built.
fn add_wall(s: &mut Scene) {
    for tri in wall() {
        let m = s.material(Mat::of(&tri));
        let v = [tri.a, tri.b, tri.c].map(|p| s.vertex(p, tri.na, [0.0; 2]));
        s.face(v, m);
    }
}

/// `n` walls: one mesh placed `n` times.
fn shared(n: usize) -> Scene {
    let mut s = Scene::default();
    let first = s.instance_of(placed(0, n), [1.0; 3], ObjectHash::of("Wall"));
    add_wall(&mut s);
    let mesh = s.mesh_of(first);
    for k in 1..n {
        s.place(mesh, placed(k, n), [1.0; 3], ObjectHash::of("Wall"));
    }
    s
}

/// `n` walls: `n` meshes, each placed once — what every object used to be.
fn copies(n: usize) -> Scene {
    let mut s = Scene::default();
    for k in 0..n {
        s.instance_of(placed(k, n), [1.0; 3], ObjectHash::of("Wall"));
        add_wall(&mut s);
    }
    s
}

/// `scene` in `mode`, every sample in.
fn settled(gpu: &phiray::Gpu, scene: Scene, mode: Mode, cam: Camera) -> (Tracer, Vec<u8>) {
    let mut t = Tracer::from_scene(gpu, scene).expect("a scene the device holds");
    t.set_mode(mode);
    t.set_samples(4);
    let (mut px, mut done) = t.render(gpu, cam, W, W);
    while !done {
        (px, done) = t.render(gpu, cam, W, W);
    }
    (t, px)
}

/// Pixels that differ by more than a step between two images.
fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(p, q)| p.iter().zip(*q).any(|(x, y)| x.abs_diff(*y) > 2)).count()
}

fn lit(px: &[u8]) -> usize {
    px.chunks_exact(4).filter(|p| p[0] > 100).count()
}

/// (a) One geometry upload: a thousand placements hold the wall's two
/// faces once, under one tree, and the device holds a fraction of what a
/// thousand copies take.
#[test]
fn a_thousand_instances_upload_one_mesh() {
    let Some(gpu) = Ray::gpu() else { return };
    let mut s = shared(1000);
    s.seal();
    assert_eq!((s.meshes.len(), s.insts.len(), s.len(), s.pos.len()), (1, 1000, 2, 6));
    assert!(s.insts.iter().all(|i| i.mesh == 0));
    let mut c = copies(1000);
    c.seal();
    assert_eq!((c.meshes.len(), c.insts.len(), c.len()), (1000, 1000, 2000));
    let once = Tracer::from_scene(&gpu, shared(1000)).expect("a scene").bytes();
    let many = Tracer::from_scene(&gpu, copies(1000)).expect("a scene").bytes();
    // The instance table is the same size either way; the copies carry a
    // thousand meshes' faces, vertices and trees besides.
    assert!(many > once + 1000 * (2 * 16 + 6 * 32), "copies {many} B, shared {once} B");
}

/// (b) The same image as a thousand copies — rasterized (Solid, the
/// instanced prepass) and traced (Rendered, the top tree's leaves).
#[test]
fn a_thousand_instances_draw_what_a_thousand_copies_draw() {
    let Some(gpu) = Ray::gpu() else { return };
    let cam = camera(0.028);
    for mode in [Mode::Solid, Mode::Rendered] {
        let (one, a) = settled(&gpu, shared(1000), mode, cam);
        let (_, b) = settled(&gpu, copies(1000), mode, cam);
        // The walls are in the frame, or the comparison below proves
        // nothing. Solid's studio rig lights them far more brightly than
        // Rendered's dim environment, so the floor is a share of the frame
        // both can clear, and the two images must agree on it.
        assert!(lit(&a) > (W * W / 20) as usize, "{mode:?}: the walls are in the frame ({} lit)", lit(&a));
        assert!(lit(&a).abs_diff(lit(&b)) <= (W * W / 100) as usize, "{mode:?}: {} lit shared against {} copied", lit(&a), lit(&b));
        let off = differing(&a, &b);
        assert!(off <= (W * W / 100) as usize, "{mode:?}: {off} pixels differ between one shared mesh and a thousand copies");
        if mode == Mode::Solid {
            assert!(one.raster() || gpu.adapter.is_software(), "hardware rasterizes the prepass: one instanced draw");
        }
    }
}

/// (c) More instances than the face word's sixteen bits ever held: seventy
/// thousand placements of one mesh draw as seventy thousand copies do.
#[test]
fn more_than_sixty_five_thousand_instances_draw() {
    let Some(gpu) = Ray::gpu() else { return };
    let n = 70_000;
    let mut s = shared(n);
    s.seal();
    assert_eq!((s.insts.len(), s.len()), (n, 2));
    let cam = camera(0.0035);
    let (_, a) = settled(&gpu, shared(n), Mode::Solid, cam);
    let (_, b) = settled(&gpu, copies(n), Mode::Solid, cam);
    assert!(lit(&a) > (W * W / 10) as usize, "the walls are in the frame ({} lit)", lit(&a));
    let off = differing(&a, &b);
    assert!(off <= (W * W / 100) as usize, "{off} pixels differ between one shared mesh and {n} copies");
}

/// A move of one instance among many leaves the mesh where it is: a new
/// matrix, the small top tree, and the image starts over.
#[test]
fn moving_one_of_many_instances_is_a_new_matrix() {
    let Some(gpu) = Ray::gpu() else { return };
    let cam = camera(0.028);
    let (mut t, before) = settled(&gpu, shared(100), Mode::Solid, cam);
    let mut worlds: Vec<[[f32; 4]; 4]> = (0..100).map(|k| placed(k, 100)).collect();
    worlds[37][3] = [40.0, 0.0, 40.0, 1.0];
    assert!(t.reinstance(&gpu, &worlds));
    let (sample, _) = t.sample(&gpu, cam, W, W);
    assert_eq!(sample, 0, "the image starts over");
    // Settle it before comparing: `before` and `fresh` are settled images,
    // and Solid's jittered samples accumulate, so a single sample of the
    // moved scene is a different image from the same scene settled.
    let mut done = false;
    while !done {
        (_, done) = t.render(&gpu, cam, W, W);
    }
    let after = t.pixels(&gpu);
    assert_ne!(before, after, "one wall left the frame");
    let mut gone = shared(100);
    gone.insts[37].world = worlds[37];
    let (_, fresh) = settled(&gpu, gone, Mode::Solid, cam);
    assert!(differing(&after, &fresh) <= (W * W / 100) as usize, "{} pixels differ from the scene built with the wall already moved", differing(&after, &fresh));
}
