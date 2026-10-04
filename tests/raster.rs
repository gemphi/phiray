//! Solid and Wireframe find their surfaces the way Workbench does — the
//! scene rasterized into a visibility buffer with a depth test — and draw
//! what tracing every pixel's ray drew: the same hit, the same shading,
//! under an orthographic and a perspective view, over placed instances.

use phiray::{Camera, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 72;

fn ortho() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        pivot: [0.3, 0.0, -0.2],
        units: 1.0,
        scale: 0.12,
        aspect: 1.0,
        persp: 0,
        dist: 10.0,
        pan: [0.05, -0.03],
        sky: [0.2, 0.22, 0.25, 0.0],
        env: [0.2, 0.22, 0.25, 1.0],
        grid: 0.0,
        ..Default::default()
    }
}

fn persp() -> Camera {
    Camera { persp: 1, dist: 9.0, scale: 0.9, ..ortho() }
}

/// Two walls at different depths (the near one half covering the far), and
/// a slanted card, as instances: a turned, scaled one and a translated one.
fn scene() -> Scene {
    let mut s = Scene::default();
    let n = [0.0, -1.0, 0.0];
    let wall = |c: f32, half: f32| {
        let q = [[-half, 0.0, -half], [half, 0.0, -half], [half, 0.0, half], [-half, 0.0, half]];
        [Tri::new([q[0], q[1], q[2]], [n; 3], [c, c, c, 1.0]), Tri::new([q[0], q[2], q[3]], [n; 3], [c, c, c, 1.0])]
    };
    let (sn, cs) = 0.5f32.sin_cos();
    let placed = [
        ([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]], 0.8),
        ([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 3.0, 0.5, 1.0]], 0.3),
        ([[1.3 * cs, 1.3 * sn, 0.0, 0.0], [-1.3 * sn, 1.3 * cs, 0.0, 0.0], [0.0, 0.0, 1.3, 0.0], [0.0, 1.5, -1.0, 1.0]], 0.55),
    ];
    for (world, c) in placed {
        s.instance(world);
        for tri in wall(c, 1.2) {
            let m = s.material(Mat::of(&tri));
            let v = [tri.a, tri.b, tri.c].map(|p| s.vertex(p, tri.na, [0.0; 2]));
            s.face(v, m);
        }
    }
    s
}

fn frame(gpu: &phiray::Gpu, mode: Mode, cam: Camera, raster: bool) -> Vec<u8> {
    let mut t = Tracer::from_scene(gpu, scene()).expect("a scene");
    t.set_mode(mode);
    t.set_raster(raster);
    assert_eq!(t.raster(), raster);
    t.render(gpu, cam, W, W).0
}

/// Pixels that differ by more than a step.
fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(p, q)| p.iter().zip(*q).any(|(x, y)| x.abs_diff(*y) > 2)).count()
}

#[test]
fn the_rasterized_frame_is_the_traced_frame() {
    let Some(gpu) = Ray::gpu() else { return };
    for (name, cam) in [("orthographic", ortho()), ("perspective", persp())] {
        for mode in [Mode::Solid, Mode::Wire] {
            // In Wireframe `env` is the wire's colour: white, over the grey sky.
            let cam = if mode == Mode::Wire { Camera { env: [1.0; 4], ..cam } } else { cam };
            let drawn = frame(&gpu, mode, cam, true);
            let traced = frame(&gpu, mode, cam, false);
            let off = differing(&drawn, &traced);
            // Edges may round to the other side of a pixel centre; surfaces may not.
            assert!(off <= (W * W / 50) as usize, "{name} {mode:?}: {off} of {} pixels differ", W * W);
            let bright = drawn.chunks_exact(4).filter(|p| p[0] > 120).count();
            assert!(bright > 20, "{name} {mode:?}: the bright wall (or the wires) are in the frame: {bright} pixels");
            eprintln!("{name} {mode:?}: {off} of {} pixels differ at the edges", W * W);
            if mode == Mode::Solid && name == "orthographic" {
                assert!(drawn.chunks_exact(4).any(|p| p[0] < 90 && p[0] > 40), "{name}: the dim wall is in the frame too");
            }
        }
    }
}

#[test]
fn x_ray_still_traces_and_a_moved_instance_is_rasterized_where_it_went() {
    let Some(gpu) = Ray::gpu() else { return };
    let mut t = Tracer::from_scene(&gpu, scene()).expect("a scene");
    t.set_mode(Mode::Solid);
    t.set_raster(true);
    let before = t.render(&gpu, ortho(), W, W).0;
    let xray = Camera { see_through: 0.5, ..ortho() };
    let seen = t.render(&gpu, xray, W, W).0;
    assert_ne!(before, seen, "X-ray sees through the near wall");
    let mut traced = Tracer::from_scene(&gpu, scene()).expect("a scene");
    traced.set_mode(Mode::Solid);
    traced.set_raster(false);
    assert_eq!(seen, traced.render(&gpu, xray, W, W).0, "X-ray is traced either way: the same pixels");
    // The bright wall moved out of the frame: the prepass draws it there.
    let mut worlds = vec![[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [40.0, 0.0, 0.0, 1.0]]];
    worlds.push([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 3.0, 0.5, 1.0]]);
    let (sn, cs) = 0.5f32.sin_cos();
    worlds.push([[1.3 * cs, 1.3 * sn, 0.0, 0.0], [-1.3 * sn, 1.3 * cs, 0.0, 0.0], [0.0, 0.0, 1.3, 0.0], [0.0, 1.5, -1.0, 1.0]]);
    assert!(t.reinstance(&gpu, &worlds));
    let after = t.render(&gpu, ortho(), W, W).0;
    assert!(!after.chunks_exact(4).any(|p| p[0] > 120), "the bright wall is gone from the frame");
    assert!(traced.reinstance(&gpu, &worlds));
    assert!(differing(&after, &traced.render(&gpu, ortho(), W, W).0) <= (W * W / 50) as usize);
}
