//! The selection outline (`outline.rs`, `outline.wgsl`): Blender's
//! `OVERLAY_outline` over the traced image — the id edges of the objects
//! the host named, in the theme colour each one's class asks for.

use phiray::trace::Outline;
use phiray::{Camera, Gpu, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

/// Looking down −Y, orthographic: 10 units across the frame, so each
/// 2-unit wall is about 10 pixels wide. No grid in shot.
fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        units: 1.0,
        scale: 0.1,
        aspect: 1.0,
        dist: 10.0,
        sky: [0.2, 0.22, 0.25, 0.0],
        env: [0.2, 0.22, 0.25, 1.0],
        // The outline on, one pixel out, the theme's three colours well
        // apart so a pixel says which class drew it.
        outline_fx: [1.0, 0.0, 0.35, 1.0],
        outline_pick: [0.0, 1.0, 0.0, 1.0],
        outline_active: [1.0, 0.0, 1.0, 1.0],
        outline_move: [0.0, 0.0, 1.0, 1.0],
        ..Default::default()
    }
}

/// A 2 × 2 wall at the origin, facing the camera, in its own space.
fn wall() -> [Tri; 2] {
    let q = [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let n = [0.0, -1.0, 0.0];
    [Tri::new([q[0], q[1], q[2]], [n; 3], [0.5, 0.5, 0.5, 1.0]), Tri::new([q[0], q[2], q[3]], [n; 3], [0.5, 0.5, 0.5, 1.0])]
}

/// Two walls, one four units left of the other, each its own instance.
fn scene() -> Scene {
    let mut out = Scene::default();
    for x in [-2.0f32, 2.0] {
        let mut world = phiray::trace::instance::IDENTITY;
        world[3] = [x, 0.0, 0.0, 1.0];
        out.instance(world);
        for tri in wall() {
            let m = out.material(Mat::of(&tri));
            let v = [tri.a, tri.b, tri.c].map(|p| out.vertex(p, tri.na, [0.0; 2]));
            out.face(v, m);
        }
    }
    out
}

/// Every pixel within 24 of `want` in each channel, as `(x, y)`.
fn colored(px: &[u8], want: [u8; 3]) -> Vec<(u32, u32)> {
    px.chunks_exact(4)
        .enumerate()
        .filter(|(_, p)| [0, 1, 2].iter().all(|&k| p[k].abs_diff(want[k]) <= 24))
        .map(|(i, _)| (i as u32 % W, i as u32 / W))
        .collect()
}

/// The box the pixels span, `(x_lo, x_hi, y_lo, y_hi)`.
fn span(at: &[(u32, u32)]) -> (u32, u32, u32, u32) {
    let f = |pick: fn(&(u32, u32)) -> u32, best: fn(u32, u32) -> u32, start| at.iter().map(pick).fold(start, best);
    (f(|p| p.0, u32::min, u32::MAX), f(|p| p.0, u32::max, 0), f(|p| p.1, u32::min, u32::MAX), f(|p| p.1, u32::max, 0))
}

fn drawn(tr: &mut Tracer, gpu: &Gpu) -> Vec<u8> {
    let (mut px, mut done) = tr.render(gpu, camera(), W, W);
    while !done {
        (px, done) = tr.render(gpu, camera(), W, W);
    }
    px
}

#[test]
fn the_outline_follows_the_object_the_host_named() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::from_scene(&gpu, scene()).expect("scene");
    tr.set_mode(Mode::Solid);
    // Nothing selected: the pass is left out of the frame altogether.
    let plain = drawn(&mut tr, &gpu);
    assert!(colored(&plain, [255, 0, 255]).is_empty(), "no outline until one is asked for");
    assert!(colored(&plain, [0, 255, 0]).is_empty());

    // The right wall active: a magenta ring about it, and nothing about the
    // left one — the class table is read by instance, not by pixel.
    assert!(tr.outline(&gpu, &[Outline::None, Outline::Active]));
    let active = drawn(&mut tr, &gpu);
    let ring = colored(&active, [255, 0, 255]);
    assert!(ring.len() > 24, "a 10-pixel wall has a ring of its own: {}", ring.len());
    let (x_lo, x_hi, y_lo, y_hi) = span(&ring);
    assert!(x_lo > W / 2, "the ring is in the right half: {x_lo}..{x_hi}");
    assert!(x_hi - x_lo >= 9 && x_hi - x_lo <= 14, "as wide as the wall: {}", x_hi - x_lo);
    assert!(y_hi - y_lo >= 9 && y_hi - y_lo <= 14, "and as tall: {}", y_hi - y_lo);
    // The wall's middle is the wall, not the line.
    let middle = ((y_lo + y_hi) / 2 * W + (x_lo + x_hi) / 2) as usize * 4;
    assert_eq!(active[middle..middle + 3], plain[middle..middle + 3], "the outline draws at the edge alone");

    // The selection moves with no new scene and no new camera: the settled
    // image takes one more sample and the line moves with it.
    assert!(tr.outline(&gpu, &[Outline::Select, Outline::None]));
    let picked = drawn(&mut tr, &gpu);
    assert!(colored(&picked, [255, 0, 255]).is_empty(), "the old ring is gone");
    let green = colored(&picked, [0, 255, 0]);
    assert!(green.len() > 24, "the left wall is outlined now: {}", green.len());
    assert!(span(&green).1 < W / 2, "and in the left half");
}

/// The three classes take the three theme colours Blender's detect pass
/// switches on (`color_id`), and a class the table does not name draws
/// nothing.
#[test]
fn each_class_takes_its_own_theme_colour() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::from_scene(&gpu, scene()).expect("scene");
    tr.set_mode(Mode::Solid);
    for (class, want) in [(Outline::Select, [0, 255, 0]), (Outline::Active, [255, 0, 255]), (Outline::Transform, [0, 0, 255])] {
        assert!(tr.outline(&gpu, &[class, Outline::None]));
        let px = drawn(&mut tr, &gpu);
        let at = colored(&px, want);
        assert!(at.len() > 24, "{class:?} draws in its own colour: {}", at.len());
        assert!(span(&at).1 < W / 2, "{class:?} about the left wall");
    }
    // A second object named beyond the scene's instances is ignored rather
    // than read past the table.
    assert!(tr.outline(&gpu, &[Outline::None, Outline::None, Outline::Active]));
    assert!(colored(&drawn(&mut tr, &gpu), [255, 0, 255]).is_empty(), "nothing selected, nothing drawn");
}
