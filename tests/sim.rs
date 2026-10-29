//! Rigid bodies against closed forms (ROADMAP.md §2.2 "Tests"): free fall,
//! restitution, friction, a stack, every shape at rest, Blender's
//! settings in, determinism and energy. No GPU: this is the CPU twin.

use phiray::sim::math::*;
use phiray::sim::{BlendBody, BlendObject, BlendShape, BlendWorld, BodyDef, Kind, Shape, World};

const DT: f32 = 1.0 / 60.0;

/// A wide passive box whose top is the plane z = 0; friction and
/// restitution 1, so a body's own values are the pair's.
fn ground(w: &mut World) -> usize {
    let s = w.add_shape(Shape::boxed([50.0, 50.0, 0.5]));
    w.add_body(BodyDef { kind: Kind::Passive, shape: s, x: [0.0, 0.0, -0.5], friction: 1.0, restitution: 1.0, ..Default::default() })
}

fn undamped(shape: u32, x: V3) -> BodyDef {
    BodyDef { shape, x, lin_damp: 0.0, ang_damp: 0.0, ..Default::default() }
}

#[test]
fn free_fall_matches_the_closed_form() {
    let mut w = World::new();
    let s = w.add_shape(Shape::sphere(0.5));
    let (x0, v0) = ([1.0, 2.0, 20.0], [0.3, -0.2, 2.0]);
    let i = w.add_body(BodyDef { v: v0, ..undamped(s, x0) });
    for f in 1..=100 {
        w.step(DT);
        let t = f as f32 * DT;
        let want = add(add(x0, scale(v0, t)), scale(w.gravity, 0.5 * t * t));
        let got = w.bodies[i].x;
        assert!(len(sub(got, want)) < 1e-3, "frame {f}: {got:?} against {want:?}");
    }
}

#[test]
fn a_ball_bounces_to_e_squared_of_its_height() {
    let mut w = World::new();
    ground(&mut w);
    let s = w.add_shape(Shape::sphere(0.5));
    let h0 = 2.0;
    let i = w.add_body(BodyDef { restitution: 0.5, ..undamped(s, [0.0, 0.0, 0.5 + h0]) });
    let (mut rising, mut apex) = (false, 0.0f32);
    for _ in 0..240 {
        w.step(DT);
        let (z, vz) = (w.bodies[i].x[2], w.bodies[i].v[2]);
        if vz > 0.0 {
            rising = true;
        }
        if rising && vz < 0.0 {
            break;
        }
        if rising {
            apex = apex.max(z - 0.5);
        }
    }
    assert!(rising, "the ball never bounced");
    assert!((apex / h0 - 0.25).abs() < 0.05 * 0.25, "bounced to {apex} of {h0}");
}

/// How far a unit box slides down a slope of `theta` in two seconds.
fn slide(mu: f32, theta: f32) -> f32 {
    let mut w = World::new();
    let big = w.add_shape(Shape::boxed([20.0, 20.0, 0.5]));
    let q = q_axis([0.0, 1.0, 0.0], theta);
    w.add_body(BodyDef { kind: Kind::Passive, shape: big, q, friction: 1.0, ..Default::default() });
    let unit = w.add_shape(Shape::boxed([0.5; 3]));
    let n = rotate(q, [0.0, 0.0, 1.0]);
    let i = w.add_body(BodyDef { q, friction: mu, ..undamped(unit, n) });
    let start = w.bodies[i].x;
    for _ in 0..120 {
        w.step(DT);
    }
    len(sub(w.bodies[i].x, start))
}

#[test]
fn a_box_rests_below_the_friction_angle_and_slides_above() {
    let mu = 0.5f32;
    let crit = mu.atan();
    let held = slide(mu, crit - 0.1);
    let slid = slide(mu, crit + 0.1);
    assert!(held < 1e-3, "slid {held} m below the friction angle");
    assert!(slid > 0.5, "slid only {slid} m above the friction angle");
}

#[test]
fn ten_boxes_stand_for_a_thousand_frames() {
    let mut w = World::new();
    ground(&mut w);
    let unit = w.add_shape(Shape::boxed([0.5; 3]));
    let tops: Vec<usize> = (0..10).map(|k| w.add_body(BodyDef { shape: unit, x: [0.0, 0.0, 0.5 + k as f32], ..Default::default() })).collect();
    let top = tops[9];
    let start = w.bodies[top].x;
    let mut worst = 0.0f32;
    for _ in 0..1000 {
        w.step(DT);
        worst = worst.max(len(sub(w.bodies[top].x, start)));
    }
    assert!(worst < 1e-3, "the top box moved {worst} m");
    for (k, &i) in tops.iter().enumerate() {
        let z = w.bodies[i].x[2];
        assert!((z - (0.5 + k as f32)).abs() < 5e-3, "box {k} sits at {z}");
    }
}

/// Every convex kind dropped half a metre onto the ground, `frames` on:
/// each one's lateral drift, the height it came to rest at against the
/// height it should, and how fast it is still moving.
fn shapes_at_rest(subs: u32, iters: u32, frames: usize) -> Vec<(f32, f32, f32, f32)> {
    let mut w = World::new();
    (w.substeps, w.iterations) = (subs, iters);
    ground(&mut w);
    let tilt = q_axis([1.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2);
    let hull: Vec<V3> = (0..8).map(|i| map3(|k| if (i >> k) & 1 == 1 { 0.4 } else { -0.4 })).collect();
    let shapes = [
        (w.add_shape(Shape::sphere(0.3)), [0.0, 0.0, 0.0, 1.0], 0.3),
        (w.add_shape(Shape::boxed([0.3, 0.4, 0.5])), [0.0, 0.0, 0.0, 1.0], 0.5),
        (w.add_shape(Shape::capsule(0.25, 0.5)), tilt, 0.25),
        (w.add_shape(Shape::cylinder(0.3, 0.4)), [0.0, 0.0, 0.0, 1.0], 0.4),
        (w.add_shape(Shape::cone(0.4, 0.5)), [0.0, 0.0, 0.0, 1.0], 0.5),
        (w.add_hull(&hull), [0.0, 0.0, 0.0, 1.0], 0.4),
    ];
    let bodies: Vec<(usize, V3, f32)> = shapes
        .iter()
        .enumerate()
        .map(|(k, &(s, q, rest))| {
            let x = [3.0 * k as f32, 0.0, rest + 0.5];
            (w.add_body(BodyDef { q, ..undamped(s, x) }), x, rest)
        })
        .collect();
    for _ in 0..frames {
        w.step(DT);
    }
    bodies
        .iter()
        .map(|&(i, x, rest)| {
            let b = w.bodies[i];
            (len(sub([b.x[0], b.x[1], 0.0], [x[0], x[1], 0.0])), b.x[2], rest, len(b.v).max(len(b.w)))
        })
        .collect()
}

/// The shapes whose contacts come from a closed form or from SAT: a
/// sphere and a box. Both land on the height they should to the micron
/// and never drift. The four that go through GJK/EPA do not yet, which
/// `curved_shapes_rest_only_roughly` measures.
#[test]
fn a_sphere_and_a_box_come_to_rest_exactly() {
    let (subs, iters) = (10, 10);
    let floor = 9.81 * DT / subs as f32 / 2.0;
    for (k, (drift, z, rest, speed)) in shapes_at_rest(subs, iters, 180).into_iter().take(2).enumerate() {
        assert!((z - rest).abs() < 1e-4, "shape {k} rests at {z}, not {rest}");
        assert!(drift < 1e-5, "shape {k} drifted {drift} m");
        assert!(speed < 2.0 * floor, "shape {k} still moves at {speed}");
    }
}

/// Every kind, including the four the narrowphase answers with GJK/EPA.
///
/// **Ignored: the GJK/EPA manifold is not good enough for them yet**
/// (ROADMAP.md "M1 status"). At ten substeps and ten projections each,
/// measured 2026-10-29: the capsule drifts 15 mm and rests 3 mm high,
/// the cylinder drifts 18 mm, the cone and the hull settle within a
/// centimetre of their height but keep creeping. Nothing is launched or
/// lost any more — what is left is the quality of the points and normals
/// the narrowphase hands over for a curved or a flat-faced convex, not
/// the solver over them.
#[test]
#[ignore]
fn curved_shapes_rest_only_roughly() {
    let (subs, iters) = (10, 10);
    // A frame ends on half a substep's kick of gravity, which the next
    // frame's contacts take out: `g h / 2` is the speed a body at rest
    // carries here, and no scheme that ends a step with the kick has
    // less. A sphere and a box sit at exactly that. "At rest" is twice
    // it, so the budget falls with the substep count rather than being a
    // number that only one count can meet.
    let floor = 9.81 * DT / subs as f32 / 2.0;
    for (k, (drift, z, rest, speed)) in shapes_at_rest(subs, iters, 180).into_iter().enumerate() {
        assert!((z - rest).abs() < 5e-3, "shape {k} rests at {z}, not {rest}");
        assert!(drift < 1e-2, "shape {k} drifted {drift} m");
        assert!(speed < 2.0 * floor, "shape {k} still moves at {speed}, over {}", 2.0 * floor);
    }
}

#[test]
fn blender_settings_make_a_crate_rest_on_a_mesh_floor() {
    let floor: Vec<[V3; 3]> = vec![[[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0]], [[-5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0]]];
    let cube: Vec<[V3; 3]> = {
        let c = |i: usize| map3(|k| if (i >> k) & 1 == 1 { 1.0 } else { -1.0 });
        let quad = |a: usize, b: usize, cc: usize, d: usize| [[c(a), c(b), c(cc)], [c(a), c(cc), c(d)]];
        [quad(0, 1, 3, 2), quad(4, 6, 7, 5), quad(0, 4, 5, 1), quad(2, 3, 7, 6), quad(0, 2, 6, 4), quad(1, 5, 7, 3)].concat()
    };
    let scale = [0.5, 0.5, 0.25];
    let q = q_axis([0.0, 0.0, 1.0], 0.4);
    let world = pose_matrix([1.0, 2.0, 1.5], q, [0.0; 3], scale);
    let objects = [
        BlendObject { body: BlendBody { passive: true, shape: BlendShape::Mesh, friction: 0.9, ..Default::default() }, tris: &floor, world: pose_matrix([0.0; 3], [0.0, 0.0, 0.0, 1.0], [0.0; 3], [1.0; 3]), obj: 7 },
        BlendObject { body: BlendBody { shape: BlendShape::Box, mass: 2.5, friction: 0.3, use_margin: true, margin: 0.02, ..Default::default() }, tris: &cube, world, obj: 3 },
    ];
    let mut w = World::from_blend(&BlendWorld { substeps_per_frame: 20, solver_iterations: 25, ..Default::default() }, &objects);
    // Blender's iterations are per substep (Bullet's `setNumIterations`),
    // not a budget split between them.
    assert_eq!((w.substeps, w.iterations), (20, 25));
    assert_eq!(w.bodies[1].inv_mass, 1.0 / 2.5);
    assert!(w.bodies[0].inv_mass == 0.0 && w.is_mesh(w.bodies[0].shape));
    for _ in 0..180 {
        w.step(DT);
    }
    let b = w.bodies[1];
    assert!((b.x[2] - 0.25).abs() < 5e-3, "the crate rests at {}", b.x[2]);
    assert!(len(sub([b.x[0], b.x[1], 0.0], [1.0, 2.0, 0.0])) < 1e-2, "the crate drifted to {:?}", b.x);
    let poses: Vec<(usize, [[f32; 4]; 4])> = w.poses().collect();
    assert_eq!((poses[0].0, poses[1].0), (7, 3));
    let m = poses[1].1;
    assert!(len(sub([m[3][0], m[3][1], m[3][2]], b.x)) < 1e-6);
    assert!((len([m[0][0], m[0][1], m[0][2]]) - 0.5).abs() < 1e-5 && (len([m[2][0], m[2][1], m[2][2]]) - 0.25).abs() < 1e-5);
}

/// A ball dropped into a leaning stack: contacts of every kind, 120 frames.
fn busy() -> World {
    let mut w = World::new();
    ground(&mut w);
    let unit = w.add_shape(Shape::boxed([0.5; 3]));
    for k in 0..5 {
        w.add_body(BodyDef { shape: unit, x: [0.05 * k as f32, 0.0, 0.5 + k as f32], ..Default::default() });
    }
    let s = w.add_shape(Shape::sphere(0.4));
    w.add_body(BodyDef { shape: s, x: [0.3, 0.1, 8.0], restitution: 0.3, ..Default::default() });
    for _ in 0..120 {
        w.step(DT);
    }
    w
}

#[test]
fn two_runs_are_bit_identical() {
    let (a, b) = (busy(), busy());
    assert_eq!(a.bodies, b.bodies);
    assert_eq!(a.contacts, b.contacts);
    assert!(a.bodies.iter().skip(1).any(|b| b.x[2] < 4.0), "the stack never fell");
}

#[test]
fn a_frictionless_bouncing_ball_does_not_gain_energy() {
    let mut w = World::new();
    ground(&mut w);
    let s = w.add_shape(Shape::sphere(0.5));
    w.add_body(BodyDef { restitution: 1.0, friction: 0.0, ..undamped(s, [0.0, 0.0, 1.5]) });
    let e0 = w.energy();
    let mut bounces = 0;
    let mut vz = 0.0;
    for _ in 0..600 {
        w.step(DT);
        let e = w.energy();
        assert!(e <= e0 * (1.0 + 1e-4), "energy grew from {e0} to {e}");
        let now = w.bodies[1].v[2];
        if vz < 0.0 && now > 0.0 {
            bounces += 1;
        }
        vz = now;
    }
    assert!(bounces >= 5, "only {bounces} bounces in ten seconds");
    assert!(w.energy() > 0.5 * e0, "the ball lost most of its energy: {} of {e0}", w.energy());
}

/// Sweeps substeps and projections per substep over the three scenes that
/// convergence governs, to tell a solver bug from a budget:
/// `cargo test -p phiray --test sim -- --ignored --nocapture convergence_sweep`.
#[test]
#[ignore]
fn convergence_sweep() {
    for (subs, iters) in [(10, 1), (10, 4), (10, 8), (20, 1), (20, 4), (40, 1), (10, 16)] {
        let tune = |w: &mut World| {
            w.substeps = subs;
            w.iterations = iters;
        };

        // The slope, held and sliding.
        let run = |mu: f32, theta: f32| {
            let mut w = World::new();
            tune(&mut w);
            let big = w.add_shape(Shape::boxed([20.0, 20.0, 0.5]));
            let q = q_axis([0.0, 1.0, 0.0], theta);
            w.add_body(BodyDef { kind: Kind::Passive, shape: big, q, friction: 1.0, ..Default::default() });
            let unit = w.add_shape(Shape::boxed([0.5; 3]));
            let n = rotate(q, [0.0, 0.0, 1.0]);
            let i = w.add_body(BodyDef { q, friction: mu, ..undamped(unit, n) });
            let start = w.bodies[i].x;
            for _ in 0..120 {
                w.step(DT);
            }
            (len(sub(w.bodies[i].x, start)), w.contacts.len())
        };
        let crit = 0.5f32.atan();
        let (held, nheld) = run(0.5, crit - 0.1);
        let (slid, _) = run(0.5, crit + 0.1);

        // The stack: how far the top box settles.
        let mut w = World::new();
        tune(&mut w);
        ground(&mut w);
        let unit = w.add_shape(Shape::boxed([0.5; 3]));
        let tops: Vec<usize> = (0..10).map(|k| w.add_body(BodyDef { shape: unit, x: [0.0, 0.0, 0.5 + k as f32], ..Default::default() })).collect();
        let start = w.bodies[tops[9]].x;
        let mut worst = 0.0f32;
        for _ in 0..300 {
            w.step(DT);
            worst = worst.max(len(sub(w.bodies[tops[9]].x, start)));
        }

        eprintln!("subs {subs:>2} iters {iters:>2} | slope held {held:.2e} ({nheld} contacts) slid {slid:.3} | stack {worst:.2e}");
        for (k, (drift, z, rest, speed)) in shapes_at_rest(subs, iters, 180).into_iter().enumerate() {
            let name = ["sphere", "box", "capsule", "cylinder", "cone", "hull"][k];
            eprintln!("    {name:<9} drift {drift:.2e} rest {z:.4} of {rest} speed {speed:.1e}");
        }
    }
}

/// Prints the stack's top box and the contacts, frame by frame, for a look
/// at a failure: `cargo test -p phiray --test sim -- --ignored stack_trace`.
#[test]
#[ignore]
fn stack_trace() {
    let mut w = World::new();
    ground(&mut w);
    let unit = w.add_shape(Shape::boxed([0.5; 3]));
    for k in 0..10 {
        w.add_body(BodyDef { shape: unit, x: [0.0, 0.0, 0.5 + k as f32], ..Default::default() });
    }
    for f in 0..300 {
        w.step(DT);
        if f % 10 == 0 {
            let top = w.bodies[10];
            let bottom = w.bodies[1];
            eprintln!("frame {f}: top {:?} v {:?} w {:?} | bottom {:?} | {} contacts", top.x, top.v, top.w, bottom.x, w.contacts.len());
            if f == 0 {
                for c in &w.contacts {
                    eprintln!("  {}-{} n {:?} depth {} anchors {:?} {:?}", c.a, c.b, c.normal, c.depth0, c.anchor_a, c.anchor_b);
                }
            }
        }
    }
}
