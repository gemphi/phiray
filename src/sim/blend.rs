//! Blender's settings into a world (ROADMAP.md §2.2 "Fitting phiber", 1
//! and 5). `from_blend` takes the values phiphx reads from the file
//! (`blend/physics/`: `BRigidWorld`, `BRigidBody`) copied into
//! `BlendWorld` and `BlendBody`, so phiray stays free of the reader;
//! phiber hands them over with each object's triangles and world matrix.
//! Shapes follow `rigidbody_validate_sim_shape` in Blender's
//! `rigidbody.cc`: dimensions from the object's bounds with its scale,
//! centred on the object origin (the centre of mass, as Blender assumes),
//! a quadric's radius the larger of x and y and its height z; a hull is
//! the scaled vertices; a Mesh shape on an active body becomes its hull
//! and is listed in `World::hull_fallback`.

use super::math::*;
use super::state::{Kind, Shape};
use super::build::BodyDef;
use super::world::World;

/// Blender's `RigidBodyWorld`, with the scene's gravity already weighted
/// by the world's effector weight for gravity.
#[derive(Clone, Debug, PartialEq)]
pub struct BlendWorld {
    pub substeps_per_frame: u32,
    pub solver_iterations: u32,
    pub time_scale: f32,
    pub gravity: V3,
}

impl Default for BlendWorld {
    fn default() -> Self {
        Self { substeps_per_frame: 10, solver_iterations: 10, time_scale: 1.0, gravity: [0.0, 0.0, -9.81] }
    }
}

/// Blender's `RigidBodyOb.shape`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendShape {
    Box,
    Sphere,
    Capsule,
    Cylinder,
    Cone,
    #[default]
    ConvexHull,
    Mesh,
    /// Compound shapes are taken as the hull of the object alone.
    Compound,
}

/// Blender's `RigidBodyOb`, the fields the solver reads.
#[derive(Clone, Debug, PartialEq)]
pub struct BlendBody {
    pub passive: bool,
    pub shape: BlendShape,
    /// Dynamic; off, the body holds its pose and collides.
    pub enabled: bool,
    /// Animated: the body follows its object.
    pub kinematic: bool,
    pub mass: f32,
    pub friction: f32,
    pub restitution: f32,
    pub use_margin: bool,
    pub margin: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub use_deactivation: bool,
    pub start_deactivated: bool,
    pub linear_sleep: f32,
    pub angular_sleep: f32,
    /// Collision Collections, a bit each.
    pub collections: u32,
}

impl Default for BlendBody {
    fn default() -> Self {
        Self {
            passive: false,
            shape: BlendShape::ConvexHull,
            enabled: true,
            kinematic: false,
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            use_margin: false,
            margin: 0.04,
            linear_damping: 0.04,
            angular_damping: 0.1,
            use_deactivation: false,
            start_deactivated: false,
            linear_sleep: 0.4,
            angular_sleep: 0.5,
            collections: 1,
        }
    }
}

/// One object with a rigid body: its settings, its mesh in object space
/// and its world matrix (Blender's `m[col][row]`).
pub struct BlendObject<'a> {
    pub body: BlendBody,
    pub tris: &'a [[V3; 3]],
    pub world: [[f32; 4]; 4],
    /// The caller's object index, handed back by `World::poses`.
    pub obj: usize,
}

impl World {
    pub fn from_blend(w: &BlendWorld, objects: &[BlendObject]) -> Self {
        let mut world = World::new();
        world.gravity = w.gravity;
        world.substeps = w.substeps_per_frame.max(1);
        // Blender hands `num_solver_iterations` to Bullet's
        // `setNumIterations`, which runs that many in *each* internal
        // step — so it is iterations per substep, not a budget to divide
        // between them. Dividing left the default file on one projection
        // a substep, and one is not enough to hold a stack: ten boxes
        // settled 2 mm instead of the 0.07 mm they settle at Blender's
        // own ten.
        world.iterations = w.solver_iterations.max(1);
        world.time_scale = w.time_scale;
        for o in objects {
            let b = &o.body;
            let (x, q, s) = decompose(o.world);
            let kind = match (b.passive, b.kinematic || !b.enabled) {
                (true, _) => Kind::Passive,
                (false, true) => Kind::Kinematic,
                (false, false) => Kind::Active,
            };
            let verts: Vec<V3> = o.tris.iter().flatten().map(|v| mul3(*v, s)).collect();
            let (lo, hi) = verts.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), v| (map3(|k| lo[k].min(v[k])), map3(|k| hi[k].max(v[k]))));
            let half = if verts.is_empty() { [0.0; 3] } else { map3(|k| 0.5 * (hi[k] - lo[k])) };
            let radius = half[0].max(half[1]);
            let shape = match (b.shape, kind) {
                (BlendShape::Box, _) => world.add_shape(Shape::boxed(half)),
                (BlendShape::Sphere, _) => world.add_shape(Shape::sphere(half[0].max(half[1]).max(half[2]))),
                (BlendShape::Capsule, _) => world.add_shape(Shape::capsule(radius, (half[2] - radius).max(0.0))),
                (BlendShape::Cylinder, _) => world.add_shape(Shape::cylinder(radius, half[2])),
                (BlendShape::Cone, _) => world.add_shape(Shape::cone(radius, half[2])),
                (BlendShape::Mesh, Kind::Passive) | (BlendShape::Mesh, Kind::Kinematic) => {
                    let tris: Vec<[V3; 3]> = o.tris.iter().map(|t| t.map(|v| mul3(v, s))).collect();
                    world.add_mesh(&tris)
                }
                (BlendShape::Mesh, Kind::Active) => {
                    world.hull_fallback.push(o.obj);
                    world.add_hull(&unique(verts))
                }
                (BlendShape::ConvexHull, _) | (BlendShape::Compound, _) => world.add_hull(&unique(verts)),
            };
            let margin = match (b.use_margin, b.shape) {
                (true, _) => b.margin,
                (false, BlendShape::ConvexHull) | (false, BlendShape::Mesh) | (false, BlendShape::Compound) => 0.04,
                (false, _) => 0.0,
            };
            world.add_body(BodyDef {
                kind,
                shape,
                mass: b.mass,
                x,
                q,
                scale: s,
                friction: b.friction,
                restitution: b.restitution,
                margin,
                lin_damp: b.linear_damping,
                ang_damp: b.angular_damping,
                sleep: b.use_deactivation.then_some([b.linear_sleep, b.angular_sleep]),
                asleep: b.start_deactivated,
                groups: b.collections,
                obj: o.obj,
                ..BodyDef::default()
            });
        }
        world
    }
}

/// The distinct points of `verts`, in first-seen order.
fn unique(verts: Vec<V3>) -> Vec<V3> {
    let mut seen = std::collections::HashSet::new();
    verts.into_iter().filter(|v| seen.insert(v.map(f32::to_bits))).collect()
}
