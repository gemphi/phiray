//! Rigid body simulation, ROADMAP.md §2 "Physics": XPBD with substeps on
//! the CPU first (milestone M1), every array a `Pod` twin of the
//! `GpuArray` the kernels of M3 will take, the broadphase over phiray's
//! `Bvh`. `World::step` is one frame; `World::poses` feeds
//! `Tracer::reinstance`; `World::from_blend` takes what phiphx reads of
//! Blender's Rigid Body World and each object's rigid body.
//!
//! Files: `state` (the arrays), `shape` (one support function for every
//! convex kind), `broad` (swept bounds, the tree, the pairs), `mesh`
//! (passive triangle meshes and their midphase), `narrow` (closed forms,
//! SAT with clipping, GJK/EPA), `contact` (positional projection with
//! static friction; dynamic friction and restitution on the velocities),
//! `build` (filling a world), `world` (the step), `blend` (Blender's
//! settings in), `math`.

pub mod blend;
mod broad;
mod build;
mod contact;
pub mod math;
mod mesh;
pub mod narrow;
pub mod shape;
mod state;
mod world;

pub use blend::{BlendBody, BlendObject, BlendShape, BlendWorld};
pub use mesh::MeshCollider;
pub use state::{Body, BodyConst, BodyPrev, Contact, DEACTIVATES, Kind, SLEEPING, Shape, ShapeKind};
pub use build::BodyDef;
pub use world::World;
