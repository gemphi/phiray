//! Filling a world (ROADMAP.md §2.2 "State layout"): shapes, hulls and
//! meshes into their tables, bodies into the `Pod` arrays with their
//! inverse mass and inertia, and what the step keeps beside them (the
//! object and scale `poses` hands back, sleep thresholds, kinematic
//! targets).

use super::math::*;
use super::mesh::MeshCollider;
use super::shape;
use super::state::{Body, BodyConst, BodyPrev, DEACTIVATES, Kind, Shape};
use super::world::World;

/// What `World::add_body` takes.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyDef {
    pub kind: Kind,
    pub shape: u32,
    pub mass: f32,
    pub x: V3,
    pub q: Q,
    pub v: V3,
    pub w: V3,
    /// The object's scale, baked into the shape, kept for `poses`.
    pub scale: V3,
    /// Centre of mass in the scaled object space.
    pub com: V3,
    pub friction: f32,
    pub restitution: f32,
    pub margin: f32,
    pub lin_damp: f32,
    pub ang_damp: f32,
    /// Linear and angular speeds under which the body may fall asleep.
    pub sleep: Option<[f32; 2]>,
    pub asleep: bool,
    pub groups: u32,
    /// The caller's object index, handed back by `poses`.
    pub obj: usize,
}

impl Default for BodyDef {
    fn default() -> Self {
        Self {
            kind: Kind::Active,
            shape: 0,
            mass: 1.0,
            x: [0.0; 3],
            q: [0.0, 0.0, 0.0, 1.0],
            v: [0.0; 3],
            w: [0.0; 3],
            scale: [1.0; 3],
            com: [0.0; 3],
            friction: 0.5,
            restitution: 0.0,
            margin: 0.0,
            lin_damp: 0.04,
            ang_damp: 0.1,
            sleep: None,
            asleep: false,
            groups: 1,
            obj: 0,
        }
    }
}

/// Per-body output: the object and its scale.
#[derive(Clone, Copy, Debug)]
pub(super) struct Place {
    pub obj: usize,
    pub scale: V3,
}

impl World {
    pub fn add_shape(&mut self, s: Shape) -> u32 {
        self.shapes.push(s);
        self.shapes.len() as u32 - 1
    }

    /// A hull over `points` (body space, about the centre of mass).
    pub fn add_hull(&mut self, points: &[V3]) -> u32 {
        let at = self.hull.len() as u32;
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for p in points {
            lo = map3(|k| lo[k].min(p[k]));
            hi = map3(|k| hi[k].max(p[k]));
            self.hull.push([p[0], p[1], p[2], 0.0]);
        }
        let half = if points.is_empty() { [0.0; 3] } else { map3(|k| 0.5 * (hi[k] - lo[k])) };
        self.add_shape(Shape::hull(at, points.len() as u32, half))
    }

    /// A passive triangle mesh (body space).
    pub fn add_mesh(&mut self, tris: &[[V3; 3]]) -> u32 {
        let mesh = MeshCollider::new(tris.to_vec());
        let shape = Shape::mesh(self.meshes.len() as u32, tris.len() as u32, mesh.half());
        self.meshes.push(mesh);
        self.add_shape(shape)
    }

    pub fn add_body(&mut self, d: BodyDef) -> usize {
        let movable = d.kind == Kind::Active && d.mass > 0.0;
        let inv_mass = if movable { 1.0 / d.mass } else { 0.0 };
        let inertia = shape::inertia(&self.shapes[d.shape as usize], d.mass);
        let inv_inertia = map3(|k| if movable && inertia[k] > 0.0 { 1.0 / inertia[k] } else { 0.0 });
        let mut flags = d.kind as u32;
        if d.sleep.is_some() {
            flags |= DEACTIVATES;
        }
        let mut body = Body { x: d.x, inv_mass, q: q_norm(d.q), v: d.v, shape: d.shape, w: d.w, flags };
        body.set_asleep(d.asleep && d.sleep.is_some());
        self.bodies.push(body);
        self.consts.push(BodyConst {
            inv_inertia,
            friction: d.friction,
            com_local: d.com,
            restitution: d.restitution,
            lin_damp: d.lin_damp,
            ang_damp: d.ang_damp,
            margin: d.margin,
            groups: d.groups,
        });
        self.prev.push(BodyPrev { x: d.x, _a: 0.0, q: body.q });
        self.places.push(Place { obj: d.obj, scale: d.scale });
        self.sleep.push(d.sleep.unwrap_or([0.0, 0.0]));
        self.timers.push(0.0);
        self.targets.push(None);
        self.bodies.len() - 1
    }

    /// Where kinematic body `i` is at the end of the next step.
    pub fn drive(&mut self, i: usize, x: V3, q: Q) {
        self.targets[i] = Some((x, q_norm(q)));
    }
}
