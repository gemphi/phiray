//! The world and its step (ROADMAP.md §2.2 "One step"): bounds, broadphase
//! and narrowphase once a frame, then `substeps` XPBD substeps of
//! integrate, project, derive velocities and the velocity pass, then
//! sleeping. Gravity goes in as two half kicks, one before the drift and
//! one after the velocity pass (kick-drift-kick): free fall is then exact
//! at any substep count, and a resting contact's derived velocity is zero
//! when the velocity pass sees it, so no impulse is spent holding a body
//! up and none of the rotation noise such impulses leave at a box's
//! corners gets in. Kinematic bodies take the pose `drive` gives them over
//! the frame; passive bodies never move.

use super::broad;
use super::build::Place;
use super::contact;
use super::math::*;
use super::mesh::MeshCollider;
use super::narrow::Ctx;
use super::state::{Body, BodyConst, BodyPrev, Contact, DEACTIVATES, Kind, Shape, ShapeKind};

pub struct World {
    pub bodies: Vec<Body>,
    pub consts: Vec<BodyConst>,
    pub prev: Vec<BodyPrev>,
    pub shapes: Vec<Shape>,
    pub hull: Vec<[f32; 4]>,
    pub meshes: Vec<MeshCollider>,
    /// This frame's contacts, in pair order.
    pub contacts: Vec<Contact>,
    pub gravity: V3,
    pub substeps: u32,
    /// Projections per substep.
    pub iterations: u32,
    pub time_scale: f32,
    /// How fast a penetration may be pushed out, in metres a second. A
    /// substep derives velocity from its pose change, so correcting a
    /// deep penetration in one substep turns it into speed; this is the
    /// rate limit that keeps it from becoming a launch, as Bullet's error
    /// reduction does (`contact::project`).
    pub max_recover: f32,
    /// Seconds under the sleep thresholds before a body sleeps (Bullet: 2).
    pub sleep_after: f32,
    /// Objects given as Mesh shapes on active bodies, simulated as hulls.
    pub hull_fallback: Vec<usize>,
    pub(super) places: Vec<Place>,
    pub(super) sleep: Vec<[f32; 2]>,
    pub(super) timers: Vec<f32>,
    pub(super) targets: Vec<Option<(V3, Q)>>,
    vn0: Vec<f32>,
    /// This substep's pure penetration recovery per body, which
    /// `velocities` leaves out of the velocity it derives.
    push: Vec<V3>,
    spin: Vec<V3>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            consts: Vec::new(),
            prev: Vec::new(),
            shapes: Vec::new(),
            hull: Vec::new(),
            meshes: Vec::new(),
            contacts: Vec::new(),
            gravity: [0.0, 0.0, -9.81],
            substeps: 10,
            // Blender's Rigid Body World defaults: 10 substeps a frame,
            // 10 solver iterations in each (`blend.rs`).
            iterations: 10,
            time_scale: 1.0,
            max_recover: 2.0,
            sleep_after: 2.0,
            hull_fallback: Vec::new(),
            places: Vec::new(),
            sleep: Vec::new(),
            timers: Vec::new(),
            targets: Vec::new(),
            vn0: Vec::new(),
            push: Vec::new(),
            spin: Vec::new(),
        }
    }

    /// One frame of `frame_dt` seconds.
    pub fn step(&mut self, frame_dt: f32) {
        let dt = frame_dt * self.time_scale;
        if dt <= 0.0 || self.substeps == 0 {
            return;
        }
        for (i, t) in self.targets.iter().enumerate() {
            if let Some((x, q)) = t {
                let b = &mut self.bodies[i];
                b.v = scale(sub(*x, b.x), 1.0 / dt);
                b.w = q_delta_w(*q, b.q, dt);
            }
        }
        self.collide(dt);
        let h = dt / self.substeps as f32;
        let g = len(self.gravity);
        for _ in 0..self.substeps {
            contact::presolve(&self.bodies, &mut self.contacts, &mut self.vn0);
            self.integrate(h);
            // The substep's recovery budget, split between its
            // projections: `velocities` measures one pose change over
            // `h`, so what the projections push out together is what
            // would turn into speed.
            let cap = (self.max_recover * h / self.iterations.max(1) as f32).max(0.0);
            self.push.iter_mut().for_each(|p| *p = [0.0; 3]);
            self.spin.iter_mut().for_each(|s| *s = [0.0; 3]);
            for _ in 0..self.iterations {
                contact::project(&mut self.bodies, &self.consts, &self.prev, &mut self.contacts, cap, &mut self.push, &mut self.spin);
            }
            self.velocities(h);
            contact::velocity_pass(&mut self.bodies, &self.consts, &mut self.contacts, &self.vn0, h, g);
            self.kick(h);
        }
        for (i, t) in self.targets.iter_mut().enumerate() {
            if let Some((x, q)) = t.take() {
                (self.bodies[i].x, self.bodies[i].q) = (x, q);
            }
        }
        self.sleep(dt);
    }

    /// Bounds, broadphase, narrowphase; a contact wakes what it touches.
    fn collide(&mut self, dt: f32) {
        let aabbs = broad::bounds(&self.bodies, &self.consts, &self.shapes, &self.hull, &self.meshes, dt);
        let pairs = broad::pairs(&aabbs, &self.bodies, &self.consts);
        self.contacts.clear();
        let ctx = Ctx { bodies: &self.bodies, consts: &self.consts, shapes: &self.shapes, hull: &self.hull, meshes: &self.meshes, aabbs: &aabbs, dt };
        for (i, j) in pairs {
            ctx.pair(i as usize, j as usize, &mut self.contacts);
        }
        for c in &self.contacts {
            for (me, other) in [(c.a as usize, c.b as usize), (c.b as usize, c.a as usize)] {
                if self.bodies[me].asleep() && self.bodies[other].moves() {
                    self.bodies[me].set_asleep(false);
                    self.timers[me] = 0.0;
                }
            }
        }
        self.vn0.resize(self.contacts.len(), 0.0);
        self.push.resize(self.bodies.len(), [0.0; 3]);
        self.spin.resize(self.bodies.len(), [0.0; 3]);
    }

    /// Half a kick of gravity, then the drift; kinematic bodies drift only.
    fn integrate(&mut self, h: f32) {
        for (b, p) in self.bodies.iter_mut().zip(&mut self.prev) {
            *p = BodyPrev { x: b.x, _a: 0.0, q: b.q };
            if !b.moves() {
                continue;
            }
            if b.kind() == Kind::Active {
                b.v = add(b.v, scale(self.gravity, 0.5 * h));
            }
            b.x = add(b.x, scale(b.v, h));
            b.q = q_integrate(b.q, b.w, h);
        }
    }

    /// Velocities from the poses: what the constraints moved a body by,
    /// over `h`, on top of the velocity it drifted with. The change is
    /// measured against the pose the drift predicted (the expression
    /// `integrate` used, so it is zero to the bit where nothing touched
    /// the body): `(x - prev) / h` at tens of metres loses the substep's
    /// gravity to f32 rounding.
    /// What the projections pushed out of a standing penetration
    /// (`push`, `spin`) is left out: it moved the pose on purpose and is
    /// not motion the body acquired. Without that, letting a deep
    /// penetration out at `max_recover` added that speed to the body
    /// every substep it took — the frictionless ball came off the ground
    /// faster than it arrived and its energy grew by 70%.
    fn velocities(&mut self, h: f32) {
        for (i, (b, p)) in self.bodies.iter_mut().zip(&self.prev).enumerate() {
            if !b.dynamic() {
                continue;
            }
            let (x_pred, q_pred) = (add(p.x, scale(b.v, h)), q_integrate(p.q, b.w, h));
            b.v = add(b.v, scale(sub(sub(b.x, x_pred), self.push[i]), 1.0 / h));
            b.w = add(b.w, sub(q_delta_w(b.q, q_pred, h), scale(self.spin[i], 1.0 / h)));
        }
    }

    /// The other half kick of gravity, then Blender's damping.
    fn kick(&mut self, h: f32) {
        for (b, c) in self.bodies.iter_mut().zip(&self.consts) {
            if !b.dynamic() {
                continue;
            }
            b.v = add(b.v, scale(self.gravity, 0.5 * h));
            b.v = scale(b.v, (1.0 - c.lin_damp).max(0.0).powf(h));
            b.w = scale(b.w, (1.0 - c.ang_damp).max(0.0).powf(h));
        }
    }

    /// Blender's deactivation: under both thresholds for `sleep_after`.
    fn sleep(&mut self, dt: f32) {
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if !b.dynamic() || b.flags & DEACTIVATES == 0 {
                continue;
            }
            let [lin, ang] = self.sleep[i];
            let still = len(b.v) < lin && len(b.w) < ang;
            self.timers[i] = if still { self.timers[i] + dt } else { 0.0 };
            if self.timers[i] >= self.sleep_after {
                b.set_asleep(true);
                (b.v, b.w) = ([0.0; 3], [0.0; 3]);
            }
        }
    }

    /// Each body's object and its matrix, for `Tracer::reinstance`.
    pub fn poses(&self) -> impl Iterator<Item = (usize, [[f32; 4]; 4])> + '_ {
        self.bodies.iter().zip(&self.consts).zip(&self.places).map(|((b, c), p)| (p.obj, pose_matrix(b.x, b.q, c.com_local, p.scale)))
    }

    /// Kinetic plus potential energy of the active bodies.
    pub fn energy(&self) -> f32 {
        self.bodies
            .iter()
            .zip(&self.consts)
            .filter(|(b, _)| b.inv_mass > 0.0)
            .map(|(b, c)| {
                let m = 1.0 / b.inv_mass;
                let wb = unrotate(b.q, b.w);
                let rot: f32 = (0..3).map(|k| if c.inv_inertia[k] > 0.0 { 0.5 * wb[k] * wb[k] / c.inv_inertia[k] } else { 0.0 }).sum();
                0.5 * m * len2(b.v) + rot - m * dot(self.gravity, b.x)
            })
            .sum()
    }

    /// Whether shape `s` is a passive triangle mesh.
    pub fn is_mesh(&self, s: u32) -> bool {
        self.shapes[s as usize].kind() == ShapeKind::Mesh
    }
}
