//! The narrowphase (ROADMAP.md §2.2 "Collision"): one `match` on the
//! ordered pair of shape kinds picks closed forms, SAT with clipping or
//! GJK/EPA; Mesh shapes run the midphase over their triangle tree and
//! take each triangle through the same `match`, with the triangle's
//! plane as the contact normal. Contacts come out with body-local anchors
//! (Muller 2020), so the substeps re-evaluate them.

mod closed;
mod epa;
mod gjk;
mod manifold;
mod sat;

pub use gjk::Convex;
pub use manifold::{Manifold, Point};
pub use sat::{Poly, sat};

use super::math::*;
use super::mesh::MeshCollider;
use super::shape::segment;
use super::state::{Body, BodyConst, Contact, Shape, ShapeKind};
use crate::bvh::Aabb;

/// How far apart two surfaces may be and still get a (speculative)
/// contact, past their margins: the substeps then catch the touch.
pub const SLOP: f32 = 0.01;

/// One body as the narrowphase sees it.
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub shape: &'a Shape,
    pub hull: &'a [[f32; 4]],
    pub x: V3,
    pub q: Q,
    pub margin: f32,
}

impl<'a> View<'a> {
    fn kind(&self) -> ShapeKind {
        self.shape.kind()
    }

    fn convex(&self) -> Convex<'a> {
        Convex::Shape { s: self.shape, hull: self.hull, x: self.x, q: self.q }
    }

    fn poly(&self) -> Poly {
        Poly::boxed(self.x, self.q, self.shape.half)
    }
}

/// GJK for the gap, EPA for the penetration.
pub fn gjk_epa(a: &Convex, b: &Convex, slop: f32) -> Manifold {
    match gjk::gjk(a, b) {
        gjk::Gjk::Apart { a: pa, b: pb, dist } => match dist <= slop {
            true => Manifold::one(norm(sub(pb, pa)), pa, pb),
            false => Manifold::none(),
        },
        gjk::Gjk::Overlap(s) => match epa::epa(a, b, s) {
            Some(e) => epa::manifold(a, b, &e, slop),
            None => Manifold::none(),
        },
    }
}

/// Two convex shapes; the normal points from `a` to `b`. `extra` widens
/// the speculative reach by how far the pair can close before the next
/// narrowphase (`Ctx::pair`).
pub fn convex_pair(a: &View, b: &View, extra: f32) -> Manifold {
    use ShapeKind::*;
    let slop = a.margin + b.margin + SLOP + extra;
    match (a.kind(), b.kind()) {
        (Sphere, Sphere) => closed::spheres(a.x, a.shape.radius, b.x, b.shape.radius, slop),
        (Sphere, Capsule) => closed::sphere_capsule(a.x, a.shape.radius, segment(b.shape, b.x, b.q), b.shape.radius, slop),
        (Capsule, Sphere) => convex_pair(b, a, extra).flip(),
        (Capsule, Capsule) => closed::capsules(segment(a.shape, a.x, a.q), a.shape.radius, segment(b.shape, b.x, b.q), b.shape.radius, slop),
        (Sphere, Box) => closed::sphere_box(a.x, a.shape.radius, b.x, b.q, b.shape.half, slop),
        (Box, Sphere) => convex_pair(b, a, extra).flip(),
        (Box, Box) => sat(&a.poly(), &b.poly(), slop, None),
        _ => gjk_epa(&a.convex(), &b.convex(), slop),
    }
}

/// A convex shape against one triangle of a mesh, the contact along the
/// triangle's plane on the body's side.
pub fn tri_pair(a: &View, t: [V3; 3], slop: f32) -> Manifold {
    let n = norm(cross(sub(t[1], t[0]), sub(t[2], t[0])));
    let m = match a.kind() {
        ShapeKind::Sphere => closed::sphere_triangle(a.x, a.shape.radius, t, slop),
        ShapeKind::Box => sat(&a.poly(), &Poly::triangle(t), slop, Some(n)),
        _ => gjk_epa(&a.convex(), &Convex::Tri(t), slop),
    };
    if m.n == 0 {
        return m;
    }
    let facing = if dot(m.normal, n) >= 0.0 { n } else { neg(n) };
    m.along(facing, t[0])
}

/// Everything the narrowphase reads.
pub struct Ctx<'a> {
    pub bodies: &'a [Body],
    pub consts: &'a [BodyConst],
    pub shapes: &'a [Shape],
    pub hull: &'a [[f32; 4]],
    pub meshes: &'a [MeshCollider],
    pub aabbs: &'a [Aabb],
    /// The frame this narrowphase serves, in seconds.
    pub dt: f32,
}

impl Ctx<'_> {
    fn view(&self, i: usize) -> View<'_> {
        let b = &self.bodies[i];
        View { shape: &self.shapes[b.shape as usize], hull: self.hull, x: b.x, q: b.q, margin: self.consts[i].margin }
    }

    /// How far pair `(i, j)` can close before the next narrowphase: a
    /// contact is made that far out, apart and inert, so the substeps
    /// meet the surface at the moment the pair reaches it rather than
    /// wherever the next frame finds them. A ball dropped from two
    /// metres closes 10 cm a frame and was 5 cm *under* the ground
    /// before any contact existed, so it bounced from down there and
    /// came back to 0.47 m instead of 0.5.
    fn reach(&self, i: usize, j: usize) -> f32 {
        (len(sub(self.bodies[i].v, self.bodies[j].v)) * self.dt).max(0.0)
    }

    /// The contacts of pair `(i, j)`, appended to `out`. Mesh against mesh
    /// is not supported (an active Mesh body is a hull already).
    pub fn pair(&self, i: usize, j: usize, out: &mut Vec<Contact>) {
        let (a, b) = (self.view(i), self.view(j));
        match (a.kind(), b.kind()) {
            (ShapeKind::Mesh, ShapeKind::Mesh) => {}
            (ShapeKind::Mesh, _) => self.against_mesh(j, i, out),
            (_, ShapeKind::Mesh) => self.against_mesh(i, j, out),
            _ => self.push(i, j, &convex_pair(&a, &b, self.reach(i, j)), out),
        }
    }

    /// Convex body `i` against the triangles of mesh body `m` under its box.
    fn against_mesh(&self, i: usize, m: usize, out: &mut Vec<Contact>) {
        let a = self.view(i);
        let mb = &self.bodies[m];
        let mesh = &self.meshes[self.shapes[mb.shape as usize].data as usize];
        let slop = a.margin + self.consts[m].margin + SLOP + self.reach(i, m);
        let mut tris = Vec::new();
        mesh.under(&self.aabbs[i], mb.x, mb.q, &mut tris);
        for t in tris {
            let tri = mesh.tris[t as usize].map(|v| add(mb.x, rotate(mb.q, v)));
            self.push(i, m, &tri_pair(&a, tri, slop), out);
        }
    }

    fn push(&self, i: usize, j: usize, m: &Manifold, out: &mut Vec<Contact>) {
        let (a, b) = (&self.bodies[i], &self.bodies[j]);
        for k in 0..m.n {
            let p = m.points[k];
            out.push(Contact {
                a: i as u32,
                b: j as u32,
                anchor_a: unrotate(a.q, sub(p.a, a.x)),
                lambda_n: 0.0,
                anchor_b: unrotate(b.q, sub(p.b, b.x)),
                lambda_t: 0.0,
                normal: m.normal,
                depth0: m.depth(k),
            });
        }
    }
}
