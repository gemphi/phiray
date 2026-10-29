//! Passive triangle meshes (ROADMAP.md §2.2 "Collision", midphase): a Mesh
//! shape keeps its triangles in its body's space under phiray's SAH tree,
//! the builder the tracer uses for its object trees, and the narrowphase
//! asks it for the triangles under a convex body's box.

use super::broad::query;
use super::math::*;
use crate::bvh::{Aabb, Bvh};

pub struct MeshCollider {
    /// Triangles in the body's space (the object's scale baked in).
    pub tris: Vec<[V3; 3]>,
    pub boxes: Vec<Aabb>,
    pub bvh: Bvh,
    /// Bounds of every triangle, body space.
    pub bounds: Aabb,
}

impl MeshCollider {
    pub fn new(tris: Vec<[V3; 3]>) -> Self {
        let boxes: Vec<Aabb> = tris.iter().map(|t| Aabb::of_tri(t[0], t[1], t[2])).collect();
        let bvh = Bvh::build_sah(&boxes, 4);
        let bounds = if boxes.is_empty() { Aabb::new([0.0; 3], [0.0; 3]) } else { Bvh::scene(&boxes) };
        Self { tris, boxes, bvh, bounds }
    }

    /// Half extents of the body-space bounds.
    pub fn half(&self) -> V3 {
        map3(|k| 0.5 * (self.bounds.hi[k] - self.bounds.lo[k]))
    }

    /// The corners of `b` taken through `f`, bounded again.
    fn corners(b: &Aabb, f: impl Fn(V3) -> V3) -> Aabb {
        (0..8).fold(Aabb::default(), |acc, i| {
            let c = map3(|k| if (i >> k) & 1 == 1 { b.hi[k] } else { b.lo[k] });
            let p = f(c);
            acc.union(Aabb::new(p, p))
        })
    }

    /// The bounds in world when the body sits at `x`, `q`.
    pub fn world_bounds(&self, x: V3, q: Q) -> Aabb {
        Self::corners(&self.bounds, |c| add(x, rotate(q, c)))
    }

    /// The triangles under the world box `b` for a body at `x`, `q`, in
    /// index order.
    pub fn under(&self, b: &Aabb, x: V3, q: Q, out: &mut Vec<u32>) {
        let local = Self::corners(b, |c| unrotate(q, sub(c, x)));
        query(&self.bvh, &self.boxes, &local, out);
        out.sort_unstable();
    }
}
