//! The broadphase (ROADMAP.md §2.2 "One step" 1-2 and "Collision"): swept
//! bounds per body, phiray's `Bvh` over them (the tree the GPU `Overlap`
//! kernel will walk with the same 32-slot stack), and the pairs whose
//! boxes overlap, whose collision groups meet and of which one is active,
//! sorted so a frame is the same every run.

use super::math::*;
use super::mesh::MeshCollider;
use super::shape;
use super::state::{Body, BodyConst, Kind, Shape, ShapeKind};
use crate::bvh::{Aabb, Bvh};

/// Padding past the margin and the narrowphase's slop, so a pair is seen
/// the frame before it touches.
const PAD: f32 = super::narrow::SLOP + 0.005;

pub fn overlaps(a: &Aabb, b: &Aabb) -> bool {
    (0..3).all(|k| a.lo[k] <= b.hi[k] && a.hi[k] >= b.lo[k])
}

/// The primitives of `bvh` (built over `prims`) whose box meets `b`,
/// appended to `out` in traversal order.
pub fn query(bvh: &Bvh, prims: &[Aabb], b: &Aabb, out: &mut Vec<u32>) {
    if bvh.nodes.is_empty() {
        return;
    }
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        let n = bvh.nodes[i];
        if !overlaps(&n.bounds(), b) {
            continue;
        }
        match n.is_leaf() {
            true => {
                for slot in n.lo.index()..n.hi.index() {
                    let p = bvh.order[slot as usize];
                    if overlaps(&prims[p as usize], b) {
                        out.push(p);
                    }
                }
            }
            false => stack.extend([n.lo.index() as usize, n.hi.index() as usize]),
        }
    }
}

/// Each body's bounds swept by its motion over `dt`, grown by its margin.
pub fn bounds(bodies: &[Body], consts: &[BodyConst], shapes: &[Shape], hull: &[[f32; 4]], meshes: &[MeshCollider], dt: f32) -> Vec<Aabb> {
    bodies
        .iter()
        .zip(consts)
        .map(|(b, c)| {
            let s = &shapes[b.shape as usize];
            let now = match s.kind() {
                ShapeKind::Mesh => meshes[s.data as usize].world_bounds(b.x, b.q),
                _ => shape::bounds(s, hull, b.x, b.q),
            };
            let shift = scale(b.v, dt);
            let swept = now.union(Aabb::new(add(now.lo, shift), add(now.hi, shift)));
            let pad = c.margin + PAD;
            Aabb::new(map3(|k| swept.lo[k] - pad), map3(|k| swept.hi[k] + pad))
        })
        .collect()
}

/// Sorted pairs `(i, j)`, `i < j`, whose boxes overlap and groups meet,
/// one of them active; a sleeping body is found by what moves into it.
pub fn pairs(aabbs: &[Aabb], bodies: &[Body], consts: &[BodyConst]) -> Vec<(u32, u32)> {
    let bvh = Bvh::build(aabbs, 4);
    let mut out = Vec::new();
    let mut hits = Vec::new();
    for (i, b) in bodies.iter().enumerate() {
        if !b.moves() {
            continue;
        }
        hits.clear();
        query(&bvh, aabbs, &aabbs[i], &mut hits);
        for &j in &hits {
            let j = j as usize;
            let other = &bodies[j];
            let active = b.kind() == Kind::Active || other.kind() == Kind::Active;
            if j == i || !active || consts[i].groups & consts[j].groups == 0 {
                continue;
            }
            out.push((i.min(j) as u32, i.max(j) as u32));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}
