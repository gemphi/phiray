//! Binned SAH builder — the tree the tracer walks.
//!
//! The Karras build (`build.rs`) sorts primitives along a Morton curve and
//! splits wherever the code's top bit changes. That is fast to build and
//! what the GPU kernels mirror, but where it splits has nothing to do with
//! what a ray pays: two halves of a figure's hair can share a node because
//! their centres share a Morton prefix, and every ray through either half
//! then visits both. Cycles builds with the Surface Area Heuristic
//! (`intern/cycles/bvh/build.cpp`, `BVHObjectBinning`), which places each
//! split where the expected cost of the two children — their surface area
//! times what they hold — is lowest. A tree built that way is visited far
//! less per ray, and traversal is most of what a sample costs.
//!
//! This is the binned form (Wald, *On fast Construction of SAH-based
//! Bounding Volume Hierarchies*, 2007): centroids in [`BINS`] bins per axis,
//! the best of the `3 × (BINS − 1)` planes, a leaf when splitting would
//! cost more than intersecting what the node holds. The node layout, the
//! slot order and the depth cap are `build_compact`'s, so the walk, the
//! refit and the GPU upload take either tree.

use super::Bvh;
use super::node::{Aabb, Node, STACK};
use rayon::prelude::*;

/// Centroid bins per axis (Cycles bins at 32; 16 loses little).
const BINS: usize = 16;
/// Below this many primitives a node's bins are filled on one thread.
const PARALLEL: usize = 1 << 15;
/// The cost of stepping into a node against intersecting one primitive.
const TRAVERSE: f32 = 1.0;

fn area(b: &Aabb) -> f32 {
    let d: [f32; 3] = std::array::from_fn(|k| (b.hi[k] - b.lo[k]).max(0.0));
    2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
}

/// One axis's bins: how many centroids each holds and their primitives'
/// bounds.
#[derive(Clone, Copy)]
struct Bins {
    count: [[u32; BINS]; 3],
    bounds: [[Aabb; BINS]; 3],
}

impl Bins {
    fn empty() -> Self {
        Self { count: [[0; BINS]; 3], bounds: [[Aabb::default(); BINS]; 3] }
    }

    fn merge(mut self, o: Self) -> Self {
        for k in 0..3 {
            for b in 0..BINS {
                self.count[k][b] += o.count[k][b];
                self.bounds[k][b] = self.bounds[k][b].union(o.bounds[k][b]);
            }
        }
        self
    }
}

/// Where a centroid falls along each axis of `span`.
struct Grid {
    lo: [f32; 3],
    scale: [f32; 3],
}

impl Grid {
    fn of(span: &Aabb) -> Self {
        let scale = std::array::from_fn(|k| {
            let e = span.hi[k] - span.lo[k];
            if e > 0.0 { BINS as f32 * (1.0 - 1e-5) / e } else { 0.0 }
        });
        Self { lo: span.lo, scale }
    }

    fn bin(&self, c: [f32; 3], k: usize) -> usize {
        (((c[k] - self.lo[k]) * self.scale[k]) as usize).min(BINS - 1)
    }
}

impl Bvh {
    /// The tracer's tree over `prims`: binned SAH, leaves of at most `leaf`
    /// primitives unless a split would cost more (or the walk's stack would
    /// overflow), in `build_compact`'s layout.
    pub fn build_sah(prims: &[Aabb], leaf: u32) -> Self {
        let n = prims.len();
        if n == 0 {
            return Self::default();
        }
        let centres: Vec<[f32; 3]> = prims.par_iter().map(Aabb::center).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = vec![Node::default()];
        let mut depth = 0;
        let mut work = vec![(0usize, 0usize, n, 0u32)];
        while let Some((slot, start, end, d)) = work.pop() {
            depth = depth.max(d);
            let cut = if d + 1 >= STACK as u32 { None } else { split(prims, &centres, &mut order[start..end], leaf as usize) };
            match cut {
                None => nodes[slot] = Node::leaf([0.0; 3], [0.0; 3], start as u32, end as u32),
                Some(m) => {
                    let at = nodes.len();
                    nodes.extend([Node::default(); 2]);
                    nodes[slot] = Node::inner([0.0; 3], [0.0; 3], at as u32, at as u32 + 1);
                    work.push((at + 1, start + m, end, d + 1));
                    work.push((at, start, start + m, d + 1));
                }
            }
        }
        let mut bvh = Self { nodes, order, depth };
        bvh.refit(prims);
        bvh
    }
}

/// The SAH split of `slots` (reordered so the left child is the first `m`),
/// or `None` for a leaf.
fn split(prims: &[Aabb], centres: &[[f32; 3]], slots: &mut [u32], leaf: usize) -> Option<usize> {
    let count = slots.len();
    if count <= 1 {
        return None;
    }
    let span = |s: &[u32]| s.iter().fold(Aabb::default(), |a, &i| a.union(Aabb::new(centres[i as usize], centres[i as usize])));
    let cbox = if count >= PARALLEL { slots.par_chunks(PARALLEL / 4).map(span).reduce(Aabb::default, Aabb::union) } else { span(slots) };
    let grid = Grid::of(&cbox);
    if grid.scale == [0.0; 3] {
        // Every centre in one point: no plane separates them, so halve the
        // list — unless it is small enough to be a leaf anyway.
        return (count > leaf).then_some(count / 2);
    }
    let fill = |s: &[u32]| {
        let mut b = Bins::empty();
        for &i in s {
            let c = centres[i as usize];
            for k in 0..3 {
                let at = grid.bin(c, k);
                b.count[k][at] += 1;
                b.bounds[k][at] = b.bounds[k][at].union(prims[i as usize]);
            }
        }
        b
    };
    let bins = if count >= PARALLEL { slots.par_chunks(PARALLEL / 4).map(fill).reduce(Bins::empty, Bins::merge) } else { fill(slots) };
    // Sweep each axis once from the right to have every right side's area
    // and count, then from the left to price each plane.
    let mut best: Option<(f32, usize, usize)> = None;
    for k in 0..3 {
        if grid.scale[k] == 0.0 {
            continue;
        }
        let mut right = [(0.0f32, 0u32); BINS];
        let (mut b, mut c) = (Aabb::default(), 0u32);
        for at in (1..BINS).rev() {
            b = b.union(bins.bounds[k][at]);
            c += bins.count[k][at];
            right[at] = (area(&b), c);
        }
        let (mut b, mut c) = (Aabb::default(), 0u32);
        for at in 0..BINS - 1 {
            b = b.union(bins.bounds[k][at]);
            c += bins.count[k][at];
            let (ra, rc) = right[at + 1];
            if c == 0 || rc == 0 {
                continue;
            }
            let cost = area(&b) * c as f32 + ra * rc as f32;
            if best.is_none_or(|(was, ..)| cost < was) {
                best = Some((cost, k, at));
            }
        }
    }
    let bounds = bins.bounds[0].iter().fold(Aabb::default(), |a, b| a.union(*b));
    let Some((cost, k, at)) = best else {
        return (count > leaf).then_some(count / 2);
    };
    // Split only where it pays: a node's own visit plus its children's
    // expected work, against testing everything it holds.
    let parent = area(&bounds).max(f32::MIN_POSITIVE);
    if count <= leaf && TRAVERSE + cost / parent >= count as f32 {
        return None;
    }
    let mut m = 0;
    for j in 0..count {
        if grid.bin(centres[slots[j] as usize], k) <= at {
            slots.swap(j, m);
            m += 1;
        }
    }
    Some(if m == 0 || m == count { count / 2 } else { m })
}
