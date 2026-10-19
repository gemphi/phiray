//! The four-wide view of a tree — what the kernel walks.
//!
//! The binary tree (`Node`, Karras links, `refit`) stays the thing that is
//! built and refitted. This is a *view* of it: every inner binary node is
//! collapsed, with up to three of its descendants, into one node of four
//! children, the way cuBQL's `wide_gpu_builder.h` does on the GPU
//! (`collapseInit` → `collapseSummarize` → `collapseExecute`, in the vendored
//! Warp under `phiray/reference/warp/native/cuBQL`). Measured offline on pose
//! 26 (`render-wavefront.md`): a four-wide walk fetches 1.7× fewer nodes and
//! pushes 1.7× fewer stack entries than the binary one, for 4% more box
//! tests — and box tests are arithmetic while fetches and pushes are memory.
//!
//! Which descendants join a node: starting from the binary node's two
//! children, the inner candidate with the largest surface area is opened
//! (replaced by its own two children) until four candidates stand or none is
//! inner. Surface area is the usual greedy choice (the child a ray is most
//! likely to enter is the one worth testing directly) and it is what cuBQL's
//! summarise step picks too.
//!
//! Layout (128 bytes, WGSL `struct Wide`): the four children's bounds as six
//! `vec4<f32>` — x then y then z of the lower corner, the same of the upper
//! — so one slab test does all four in vector arithmetic; then `c`, the
//! children, and `e`, the leaf ends. A child word is an index into the wide
//! array, or `LEAF | start` with the end in `e` (and, in the top tree, the
//! `INST` flag over instance slots, exactly as the binary leaf carried it).
//! An empty slot has an inverted box (`lo = +INF`, `hi = -INF`) and the
//! child word `0`, which is never a child (the root is node 0): the kernel
//! masks on the word, because the box alone is not enough — under the slab
//! arithmetic infinities give `tmin = -INF, tmax = +INF`, a hit.

use super::node::{Aabb, LEAF, Node};
use bytemuck::{Pod, Zeroable};

/// How many children a node holds.
pub const WIDTH: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Wide {
    pub lox: [f32; 4],
    pub loy: [f32; 4],
    pub loz: [f32; 4],
    pub hix: [f32; 4],
    pub hiy: [f32; 4],
    pub hiz: [f32; 4],
    /// Children: a wide index, or `LEAF | start` (plus `INST` in the top tree).
    pub c: [u32; 4],
    /// A leaf child's end; `0` for an inner child or an empty slot.
    pub e: [u32; 4],
}

impl Default for Wide {
    fn default() -> Self {
        Self {
            lox: [f32::INFINITY; 4],
            loy: [f32::INFINITY; 4],
            loz: [f32::INFINITY; 4],
            hix: [f32::NEG_INFINITY; 4],
            hiy: [f32::NEG_INFINITY; 4],
            hiz: [f32::NEG_INFINITY; 4],
            c: [0; 4],
            e: [0; 4],
        }
    }
}

impl Wide {
    /// Set child `k` to `b` with the child word `c` (and end `e`).
    pub fn set(&mut self, k: usize, b: Aabb, c: u32, e: u32) {
        (self.lox[k], self.loy[k], self.loz[k]) = (b.lo[0], b.lo[1], b.lo[2]);
        (self.hix[k], self.hiy[k], self.hiz[k]) = (b.hi[0], b.hi[1], b.hi[2]);
        self.c[k] = c;
        self.e[k] = e;
    }

    /// Child `k`'s box.
    pub fn child_bounds(&self, k: usize) -> Aabb {
        Aabb::new([self.lox[k], self.loy[k], self.loz[k]], [self.hix[k], self.hiy[k], self.hiz[k]])
    }

    /// Is slot `k` filled?
    pub fn filled(&self, k: usize) -> bool {
        self.lox[k] != f32::INFINITY
    }

    pub fn is_leaf_child(&self, k: usize) -> bool {
        self.c[k] & LEAF != 0
    }
}

fn area(b: &Aabb) -> f32 {
    let d: [f32; 3] = std::array::from_fn(|k| (b.hi[k] - b.lo[k]).max(0.0));
    2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
}

/// The wide view of `nodes` (a binary tree with root 0, bounds already
/// fitted). Returns the wide nodes — root first — and, per wide node, the
/// binary index each child slot stands for (`u32::MAX` for an empty slot),
/// so a refit of the binary bounds can be carried over without collapsing
/// again (`rebound`).
pub fn collapse(nodes: &[Node]) -> (Vec<Wide>, Vec<[u32; WIDTH]>) {
    if nodes.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut out: Vec<Wide> = Vec::with_capacity(nodes.len() / 2 + 1);
    let mut of: Vec<[u32; WIDTH]> = Vec::with_capacity(nodes.len() / 2 + 1);
    // (binary node that becomes a wide node, its slot in `out`).
    let mut todo: Vec<(u32, usize)> = vec![(0, 0)];
    out.push(Wide::default());
    of.push([u32::MAX; WIDTH]);
    while let Some((bi, slot)) = todo.pop() {
        let root = nodes[bi as usize];
        // The candidates: the binary children of this node — or, for a root
        // that is itself a leaf (a one-leaf tree), the leaf alone.
        let mut cand: Vec<u32> = if root.is_leaf() { vec![bi] } else { vec![root.lo.index(), root.hi.index()] };
        loop {
            if cand.len() >= WIDTH {
                break;
            }
            // Open the inner candidate with the largest surface area.
            let pick = cand
                .iter()
                .enumerate()
                .filter(|(_, c)| !nodes[**c as usize].is_leaf())
                .max_by(|a, b| area(&nodes[*a.1 as usize].bounds()).total_cmp(&area(&nodes[*b.1 as usize].bounds())))
                .map(|(k, _)| k);
            let Some(k) = pick else { break };
            let open = nodes[cand[k] as usize];
            cand[k] = open.lo.index();
            cand.push(open.hi.index());
        }
        let mut w = Wide::default();
        let mut map = [u32::MAX; WIDTH];
        for (k, &c) in cand.iter().enumerate() {
            let n = nodes[c as usize];
            map[k] = c;
            if n.is_leaf() {
                w.set(k, n.bounds(), n.lo.i, n.hi.i);
            } else {
                let at = out.len();
                out.push(Wide::default());
                of.push([u32::MAX; WIDTH]);
                todo.push((c, at));
                w.set(k, n.bounds(), at as u32, 0);
            }
        }
        out[slot] = w;
        of[slot] = map;
    }
    (out, of)
}

/// The wide bounds again from refitted binary `nodes`, the structure kept.
pub fn rebound(wide: &mut [Wide], of: &[[u32; WIDTH]], nodes: &[Node]) {
    for (w, map) in wide.iter_mut().zip(of) {
        for k in 0..WIDTH {
            if map[k] != u32::MAX {
                let b = nodes[map[k] as usize].bounds();
                (w.lox[k], w.loy[k], w.loz[k]) = (b.lo[0], b.lo[1], b.lo[2]);
                (w.hix[k], w.hiy[k], w.hiz[k]) = (b.hi[0], b.hi[1], b.hi[2]);
            }
        }
    }
}

/// The deepest wide node from the root, for the kernel's stack.
pub fn depth(wide: &[Wide]) -> u32 {
    let mut best = 0;
    let mut stack = if wide.is_empty() { vec![] } else { vec![(0usize, 0u32)] };
    while let Some((i, d)) = stack.pop() {
        best = best.max(d);
        let w = &wide[i];
        for k in 0..WIDTH {
            if w.filled(k) && !w.is_leaf_child(k) {
                stack.push((w.c[k] as usize, d + 1));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::Bvh;

    fn boxes(n: usize) -> Vec<Aabb> {
        (0..n)
            .map(|i| {
                let f = i as f32;
                let c = [(f * 7.3) % 50.0, (f * 3.1) % 40.0, (f * 11.7) % 30.0];
                Aabb::new([c[0] - 0.3, c[1] - 0.3, c[2] - 0.3], [c[0] + 0.3, c[1] + 0.3, c[2] + 0.3])
            })
            .collect()
    }

    #[test]
    fn layout_is_128_bytes() {
        assert_eq!(std::mem::size_of::<Wide>(), 128);
    }

    /// Every leaf of the binary tree is reached exactly once through the
    /// wide one, and every wide child box is the binary node's box.
    #[test]
    fn the_wide_tree_reaches_every_leaf_once_with_the_same_bounds() {
        let b = boxes(500);
        let bvh = Bvh::build_compact(&b, 4);
        let (wide, of) = collapse(&bvh.nodes);
        assert!(wide.len() < bvh.nodes.len(), "fewer nodes: {} of {}", wide.len(), bvh.nodes.len());
        let mut seen = vec![0u32; b.len()];
        let mut stack = vec![0usize];
        let mut inner_seen = 0;
        while let Some(i) = stack.pop() {
            let w = &wide[i];
            for k in 0..WIDTH {
                if !w.filled(k) {
                    continue;
                }
                let bi = of[i][k] as usize;
                assert_eq!(w.child_bounds(k), bvh.nodes[bi].bounds(), "child {k} of wide {i}");
                if w.is_leaf_child(k) {
                    let n = bvh.nodes[bi];
                    assert!(n.is_leaf());
                    assert_eq!((w.c[k], w.e[k]), (n.lo.i, n.hi.i));
                    for s in n.lo.index()..n.hi.index() {
                        seen[bvh.order[s as usize] as usize] += 1;
                    }
                } else {
                    inner_seen += 1;
                    stack.push(w.c[k] as usize);
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "every primitive once");
        assert_eq!(inner_seen + 1, wide.len(), "every wide node reached once");
        assert!(depth(&wide) <= bvh.depth, "no deeper than the binary tree");
    }

    #[test]
    fn a_rebound_matches_a_fresh_collapse() {
        let b = boxes(300);
        let mut bvh = Bvh::build_compact(&b, 4);
        let (mut wide, of) = collapse(&bvh.nodes);
        let moved: Vec<Aabb> = b.iter().map(|a| Aabb::new([a.lo[0] + 5.0, a.lo[1], a.lo[2]], [a.hi[0] + 5.0, a.hi[1], a.hi[2]])).collect();
        bvh.refit(&moved);
        rebound(&mut wide, &of, &bvh.nodes);
        let (fresh, _) = collapse(&bvh.nodes);
        assert_eq!(wide, fresh);
    }

    #[test]
    fn one_leaf_and_no_leaves() {
        let (w, _) = collapse(&[]);
        assert!(w.is_empty());
        let bvh = Bvh::build_compact(&boxes(1), 4);
        let (w, of) = collapse(&bvh.nodes);
        assert_eq!(w.len(), 1);
        assert!(w[0].filled(0) && w[0].is_leaf_child(0) && !w[0].filled(1));
        assert_eq!(of[0][0], 0);
    }
}
