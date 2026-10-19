//! CPU builder — the reference the GPU kernels must match.

use super::node::{Aabb, LEAF, Node, STACK};
use super::wide::{self, Wide, WIDTH};
use super::{code, delta};
use rayon::prelude::*;

/// Karras result for one internal node: children (already offset) and its slot range.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Link {
    pub left: u32,
    pub right: u32,
    pub first: u32,
    pub last: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bvh {
    pub nodes: Vec<Node>,
    /// Sorted slot → primitive index.
    pub order: Vec<u32>,
    /// Deepest reachable node (root = 0), ≤ `STACK - 1`.
    pub depth: u32,
    /// The four-wide view the kernel walks (`wide.rs`), kept in step with
    /// `nodes` by `refit`; `wide_of` names the binary node behind each
    /// child slot so a refit carries bounds over without collapsing again.
    pub wide: Vec<Wide>,
    pub wide_of: Vec<[u32; WIDTH]>,
}

impl Link {
    /// Karras split for internal node `i` over `codes` (sorted).
    pub fn of(codes: &[u32], i: usize) -> Self {
        let (i, n) = (i as i64, codes.len() as i64);
        let d: i64 = if delta(codes, i, i + 1) - delta(codes, i, i - 1) > 0 { 1 } else { -1 };
        let dmin = delta(codes, i, i - d);
        let mut lmax = 2i64;
        while delta(codes, i, i + lmax * d) > dmin {
            lmax *= 2;
        }
        let (mut l, mut t) = (0i64, lmax / 2);
        while t >= 1 {
            if delta(codes, i, i + (l + t) * d) > dmin {
                l += t;
            }
            t /= 2;
        }
        let j = i + l * d;
        let dnode = delta(codes, i, j);
        let (mut s, mut t) = (0i64, l);
        loop {
            t = (t + 1) >> 1;
            if delta(codes, i, i + (s + t) * d) > dnode {
                s += t;
            }
            if t <= 1 {
                break;
            }
        }
        let g = i + s * d + d.min(0);
        let (first, last) = (i.min(j), i.max(j));
        let left = if first == g { n - 1 + g } else { g };
        let right = if last == g + 1 { n + g } else { g + 1 };
        Self { left: left as u32, right: right as u32, first: first as u32, last: last as u32 }
    }
}

impl Bvh {
    /// Sorted order + codes for `prims`.
    pub fn sort(prims: &[Aabb]) -> (Vec<u32>, Vec<u32>) {
        let codes = Self::codes(prims);
        Self::sort_codes(&codes)
    }

    pub fn scene(prims: &[Aabb]) -> Aabb {
        prims.iter().fold(Aabb::default(), |a, b| a.union(*b))
    }

    /// Unsorted Morton codes of prim centres.
    pub fn codes(prims: &[Aabb]) -> Vec<u32> {
        let s = Self::scene(prims);
        let inv: [f32; 3] = std::array::from_fn(|k| {
            let e = s.hi[k] - s.lo[k];
            if e > 0.0 { 1.0 / e } else { 0.0 }
        });
        prims.par_iter().map(|p| {
            let c = p.center();
            code(std::array::from_fn(|k| (c[k] - s.lo[k]) * inv[k]))
        }).collect()
    }

    pub fn sort_codes(codes: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut order: Vec<u32> = (0..codes.len() as u32).collect();
        order.par_sort_unstable_by_key(|&i| (codes[i as usize], i));
        let sorted = order.iter().map(|&i| codes[i as usize]).collect();
        (order, sorted)
    }

    pub fn build(prims: &[Aabb], leaf: u32) -> Self {
        let (order, sorted) = Self::sort(prims);
        let links: Vec<Link> = (0..sorted.len().saturating_sub(1)).into_par_iter().map(|i| Link::of(&sorted, i)).collect();
        let mut bvh = Self::assemble(order, &links, leaf);
        bvh.refit(prims);
        bvh
    }

    /// Same tree as `build` + `compact`, without ever holding `2n - 1` nodes
    /// (large scenes: ~n / 2 nodes are kept).
    pub fn build_compact(prims: &[Aabb], leaf: u32) -> Self {
        let (order, sorted) = Self::sort(prims);
        let n = order.len();
        let links: Vec<Link> = (0..n.saturating_sub(1)).into_par_iter().map(|i| Link::of(&sorted, i)).collect();
        drop(sorted);
        if n == 0 {
            return Self { order, ..Default::default() };
        }
        let mut nodes = vec![Node::default()];
        let mut queue = std::collections::VecDeque::from([(0usize, 0usize, 0u32)]);
        let mut depth = 0;
        while let Some((i, slot, d)) = queue.pop_front() {
            depth = depth.max(d);
            if i >= n - 1 {
                let k = (i - (n - 1)) as u32;
                nodes[slot] = Node::leaf([0.0; 3], [0.0; 3], k, k + 1);
                continue;
            }
            let l = links[i];
            if l.last - l.first < leaf || d + 1 >= STACK as u32 {
                nodes[slot] = Node::leaf([0.0; 3], [0.0; 3], l.first, l.last + 1);
                continue;
            }
            let at = nodes.len();
            nodes.extend([Node::default(); 2]);
            nodes[slot] = Node::inner([0.0; 3], [0.0; 3], at as u32, at as u32 + 1);
            queue.extend([(l.left as usize, at, d + 1), (l.right as usize, at + 1, d + 1)]);
        }
        drop(links);
        let mut bvh = Self { nodes, order, depth, ..Default::default() };
        bvh.refit(prims);
        bvh
    }

    /// Nodes from links, then pack subtrees with ≤ `leaf` prims or at depth `STACK - 1` into leaves.
    pub fn assemble(order: Vec<u32>, links: &[Link], leaf: u32) -> Self {
        let n = order.len();
        if n == 0 {
            return Self { order, ..Default::default() };
        }
        let mut nodes = vec![Node::default(); 2 * n - 1];
        for (i, l) in links.iter().enumerate() {
            nodes[i] = Node::inner([0.0; 3], [0.0; 3], l.left, l.right);
        }
        for k in 0..n {
            nodes[n - 1 + k] = Node::leaf([0.0; 3], [0.0; 3], k as u32, k as u32 + 1);
        }
        let mut depth = 0;
        let mut stack = vec![(0usize, 0u32)];
        while let Some((i, d)) = stack.pop() {
            depth = depth.max(d);
            if i >= n - 1 {
                continue;
            }
            let l = links[i];
            if l.last - l.first < leaf || d + 1 >= STACK as u32 {
                nodes[i].lo.i = l.first | LEAF;
                nodes[i].hi.i = l.last + 1;
                continue;
            }
            stack.push((l.left as usize, d + 1));
            stack.push((l.right as usize, d + 1));
        }
        Self { nodes, order, depth, ..Default::default() }
    }

    /// Bounds again over `boxes` given by sorted slot (prims already in leaf
    /// order, as the tracer keeps its faces): the same tree over a new pose.
    pub fn refit_slots(&mut self, boxes: &[Aabb]) {
        let order = std::mem::take(&mut self.order);
        self.order = (0..boxes.len() as u32).collect();
        self.refit(boxes);
        self.order = order;
    }

    /// Exact bounds of every reachable node (iterative post-order from the
    /// root), and the wide view brought in step.
    pub fn refit(&mut self, prims: &[Aabb]) {
        self.refit_binary(prims);
        self.widen();
    }

    /// The wide view from `nodes`: bounds carried over when the structure is
    /// the one it was collapsed from, collapsed afresh otherwise.
    pub fn widen(&mut self) {
        if !self.wide.is_empty() && self.wide_of.len() == self.wide.len() {
            wide::rebound(&mut self.wide, &self.wide_of, &self.nodes);
        } else {
            (self.wide, self.wide_of) = wide::collapse(&self.nodes);
        }
    }

    fn refit_binary(&mut self, prims: &[Aabb]) {
        let mut stack = if self.nodes.is_empty() { vec![] } else { vec![(0usize, false)] };
        while let Some((i, done)) = stack.pop() {
            let nd = self.nodes[i];
            let b = if nd.is_leaf() {
                (nd.lo.index()..nd.hi.index()).fold(Aabb::default(), |a, s| a.union(prims[self.order[s as usize] as usize]))
            } else if done {
                self.nodes[nd.lo.index() as usize].bounds().union(self.nodes[nd.hi.index() as usize].bounds())
            } else {
                stack.extend([(i, true), (nd.lo.index() as usize, false), (nd.hi.index() as usize, false)]);
                continue;
            };
            self.nodes[i].set_bounds(b);
        }
    }
}
