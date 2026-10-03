//! Node layout shared by the CPU builder and WGSL (Warp `BVHPackedNodeHalf`).

use bytemuck::{Pod, Zeroable};

/// Traversal stack depth; builders turn deeper subtrees into leaves.
pub const STACK: usize = 32;
pub(crate) const LEAF: u32 = 1 << 31;

/// Half a node: one bound corner + index with a leaf bit. 16 bytes, WGSL `struct Half { p: vec3f, i: u32 }`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Half {
    pub p: [f32; 3],
    pub i: u32,
}

impl Half {
    pub fn index(self) -> u32 {
        self.i & !LEAF
    }
    pub fn leaf(self) -> bool {
        self.i & LEAF != 0
    }
}

/// A node = lower + upper half. Inner: children `lo.i`, `hi.i`. Leaf: prims `[lo.i, hi.i)`
/// in sorted order (`Bvh::order`). 32 bytes, WGSL `struct Node { lo: vec3f, li: u32, hi: vec3f, ri: u32 }`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Node {
    pub lo: Half,
    pub hi: Half,
}

impl Node {
    pub fn inner(lo: [f32; 3], hi: [f32; 3], left: u32, right: u32) -> Self {
        Self { lo: Half { p: lo, i: left }, hi: Half { p: hi, i: right } }
    }

    pub fn leaf(lo: [f32; 3], hi: [f32; 3], start: u32, end: u32) -> Self {
        Self { lo: Half { p: lo, i: start | LEAF }, hi: Half { p: hi, i: end } }
    }

    pub fn is_leaf(&self) -> bool {
        self.lo.leaf()
    }

    pub fn bounds(&self) -> Aabb {
        Aabb::new(self.lo.p, self.hi.p)
    }

    pub fn set_bounds(&mut self, b: Aabb) {
        (self.lo.p, self.hi.p) = (b.lo, b.hi);
    }
}

/// Axis-aligned box, 32 bytes (padded for WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Aabb {
    pub lo: [f32; 3],
    _a: f32,
    pub hi: [f32; 3],
    _b: f32,
}

impl Default for Aabb {
    fn default() -> Self {
        Self::new([f32::MAX; 3], [f32::MIN; 3])
    }
}

impl Aabb {
    pub fn new(lo: [f32; 3], hi: [f32; 3]) -> Self {
        Self { lo, _a: 0.0, hi, _b: 0.0 }
    }

    pub fn of_tri(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Self {
        Self::new(std::array::from_fn(|k| a[k].min(b[k]).min(c[k])), std::array::from_fn(|k| a[k].max(b[k]).max(c[k])))
    }

    pub fn union(self, o: Self) -> Self {
        Self::new(std::array::from_fn(|k| self.lo[k].min(o.lo[k])), std::array::from_fn(|k| self.hi[k].max(o.hi[k])))
    }

    pub fn contains(&self, o: &Self) -> bool {
        (0..3).all(|k| self.lo[k] <= o.lo[k] && self.hi[k] >= o.hi[k])
    }

    pub fn center(&self) -> [f32; 3] {
        std::array::from_fn(|k| 0.5 * (self.lo[k] + self.hi[k]))
    }
}

/// 30-bit Morton code of a point already normalised to `[0, 1]`, group in the high 32 bits.
pub fn morton(p: [f32; 3], group: u32) -> u64 {
    fn spread(v: f32) -> u64 {
        let mut x = (v.clamp(0.0, 1.0) * 1023.0) as u64;
        x = (x | (x << 16)) & 0x0300_00FF;
        x = (x | (x << 8)) & 0x0300_F00F;
        x = (x | (x << 4)) & 0x030C_30C3;
        (x | (x << 2)) & 0x0924_9249
    }
    ((group as u64) << 32) | (spread(p[0]) << 2) | (spread(p[1]) << 1) | spread(p[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_wgsl() {
        assert_eq!(std::mem::size_of::<Half>(), 16);
        let n = Node::leaf([0.0; 3], [1.0; 3], 7, 9);
        assert!(n.is_leaf() && n.lo.index() == 7 && n.hi.index() == 9);
        assert!(!Node::inner([0.0; 3], [1.0; 3], 1, 2).is_leaf());
    }

    #[test]
    fn morton_orders_and_groups() {
        assert_eq!(morton([1.0; 3], 0), (1 << 30) - 1);
        assert!(morton([0.1; 3], 0) < morton([0.9; 3], 0));
        assert!(morton([1.0; 3], 0) < morton([0.0; 3], 1));
    }
}
