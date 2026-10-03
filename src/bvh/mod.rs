//! Linear BVH (Karras 2012) with packed leaves and a 32-deep cap (Warp `bvh.cu`).
//!
//! Layout: internal nodes `0..n-1` (root = 0), leaves `n-1..2n-1`.
//! Build: Morton codes → sort → per-node Karras split (parallel, no races) →
//! pack small/deep subtrees into leaves → bounds. Refit: Jacobi passes
//! (`depth + 1`, ping-pong on the GPU), so no atomics are needed.

mod build;
mod gpu;
mod kernels;
mod node;
mod shape;

pub use build::{Bvh, Link};
pub use gpu::GpuBvh;
pub use kernels::{Karras, Morton, Refit};
pub use node::{Aabb, Half, Node, STACK, morton};
pub(crate) use node::LEAF;

/// 30-bit Morton code of a point normalised to `[0, 1]` (the WGSL `Morton` kernel matches this).
pub fn code(p: [f32; 3]) -> u32 {
    morton(p, 0) as u32
}

/// Karras prefix length between sorted slots `i` and `j` (key = code, then slot); -1 outside.
pub fn delta(codes: &[u32], i: i64, j: i64) -> i32 {
    if j < 0 || j >= codes.len() as i64 {
        return -1;
    }
    let (a, b) = (codes[i as usize], codes[j as usize]);
    if a != b {
        (a ^ b).leading_zeros() as i32
    } else {
        32 + ((i as u32) ^ (j as u32)).leading_zeros() as i32
    }
}

#[cfg(test)]
mod tests;
