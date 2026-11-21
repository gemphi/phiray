//! Compute kernel model (Warp `block_dim` + lean index, mapped to WGSL),
//! and the built-in kernels: each has a WGSL body and a CPU twin used for
//! tests and fallback.
mod denoise;
mod image;
mod skin;
pub use denoise::{Denoise, DenoiseParam};
pub use image::{Bilateral, BilateralParam, Convolve, ConvolveParam, ImageBilateralFilter, ImageConvolution, MatMul, MatMulParam, MatrixMultiplication};
pub use skin::{Pair, Skin, SkinParam};

/// Invocations per workgroup; matches WGSL `@workgroup_size(256)`.
pub const BLOCK: u32 = 256;
/// Max workgroups per axis (WebGPU default limit).
pub const MAX_GROUP: u32 = 65_535;

/// How a kernel sees one binding (group 0, in order; binding 0 is always the params uniform).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    Read,
    Write,
}

/// FNV-1a's offset basis.
const FNV_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a over `s`, continuing from `h` — **at compile time**.
///
/// `const fn` so a kernel's cache key is a number baked into the binary
/// rather than a loop run per dispatch. The tracer's own WGSL is 168 KB
/// (seventeen concatenated files), and [`Kernel::key`] used to hash all of
/// it on every call while `Gpu::pass` called `key()` twice per dispatch —
/// a third of a megabyte of hashing per dispatch, for a number that cannot
/// change. Warp makes the same point about its own speed
/// (`reference/warp/ANALYSIS.md` §2): it is "no per-call allocation, no
/// per-call argument marshalling", and Warp hashes a kernel's source once
/// into a cache name, never per launch.
const fn fnv1a(s: &str, mut h: u64) -> u64 {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        h = (h ^ b[i] as u64).wrapping_mul(0x0100_0000_01b3);
        i += 1;
    }
    h
}

/// A GPU kernel: WGSL body plus its bindings. `Dispatch::PRELUDE` is prepended.
///
/// The body declares `@group(0) @binding(0) var<uniform> p: Param;` whose first
/// two fields are `n: u32, groups_x: u32`, then its storage bindings.
pub trait Kernel {
    const NAME: &'static str;
    const WGSL: &'static str;
    /// Storage bindings after the params uniform.
    const BIND_LIST: &'static [Bind];
    /// Param block (must start with `n: u32, groups_x: u32`).
    type Param: bytemuck::Pod;

    /// The compute entry point. `main` unless one source carries two
    /// kernels (`Adapt` and `AdaptDilate` share `adapt.wgsl`).
    const ENTRY: &'static str = "main";

    /// The `wgsl/` files `WGSL` is the concatenation of, in that order —
    /// what shader hot reload re-reads (`crate::hot`). Empty means "always
    /// the baked text", which is every kernel until it opts in: a kernel's
    /// source is a concatenation of many files and only the macro that
    /// built it knows which, so the list has to be written beside it.
    const FILE_LIST: &'static [&'static str] = &[];

    /// Full shader source: the files from disk when hot reload is live and
    /// has all of them, else the text baked in at compile time.
    fn source() -> String {
        match crate::hot::assemble(Self::FILE_LIST) {
            Some(disk) => format!("{}{}", Dispatch::PRELUDE, disk),
            None => format!("{}{}", Dispatch::PRELUDE, Self::WGSL),
        }
    }

    /// This kernel's cache key for the shipping path, FNV-1a over name +
    /// entry + source, **evaluated by the compiler**.
    ///
    /// Never override it. It is an associated const rather than a line
    /// inside [`Self::key`] so the hash is a constant in the binary: the
    /// key is taken on the hot path of every dispatch, and the text it
    /// hashes is `'static` and cannot change between two calls.
    const BAKED_KEY: u64 = fnv1a(Self::WGSL, fnv1a(Self::ENTRY, fnv1a(Self::NAME, FNV_SEED)));

    /// Stable cache key: the baked constant, with `hot::generation()`
    /// folded in.
    ///
    /// Taken per dispatch, so this is one relaxed atomic load and — in the
    /// shipping case, where no shader has been reloaded — a return of
    /// [`Self::BAKED_KEY`]. It used to hash [`Self::WGSL`] here, byte by
    /// byte, on every call; see [`fnv1a`].
    ///
    /// A reload bumps the generation, every key changes, and `Gpu::pipeline`
    /// is asked for a pipeline it has never compiled — so the cache does
    /// the invalidation and there is none to write.
    fn key() -> u64 {
        match crate::hot::generation() {
            0 => Self::BAKED_KEY,
            g => Self::BAKED_KEY ^ g.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        }
    }
}

/// Workgroup counts for `n` items: fill x, spill into y (Warp "lean" grid).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dispatch {
    pub n: u32,
    pub x: u32,
    pub y: u32,
}

impl Dispatch {
    pub fn new(n: u32) -> Self {
        let blocks = n.div_ceil(BLOCK).max(1);
        let x = blocks.min(MAX_GROUP);
        Self { n, x, y: blocks.div_ceil(x) }
    }

    /// Flat item index for an invocation id; `None` past the end.
    pub fn index(&self, gid: [u32; 2]) -> Option<u32> {
        let i = gid[0] as u64 + gid[1] as u64 * (self.x as u64 * BLOCK as u64);
        (i < self.n as u64).then_some(i as u32)
    }

    /// WGSL prelude every kernel includes.
    pub const PRELUDE: &'static str = "\
const BLOCK: u32 = 256u;
fn item(gid: vec3<u32>, groups_x: u32) -> u32 { return gid.x + gid.y * groups_x * BLOCK; }
";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_every_item_once() {
        for n in [0, 1, 255, 256, 257, 16_777_216, 20_000_000] {
            let d = Dispatch::new(n);
            assert!(d.x <= MAX_GROUP && d.y <= MAX_GROUP);
            assert!(d.x as u64 * d.y as u64 * BLOCK as u64 >= n as u64);
            let last = [(d.x * BLOCK) - 1, d.y - 1];
            assert!(n == 0 || d.index(last).is_none() || n as u64 == d.x as u64 * d.y as u64 * BLOCK as u64);
        }
        let d = Dispatch::new(20_000_000);
        assert_eq!(d.index([5, 1]), Some(5 + 65_535 * 256));
    }

    struct Fill;
    impl Kernel for Fill {
        const NAME: &'static str = "fill";
        const WGSL: &'static str = "@compute @workgroup_size(256) fn main() {}";
        const BIND_LIST: &'static [Bind] = &[];
        type Param = [u32; 2];
    }

    #[test]
    fn key_is_stable() {
        assert_eq!(Fill::key(), Fill::key());
        assert_ne!(Fill::key(), 0);
    }
}
