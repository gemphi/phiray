//! Compute kernel model (Warp `block_dim` + lean index, mapped to WGSL).

/// Invocations per workgroup; matches WGSL `@workgroup_size(256)`.
pub const BLOCK: u32 = 256;
/// Max workgroups per axis (WebGPU default limit).
pub const MAX_GROUPS: u32 = 65_535;

/// How a kernel sees one binding (group 0, in order; binding 0 is always the params uniform).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    Read,
    Write,
}

/// A GPU kernel: WGSL body plus its bindings. `Dispatch::PRELUDE` is prepended.
///
/// The body declares `@group(0) @binding(0) var<uniform> p: Params;` whose first
/// two fields are `n: u32, groups_x: u32`, then its storage bindings.
pub trait Kernel {
    const NAME: &'static str;
    const WGSL: &'static str;
    /// Storage bindings after the params uniform.
    const BINDS: &'static [Bind];
    /// Params block (must start with `n: u32, groups_x: u32`).
    type Params: bytemuck::Pod;

    /// The compute entry point. `main` unless one source carries two
    /// kernels (`Adapt` and `AdaptDilate` share `adapt.wgsl`).
    const ENTRY: &'static str = "main";

    /// Full shader source.
    fn source() -> String {
        format!("{}{}", Dispatch::PRELUDE, Self::WGSL)
    }

    /// Stable cache key (FNV-1a over name + source).
    fn key() -> u64 {
        Self::NAME.bytes().chain(Self::ENTRY.bytes()).chain(Self::WGSL.bytes()).fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
        })
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
        let x = blocks.min(MAX_GROUPS);
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
            assert!(d.x <= MAX_GROUPS && d.y <= MAX_GROUPS);
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
        const BINDS: &'static [Bind] = &[];
        type Params = [u32; 2];
    }

    #[test]
    fn key_is_stable() {
        assert_eq!(Fill::key(), Fill::key());
        assert_ne!(Fill::key(), 0);
    }
}
