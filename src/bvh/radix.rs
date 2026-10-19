//! Radix sort on the GPU: `cub::DeviceRadixSort` (Warp `bvh.cu:184`) translated
//! to WGSL, so the LBVH build stops coming back to the CPU to sort.
//!
//! ## Why the key is 64 bits
//!
//! The CPU sort this replaces is one line of `build.rs`:
//!
//! ```ignore
//! order.par_sort_unstable_by_key(|&i| (codes[i as usize], i));
//! ```
//!
//! Code first, original slot as the tie-break — which is a *stable* sort by
//! code, and Morton codes collide constantly (10 bits an axis, so any two
//! prims in the same cell share one). A least-significant-digit radix sort
//! is only correct if every pass is stable, and the fast way to scatter on a
//! GPU is `atomicAdd` on a per-digit cursor, whose order is whatever the
//! hardware decides. Stable and atomic do not go together.
//!
//! So the slot goes *into* the key: `(code, index)` as one 64-bit value,
//! `vec2<u32>(index, code)`. Every key is then distinct, stability stops
//! meaning anything, and the atomic scatter is free to race — the result is
//! the same order the CPU produced, value for value, because sorting
//! `(code, index)` ascending *is* a stable sort by code.
//!
//! It also makes the sort deterministic, which a renderer needs and
//! `atomicAdd` on its own would not give.
//!
//! ## Four dispatches, eight digits
//!
//! One `tally` buffer of 512 `u32` serves both halves of each pass:
//! `[0, 256)` counts the digit, `[256, 512)` is the cursor the scatter
//! advances. [`Offset`] turns the first half into the exclusive prefix sum
//! it writes into the second, and zeroes the first on its way out — so the
//! next digit's [`Count`] starts clean and [`Clear`] runs once per sort
//! rather than once per pass.

use crate::kernel::{Bind, Kernel};
use bytemuck::{Pod, Zeroable};

/// Digit values in one pass, and the length of each half of `tally`.
pub const BINS: u32 = 256;
/// Bits a pass sorts by.
pub const DIGIT: u32 = 8;
/// Passes to cover a 64-bit key.
pub const PASSES: u32 = 64 / DIGIT;

/// `shift` says which 8 bits of the key this dispatch sorts by.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RadixParams {
    pub n: u32,
    pub groups_x: u32,
    pub shift: u32,
    pub _a: u32,
}

/// The key's two halves and how a pass reads a digit out of them, in front
/// of each kernel's own body — so there is one definition of what a key is.
///
/// A macro rather than a `const`, because `concat!` takes literals only and
/// every kernel's `WGSL` has to be one.
macro_rules! key {
    ($tail:expr) => {
        concat!(
            r#"
struct Params { n: u32, groups_x: u32, shift: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: Params;

fn digit(k: vec2<u32>, shift: u32) -> u32 {
    if (shift < 32u) { return (k.x >> shift) & 0xFFu; }
    return (k.y >> (shift - 32u)) & 0xFFu;
}
"#,
            $tail
        )
    };
}

/// `key[i] = (i, codes[i])` — the 64-bit key the sort works on.
pub struct Seed;

impl Kernel for Seed {
    const NAME: &'static str = "radix.seed";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Write];
    type Params = RadixParams;
    const WGSL: &'static str = key!(
        r#"
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> key: array<vec2<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    key[i] = vec2<u32>(i, codes[i]);
}
"#
    );
}

/// Zero both halves of `tally`. Once per sort, not once per pass.
pub struct Clear;

impl Kernel for Clear {
    const NAME: &'static str = "radix.clear";
    const BINDS: &'static [Bind] = &[Bind::Write];
    type Params = RadixParams;
    const WGSL: &'static str = key!(
        r#"
@group(0) @binding(1) var<storage, read_write> tally: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    tally[i] = 0u;
}
"#
    );
}

/// Histogram of this pass's digit into `tally[0, 256)`.
pub struct Count;

impl Kernel for Count {
    const NAME: &'static str = "radix.count";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Write];
    type Params = RadixParams;
    const WGSL: &'static str = key!(
        r#"
@group(0) @binding(1) var<storage, read> key: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> tally: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    atomicAdd(&tally[digit(key[i], p.shift)], 1u);
}
"#
    );
}

/// Exclusive prefix sum of the counts into the cursor half, and clear the
/// counts for the next pass.
///
/// One invocation over 256 bins. A parallel scan would be the textbook
/// answer and is not worth it here: 256 serial adds once per pass is 2048
/// adds for a whole sort, against a dispatch that touches every primitive.
pub struct Offset;

impl Kernel for Offset {
    const NAME: &'static str = "radix.offset";
    const BINDS: &'static [Bind] = &[Bind::Write];
    type Params = RadixParams;
    const WGSL: &'static str = key!(
        r#"
@group(0) @binding(1) var<storage, read_write> tally: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Addressed through `item` like every other kernel, which also keeps
    // the params uniform statically used. The bind group layout is derived
    // from the shader (`Pass::add` -> `get_bind_group_layout`), so a kernel
    // that never reads `p` loses binding 0 from its layout while `add` goes
    // on binding one there — which is the mismatch this pass used to hit.
    if (item(gid, p.groups_x) != 0u) { return; }
    var running = 0u;
    for (var k = 0u; k < 256u; k = k + 1u) {
        let c = tally[k];
        tally[256u + k] = running;
        running = running + c;
        tally[k] = 0u;
    }
}
"#
    );
}

/// Move each key to the slot its digit's cursor hands out.
pub struct Scatter;

impl Kernel for Scatter {
    const NAME: &'static str = "radix.scatter";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Write, Bind::Write];
    type Params = RadixParams;
    const WGSL: &'static str = key!(
        r#"
@group(0) @binding(1) var<storage, read> src: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> tally: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let k = src[i];
    dst[atomicAdd(&tally[256u + digit(k, p.shift)], 1u)] = k;
}
"#
    );
}
