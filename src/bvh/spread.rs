//! `Spread`: the sorted codes back out as a plain `array<u32>`.
//!
//! The sort works on one 64-bit key (`radix.rs`) because that is what makes
//! it deterministic. `Karras` wants the codes on their own, as it always
//! has. This kernel is the seam between the two, and it exists so the codes
//! never leave the device: without it the build would read the keys back,
//! split them on the CPU and upload the halves again — which is three
//! quarters of the hop the GPU sort was written to remove.

use crate::kernel::{Bind, Kernel};
use super::radix::RadixParams;

pub struct Spread;

impl Kernel for Spread {
    const NAME: &'static str = "radix.spread";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Write];
    type Params = RadixParams;
    const WGSL: &'static str = r#"
struct Params { n: u32, groups_x: u32, shift: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> key: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> codes: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    codes[i] = key[i].y;
}
"#;
}
