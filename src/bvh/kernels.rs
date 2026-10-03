//! BVH kernels: Morton codes, Karras links, Jacobi refit. Each matches `build.rs`.

use crate::kernel::{Bind, Kernel};
use bytemuck::{Pod, Zeroable};

/// Scene box for Morton normalisation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MortonParams {
    pub n: u32,
    pub groups_x: u32,
    pub _a: [u32; 2],
    pub lo: [f32; 3],
    pub _b: f32,
    pub inv: [f32; 3],
    pub _c: f32,
}

pub struct Morton;

impl Kernel for Morton {
    const NAME: &'static str = "bvh.morton";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Write];
    type Params = MortonParams;
    const WGSL: &'static str = concat!(
        "struct Aabb { lo: vec3<f32>, _a: f32, hi: vec3<f32>, _b: f32 }\n",
        r#"
struct Params { n: u32, groups_x: u32, _a: u32, _b: u32, lo: vec3<f32>, _c: f32, inv: vec3<f32>, _d: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> prims: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> codes: array<u32>;

fn spread(v: f32) -> u32 {
    var x = u32(clamp(v, 0.0, 1.0) * 1023.0);
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x << 8u)) & 0x0300F00Fu;
    x = (x | (x << 4u)) & 0x030C30C3u;
    return (x | (x << 2u)) & 0x09249249u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let c = 0.5 * (prims[i].lo + prims[i].hi);
    let u = (c - p.lo) * p.inv;
    codes[i] = (spread(u.x) << 2u) | (spread(u.y) << 1u) | spread(u.z);
}
"#
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct KarrasParams {
    pub n: u32,
    pub groups_x: u32,
    /// Number of leaves (= sorted codes).
    pub leaves: u32,
    pub _a: u32,
}

pub struct Karras;

impl Kernel for Karras {
    const NAME: &'static str = "bvh.karras";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Write];
    type Params = KarrasParams;
    const WGSL: &'static str = r#"
struct Params { n: u32, groups_x: u32, leaves: u32, _a: u32 }
struct Link { left: u32, right: u32, first: u32, last: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> links: array<Link>;

fn delta(i: i32, j: i32) -> i32 {
    if (j < 0 || j >= i32(p.leaves)) { return -1; }
    let a = codes[i];
    let b = codes[j];
    if (a != b) { return i32(countLeadingZeros(a ^ b)); }
    return 32 + i32(countLeadingZeros(u32(i) ^ u32(j)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = item(gid, p.groups_x);
    if (k >= p.n) { return; }
    let i = i32(k);
    let n = i32(p.leaves);
    let d = select(-1, 1, delta(i, i + 1) - delta(i, i - 1) > 0);
    let dmin = delta(i, i - d);
    var lmax = 2;
    while (delta(i, i + lmax * d) > dmin) { lmax *= 2; }
    var l = 0;
    var t = lmax / 2;
    while (t >= 1) {
        if (delta(i, i + (l + t) * d) > dmin) { l += t; }
        t /= 2;
    }
    let j = i + l * d;
    let dnode = delta(i, j);
    var s = 0;
    t = l;
    loop {
        t = (t + 1) >> 1u;
        if (delta(i, i + (s + t) * d) > dnode) { s += t; }
        if (t <= 1) { break; }
    }
    let g = i + s * d + min(d, 0);
    let first = min(i, j);
    let last = max(i, j);
    let left = select(g, n - 1 + g, first == g);
    let right = select(g + 1, n + g, last == g + 1);
    links[k] = Link(u32(left), u32(right), u32(first), u32(last));
}
"#;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RefitParams {
    pub n: u32,
    pub groups_x: u32,
    pub _a: [u32; 2],
}

/// One Jacobi step: `dst[k]` = bounds of its prims (leaf) or of `src` children.
pub struct Refit;

impl Kernel for Refit {
    const NAME: &'static str = "bvh.refit";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Read, Bind::Read, Bind::Write];
    type Params = RefitParams;
    const WGSL: &'static str = concat!(
        "struct Aabb { lo: vec3<f32>, _a: f32, hi: vec3<f32>, _b: f32 }\n",
        "struct Node { lo: vec3<f32>, li: u32, hi: vec3<f32>, ri: u32 }\nconst LEAF: u32 = 0x80000000u;\n",
        r#"
struct Params { n: u32, groups_x: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> prims: array<Aabb>;
@group(0) @binding(2) var<storage, read> order: array<u32>;
@group(0) @binding(3) var<storage, read> src: array<Node>;
@group(0) @binding(4) var<storage, read_write> dst: array<Node>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = item(gid, p.groups_x);
    if (k >= p.n) { return; }
    var nd = src[k];
    var lo = vec3<f32>(3.4028235e38);
    var hi = vec3<f32>(-3.4028235e38);
    if ((nd.li & LEAF) != 0u) {
        for (var s = nd.li & ~LEAF; s < nd.ri; s++) {
            let b = prims[order[s]];
            lo = min(lo, b.lo);
            hi = max(hi, b.hi);
        }
    } else {
        let a = src[nd.li];
        let b = src[nd.ri];
        lo = min(a.lo, b.lo);
        hi = max(a.hi, b.hi);
    }
    nd.lo = lo;
    nd.hi = hi;
    dst[k] = nd;
}
"#
    );
}
