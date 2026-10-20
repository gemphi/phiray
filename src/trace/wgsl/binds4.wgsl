// Compact scene over 4 bindings per large array (see `binds2.wgsl`).
// A chunk is found by comparison, not by division. `i / p.split.x` is a
// 32-bit integer divide by a runtime value, which a GPU has no instruction
// for — it is emulated in ~20 instructions — and `node`, `face` and `vert`
// are the three hottest reads in the tracer: a closest-hit ray does one
// `node` per BVH step and one `face` plus three `vert` per triangle. The
// chain below is two (or four) compares and a subtract, and returns exactly
// what the division returned for every in-range index.
@group(0) @binding(1) var<storage, read> nodes0: array<Node>;
@group(0) @binding(2) var<storage, read> nodes1: array<Node>;
@group(0) @binding(3) var<storage, read> nodes2: array<Node>;
@group(0) @binding(4) var<storage, read> nodes3: array<Node>;
@group(0) @binding(5) var<storage, read> faces0: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> faces1: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> faces2: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> faces3: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read> verts0: array<Vert>;
@group(0) @binding(10) var<storage, read> verts1: array<Vert>;
@group(0) @binding(11) var<storage, read> verts2: array<Vert>;
@group(0) @binding(12) var<storage, read> verts3: array<Vert>;
@group(0) @binding(13) var<storage, read> data: array<u32>;
@group(0) @binding(14) var<storage, read_write> px: array<u32>;

fn node(i: u32) -> Node {
    let s = p.split.x;
    if (i < s) { return nodes0[i]; }
    if (i < 2u * s) { return nodes1[i - s]; }
    if (i < 3u * s) { return nodes2[i - 2u * s]; }
    return nodes3[i - 3u * s];
}
fn face(i: u32) -> vec4<u32> {
    let s = p.split.y;
    if (i < s) { return faces0[i]; }
    if (i < 2u * s) { return faces1[i - s]; }
    if (i < 3u * s) { return faces2[i - 2u * s]; }
    return faces3[i - 3u * s];
}
fn vert(i: u32) -> Vert {
    let s = p.split.z;
    if (i < s) { return verts0[i]; }
    if (i < 2u * s) { return verts1[i - s]; }
    if (i < 3u * s) { return verts2[i - 2u * s]; }
    return verts3[i - 3u * s];
}
