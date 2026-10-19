// Compact scene (`scene.rs`), each large array split over 2 bindings (4 in
// `binds4.wgsl`) of `p.split` items so one binding never exceeds the device limit:
// nodes, faces (BVH leaf order), vertices; `data = [table…, atlas…]` with
// `table = [lamp count L, lamp faces × L, lamp cdf × L, materials × 14 words]`
// and the atlas from `p.split.w`; `px = [packed pixel × n, accum rgb × n]`.
// A chunk is found by comparison, not by division. `i / p.split.x` is a
// 32-bit integer divide by a runtime value, which a GPU has no instruction
// for — it is emulated in ~20 instructions — and `node`, `face` and `vert`
// are the three hottest reads in the tracer: a closest-hit ray does one
// `node` per BVH step and one `face` plus three `vert` per triangle. The
// chain below is two (or four) compares and a subtract, and returns exactly
// what the division returned for every in-range index.
@group(0) @binding(1) var<storage, read> nodes0: array<Wide>;
@group(0) @binding(2) var<storage, read> nodes1: array<Wide>;
@group(0) @binding(3) var<storage, read> faces0: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> faces1: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> verts0: array<Vert>;
@group(0) @binding(6) var<storage, read> verts1: array<Vert>;
@group(0) @binding(7) var<storage, read> data: array<u32>;
@group(0) @binding(8) var<storage, read_write> px: array<u32>;

fn node(i: u32) -> Wide { let s = p.split.x; if (i < s) { return nodes0[i]; } return nodes1[i - s]; }
fn face(i: u32) -> vec4<u32> { let s = p.split.y; if (i < s) { return faces0[i]; } return faces1[i - s]; }
fn vert(i: u32) -> Vert { let s = p.split.z; if (i < s) { return verts0[i]; } return verts1[i - s]; }
