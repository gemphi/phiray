// Compact scene (`scene.rs`), each large array split over 2 bindings (4 in
// `binds4.wgsl`) of `p.split` items so one binding never exceeds the device limit:
// nodes, faces (BVH leaf order), vertices; `data = [table…, atlas…]` with
// `table = [lamp count L, lamp faces × L, lamp cdf × L, materials × 14 words]`
// and the atlas from `p.split.w`; `px = [packed pixel × n, accum rgb × n]`.
@group(0) @binding(1) var<storage, read> nodes0: array<Node>;
@group(0) @binding(2) var<storage, read> nodes1: array<Node>;
@group(0) @binding(3) var<storage, read> faces0: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> faces1: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> verts0: array<Vert>;
@group(0) @binding(6) var<storage, read> verts1: array<Vert>;
@group(0) @binding(7) var<storage, read> data: array<u32>;
@group(0) @binding(8) var<storage, read_write> px: array<u32>;

fn node(i: u32) -> Node { let k = i / p.split.x; let j = i - k * p.split.x; if (k == 0u) { return nodes0[j]; } return nodes1[j]; }
fn face(i: u32) -> vec4<u32> { let k = i / p.split.y; let j = i - k * p.split.y; if (k == 0u) { return faces0[j]; } return faces1[j]; }
fn vert(i: u32) -> Vert { let k = i / p.split.z; let j = i - k * p.split.z; if (k == 0u) { return verts0[j]; } return verts1[j]; }
