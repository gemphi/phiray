// The visibility prepass's bindings (`raster.wgsl`): the faces, vertices
// and data of `binds2.wgsl`, read-only — a vertex stage may not bind the
// image — at bindings 1..5.
@group(0) @binding(1) var<storage, read> faces0: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> faces1: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> verts0: array<Vert>;
@group(0) @binding(4) var<storage, read> verts1: array<Vert>;
@group(0) @binding(5) var<storage, read> data: array<u32>;

fn face(i: u32) -> vec4<u32> { let k = i / p.split.y; let j = i - k * p.split.y; if (k == 0u) { return faces0[j]; } return faces1[j]; }
fn vert(i: u32) -> Vert { let k = i / p.split.z; let j = i - k * p.split.z; if (k == 0u) { return verts0[j]; } return verts1[j]; }
