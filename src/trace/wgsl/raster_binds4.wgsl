// `raster_binds2.wgsl` over four bindings per array (`binds4.wgsl`).
@group(0) @binding(1) var<storage, read> faces0: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> faces1: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> faces2: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> faces3: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> verts0: array<Vert>;
@group(0) @binding(6) var<storage, read> verts1: array<Vert>;
@group(0) @binding(7) var<storage, read> verts2: array<Vert>;
@group(0) @binding(8) var<storage, read> verts3: array<Vert>;
@group(0) @binding(9) var<storage, read> data: array<u32>;

fn face(i: u32) -> vec4<u32> {
    let k = i / p.split.y; let j = i - k * p.split.y;
    if (k == 0u) { return faces0[j]; } if (k == 1u) { return faces1[j]; } if (k == 2u) { return faces2[j]; } return faces3[j];
}
fn vert(i: u32) -> Vert {
    let k = i / p.split.z; let j = i - k * p.split.z;
    if (k == 0u) { return verts0[j]; } if (k == 1u) { return verts1[j]; } if (k == 2u) { return verts2[j]; } return verts3[j];
}
