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
