// Compact scene over 4 bindings per large array (see `binds2.wgsl`).
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
    let k = i / p.split.x;
    let j = i - k * p.split.x;
    switch k {
        case 0u: { return nodes0[j]; }
        case 1u: { return nodes1[j]; }
        case 2u: { return nodes2[j]; }
        default: { return nodes3[j]; }
    }
}
fn face(i: u32) -> vec4<u32> {
    let k = i / p.split.y;
    let j = i - k * p.split.y;
    switch k {
        case 0u: { return faces0[j]; }
        case 1u: { return faces1[j]; }
        case 2u: { return faces2[j]; }
        default: { return faces3[j]; }
    }
}
fn vert(i: u32) -> Vert {
    let k = i / p.split.z;
    let j = i - k * p.split.z;
    switch k {
        case 0u: { return verts0[j]; }
        case 1u: { return verts1[j]; }
        case 2u: { return verts2[j]; }
        default: { return verts3[j]; }
    }
}
