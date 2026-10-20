struct Node { lo: vec3<f32>, li: u32, hi: vec3<f32>, ri: u32 }
struct Tri {
    a: vec3<f32>, rough: f32, na: vec3<f32>, trans: f32,
    b: vec3<f32>, metal: f32, nb: vec3<f32>, ior: f32,
    c: vec3<f32>, _c: f32, nc: vec3<f32>, _nc: f32,
    color: vec4<f32>,
    us: vec3<f32>, tex: u32,
    vs: vec3<f32>, nmap: u32,
    emit: vec3<f32>, bump: f32,
    // Roughness / metallic / emission textures (NO_TEX = none).
    maps: vec3<u32>,
    // Subsurface weight and scattering distance per channel (world units).
    sss: f32, sss_radius: vec3<f32>,
    // Fabric sheen: weight, roughness, tint.
    sheen: f32, sheen_rough: f32, sheen_tint: vec3<f32>,
    // Clearcoat: weight and roughness.
    coat: f32, coat_rough: f32,
    // Anisotropy and rotation (carried; not yet shaded).
    aniso: f32, aniso_rot: f32,
    // A second lobe: another material's id (NO_MAT = none) and its weight.
    second: u32, mix: f32,
}
struct Hit { t: f32, tri: u32, u: f32, v: f32 }

const PI: f32 = 3.14159265;
const LEAF: u32 = 0x80000000u;
// A top-tree leaf: its range is instance slots (`inst_order`), not faces.
const INST: u32 = 0x40000000u;
// Words per instance record (`Instance::words`).
const INST_WORDS: u32 = 40u;
// Words per material record (`Mat`).
const MAT_WORDS: u32 = 120u;
const NO_MAT: u32 = 0xffffffffu;
const NO_TEX: u32 = 0xffffffffu;
const INF: f32 = 3.0e38;

@group(0) @binding(0) var<uniform> p: Camera;
struct Vert { p: vec4<f32>, a: vec4<f32> }

fn atlas(i: u32) -> u32 { return data[p.split.w + i]; }

fn tf(i: u32) -> f32 { return bitcast<f32>(data[i]); }

// First word of face `f`'s material (the low half of its fourth word; the
// high half names its instance).
fn mat_at(f: u32) -> u32 { return mat_word(face(f).w & 0xffffu); }
// First word of material `id`.
fn mat_word(id: u32) -> u32 { return 1u + 2u * data[0] + MAT_WORDS * id; }
// A material's second lobe (`Mat::second`, `NO_MAT` for none) and its weight.
fn mat_second(m: u32) -> u32 { return data[m + 30u]; }
fn mat_mix(m: u32) -> f32 { return tf(m + 31u); }

// Instance `i`'s record: world rows 0..12, inverse rows 12..24, 24 the root
// of its tree, 25..27 its faces, 27 its flags: bit 0 set when it sits at
// the identity, bit 1 when it casts no shadow; 28..31 its colour and hash,
// 32..38 its bounding box (`procedural.wgsl`).
fn inst_at(i: u32) -> u32 { return p.insts.x + i * INST_WORDS; }
fn inst_row(w: u32) -> vec4<f32> { return vec4<f32>(tf(w), tf(w + 1u), tf(w + 2u), tf(w + 3u)); }
// Top-tree slot `s` → instance id.
fn inst_order(s: u32) -> u32 { return data[p.insts.x + p.insts.y * INST_WORDS + s]; }

// A normal under the inverse-transpose of an instance's 3 × 3 (`i0..i2`
// its inverse's rows), unit again; a zero (flat) normal stays zero.
fn place_n(i0: vec4<f32>, i1: vec4<f32>, i2: vec4<f32>, n: vec3<f32>) -> vec3<f32> {
    let m = n.x * i0.xyz + n.y * i1.xyz + n.z * i2.xyz;
    let l = length(m);
    if (l < 1e-12) { return vec3<f32>(0.0); }
    return m / l;
}

fn unoct(e: vec2<f32>) -> vec3<f32> {
    if (e.x > 1.5) { return vec3<f32>(0.0); }
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n = vec3<f32>((1.0 - abs(e.y)) * sign(e.x), (1.0 - abs(e.x)) * sign(e.y), n.z);
    }
    return normalize(n);
}

// Assemble face `f` as a full triangle record, placed in the world by its
// instance (its vertices are kept in the object's space), with its own
// material.
fn tri_at(f: u32) -> Tri { return tri_with(f, mat_at(f)); }

// Face `f` with the material whose first word is `m` — its own, or the
// second lobe of its own.
fn tri_with(f: u32, m: u32) -> Tri {
    let v = face(f);
    let x = vert(v.x);
    let y = vert(v.y);
    let z = vert(v.z);
    let a0 = x.a;
    let a1 = y.a;
    let a2 = z.a;
    var pa = x.p.xyz;
    var pb = y.p.xyz;
    var pc = z.p.xyz;
    var na = unoct(a0.xy);
    var nb = unoct(a1.xy);
    var nc = unoct(a2.xy);
    let w = inst_at(v.w >> 16u);
    if ((data[w + 27u] & 1u) == 0u) {
        let r0 = inst_row(w);
        let r1 = inst_row(w + 4u);
        let r2 = inst_row(w + 8u);
        pa = vec3<f32>(dot(r0, x.p), dot(r1, x.p), dot(r2, x.p));
        pb = vec3<f32>(dot(r0, y.p), dot(r1, y.p), dot(r2, y.p));
        pc = vec3<f32>(dot(r0, z.p), dot(r1, z.p), dot(r2, z.p));
        let i0 = inst_row(w + 12u);
        let i1 = inst_row(w + 16u);
        let i2 = inst_row(w + 20u);
        na = place_n(i0, i1, i2, na);
        nb = place_n(i0, i1, i2, nb);
        nc = place_n(i0, i1, i2, nc);
    }
    return Tri(
        pa, tf(m + 4u), na, tf(m + 6u),
        pb, tf(m + 5u), nb, tf(m + 7u),
        pc, 0.0, nc, 0.0,
        vec4<f32>(tf(m), tf(m + 1u), tf(m + 2u), tf(m + 3u)),
        vec3<f32>(a0.z, a1.z, a2.z), data[m + 12u],
        vec3<f32>(a0.w, a1.w, a2.w), data[m + 13u],
        vec3<f32>(tf(m + 8u), tf(m + 9u), tf(m + 10u)), tf(m + 11u),
        vec3<u32>(data[m + 14u], data[m + 15u], data[m + 16u]),
        tf(m + 17u), vec3<f32>(tf(m + 18u), tf(m + 19u), tf(m + 20u)),
        tf(m + 21u), tf(m + 22u), vec3<f32>(tf(m + 23u), tf(m + 24u), tf(m + 25u)),
        tf(m + 26u), tf(m + 27u),
        tf(m + 28u), tf(m + 29u),
        data[m + 30u], tf(m + 31u),
    );
}

