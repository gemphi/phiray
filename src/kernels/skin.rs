//! Linear blend skinning with any number of influences (Blender Armature deform).
//!
//! Influences are CSR: vertex `i` uses `pairs[offsets[i]..offsets[i + 1]]`.
//! An influence names two matrices and how far the vertex leans from the
//! first to the second — that is a B-Bone's two neighbouring segments. A rigid
//! bone sets `next = bone` and `blend = 0`, which costs one extra lerp.
//! `out = Σ w·Bᵢ·rest / Σ w` when `Σ w > min`, else `rest`.
//! Bindings: rest `vec4`, offsets `u32` (n + 1), pairs, bone matrices, out `vec4`.

use crate::kernel::{Bind, Kernel};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Pair {
    pub bone: u32,
    pub weight: f32,
    /// The next B-Bone segment (`== bone` for a rigid bone).
    pub next: u32,
    /// How far towards `next` this vertex lies (0 for a rigid bone).
    pub blend: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SkinParams {
    pub n: u32,
    pub groups_x: u32,
    /// Minimum total weight to move a vertex (Blender: 0.0001).
    pub min: f32,
    pub _pad: u32,
}

pub struct Skin;

impl Kernel for Skin {
    const NAME: &'static str = "skin";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Read, Bind::Read, Bind::Read, Bind::Write];
    type Params = SkinParams;
    const WGSL: &'static str = r#"
struct Params { n: u32, groups_x: u32, min: f32, _pad: u32 }
struct Pair { bone: u32, weight: f32, next: u32, blend: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> rest: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> pairs: array<Pair>;
@group(0) @binding(4) var<storage, read> bones: array<mat4x4<f32>>;
@group(0) @binding(5) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let v = vec4<f32>(rest[i].xyz, 1.0);
    var acc = vec3<f32>(0.0);
    var total = 0.0;
    for (var k = offsets[i]; k < offsets[i + 1u]; k++) {
        let q = pairs[k];
        let m = bones[q.bone] + (bones[q.next] - bones[q.bone]) * q.blend;
        acc += q.weight * (m * v).xyz;
        total += q.weight;
    }
    out[i] = select(v, vec4<f32>(acc / total, 1.0), total > p.min);
}
"#;
}

impl Skin {
    /// CPU twin of the WGSL body (column-major matrices).
    pub fn cpu(rest: &[[f32; 4]], offsets: &[u32], pairs: &[Pair], bones: &[[f32; 16]], min: f32) -> Vec<[f32; 4]> {
        rest.iter()
            .enumerate()
            .map(|(i, r)| {
                let (mut acc, mut total) = ([0.0f32; 3], 0.0f32);
                for q in &pairs[offsets[i] as usize..offsets[i + 1] as usize] {
                    let (m, n) = (&bones[q.bone as usize], &bones[q.next as usize]);
                    let at = |j: usize| m[j] + (n[j] - m[j]) * q.blend;
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a += q.weight * (at(k) * r[0] + at(4 + k) * r[1] + at(8 + k) * r[2] + at(12 + k));
                    }
                    total += q.weight;
                }
                if total > min {
                    [acc[0] / total, acc[1] / total, acc[2] / total, 1.0]
                } else {
                    [r[0], r[1], r[2], 1.0]
                }
            })
            .collect()
    }
}
