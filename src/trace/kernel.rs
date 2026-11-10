//! The triangle the kernels read (the kernels are in `shader.rs`; the
//! camera, lamps and atlas in `camera.rs`, `light.rs`, `atlas.rs`).

use crate::bvh::Aabb;
use bytemuck::{Pod, Zeroable};

/// A triangle: vertex normals, linear base colour (alpha = Principled Alpha),
/// PBR inputs and optional texture (160 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Tri {
    pub a: [f32; 3],
    pub rough: f32,
    pub na: [f32; 3],
    /// Transmission weight (glass).
    pub trans: f32,
    pub b: [f32; 3],
    pub metal: f32,
    pub nb: [f32; 3],
    /// Index of refraction.
    pub ior: f32,
    pub c: [f32; 3],
    pub _c: f32,
    pub nc: [f32; 3],
    pub _nc: f32,
    pub color: [f32; 4],
    /// Corner UVs (u of a, b, c) and texture id (`NO_TEX` = none).
    pub us: [f32; 3],
    pub tex: u32,
    /// Corner UVs (v of a, b, c).
    pub vs: [f32; 3],
    /// Normal map texture id (`NO_TEX` = none).
    pub nmap: u32,
    /// Emitted radiance (linear).
    pub emit: [f32; 3],
    /// Normal Map node strength.
    pub bump: f32,
    /// Roughness / metallic / emission texture ids (`NO_TEX` = none); each
    /// multiplies the matching value (Blender replaces the socket, so the
    /// caller passes 1 there).
    pub maps: [u32; 3],
    /// Subsurface weight (Principled "Subsurface Weight").
    pub sss: f32,
    /// Scattering distance per channel in world units (radius × scale).
    pub sss_radius: [f32; 3],
    /// Sheen weight, its roughness and its tint (fabric retro-reflection).
    pub sheen: f32,
    pub sheen_rough: f32,
    pub sheen_tint: [f32; 3],
    /// Coat weight and roughness (clear lacquer over the base).
    pub coat: f32,
    pub coat_rough: f32,
    /// Anisotropy and its rotation; carried for parity, not yet shaded
    /// (per-triangle tangents are needed first).
    pub aniso: f32,
    pub aniso_rot: f32,
    /// A second lobe (`Mat::second`, `mix`): the id of another material
    /// in the scene and the weight it takes; `NO_MAT`, 0 for one lobe.
    pub second: u32,
    pub mix: f32,
    /// Procedural textures (`Mat::color_proc`, `bump_proc`, `color_proc2`).
    pub color_proc: super::mat::Proc,
    pub bump_proc: super::mat::Proc,
    pub color_proc2: super::mat::Proc,
}

/// `Tri::tex` value for untextured triangles.
pub const NO_TEX: u32 = u32::MAX;
pub use super::mat::NO_MAT;

impl Tri {
    pub fn new(p: [[f32; 3]; 3], n: [[f32; 3]; 3], color: [f32; 4]) -> Self {
        Self { a: p[0], b: p[1], c: p[2], na: n[0], nb: n[1], nc: n[2], color, rough: 0.5, ior: 1.5, tex: NO_TEX, nmap: NO_TEX, maps: [NO_TEX; 3], sheen_rough: 0.5, sheen_tint: [1.0; 3], coat_rough: 0.03, sss_radius: [0.05, 0.01, 0.005], second: NO_MAT, ..Default::default() }
    }

    /// Texture `tex` (from [`Atlas::add`]) mapped by corner UVs; it multiplies `color`.
    pub fn with_texture(mut self, tex: u32, uv: [[f32; 2]; 3]) -> Self {
        self.tex = tex;
        self.us = [uv[0][0], uv[1][0], uv[2][0]];
        self.vs = [uv[0][1], uv[1][1], uv[2][1]];
        self
    }

    /// Tangent-space normal map (atlas id, raw bytes) with the node's strength;
    /// uses the same corner UVs as [`Self::with_texture`].
    pub fn with_normal_map(mut self, id: u32, strength: f32, uv: [[f32; 2]; 3]) -> Self {
        (self.nmap, self.bump) = (id, strength);
        self.us = [uv[0][0], uv[1][0], uv[2][0]];
        self.vs = [uv[0][1], uv[1][1], uv[2][1]];
        self
    }

    /// Emission (colour × strength), transmission weight and IOR.
    pub fn with_optics(mut self, emit: [f32; 3], trans: f32, ior: f32) -> Self {
        (self.emit, self.trans, self.ior) = (emit, trans.clamp(0.0, 1.0), ior.max(1.0));
        self
    }

    /// Roughness / metallic / emission-colour textures (atlas ids, `NO_TEX` for
    /// none); each multiplies its value, with the same corner UVs as
    /// [`Self::with_texture`].
    pub fn with_maps(mut self, maps: [u32; 3], uv: [[f32; 2]; 3]) -> Self {
        self.maps = maps;
        self.us = [uv[0][0], uv[1][0], uv[2][0]];
        self.vs = [uv[0][1], uv[1][1], uv[2][1]];
        self
    }

    /// Principled roughness / metallic.
    pub fn with_pbr(mut self, rough: f32, metal: f32) -> Self {
        (self.rough, self.metal) = (rough.clamp(0.02, 1.0), metal.clamp(0.0, 1.0));
        self
    }

    /// Subsurface weight and its per-channel scattering distance in world
    /// units (Principled "Subsurface Radius" × "Subsurface Scale").
    pub fn with_subsurface(mut self, weight: f32, radius: [f32; 3]) -> Self {
        self.sss = weight.clamp(0.0, 1.0);
        self.sss_radius = radius.map(|r| r.max(0.0));
        self
    }

    /// Sheen weight, roughness and tint (velvet, cotton fuzz, brushed wool).
    pub fn with_sheen(mut self, weight: f32, rough: f32, tint: [f32; 3]) -> Self {
        (self.sheen, self.sheen_rough, self.sheen_tint) = (weight.clamp(0.0, 1.0), rough.clamp(0.07, 1.0), tint);
        self
    }

    /// Coat weight and roughness (clearcoat over the base lobes).
    pub fn with_coat(mut self, weight: f32, rough: f32) -> Self {
        (self.coat, self.coat_rough) = (weight.clamp(0.0, 1.0), rough.clamp(0.02, 1.0));
        self
    }

    /// Anisotropy and its rotation (carried for parity; not yet shaded).
    pub fn with_anisotropy(mut self, amount: f32, rotation: f32) -> Self {
        (self.aniso, self.aniso_rot) = (amount.clamp(-1.0, 1.0), rotation);
        self
    }

    /// A Mix Shader's other closure: material `second` (`Scene::material`)
    /// at weight `mix` (0 all this, 1 all the other), the UVs shared.
    pub fn with_second(mut self, second: u32, mix: f32) -> Self {
        (self.second, self.mix) = (second, mix.clamp(0.0, 1.0));
        self
    }

    /// A procedural texture on the base colour or the mix weight
    /// (`Proc::target`), and one on the normal through a Bump node.
    /// The other colour of a Base Color Mix node, blended over the first
    /// colour slot by the mode and factor it carries ([`Proc::mixed_over`]).
    pub fn with_base_mix(mut self, second: Option<super::mat::Proc>) -> Self {
        self.color_proc2 = second.unwrap_or_default();
        self
    }

    pub fn with_procs(mut self, color: Option<super::mat::Proc>, bump: Option<super::mat::Proc>) -> Self {
        self.color_proc = color.unwrap_or_default();
        self.bump_proc = bump.unwrap_or_default();
        self
    }

    pub fn bounds(&self) -> Aabb {
        Aabb::of_tri(self.a, self.b, self.c)
    }
}

