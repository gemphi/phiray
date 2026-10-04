//! Material table entry shared by many faces.

use super::kernel::Tri;
use bytemuck::{Pod, Zeroable};

/// Surface inputs shared by many faces (32 words on the GPU).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Mat {
    pub color: [f32; 4],
    pub rough: f32,
    pub metal: f32,
    pub trans: f32,
    pub ior: f32,
    pub emit: [f32; 3],
    pub bump: f32,
    pub tex: u32,
    pub nmap: u32,
    /// Roughness / metallic / emission textures (see [`Tri::with_maps`]).
    pub maps: [u32; 3],
    pub sss: f32,
    pub sss_radius: [f32; 3],
    pub sheen: f32,
    pub sheen_rough: f32,
    pub sheen_tint: [f32; 3],
    pub coat: f32,
    pub coat_rough: f32,
    pub aniso: f32,
    pub aniso_rot: f32,
    /// A Mix Shader's other closure: the id of a second material, `NO_MAT`
    /// for none, and the weight the second takes (`Tri::with_second`). The
    /// path tracer shades one of the two per hit, chosen by the weight —
    /// how Cycles samples a mix of closures — and Material Preview shades
    /// both and lerps.
    pub second: u32,
    pub mix: f32,
}

/// `Mat::second` of a material with one lobe.
pub const NO_MAT: u32 = u32::MAX;

impl Mat {
    pub fn of(t: &Tri) -> Self {
        Self {
            color: t.color, rough: t.rough, metal: t.metal, trans: t.trans, ior: t.ior,
            emit: t.emit, bump: t.bump, tex: t.tex, nmap: t.nmap, maps: t.maps,
            sss: t.sss, sss_radius: t.sss_radius,
            sheen: t.sheen, sheen_rough: t.sheen_rough, sheen_tint: t.sheen_tint,
            coat: t.coat, coat_rough: t.coat_rough, aniso: t.aniso, aniso_rot: t.aniso_rot,
            second: t.second, mix: t.mix,
        }
    }
    pub(super) fn key(&self) -> [u32; 32] {
        bytemuck::cast(*self)
    }
    pub(super) fn power(&self) -> f32 {
        0.2126 * self.emit[0] + 0.7152 * self.emit[1] + 0.0722 * self.emit[2]
    }

    /// The emitted power of this material with its second lobe, as the
    /// mix shades it: the lobes' powers by their weights.
    pub(super) fn power_in(&self, mats: &[Mat]) -> f32 {
        match mats.get(self.second as usize) {
            Some(other) if self.second != NO_MAT => self.power() * (1.0 - self.mix) + other.power() * self.mix,
            _ => self.power(),
        }
    }
}

impl Tri {
    /// This triangle's material inputs.
    pub fn mat(&self) -> Mat {
        Mat::of(self)
    }
}

