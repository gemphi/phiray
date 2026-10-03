//! Viewport shading modes served by phiray.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Edges only, depth-tested (`V3D_SHADING_WIREFRAME`).
    Wire,
    /// Flat/studio shading (phiber canvas today).
    Solid,
    /// Material Preview: materials and textures under the studio light, in
    /// one deterministic dispatch — prefiltered environment lighting, hard
    /// shadows from the lamps, mipmapped textures (`preview.wgsl`), the way
    /// EEVEE draws Blender's — never a sampled, refining image.
    Preview,
    /// wgpu compute path tracer.
    Rendered,
}

impl Mode {
    /// Blender's viewport order: wireframe, solid, material, rendered.
    pub const ALL: &'static [Mode] = &[Mode::Wire, Mode::Solid, Mode::Preview, Mode::Rendered];

    /// Modes that finish in a single dispatch: no sampling, no refinement.
    pub fn deterministic(self) -> bool {
        matches!(self, Mode::Wire | Mode::Solid | Mode::Preview)
    }

    /// Blender's `View3DShading.type` name.
    pub fn blender(self) -> &'static str {
        match self {
            Mode::Wire => "WIREFRAME",
            Mode::Solid => "SOLID",
            Mode::Preview => "MATERIAL",
            Mode::Rendered => "RENDERED",
        }
    }

    pub fn from_blender(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|m| m.blender() == name)
    }
}
