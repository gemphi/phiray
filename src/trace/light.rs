//! A file's lamps as the kernels read them.
use bytemuck::{Pod, Zeroable};

/// A Blender light (`kind`: 0 sun, 1 point, 2 spot, 3 area).
/// `color` is colour × strength (sun, W/m²) or colour × power (W).
///
/// The shape fields are what give the light's shadow its penumbra
/// (`shadow.wgsl`): `size` is a sun's angular diameter in radians, a point's
/// or spot's radius in metres, or an area light's edge along its own X; an
/// area light adds `right`, `size_y` and `shape`; a spot adds `cone`,
/// `blend` and `spot_scale`, and shares `right`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Light {
    pub pos: [f32; 3],
    pub kind: u32,
    pub color: [f32; 3],
    /// Soft-shadow size: sun angle (rad), radius (m), or an area light's
    /// edge along `right` (m, the object's scale in it).
    pub size: f32,
    /// Direction the light shines (sun, spot, area).
    pub dir: [f32; 3],
    /// Spot: cosine of the half cone angle.
    pub cone: f32,
    /// An area light's own X axis, normalized; its plane is `dir`'s normal.
    pub right: [f32; 3],
    /// An area light's edge across `right`, in its plane (m, scaled).
    /// Blender gives Square and Disk one size, so this equals `size` for
    /// those (`eevee_light.cc`: Size Y is read for Rectangle and Ellipse).
    pub size_y: f32,
    /// An area light's `Light.area_shape`: 0 square, 1 rectangle, 4 disk,
    /// 5 ellipse — a disk and an ellipse are rounded, the others not.
    pub shape: u32,
    /// A spot's Blend as a width in cosine, `(1 − cone) × Light.spotblend`
    /// (`eevee_light.cc`'s `spot_blend`): the cone fades over
    /// `cone ..= cone + blend`, so 0 is a hard edge.
    pub blend: f32,
    /// A spot's cone across its own X and Y, `scale.z / scale.xy` of the
    /// object (`eevee_light.cc`'s `spot_size_inv`): `[1, 1]` leaves the
    /// cone round, and anything else makes its cross-section an ellipse
    /// (`spot_mask` in `light.wgsl`). Zero for every other kind.
    pub spot_scale: [f32; 2],
}

/// `Light::shape` (`eLightAreaShape`).
pub const AREA_SQUARE: u32 = 0;
pub const AREA_RECT: u32 = 1;
pub const AREA_DISK: u32 = 4;
pub const AREA_ELLIPSE: u32 = 5;
