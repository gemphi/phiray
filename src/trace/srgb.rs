//! The sRGB transfer function on the CPU — the one owner of it, as
//! `view.wgsl`'s `srgb` and `srgb_to_linear` are on the GPU (Blender's
//! `srgb_to_linearrgb` / `linearrgb_to_srgb`): what a display-encoded
//! colour, a byte colour or a swatch reads as in scene-linear, and back.

/// The transfer, over floats in `0 ..= 1` and over bytes.
pub struct Srgb;

impl Srgb {
    /// `srgb_to_linearrgb`.
    pub fn to_linear(c: f32) -> f32 {
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    }

    /// `linearrgb_to_srgb`.
    pub fn from_linear(c: f32) -> f32 {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
    }

    /// A byte colour's channel, scene-linear.
    pub fn byte_to_linear(c: u8) -> f32 {
        Self::to_linear(f32::from(c) / 255.0)
    }

    /// A scene-linear channel as a byte colour's, rounded.
    pub fn linear_to_byte(c: f32) -> u8 {
        (Self::from_linear(c) * 255.0 + 0.5) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::Srgb;

    #[test]
    fn the_transfer_round_trips_and_meets_blenders_values() {
        assert!((Srgb::to_linear(0.5) - 0.214_041).abs() < 1e-5, "{}", Srgb::to_linear(0.5));
        assert!((Srgb::from_linear(0.214_041) - 0.5).abs() < 1e-5);
        assert_eq!(Srgb::linear_to_byte(1.0), 255);
        assert_eq!(Srgb::linear_to_byte(0.0), 0);
        assert_eq!(Srgb::linear_to_byte(0.5), 188, "half grey");
        for b in [0u8, 1, 17, 128, 254, 255] {
            assert_eq!(Srgb::linear_to_byte(Srgb::byte_to_linear(b)), b);
        }
    }
}
