//! A procedural texture slot of the material record (`procedural.wgsl`,
//! 40 words): one of Blender's texture nodes — Noise, Voronoi, Checker,
//! Brick, Gradient, Magic, Wave, White Noise, Gabor — with its inputs as
//! the node ships them,
//! its coordinates, a colour ramp baked to an atlas image, and what it
//! drives. The word layout is the kernel's (`procedural.wgsl`'s head).

use super::kernel::NO_TEX;
use bytemuck::{Pod, Zeroable};

/// `Proc::kind`'s low byte: the texture and the output read.
pub const PROC_NOISE_FAC: u32 = 1;
pub const PROC_NOISE_COLOR: u32 = 2;
pub const PROC_VORONOI_DISTANCE: u32 = 3;
pub const PROC_VORONOI_COLOR: u32 = 4;
pub const PROC_VORONOI_RADIUS: u32 = 5;
pub const PROC_CHECKER_FAC: u32 = 6;
pub const PROC_CHECKER_COLOR: u32 = 7;
pub const PROC_BRICK_FAC: u32 = 8;
pub const PROC_BRICK_COLOR: u32 = 9;
pub const PROC_GRADIENT: u32 = 10;
pub const PROC_MAGIC_FAC: u32 = 11;
pub const PROC_MAGIC_COLOR: u32 = 12;
pub const PROC_WAVE: u32 = 13;
pub const PROC_WHITE_NOISE_FAC: u32 = 14;
pub const PROC_WHITE_NOISE_COLOR: u32 = 15;
/// Gabor ▸ Value, Phase and Intensity — one phasor, three outputs.
pub const PROC_GABOR_VALUE: u32 = 16;
pub const PROC_GABOR_PHASE: u32 = 17;
pub const PROC_GABOR_INTENSITY: u32 = 18;
/// `Proc::kind`'s flag: the node's Normalize is on (Voronoi; Noise).
pub const PROC_NORMALIZE: u32 = 256;

/// A procedural texture slot. `kind` 0 is an empty slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Proc {
    /// `PROC_*` in the low byte, `PROC_NORMALIZE` above.
    pub kind: u32,
    /// 0 Generated, 1 Object, 2 UV.
    pub coords: u32,
    /// The ramp's atlas image (`Atlas::add`, 256 × 1), `NO_TEX` for none.
    pub ramp: u32,
    /// The first slot: bit 0 the base colour, bit 1 the mix weight. The
    /// bump slot: the Bump node's Distance.
    pub target: u32,
    /// The Mapping node's Location and Scale (identity when none); its
    /// Rotation and Type are at the slot's tail (`rotation`, `mapping`).
    pub location: [f32; 3],
    pub scale3: [f32; 3],
    /// The texture's Scale, Detail, Roughness, Lacunarity, and Distortion
    /// (Noise, Magic, Wave) or Randomness (Voronoi). A Gabor keeps its
    /// Frequency in `detail`, its Anisotropy in `roughness` and its
    /// Orientation 2D in `lacunarity`; its Orientation 3D is `color1`.
    pub scale: f32,
    pub detail: f32,
    pub roughness: f32,
    pub lacunarity: f32,
    pub distortion: f32,
    /// The Bump node's Strength (the bump slot).
    pub strength: f32,
    /// The node's type: `NodeTexNoise.type`, `NodeTexVoronoi.feature`,
    /// `NodeTexGradient.gradient_type`, `NodeTexWave.wave_type`, White
    /// Noise's dimensions (`bNode.custom1`), `NodeTexGabor.type`.
    pub variant: u32,
    /// Voronoi's metric (`NodeTexVoronoi.distance`), Wave's bands
    /// direction, Brick's offset frequency.
    pub direction: u32,
    /// Wave's rings direction, Brick's squash frequency, Magic's depth.
    pub rings: u32,
    /// Wave's profile.
    pub profile: u32,
    /// Noise Offset; Brick's offset amount; Wave Detail Scale; White
    /// Noise's W, which its 1D and 4D forms hash.
    pub offset: f32,
    /// Noise Gain; Brick's squash amount; Wave Detail Roughness.
    pub gain: f32,
    /// Voronoi Smoothness; Wave Phase Offset; Brick Mortar Size.
    pub smoothness: f32,
    /// Voronoi Exponent; Brick Mortar Smooth.
    pub exponent: f32,
    /// Brick Bias, Brick Width, Row Height.
    pub bias: f32,
    pub width: f32,
    pub height: f32,
    /// The Mapping node's Type (`NodeMappingType`, `bNode.custom1`):
    /// 0 Point, 1 Texture, 2 Vector, 3 Normal — what `proc_coord` does
    /// with the three vectors (`svm_mapping` in `mapping_util.h`).
    pub mapping: u32,
    /// Checker / Brick Color1, Color2; Brick Mortar.
    pub color1: [f32; 3],
    pub color2: [f32; 3],
    pub color3: [f32; 3],
    /// The Mapping node's Rotation, Euler XYZ in radians.
    pub rotation: [f32; 3],
    /// The Math node between this texture and the socket it drives
    /// (`NodeMathOperation`, `svm_math`), `MATH_NONE` for none, with its
    /// second and third Value sockets and its Clamp checkbox
    /// (`bNode.custom2`).
    pub math: u32,
    pub math_b: f32,
    pub math_c: f32,
    pub math_clamp: u32,
}

/// `Proc::math` of a slot with no Math node after it.
pub const MATH_NONE: u32 = u32::MAX;

impl Proc {
    /// A White Noise slot: three dimensions, the node's default
    /// (`node_shader_init_tex_white_noise`), and no Scale socket at all.
    pub fn white_noise(target: u32) -> Self {
        Self { kind: PROC_WHITE_NOISE_FAC, variant: 3, scale: 1.0, ..Self::of(0, target) }
    }

    /// A Gabor slot at the node's socket defaults
    /// (`node_shader_tex_gabor.cc`): Scale 5, Frequency 2, Anisotropy 1,
    /// Orientation 2D π/4, Orientation 3D (√2, √2, 0), type 2D.
    pub fn gabor(target: u32) -> Self {
        let r = std::f32::consts::SQRT_2;
        Self {
            kind: PROC_GABOR_VALUE,
            detail: 2.0,
            roughness: 1.0,
            lacunarity: std::f32::consts::FRAC_PI_4,
            color1: [r, r, 0.0],
            ..Self::of(0, target)
        }
    }

    /// An empty slot with `kind` and `target`, the scale 5 and the
    /// Mapping identity — what every node starts from.
    fn of(kind: u32, target: u32) -> Self {
        Self { kind, ramp: NO_TEX, target, scale3: [1.0; 3], scale: 5.0, math: MATH_NONE, ..Default::default() }
    }

    /// This texture with a Math node between it and the socket: the
    /// operation, its second and third Value sockets, and Clamp.
    pub fn mathed(self, math: u32, b: f32, c: f32, clamp: bool) -> Self {
        Self { math, math_b: b, math_c: c, math_clamp: u32::from(clamp), ..self }
    }

    /// Noise as Blender ships it: fBm, Detail 2, Roughness 0.5, Lacunarity
    /// 2, Normalize on, the Fac output.
    pub fn noise(target: u32) -> Self {
        Self { kind: PROC_NOISE_FAC | PROC_NORMALIZE, variant: 1, detail: 2.0, roughness: 0.5, lacunarity: 2.0, offset: 0.0, gain: 1.0, ..Self::of(0, target) }
    }

    /// Voronoi as Blender ships it: F1, Euclidean, Randomness 1, the
    /// Distance output, Normalize off.
    pub fn voronoi(target: u32) -> Self {
        Self { kind: PROC_VORONOI_DISTANCE, roughness: 0.5, lacunarity: 2.0, distortion: 1.0, smoothness: 1.0, exponent: 0.5, ..Self::of(0, target) }
    }

    /// Checker: light grey and dark grey at scale 5, the Fac output.
    pub fn checker(target: u32) -> Self {
        Self { color1: [0.8; 3], color2: [0.2; 3], ..Self::of(PROC_CHECKER_FAC, target) }
    }

    /// Brick as Blender ships it, the Color output.
    pub fn brick(target: u32) -> Self {
        Self {
            color1: [0.8; 3], color2: [0.2; 3], color3: [0.0; 3],
            smoothness: 0.02, exponent: 0.1, bias: 0.0, width: 0.5, height: 0.25,
            offset: 0.5, direction: 2, gain: 1.0, rings: 2,
            ..Self::of(PROC_BRICK_COLOR, target)
        }
    }

    /// Gradient: Linear (it has no Scale).
    pub fn gradient(target: u32) -> Self {
        Self { scale: 1.0, ..Self::of(PROC_GRADIENT, target) }
    }

    /// Magic: depth 2, Distortion 1, the Color output.
    pub fn magic(target: u32) -> Self {
        Self { rings: 2, distortion: 1.0, ..Self::of(PROC_MAGIC_COLOR, target) }
    }

    /// Wave: bands along X, a sine, Detail 2 at scale 1 and roughness 0.5.
    pub fn wave(target: u32) -> Self {
        Self { detail: 2.0, offset: 1.0, gain: 0.5, ..Self::of(PROC_WAVE, target) }
    }

    /// Whether the slot holds a texture.
    pub fn is_some(&self) -> bool {
        self.kind != 0
    }

    /// The texture's kind without its flags.
    pub fn kind(&self) -> u32 {
        self.kind & 0xff
    }

    /// Normalize on: a Voronoi's distance over the largest it can be, a
    /// Noise's fBm into 0 ..= 1.
    pub fn normalized(self, on: bool) -> Self {
        Self { kind: if on { self.kind | PROC_NORMALIZE } else { self.kind & !PROC_NORMALIZE }, ..self }
    }

    /// The same texture read at its other output (Fac ↔ Color; Voronoi's
    /// Distance ↔ Color).
    pub fn colored(self) -> Self {
        let kind = match self.kind() {
            PROC_NOISE_FAC => PROC_NOISE_COLOR,
            PROC_VORONOI_DISTANCE => PROC_VORONOI_COLOR,
            PROC_CHECKER_FAC => PROC_CHECKER_COLOR,
            PROC_BRICK_FAC => PROC_BRICK_COLOR,
            PROC_MAGIC_FAC => PROC_MAGIC_COLOR,
            PROC_WHITE_NOISE_FAC => PROC_WHITE_NOISE_COLOR,
            k => k,
        };
        Self { kind: (self.kind & !0xff) | kind, ..self }
    }

    /// This texture as a Bump node's height, with the node's Strength and
    /// Distance (the bump slot keeps the distance where the colour slot
    /// keeps its target).
    pub fn bump(self, strength: f32, distance: f32) -> Self {
        Self { target: distance.to_bits(), strength, ..self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_slot_is_forty_four_words_and_the_kind_keeps_its_flag() {
        assert_eq!(std::mem::size_of::<Proc>(), 44 * 4);
        let noise = Proc::noise(1);
        assert_eq!((noise.kind(), noise.kind & PROC_NORMALIZE, noise.variant), (PROC_NOISE_FAC, PROC_NORMALIZE, 1));
        assert_eq!(noise.colored().kind, PROC_NOISE_COLOR | PROC_NORMALIZE);
        assert_eq!(Proc::voronoi(2).normalized(true).kind, PROC_VORONOI_DISTANCE | PROC_NORMALIZE);
        assert_eq!(Proc::noise(1).normalized(false).kind, PROC_NOISE_FAC);
        assert!(!Proc::default().is_some() && Proc::brick(1).is_some());
        assert_eq!(Proc::noise(1).math, MATH_NONE, "no Math node by default");
        let m = Proc::noise(1).mathed(2, 3.0, 0.0, true);
        assert_eq!((m.math, m.math_b, m.math_clamp), (2, 3.0, 1));
        assert_eq!((Proc::brick(1).kind(), Proc::gradient(1).scale, Proc::magic(1).rings), (PROC_BRICK_COLOR, 1.0, 2));
    }
}
