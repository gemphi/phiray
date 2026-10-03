//! Edge-avoiding a-trous wavelet denoiser (Dammertz et al. 2010), guided by
//! the first hit's albedo, normal and depth.
//!
//! Path tracing's noise is high-frequency and the scene's real detail is not,
//! so a blur that refuses to cross an edge removes one and keeps the other.
//! Each pass blurs with a 5x5 kernel whose taps are `step` pixels apart —
//! 1, 2, 4, 8, 16 — which reaches a wide radius in five passes rather than one
//! enormous kernel. A tap is weighted down when its normal, albedo or depth
//! disagrees with the centre's, which is what keeps a silhouette or a seam
//! from smearing.
//!
//! Radiance is filtered *divided by albedo*, so texture detail is put back
//! after filtering rather than blurred away with the noise.
//!
//! Buffer layout (`Tracer::PX_WORDS`): 0..n packed display, n..4n linear sums,
//! 4n..7n display sums, 7n..10n albedo, 10n..13n normal, 13n..14n depth,
//! 14n..17n and 17n..20n the two scratch images this ping-pongs between,
//! 20n..24n the grid overlay and its coverage, 24n..25n the film coverage
//! (the pixel's alpha).

use crate::kernel::{Bind, Kernel};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DenoiseParams {
    pub n: u32,
    pub groups_x: u32,
    pub w: u32,
    pub h: u32,
    /// Distance between taps for this pass (1, 2, 4, …).
    pub step: u32,
    /// Which pass this is. Even passes read scratch A and write B, odd the
    /// other way; round 0 reads the accumulated radiance instead. (`pass` is
    /// a reserved word in WGSL, hence `round`.)
    pub round: u32,
    /// 1 on the last pass: put the albedo back, apply the view transform and
    /// write the packed display word.
    pub last: u32,
    /// 1 to leave the linear sums as they are and filter only what is shown:
    /// a viewport refining under the denoiser adds its next sample to the
    /// true sums, not to a filtered copy of them (`Tracer::denoise_shown`).
    pub keep: u32,
    /// How much a disagreement costs, as `1 / sigma`, for normal, albedo and
    /// depth; the fourth component is 1 / sample count.
    pub strength: [f32; 4],
    /// `Camera::view` — the colour management the final pass applies.
    pub view: [f32; 4],
}

pub struct Denoise;

impl Kernel for Denoise {
    const NAME: &'static str = "denoise";
    const BINDS: &'static [Bind] = &[Bind::Write];
    type Params = DenoiseParams;
    const WGSL: &'static str = concat!(
        include_str!("denoise_params.wgsl"),
        include_str!("../trace/wgsl/view.wgsl"),
        include_str!("denoise.wgsl"),
    );
}
