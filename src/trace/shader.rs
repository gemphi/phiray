//! The path-tracing kernels (WGSL in `wgsl/`).

use super::kernel::Camera;
use crate::kernel::{Bind, Kernel};

/// The path tracer with each large array over 2 bindings (8 storage bindings).
pub struct Trace;

/// The same tracer over 4 bindings per array (14 storage bindings).
pub struct Trace4;

const R: Bind = Bind::Read;

/// Workbench (Solid) over 2 bindings per array.
pub struct Solid;

/// Wireframe over 2 bindings per array.
pub struct Wire;

/// Wireframe over 4 bindings per array.
pub struct Wire4;

/// Workbench (Solid) over 4 bindings per array.
pub struct Solid4;

/// Material Preview over 2 bindings per array.
pub struct Preview;

/// Material Preview over 4 bindings per array.
pub struct Preview4;

/// Adaptive Sampling's convergence check (`adapt.wgsl`): Cycles'
/// `film_adaptive_sampling_convergence_check` over the full and half
/// images the path tracer accumulates, marking pixels that have settled
/// so the tracer stops sampling them. It binds the image alone.
pub struct Adapt;

impl Kernel for Adapt {
    const NAME: &'static str = "adapt";
    const BINDS: &'static [Bind] = &[Bind::Write];
    type Params = Camera;
    const WGSL: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/adapt.wgsl"));
}

/// Adaptive Sampling's dilation (`adapt.wgsl`'s `dilate`): Cycles'
/// `film_adaptive_sampling_filter_x` / `_y` as one parallel pass, so a
/// settled pixel beside an unsettled one keeps refining. Same source as
/// [`Adapt`], a different entry point.
pub struct AdaptDilate;

impl Kernel for AdaptDilate {
    const NAME: &'static str = "adapt_dilate";
    const ENTRY: &'static str = "dilate";
    const BINDS: &'static [Bind] = &[Bind::Write];
    type Params = Camera;
    const WGSL: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/adapt.wgsl"));
}

/// Workbench's anti-aliasing gather (`taa.wgsl`): one sample's image
/// filtered into the running sums and resolved, after each Solid or
/// Wireframe dispatch. It binds the image alone.
pub struct Taa;

impl Kernel for Taa {
    const NAME: &'static str = "taa";
    const BINDS: &'static [Bind] = &[Bind::Write];
    type Params = Camera;
    const WGSL: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/taa.wgsl"));
}

/// SMAA over the gather's first samples (`smaa.wgsl`, `smaa_lib.wgsl`):
/// Workbench's `smaa_edge_detect_ps_`, `smaa_aa_weight_ps_` and
/// `smaa_resolve_ps_` as three stages (`Camera::slice.z` 1–3). It binds
/// the image and the area and search tables (`smaa.rs`).
pub struct Smaa;

impl Kernel for Smaa {
    const NAME: &'static str = "smaa";
    const BINDS: &'static [Bind] = &[Bind::Write, Bind::Read];
    type Params = Camera;
    const WGSL: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/smaa_lib.wgsl"), include_str!("wgsl/smaa.wgsl"));
}

/// Workbench's Solid effects (`fx.wgsl`): the cavity (SSAO) and curvature
/// effects and the object outline over the opaque pass's buffers, before
/// the gather, and the Depth of Field's stages (`dof.wgsl`, `Camera::
/// slice.z`). It binds the image and the blue-noise jitter table
/// (`blue_noise.rs`).
pub struct Fx;

impl Kernel for Fx {
    const NAME: &'static str = "fx";
    const BINDS: &'static [Bind] = &[Bind::Write, Bind::Read];
    type Params = Camera;
    const WGSL: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/view.wgsl"), include_str!("wgsl/fx.wgsl"), include_str!("wgsl/dof.wgsl"));
}

/// The display step alone (`show.wgsl`): the view transform and the grid
/// over the linear sums already in the image, so a host that has
/// composited the linear pixels can have them shown without a second
/// implementation of the transform (`Tracer::show`). It binds the image.
pub struct Show;

impl Kernel for Show {
    const NAME: &'static str = "show";
    const BINDS: &'static [Bind] = &[Bind::Write];
    type Params = Camera;
    const WGSL: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/view.wgsl"), include_str!("wgsl/show.wgsl"));
}

/// The selection outline (`outline.wgsl`): the id edges of the selected
/// objects over the finished image, after the gather and SMAA — Blender's
/// overlay pass. It binds the image and the per-instance class table
/// (`outline.rs`).
pub struct Edge;

impl Kernel for Edge {
    const NAME: &'static str = "edge";
    const BINDS: &'static [Bind] = &[Bind::Write, Bind::Read];
    type Params = Camera;
    const WGSL: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/view.wgsl"), include_str!("wgsl/outline.wgsl"));
}

macro_rules! trace_wgsl {
    ($binds:literal) => {
        concat!(
            include_str!("wgsl/camera.wgsl"),
            include_str!("wgsl/layout.wgsl"),
            include_str!($binds),
            include_str!("wgsl/texture.wgsl"),
            include_str!("wgsl/noise.wgsl"),
            include_str!("wgsl/hash.wgsl"),
            include_str!("wgsl/gabor.wgsl"),
            include_str!("wgsl/math.wgsl"),
            include_str!("wgsl/voronoi.wgsl"),
            include_str!("wgsl/pattern.wgsl"),
            include_str!("wgsl/procedural.wgsl"),
            include_str!("wgsl/bvh.wgsl"),
            include_str!("wgsl/view.wgsl"),
            include_str!("wgsl/util.wgsl"),
            include_str!("wgsl/light.wgsl"),
            include_str!("wgsl/shade.wgsl"),
        )
    };
}

// Material Preview lights every pixel in one pass from the prefiltered
// environment and the lamps (`preview.wgsl`); it takes the tracer's light
// library for the lamps' BRDF and the environment's mapping.
macro_rules! preview_wgsl {
    ($binds:literal) => {
        concat!(
            include_str!("wgsl/camera.wgsl"),
            include_str!("wgsl/layout.wgsl"),
            include_str!($binds),
            include_str!("wgsl/texture.wgsl"),
            include_str!("wgsl/noise.wgsl"),
            include_str!("wgsl/hash.wgsl"),
            include_str!("wgsl/gabor.wgsl"),
            include_str!("wgsl/math.wgsl"),
            include_str!("wgsl/voronoi.wgsl"),
            include_str!("wgsl/pattern.wgsl"),
            include_str!("wgsl/procedural.wgsl"),
            include_str!("wgsl/bvh.wgsl"),
            include_str!("wgsl/view.wgsl"),
            include_str!("wgsl/util.wgsl"),
            include_str!("wgsl/light.wgsl"),
            include_str!("wgsl/shadow.wgsl"),
            include_str!("wgsl/preview.wgsl"),
        )
    };
}

// Solid and Wire need no lighting integrator: one primary ray, done. They
// differ only in the entry point, so they share the prelude.
macro_rules! direct_wgsl {
    ($binds:literal, $vis:literal, $main:literal) => {
        concat!(
            include_str!("wgsl/camera.wgsl"),
            include_str!("wgsl/layout.wgsl"),
            include_str!($binds),
            include_str!($vis),
            include_str!("wgsl/texture.wgsl"),
            include_str!("wgsl/bvh.wgsl"),
            include_str!("wgsl/view.wgsl"),
            include_str!("wgsl/util.wgsl"),
            include_str!("wgsl/workbench.wgsl"),
            include_str!($main),
        )
    };
}

impl Kernel for Trace {
    const NAME: &'static str = "trace";
    const BINDS: &'static [Bind] = &[R, R, R, R, R, R, R, Bind::Write];
    type Params = Camera;
    const WGSL: &'static str = trace_wgsl!("wgsl/binds2.wgsl");
}

impl Kernel for Trace4 {
    const NAME: &'static str = "trace4";
    const BINDS: &'static [Bind] = &[R, R, R, R, R, R, R, R, R, R, R, R, R, Bind::Write];
    type Params = Camera;
    const WGSL: &'static str = trace_wgsl!("wgsl/binds4.wgsl");
}

impl Kernel for Solid {
    const NAME: &'static str = "solid";
    const BINDS: &'static [Bind] = &[R, R, R, R, R, R, R, Bind::Write, R];
    type Params = Camera;
    const WGSL: &'static str = direct_wgsl!("wgsl/binds2.wgsl", "wgsl/vis2.wgsl", "wgsl/solid.wgsl");
}

impl Kernel for Solid4 {
    const NAME: &'static str = "solid4";
    const BINDS: &'static [Bind] = &[R, R, R, R, R, R, R, R, R, R, R, R, R, Bind::Write, R];
    type Params = Camera;
    const WGSL: &'static str = direct_wgsl!("wgsl/binds4.wgsl", "wgsl/vis4.wgsl", "wgsl/solid.wgsl");
}

impl Kernel for Wire {
    const NAME: &'static str = "wire";
    const BINDS: &'static [Bind] = &[R, R, R, R, R, R, R, Bind::Write, R];
    type Params = Camera;
    const WGSL: &'static str = direct_wgsl!("wgsl/binds2.wgsl", "wgsl/vis2.wgsl", "wgsl/wire.wgsl");
}

impl Kernel for Wire4 {
    const NAME: &'static str = "wire4";
    const BINDS: &'static [Bind] = &[R, R, R, R, R, R, R, R, R, R, R, R, R, Bind::Write, R];
    type Params = Camera;
    const WGSL: &'static str = direct_wgsl!("wgsl/binds4.wgsl", "wgsl/vis4.wgsl", "wgsl/wire.wgsl");
}

impl Kernel for Preview {
    const NAME: &'static str = "preview";
    const BINDS: &'static [Bind] = &[R, R, R, R, R, R, R, Bind::Write];
    type Params = Camera;
    const WGSL: &'static str = preview_wgsl!("wgsl/binds2.wgsl");
}

impl Kernel for Preview4 {
    const NAME: &'static str = "preview4";
    const BINDS: &'static [Bind] = &[R, R, R, R, R, R, R, R, R, R, R, R, R, Bind::Write];
    type Params = Camera;
    const WGSL: &'static str = preview_wgsl!("wgsl/binds4.wgsl");
}

/// The visibility prepass (`raster.wgsl`): a render pipeline, not a kernel,
/// over the faces, vertices and data of the same scene — in two or four
/// bindings per array (`gpu::Visibility`).
pub struct Raster;

impl Raster {
    pub const WGSL2: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/layout.wgsl"), include_str!("wgsl/raster_binds2.wgsl"), include_str!("wgsl/raster.wgsl"));
    pub const WGSL4: &'static str = concat!(include_str!("wgsl/camera.wgsl"), include_str!("wgsl/layout.wgsl"), include_str!("wgsl/raster_binds4.wgsl"), include_str!("wgsl/raster.wgsl"));

    /// The source for `chunks` bindings per array, and its cache key.
    pub fn source(chunks: usize) -> (&'static str, u64) {
        let src = if chunks == 4 { Self::WGSL4 } else { Self::WGSL2 };
        let key = format!("raster{chunks}").bytes().chain(src.bytes()).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
        (src, key)
    }
}
