//! phiray — Phiber's own renderer, alongside Cycles.
//!
//! Three modes, one GPU core: Solid, Material Preview, Rendered (path tracer).
//! The GPU core follows the Warp kernel model (`reference/warp/ANALYSIS.md`):
//! fixed 256-wide blocks, no per-call allocation, recorded passes replayed per frame.

pub mod bvh;
pub mod gpu;
pub mod kernel;
pub mod kernels;
pub mod mode;
pub mod trace;

pub use bvh::{Aabb, Bvh, GpuBvh, Half, Node};
pub use gpu::{Gpu, GpuAdapter, GpuAdapters, GpuArray, Pass};
pub use kernel::{BLOCK, Bind, Dispatch, Kernel};
pub use mode::Mode;
pub use trace::{Atlas, Camera, Instance, LIGHTS, Lamps, Light, Mat, NO_TEX, Scene, Srgb, Tracer, Tri, View};

/// Entry point.
pub struct Ray;

impl Ray {
    /// Renderer modes phiray provides, in viewport order.
    pub fn modes() -> &'static [Mode] {
        Mode::ALL
    }

    /// Open a GPU of its own on the best adapter (headless). `None` when no
    /// adapter exists.
    pub fn gpu() -> Option<Gpu> {
        Gpu::new()
    }
}
