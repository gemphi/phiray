//! The visibility prepass in the tracer: for Solid and Wireframe — the
//! modes Workbench draws — the primary hit of every pixel is rasterized
//! (`gpu::Visibility`, `wgsl/raster.wgsl`) before the kernel runs, and the
//! kernel reads it instead of walking the tree (`vis2.wgsl`'s `primary`).
//! X-ray, which wants every surface along the ray, and a tile of a larger
//! frame still trace; so does a software adapter, where rasterizing
//! 436 k faces of THYMUS pulled from storage took 378 ms a frame against
//! 83 ms traced (lavapipe, 1200 × 700) — the hardware the prepass is for
//! rasterizes that many in under a millisecond. `Tracer::set_raster` picks
//! either way (the tests compare the two: pixel for pixel the same).

use super::kernel::Camera;
use super::shader::Raster;
use super::Tracer;
use crate::gpu::{Gpu, Visibility};
use crate::mode::Mode;

impl Tracer {
    /// Whether to rasterize the primary hits for this frame.
    pub(super) fn rasterizes(&self, cam: &Camera) -> bool {
        self.raster && matches!(self.mode, Mode::Solid | Mode::Wire) && cam.see_through <= 0.0 && cam.tile == [0; 4]
    }

    /// `Camera.vis` for a frame `w` wide: drawn or not, the hits' row
    /// stride, the orthographic depth reach — the farthest scene corner
    /// from the pivot — and the perspective near plane.
    pub(super) fn vis_words(&self, cam: &Camera, drawn: bool, w: u32) -> [u32; 4] {
        if !drawn {
            return [0; 4];
        }
        let b = self.bounds;
        let reach = (0..8)
            .map(|k| {
                let c = [if k & 1 == 0 { b.lo[0] } else { b.hi[0] }, if k & 2 == 0 { b.lo[1] } else { b.hi[1] }, if k & 4 == 0 { b.lo[2] } else { b.hi[2] }];
                (0..3).map(|i| (c[i] - cam.pivot[i]).powi(2)).sum::<f32>().sqrt()
            })
            .fold(0.0f32, f32::max)
            * 1.01
            + self.extent * 1e-3;
        let near = (self.extent * 1e-4).max(1e-6);
        [1, Visibility::stride_for(w), reach.to_bits(), near.to_bits()]
    }

    /// The prepass over `w × h`, bound to the scene's buffers as they are.
    pub(super) fn visibility(&self, gpu: &Gpu, cam: &Camera, w: u32, h: u32) -> Visibility {
        let (source, key) = Raster::source(self.chunks());
        let bufs: Vec<&wgpu::Buffer> = self.faces.iter().map(|a| a.buffer()).chain(self.verts.iter().map(|a| a.buffer())).chain([self.data.buffer()]).collect();
        Visibility::new(gpu, key, "phiray.raster", source, &bufs, cam, w, h)
    }

    /// Rasterize or trace the primary hits (Workbench's prepass). Off, the
    /// Solid and Wireframe kernels trace every pixel's ray as the others do.
    pub fn set_raster(&mut self, on: bool) {
        if self.raster != on {
            self.raster = on;
            self.pass.clear();
            self.last = None;
        }
    }

    pub fn raster(&self) -> bool {
        self.raster
    }
}
