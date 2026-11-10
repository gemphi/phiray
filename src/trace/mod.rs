//! Rendered mode: GPU ray tracing over a [`GpuBvh`] — primary ray, sun with
//! shadow, short-range ambient occlusion, sky gradient. One dispatch per frame.

pub mod blue_noise;
pub mod hash;
pub mod fresnel;
pub mod srgb;
pub mod ibl;
pub mod instance;
pub mod mesh;
pub mod outline;
pub mod proc;
pub mod tri;
mod atlas;
mod build;
mod camera;
mod kernel;
mod lamp;
mod light;
mod mat;
mod pose;
mod read;
mod sample;
mod scene;
mod shader;
pub mod smaa;
pub mod taa;
mod forest;
mod visible;

pub use atlas::Atlas;
pub use camera::{Camera, MAX_LIGHT, Look, PixelFilter, View};
pub use kernel::{NO_MAT, NO_TEX, Tri};
pub use light::{AREA_DISK, AREA_ELLIPSE, AREA_RECT, AREA_SQUARE, Light};
pub use lamp::LampList;

pub use shader::{Adapt, AdaptClear, AdaptDilate, AdaptGather, Edge, Fx, Glow, Preview, Preview4, Show, Smaa, Solid, Solid4, Taa, Trace, Trace4, Wire, Wire4};
pub use outline::Outline;
pub use instance::Instance;
pub use mesh::Mesh;
pub use mat::Mat;
pub use proc::{PROC_NORMALIZE, Proc};
pub use srgb::Srgb;
pub use scene::Scene;

use crate::bvh::{Aabb, Node};
use crate::gpu::{Clock, Gpu, GpuArray, Pass, Time, Visibility};
use crate::mode::Mode;

pub struct Tracer {
    /// Each large array over `chunks()` bindings.
    nodes: Vec<GpuArray<Node>>,
    faces: Vec<GpuArray<[u32; 4]>>,
    verts: Vec<GpuArray<[f32; 8]>>,
    /// `[table…, atlas…, instances…]` (`Scene::table`, `Atlas::words`,
    /// `Forest::inst_words`: each record, the top-tree order, the draw order).
    data: GpuArray<u32>,
    /// `[packed pixel × n, accumulated rgb × n]`.
    px: Option<GpuArray<u32>>,
    /// The visibility prepass for the frame in hand, when Solid or Wire
    /// rasterize their primary hits (`visible.rs`); `blank` stands in its
    /// binding when they trace.
    vis: Option<Visibility>,
    blank: GpuArray<u32>,
    /// EEVEE's blue-noise table for the SSAO's per-pixel jitter
    /// (`blue_noise.rs`), bound to the effects pass.
    noise: GpuArray<u32>,
    /// SMAA's area and search tables (`smaa.rs`), bound to its passes.
    smaa: GpuArray<u32>,
    /// One word an instance: its selection-outline class plus one
    /// (`outline.rs`), bound to the outline pass. All zero — nothing
    /// outlined — until [`Self::outline`] says otherwise.
    class: GpuArray<u32>,
    /// The classes held, so a selection that has not moved uploads nothing.
    classes: Vec<u32>,
    /// A settled image has one more sample to draw: the outline changed
    /// under it (`Self::outline`), and the pass that lays it on runs with
    /// the samples, not on its own.
    reshow: bool,
    raster: bool,
    /// The prepass's draws: one instanced draw a mesh (`Forest::draws`).
    draws: Vec<[u32; 4]>,
    /// The scene's world bounds (the prepass's orthographic depth reach).
    bounds: Aabb,
    /// Items per binding (nodes, faces, vertices) and the atlas offset.
    split: [u32; 4],
    /// The instance table's first word in `data` and the instance count.
    insts: [u32; 4],
    last: Option<Camera>,
    frame: u32,
    pass: Pass,
    /// The GPU's own time for a sample (gpu/clock.rs); None where the
    /// device has no timestamp queries, and a sync a sample stands in.
    clock: Option<Clock>,
    /// What the last sample cost, submit and GPU apart ([Self::time]).
    time: Time,
    size: (u32, u32),
    /// Emissive faces (Camera::glow).
    glow: u32,
    /// Scene diagonal (for ray offsets and AO reach).
    pub extent: f32,
    /// Which kernel draws a frame. Solid is Workbench — one deterministic
    /// dispatch; the others path trace and refine.
    mode: Mode,
    /// Samples a still view refines to (`SAMPLE_COUNT` unless set).
    target: u32,
    /// Solid and Wireframe's anti-aliasing count (`set_aa_samples`).
    aa: u32,
    /// What a new pose of the scene reuses (`pose.rs`).
    pose: Option<pose::Pose>,
    /// Adaptive Sampling as Cycles resolved it: `[threshold, min samples,
    /// step, on]`, stamped onto every camera (`Camera::adapt`). All zero —
    /// off — until [`Self::set_adaptive`] says otherwise.
    adapt: [f32; 4],
}

impl Tracer {
    /// Items per binding for nodes, faces, vertices (and the atlas offset).
    pub fn split(&self) -> [u32; 4] {
        self.split
    }

    /// Bindings per large array (2 or 4).
    pub fn chunks(&self) -> usize {
        self.nodes.len()
    }

    /// Samples averaged before a still view stops refining.
    pub const SAMPLE_COUNT: u32 = 32;

    /// Pick the kernel. Changing it drops the recorded pass so the next frame
    /// records the new one.
    pub fn set_mode(&mut self, mode: Mode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        self.pass.clear();
        self.size = (0, 0);
        self.last = None;
        self.frame = 0;
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// How many samples this mode averages. Material Preview is one
    /// deterministic dispatch; Solid and Wireframe draw Workbench's
    /// jittered samples — `aa_samples`, at least one — and gather them
    /// (`taa.rs`); the path tracer refines to the target.
    pub fn samples(&self) -> u32 {
        match self.mode {
            Mode::Preview => 1,
            Mode::Solid | Mode::Wire => self.aa_samples().max(1),
            _ => self.target,
        }
    }

    /// Workbench's anti-aliasing count (`SceneState::samples_len`,
    /// `set_aa_samples`'s number as `taa::count` takes it): 0 is no
    /// anti-aliasing (`SCE_DISPLAY_AA_OFF`: the one sample, no SMAA), 1 the
    /// single pass with SMAA alone (`_FXAA`), else that many jittered
    /// samples with SMAA over the first four.
    pub fn aa_samples(&self) -> u32 {
        taa::count(self.aa)
    }

    /// How many rows a tile must be drawn over into its neighbour for the
    /// seam not to show (`rendered/still.rs`): the effects and the gather
    /// read a pixel's neighbours, two rows; a Solid or Wireframe frame the
    /// single pass leaves SMAA on reads as far as its searches go — 16
    /// steps of two texels and the step the search table adds, the edges
    /// two texels further and the blend one more (`smaa_lib.wgsl`): 40.
    pub fn seam_rows(&self) -> u32 {
        if self.jittered() && self.aa_samples() == 1 { 40 } else { 2 }
    }

    /// Anti-alias Solid and Wireframe with `n` samples (`U.viewport_aa` in
    /// the viewport, `Scene.display.render_aa` for a render; 0 off, 1 a
    /// single pass). Eight until set. Takes effect from the next view.
    pub fn set_aa_samples(&mut self, n: u32) {
        self.aa = n;
    }

    /// Which objects the selection outline draws around
    /// (`outline.rs`), one class an instance in the order the scene was
    /// built in — a shorter list leaves the rest unoutlined. A selection
    /// that has not moved uploads nothing; one that has leaves the samples
    /// traced alone and asks for one more, so a settled image picks the new
    /// outline up without starting over.
    pub fn outline(&mut self, gpu: &Gpu, classes: &[Outline]) -> bool {
        let mut words = vec![0u32; self.classes.len()];
        for (word, class) in words.iter_mut().zip(classes) {
            *word = class.word();
        }
        if words == self.classes {
            return true;
        }
        self.classes = words;
        self.reshow = true;
        self.class.write_front(gpu, &self.classes)
    }

    /// Whether any object is outlined — whether the pass is worth a
    /// dispatch at all.
    fn outlined(&self) -> bool {
        self.classes.iter().any(|w| *w != 0)
    }

    /// Whether the mode anti-aliases by Workbench's jittered samples: Solid
    /// and Wireframe, in the viewport (`viewport_aa`) and in a final render
    /// (`render_aa`) alike, so a tile of a larger frame too. The gather
    /// clamps at a tile's edge, so tiles that must join seamlessly are
    /// rendered a row over and laid in without it.
    fn jittered(&self) -> bool {
        matches!(self.mode, Mode::Solid | Mode::Wire)
    }

    /// Reads one word: the length of the list of open pixels the last
    /// sample's tail gathered (`gather.wgsl`, the word after the image).
    /// It read the whole flag plane (30n..31n) and counted it on the CPU —
    /// eight megabytes back for one number on a 1440² frame.
    pub fn unconverged(&self, gpu: &Gpu) -> Option<u32> {
        if self.adapt[3] == 0.0 {
            return None;
        }
        let n = (self.size.0 * self.size.1) as usize;
        let px = self.px.as_ref()?;
        // A short read means the device declined; say nothing rather than
        // claim a converged frame and stop early.
        px.read_range(gpu, Self::PX_LEN * n, 1).first().copied()
    }

    pub fn set_adaptive(&mut self, adapt: [f32; 4]) {
        if self.adapt != adapt {
            self.adapt = adapt;
            self.pass.clear();
        }
    }

    pub fn set_samples(&mut self, n: u32) {
        self.target = n.max(1);
    }

}
