//! Rendered mode: GPU ray tracing over a [`GpuBvh`] — primary ray, sun with
//! shadow, short-range ambient occlusion, sky gradient. One dispatch per frame.

pub mod blue_noise;
pub mod hash;
pub mod srgb;
pub mod ibl;
pub mod instance;
pub mod outline;
pub mod proc;
mod kernel;
mod lamps;
mod mat;
mod pose;
mod scene;
mod shader;
pub mod smaa;
pub mod taa;
mod trees;
mod visible;

pub use kernel::{AREA_DISK, AREA_ELLIPSE, AREA_RECT, AREA_SQUARE, Atlas, Camera, LIGHTS, Light, NO_MAT, NO_TEX, Tri, View};
pub use lamps::Lamps;
/// A dispatch after the slices (`Tracer::tail`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BTail {
    Fx,
    Taa,
    Smaa,
    Edge,
}

pub use shader::{Edge, Fx, Preview, Preview4, Show, Smaa, Solid, Solid4, Taa, Trace, Trace4, Wire, Wire4};
pub use outline::Outline;
pub use instance::Instance;
pub use mat::Mat;
pub use proc::{PROC_NORMALIZE, Proc};
pub use srgb::Srgb;
pub use scene::Scene;

use crate::bvh::{Aabb, Node};
use crate::gpu::{Gpu, GpuArray, Pass, Visibility};
use crate::kernel::Dispatch;
use crate::mode::Mode;

pub struct Tracer {
    /// Each large array over `chunks()` bindings.
    nodes: Vec<GpuArray<Node>>,
    faces: Vec<GpuArray<[u32; 4]>>,
    verts: Vec<GpuArray<[f32; 8]>>,
    /// `[table…, atlas…, instances…]` (`Scene::table`, `Atlas::words`,
    /// `Trees::inst_words`).
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
    /// Faces in the scene (the prepass draws three vertices each).
    n_faces: u32,
    /// The scene's world bounds (the prepass's orthographic depth reach).
    bounds: Aabb,
    /// Items per binding (nodes, faces, vertices) and the atlas offset.
    split: [u32; 4],
    /// The instance table's first word in `data` and the instance count.
    insts: [u32; 4],
    last: Option<Camera>,
    frame: u32,
    pass: Pass,
    size: (u32, u32),
    /// Emissive faces (Camera::glow).
    glow: u32,
    /// Scene diagonal (for ray offsets and AO reach).
    pub extent: f32,
    /// Which kernel draws a frame. Solid is Workbench — one deterministic
    /// dispatch; the others path trace and refine.
    mode: Mode,
    /// Samples a still view refines to (`SAMPLES` unless set).
    target: u32,
    /// Solid and Wireframe's anti-aliasing count (`set_aa_samples`).
    aa: u32,
    /// What a new pose of the scene reuses (`pose.rs`).
    pose: Option<pose::Pose>,
}

impl Tracer {
    /// `None` for an empty scene.
    pub fn new(gpu: &Gpu, tris: &[Tri], atlas: &Atlas) -> Option<Self> {
        Self::from_scene(gpu, Scene::from_tris(tris, atlas))
    }

    /// Build from a compact scene; `None` when empty or larger than the device allows.
    pub fn from_scene(gpu: &Gpu, scene: Scene) -> Option<Self> {
        Self::with_limit(gpu, scene, gpu.max_binding())
    }

    /// Like [`Self::from_scene`] with an explicit per-binding byte limit, over as
    /// many bindings per array as the device allows ([`Self::chunks_for`]).
    pub fn with_limit(gpu: &Gpu, scene: Scene, limit: u64) -> Option<Self> {
        Self::with_chunks(gpu, scene, limit, Self::chunks_for(gpu))
    }

    /// Compile every viewport kernel on `gpu` — the ones a scene of this
    /// device's binding count will use — so switching shading does not wait
    /// a second for the first frame of a mode while its kernel compiles.
    /// Each compiles once per device; a kernel already compiled costs
    /// nothing here. Meant for a thread of its own once the first image is
    /// up: the compiles hold the device's pipeline lock one at a time.
    /// Returns each kernel's name and how long it took, in milliseconds (a
    /// first run on a driver with no shader cache can take seconds each).
    pub fn warm(gpu: &Gpu) -> Vec<(&'static str, f64)> {
        let timed = |name, compile: fn(&Gpu)| {
            let at = std::time::Instant::now();
            compile(gpu);
            (name, at.elapsed().as_secs_f64() * 1000.0)
        };
        if Self::chunks_for(gpu) == 4 {
            vec![timed("Solid", Pass::compile::<Solid4>), timed("Preview", Pass::compile::<Preview4>), timed("Wire", Pass::compile::<Wire4>), timed("Trace", Pass::compile::<Trace4>)]
        } else {
            vec![timed("Solid", Pass::compile::<Solid>), timed("Preview", Pass::compile::<Preview>), timed("Wire", Pass::compile::<Wire>), timed("Trace", Pass::compile::<Trace>)]
        }
    }

    /// 4 bindings per array when the device has 14 storage bindings, else 2.
    pub fn chunks_for(gpu: &Gpu) -> usize {
        if gpu.device.limits().max_storage_buffers_per_shader_stage >= 14 { 4 } else { 2 }
    }

    /// Each instance's tree is built on the CPU (all cores) over its faces
    /// in its own space, and a top tree over the instances as placed
    /// (`trees.rs`); nodes, faces and vertices are each split over `chunks`
    /// (2 or 4) bindings, so a scene may use `chunks ×` the limit per array
    /// (as Blender's GPU backends chunk large buffers).
    /// Why this scene will not build on a device with this binding limit,
    /// in the words the Info report shows — `None` when it fits.
    ///
    /// Every `None` out of [`Self::with_chunks`] used to look the same from
    /// outside: the app said "no adapter, or a scene the GPU declined" and
    /// the Render Result was blank. A frame that does not render has one
    /// specific reason and the user is entitled to it, so the checks state
    /// theirs and `with_chunks` reads from here rather than keeping a
    /// second copy of the arithmetic.
    pub fn why_not(scene: &Scene, limit: u64, chunks: usize) -> Option<String> {
        let chunks = if chunks > 2 { 4 } else { 2 } as u64;
        let mib = |b: u64| format!("{:.0} MB", b as f64 / 1048576.0);
        if scene.is_empty() {
            return Some("the scene has nothing in it".into());
        }
        let biggest = scene.biggest();
        if biggest > limit * chunks {
            return Some(format!(
                "its largest array is {} and this device binds {} over {chunks} chunks",
                mib(biggest),
                mib(limit * chunks)
            ));
        }
        // The one array that is NOT chunked: the scene table, the texture
        // atlas, the instance words and — when any object carries a colour
        // attribute — one word per vertex in the whole scene.
        let data = Self::data_words(scene) * 4;
        if data > limit {
            return Some(format!(
                "its material/texture table is {} and this device binds {} in one piece (it is not chunked: the atlas is {}, and a colour attribute adds one word per vertex over {} vertices)",
                mib(data),
                mib(limit),
                mib(scene.atlas.words().len() as u64 * 4),
                scene.pos.len()
            ));
        }
        for (what, items, each) in [
            ("vertices", scene.pos.len() as u64, 32u64),
            ("faces", scene.faces.len() as u64, 16),
        ] {
            let cap = (limit / each).max(1) * chunks;
            if items > cap {
                return Some(format!("it has {items} {what} and this device binds {cap} over {chunks} chunks"));
            }
        }
        None
    }

    /// The length, in words, of the unchunked `data` buffer `with_chunks`
    /// assembles — the figure [`Self::why_not`] checks and the one the
    /// builder itself lays out below.
    fn data_words(scene: &Scene) -> u64 {
        let colors = if scene.colors.is_empty() { 0 } else { scene.pos.len() as u64 };
        scene.table().len() as u64 + scene.atlas.words().len() as u64 + scene.insts.len() as u64 * (instance::INSTANCE_WORDS as u64 + 1) + colors
    }

    pub fn with_chunks(gpu: &Gpu, mut scene: Scene, limit: u64, chunks: usize) -> Option<Self> {
        let chunks = if chunks > 2 { 4 } else { 2 };
        if scene.biggest() > limit * chunks as u64 {
            return None;
        }
        if scene.is_empty() {
            return None;
        }
        scene.seal();
        let (trees, order) = trees::Trees::build(&scene);
        // Faces in leaf order, so leaves index faces directly.
        scene.reorder(&order);
        let extent = trees.extent();
        let bounds = trees.bounds();
        let mut data = scene.table();
        let (glow, atlas_at) = (data[0], data.len() as u32);
        data.extend(scene.atlas.words());
        let insts_at = data.len() as u32;
        data.extend(trees.inst_words(&scene.insts));
        // The vertices' colour attribute, when any object has one (`Scene::paint`).
        let colors_at = if scene.colors.is_empty() { 0 } else { data.len() as u32 };
        if colors_at != 0 {
            data.extend(&scene.colors);
            data.resize(colors_at as usize + scene.pos.len(), 0xffff_ffff);
        }
        if data.len() as u64 * 4 > limit {
            return None;
        }
        let verts: Vec<[f32; 8]> = scene.pos.iter().zip(&scene.attr).map(|(p, a)| [p[0], p[1], p[2], p[3], a[0], a[1], a[2], a[3]]).collect();
        let (nodes, n_at) = GpuArray::split(gpu, &trees.nodes(&scene.insts), limit, chunks)?;
        let (faces, f_at) = GpuArray::split(gpu, &scene.faces, limit, chunks)?;
        let (verts, v_at) = GpuArray::split(gpu, &verts, limit, chunks)?;
        let n_faces = scene.faces.len() as u32;
        let pose = pose::Pose {
            pos: std::mem::take(&mut scene.pos),
            faces: std::mem::take(&mut scene.faces),
            mats: std::mem::take(&mut scene.mats),
            insts: std::mem::take(&mut scene.insts),
            order,
            trees,
            limit,
        };
        Some(Self {
            glow,
            nodes,
            faces,
            verts,
            data: GpuArray::from(gpu, &data),
            px: None,
            vis: None,
            blank: GpuArray::zeroed(gpu, 4),
            noise: GpuArray::from(gpu, &blue_noise::BlueNoise::words()),
            smaa: GpuArray::from(gpu, &smaa::SmaaTables::words()),
            class: GpuArray::zeroed(gpu, pose.insts.len().max(1)),
            classes: vec![0; pose.insts.len()],
            reshow: false,
            // Hardware rasterizes its primary hits (Workbench's prepass); a
            // software adapter traces them, where walking the tree is the
            // cheaper of the two (`visible.rs`).
            raster: !gpu.adapter.is_software(),
            n_faces,
            bounds,
            split: [n_at, f_at, v_at, atlas_at],
            insts: [insts_at, pose.insts.len() as u32, colors_at, 0],
            last: None,
            frame: 0,
            pass: Pass::default(),
            size: (0, 0),
            extent,
            mode: Mode::Rendered,
            target: Self::SAMPLES,
            aa: taa::SAMPLES,
            pose: Some(pose),
        })
    }

    /// Items per binding for nodes, faces, vertices (and the atlas offset).
    pub fn split(&self) -> [u32; 4] {
        self.split
    }

    /// Bindings per large array (2 or 4).
    pub fn chunks(&self) -> usize {
        self.nodes.len()
    }

    /// Samples averaged before a still view stops refining.
    pub const SAMPLES: u32 = 32;

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

    /// Whether Solid runs Workbench's effects pass (`fx.wgsl`) after the
    /// opaque pass: the object outline, the cavity (SSAO) or the curvature
    /// effect asked for, and X-ray off — the transparent pass has none.
    fn effects(&self, cam: &Camera) -> bool {
        self.mode == Mode::Solid && cam.see_through <= 0.0 && (cam.fx[0] > 0.0 || cam.fx[1] > 0.0 || cam.fx[2] > 0.0 || cam.ssao[0] > 0 || cam.dof[0] > 0.0)
    }

    /// The Depth of Field's stages after the composite (`dof.wgsl`,
    /// `SceneState::draw_dof`): `(stage, pixels)` for the half, quarter and
    /// eighth images, the two blurs and the full-size resolve — none when
    /// it is off, under X-ray, or the image is too small to halve thrice.
    fn dof_steps(&self, cam: &Camera, w: u32, h: u32) -> Vec<(u32, u32)> {
        if !self.effects(cam) || cam.dof[0] <= 0.0 || w < 8 || h < 8 {
            return Vec::new();
        }
        let (half, quarter, eighth) = ((w / 2) * (h / 2), (w / 4) * (h / 4), (w / 8) * (h / 8));
        vec![(1, half), (2, quarter), (3, eighth), (4, half), (5, half), (6, w * h)]
    }

    /// The dispatches after the slices, `(kernel, stage, pixels)`: the
    /// effects pass and the Depth of Field's stages (`dof_steps`), then
    /// for a jittered mode the anti-aliasing gather and SMAA's three
    /// stages (`smaa.wgsl`), which return at once from the fourth sample.
    fn tail(&self, effects: bool, dof: &[(u32, u32)], n: u32) -> Vec<(BTail, u32, u32)> {
        let mut tail = Vec::new();
        if effects {
            tail.push((BTail::Fx, 0, n));
        }
        tail.extend(dof.iter().map(|&(stage, items)| (BTail::Fx, stage, items)));
        if self.jittered() {
            tail.push((BTail::Taa, 0, n));
            tail.extend((1..=3).map(|stage| (BTail::Smaa, stage, n)));
        }
        // The overlay goes on last, over the anti-aliased image, as
        // Blender's overlay engine draws after the shading.
        if self.outlined() {
            tail.push((BTail::Edge, 0, n));
        }
        tail
    }

    /// Refine a still view to `n` samples (Blender's viewport / render
    /// sample count). Takes effect from the next view.
    pub fn set_samples(&mut self, n: u32) {
        self.target = n.max(1);
    }

    /// Render `w × h` RGBA8 pixels (row 0 at the top). A repeated camera adds a
    /// sample to the running average; the flag is `true` once refinement is done.
    pub fn render(&mut self, gpu: &Gpu, cam: Camera, w: u32, h: u32) -> (Vec<u8>, bool) {
        let (_, done) = self.sample(gpu, cam, w, h);
        (self.pixels(gpu), done)
    }

    /// Trace one sample without reading it back: `(sample index, done)`.
    pub fn sample(&mut self, gpu: &Gpu, cam: Camera, w: u32, h: u32) -> (u32, bool) {
        self.sample_in(gpu, cam, w, h, 1, || true).expect("one slice is never interrupted")
    }

    /// [`Self::sample`] in `slices` dispatches, the device drained after
    /// each and `go` asked whether to carry on: `None` when it said no —
    /// the sample is abandoned, and the next one starts the image over,
    /// since a frame half of whose pixels have one sample more is no
    /// frame. A sample of a heavy scene at full size can take a second; a
    /// view posted meanwhile used to wait it out.
    pub fn sample_in(&mut self, gpu: &Gpu, mut cam: Camera, w: u32, h: u32, slices: u32, mut go: impl FnMut() -> bool) -> Option<(u32, bool)> {
        let (w, h) = (w.max(1), h.max(1));
        let n = w * h;
        let slices = slices.clamp(1, n);
        cam.w = w;
        cam.h = h;
        if cam.ao <= 0.0 {
            cam.ao = self.extent * 0.08;
        }
        cam.eps = self.extent * 1e-5;
        cam.glow = self.glow;
        cam.split = self.split;
        cam.insts = self.insts;
        cam.frame = 0;
        cam.slice = [0; 4];
        // The anti-aliasing count is part of the view: a new count starts
        // the samples over (`SceneState`: `reset_taa` when `samples_len`
        // changes).
        cam.taa = [0.0, 0.0, 0.0, if self.jittered() { self.aa_samples() as f32 } else { 0.0 }];
        let drawn = self.rasterizes(&cam);
        cam.vis = self.vis_words(&cam, drawn, w);
        let same = self.last.is_some_and(|l| bytemuck::bytes_of(&l)[8..] == bytemuck::bytes_of(&cam)[8..]) && self.size == (w, h);
        if same && !self.reshow && self.frame + 1 >= self.samples() {
            return Some((self.frame, true));
        }
        // The outline changed: this one more sample lays it on.
        self.reshow = false;
        self.frame = if same { self.frame + 1 } else { 0 };
        self.last = Some(cam);
        cam.frame = self.frame;
        if self.jittered() {
            let aa = self.aa_samples();
            let [jx, jy] = taa::offset(self.frame, aa);
            cam.taa = [jx, jy, taa::accumulated_weight(self.frame, aa), aa as f32];
        }
        // One slice's pixels: `[first, count]` of slice `k`.
        let part = |k: u32| {
            let first = (n as u64 * k as u64 / slices as u64) as u32;
            let last = (n as u64 * (k as u64 + 1) / slices as u64) as u32;
            [first, last - first, 0, 0]
        };
        let effects = self.effects(&cam);
        let dof = self.dof_steps(&cam, w, h);
        let tail = self.tail(effects, &dof, n);
        let steps = slices as usize + tail.len();
        if self.size != (w, h) || self.pass.is_empty() || self.pass.len() != steps || drawn != self.vis.is_some() {
            // The image a new tree (`retree`) records the pass again over
            // keeps its samples; a new size or kernel starts from zero —
            // a new buffer, which the device clears, nothing uploaded.
            let px = match self.px.take() {
                Some(px) if self.size == (w, h) => px,
                _ => GpuArray::zeroed(gpu, Self::PX_WORDS * n as usize),
            };
            self.pass.clear();
            self.vis = drawn.then(|| self.visibility(gpu, &cam, w, h));
            let arrays = self.nodes.iter().map(|a| a.buffer());
            let arrays = arrays.chain(self.faces.iter().map(|a| a.buffer())).chain(self.verts.iter().map(|a| a.buffer()));
            let bufs: Vec<&wgpu::Buffer> = arrays.chain([self.data.buffer(), px.buffer()]).collect();
            // Solid and Wire bind the prepass's hits after the image.
            let hits = self.vis.as_ref().map_or(self.blank.buffer(), |v| v.hits().buffer());
            let seen: Vec<&wgpu::Buffer> = bufs.iter().copied().chain([hits]).collect();
            for k in 0..slices {
                let cam = Camera { slice: part(k), ..cam };
                let items = cam.slice[1];
                match (self.mode, self.chunks()) {
                    (Mode::Wire, 4) => self.pass.add_over::<Wire4>(gpu, n, items, cam, &seen),
                    (Mode::Wire, _) => self.pass.add_over::<Wire>(gpu, n, items, cam, &seen),
                    (Mode::Solid, 4) => self.pass.add_over::<Solid4>(gpu, n, items, cam, &seen),
                    (Mode::Solid, _) => self.pass.add_over::<Solid>(gpu, n, items, cam, &seen),
                    (Mode::Preview, 4) => self.pass.add_over::<Preview4>(gpu, n, items, cam, &bufs),
                    (Mode::Preview, _) => self.pass.add_over::<Preview>(gpu, n, items, cam, &bufs),
                    (_, 4) => self.pass.add_over::<Trace4>(gpu, n, items, cam, &bufs),
                    _ => self.pass.add_over::<Trace>(gpu, n, items, cam, &bufs),
                };
            }
            // The effects, then the anti-aliasing gather and SMAA, over the
            // whole image after the slices.
            for &(step, stage, items) in &tail {
                let cam = Camera { slice: [0, 0, stage, 0], ..cam };
                match step {
                    BTail::Fx => self.pass.add_over::<Fx>(gpu, n, items, cam, &[px.buffer(), self.noise.buffer()]),
                    BTail::Taa => self.pass.add::<Taa>(gpu, n, cam, &[px.buffer()]),
                    BTail::Smaa => self.pass.add::<Smaa>(gpu, n, cam, &[px.buffer(), self.smaa.buffer()]),
                    BTail::Edge => self.pass.add::<Edge>(gpu, n, cam, &[px.buffer(), self.class.buffer()]),
                };
            }
            self.px = Some(px);
            self.size = (w, h);
        } else {
            for k in 0..slices {
                let cam = Camera { n, groups_x: Dispatch::new(part(k)[1]).x, slice: part(k), ..cam };
                self.pass.set(gpu, k as usize, 0, &cam);
            }
            for (k, &(_, stage, items)) in tail.iter().enumerate() {
                // A Depth of Field stage runs over its own image size.
                self.pass.set(gpu, slices as usize + k, 0, &Camera { n, groups_x: Dispatch::new(items).x, slice: [0, 0, stage, 0], ..cam });
            }
        }
        if let Some(vis) = &self.vis {
            vis.draw(gpu, &cam, self.n_faces);
        }
        if slices == 1 {
            self.pass.run(gpu);
            return Some((self.frame, self.frame + 1 >= self.samples()));
        }
        for k in 0..slices {
            self.pass.run_step(gpu, k as usize);
            gpu.wait();
            if k + 1 < slices && !go() {
                // Abandoned: the next sample starts the image over.
                self.last = None;
                return None;
            }
        }
        for k in slices as usize..steps {
            self.pass.run_step(gpu, k);
        }
        Some((self.frame, self.frame + 1 >= self.samples()))
    }

    /// Words per pixel in the image buffer: the packed display pixel, then
    /// the scene-linear sums (n..4n), the shown value after the view transform
    /// (4n..7n), the denoiser's albedo / normal / depth guides (7n..14n), two
    /// scratch images the a-trous filter ping-pongs between (14n..20n), and
    /// the grid overlay's premultiplied colour and coverage sums (20n..24n),
    /// and the film coverage sum — the pixel's alpha under a transparent
    /// film (24n..25n), and the selection outline's prepass — the instance
    /// under the pixel plus one (25n..26n, `outline_store`). See
    /// `util.wgsl`'s `emit` and `guide`, and `denoise.wgsl`.
    pub(crate) const PX_WORDS: usize = 26;

    /// The most pixels one image may hold on `gpu`. The image is one
    /// binding of `PX_WORDS` words a pixel, and a device caps a binding —
    /// at 128 MiB on many, about 1.4 million pixels — past which creating
    /// the pass is a validation error that ends the render thread.
    pub fn max_pixels(gpu: &Gpu) -> u64 {
        gpu.max_binding() / (Self::PX_WORDS as u64 * 4)
    }

    /// Denoise what has been traced so far, replacing the displayed image.
    ///
    /// Five a-trous passes at widening tap spacing (`kernels/denoise.rs`),
    /// guided by the albedo, normal and depth the tracer recorded for each
    /// pixel's first hit. Call it once a render has settled; `pixels` then
    /// returns the filtered image. Tracing another sample overwrites it, so
    /// denoise again after.
    pub fn denoise(&mut self, gpu: &Gpu) {
        self.denoise_with(gpu, false);
    }

    /// Denoise what is shown, leaving the samples as they are: for a
    /// viewport that goes on refining under the filter (Blender's Viewport
    /// Denoising), where filtering the sums themselves would blur every
    /// sample into the next.
    pub fn denoise_shown(&mut self, gpu: &Gpu) {
        self.denoise_with(gpu, true);
    }

    fn denoise_with(&mut self, gpu: &Gpu, keep: bool) {
        use crate::kernels::{Denoise, DenoiseParams};
        let (w, h) = self.size;
        let n = w * h;
        let (Some(px), Some(cam)) = (self.px.as_ref(), self.last) else { return };
        // A deterministic frame has no noise to take out, and wrote no
        // guides for the filter to follow.
        if n == 0 || self.mode.deterministic() {
            return;
        }
        let d = Dispatch::new(n);
        let base = DenoiseParams {
            n,
            groups_x: d.x,
            w,
            h,
            // Tuned on Female-01: a normal has to agree closely, an albedo
            // less so (a noisy albedo is still a texture), and depth is
            // relative so the same figure works at any scale.
            strength: [64.0, 8.0, 40.0, 1.0 / (self.frame + 1) as f32],
            view: cam.view,
            keep: u32::from(keep),
            ..Default::default()
        };
        let mut pass = Pass::default();
        for (i, step) in [1u32, 2, 4, 8, 16].into_iter().enumerate() {
            let params = DenoiseParams { step, round: i as u32, last: u32::from(step == 16), ..base };
            pass.add::<Denoise>(gpu, n, params, &[px.buffer()]);
        }
        pass.run(gpu);
    }

    /// The image's size, `(w, h)` — what the last view was rendered at,
    /// and so the shape every readback comes back in.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Show `linear` as this image: the scene-linear radiance a host has
    /// *changed* — composited (`phiphx::comp`), graded, anything — put
    /// back in place of the traced mean and run through the display step,
    /// so [`Self::pixels`], [`Self::display`] and [`Self::radiance`] all
    /// come back agreeing with it.
    ///
    /// Blender composites the render's scene-linear pixels and applies the
    /// view transform **after** (`RE_RenderFrame`, then the display
    /// pipeline). A host that composited the *displayed* pixels instead
    /// would be grading after AgX rather than before it — wrong, and wrong
    /// in a way that looks plausible — so the way back in is here rather
    /// than a second CPU implementation of the transform.
    ///
    /// `linear` is one triple a pixel, row 0 first, and is truncated or
    /// zero-filled to the image's own size. `false` when there is no image
    /// yet, no camera to transform with, or the device refuses the write.
    pub fn show(&mut self, gpu: &Gpu, linear: &[[f32; 3]]) -> bool {
        let (w, h) = self.size;
        let n = (w as usize) * (h as usize);
        let (Some(px), Some(cam)) = (self.px.as_mut(), self.last) else { return false };
        if n == 0 {
            return false;
        }
        // The sums are `k` times the mean, so what goes in is scaled by the
        // sample count the buffer is at — otherwise showing a composited
        // image would darken it by `1 / k`.
        let k = (self.frame + 1) as f32;
        let mut words = Vec::with_capacity(3 * n);
        for i in 0..n {
            let c = linear.get(i).copied().unwrap_or([0.0; 3]);
            words.extend(c.iter().map(|v| (v * k).to_bits()));
        }
        if !px.write_at(gpu, n, &words) {
            return false;
        }
        let d = Dispatch::new(n as u32);
        // `self.last` keeps the camera as it was *compared* — frame 0, since
        // the frame is not part of what makes two views the same — so the
        // sample count has to be put back by hand. Taking `last.frame` gave
        // `k = 1` in the kernel and an image 32 times too bright, which the
        // round-trip test caught at once.
        let cam = Camera { n: n as u32, groups_x: d.x, slice: [0; 4], frame: self.frame, ..cam };
        let mut pass = Pass::default();
        pass.add::<Show>(gpu, n as u32, cam, &[px.buffer()]);
        pass.run(gpu);
        true
    }

    /// Linear radiance per pixel (mean of the samples so far), before tone mapping.
    pub fn radiance(&self, gpu: &Gpu) -> Vec<[f32; 3]> {
        let n = (self.size.0 * self.size.1) as usize;
        let words = self.px.as_ref().map(|o| o.read_first(gpu, 4 * n)).unwrap_or_default();
        // Words n..4n: the scene-linear sums (4n..7n hold the display sums).
        let k = (self.frame + 1) as f32;
        words[n.min(words.len())..].chunks_exact(3).map(|c| [0, 1, 2].map(|i| f32::from_bits(c[i]) / k)).collect()
    }

    /// The display-space image (the view transform applied, mean of the
    /// samples so far) as floats, before it is quantised to 8 bits. A 16-bit
    /// save reads this rather than `pixels`, so that the extra bits carry
    /// real precision rather than a stretched byte.
    pub fn display(&self, gpu: &Gpu) -> Vec<[f32; 3]> {
        let n = (self.size.0 * self.size.1) as usize;
        let words = self.px.as_ref().map(|o| o.read_first(gpu, 7 * n)).unwrap_or_default();
        // Words 4n..7n hold the shown value — the view transform of the
        // linear *mean*, grid composited — not a sum, so no division here.
        words[(4 * n).min(words.len())..].chunks_exact(3).take(n).map(|c| [0, 1, 2].map(|i| f32::from_bits(c[i]))).collect()
    }

    /// The diffuse colour under each pixel, scene-linear — Blender's
    /// "Diffuse Color" pass, and the same buffer the denoiser uses as its
    /// albedo guide. Raw, not gamma-encoded: the denoiser applies a transfer
    /// curve of its own before comparing neighbours, and a pass must not.
    pub fn albedo(&self, gpu: &Gpu) -> Vec<[f32; 3]> {
        self.guide3(gpu, 7)
    }

    /// The world-space shading normal at each first hit — Blender's "Normal"
    /// pass. Re-normalised, because what is stored is a *sum* over samples
    /// and averaging unit vectors does not give a unit vector.
    pub fn normal(&self, gpu: &Gpu) -> Vec<[f32; 3]> {
        self.guide3(gpu, 10)
            .into_iter()
            .map(|n| {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 1e-9 { [n[0] / len, n[1] / len, n[2] / len] } else { [0.0; 3] }
            })
            .collect()
    }

    /// Distance from the camera to the first hit — Blender's "Depth" pass.
    ///
    /// A ray that met nothing reads `1e10`, which is what Cycles writes
    /// (`film_get_pass_pixel_depth`, `kernel/film/read.h`) rather than zero
    /// or infinity: compositors treat it as "far away", and zero would read
    /// as "touching the lens".
    pub fn depth(&self, gpu: &Gpu) -> Vec<f32> {
        let n = (self.size.0 * self.size.1) as usize;
        let words = self.px.as_ref().map(|o| o.read_first(gpu, 14 * n)).unwrap_or_default();
        let k = (self.frame + 1) as f32;
        words[(13 * n).min(words.len())..]
            .iter()
            .take(n)
            .map(|w| {
                // Misses record -1 a sample, so a pixel that never hit
                // anything sums to -k.
                let mean = f32::from_bits(*w) / k;
                if mean <= 0.0 { 1.0e10 } else { mean }
            })
            .collect()
    }

    /// One three-channel guide buffer, averaged over the samples taken.
    fn guide3(&self, gpu: &Gpu, start: usize) -> Vec<[f32; 3]> {
        let n = (self.size.0 * self.size.1) as usize;
        let words = self.px.as_ref().map(|o| o.read_first(gpu, (start + 3) * n)).unwrap_or_default();
        let k = (self.frame + 1) as f32;
        words[(start * n).min(words.len())..].chunks_exact(3).take(n).map(|c| [0, 1, 2].map(|i| f32::from_bits(c[i]) / k)).collect()
    }

    /// The current image (RGBA8).
    pub fn pixels(&self, gpu: &Gpu) -> Vec<u8> {
        let n = (self.size.0 * self.size.1) as usize;
        let px = self.px.as_ref().map(|o| o.read_first(gpu, n)).unwrap_or_default();
        bytemuck::cast_slice(&px).to_vec()
    }
}
