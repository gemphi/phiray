//! One sample: the camera stamped, the pass recorded or its params set,
//! the slices and the tail run and timed (`gpu/clock.rs`).
use super::{Adapt, AdaptClear, AdaptDilate, AdaptGather, Camera, Edge, Fx, Preview, Preview4, Smaa, Solid, Solid4, Taa, Trace, Trace4, Tracer, Wire, Wire4};
use super::taa;
use crate::gpu::{Clock, Gpu, GpuArray, Time};
use crate::kernel::Dispatch;
use crate::mode::Mode;

/// A dispatch after the slices (`Tracer::tail`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BTail {
    Fx,
    Taa,
    Smaa,
    Edge,
    /// Adaptive Sampling: stage 0 tests convergence, stage 1 widens the
    /// unconverged set by a pixel, stages 2 and 3 pack the pixels left into
    /// the list the next sample traces (`gather.wgsl`). Recorded on every
    /// sample; the test returns at once unless the sample is a step
    /// boundary, so the pass never has to be re-recorded for it.
    Adapt,
}

impl Tracer {
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
        if self.adapt[3] != 0.0 {
            tail.extend((0..4).map(|stage| (BTail::Adapt, stage, n)));
        }
        tail
    }

    /// Refine a still view to `n` samples (Blender's viewport / render
    /// sample count). Takes effect from the next view.
    /// Adaptive Sampling, already resolved against Cycles' automatic
    /// rules (`BAdaptive::of` in phiber): `[threshold, min samples, step,
    /// on]`. Changing it changes whether the tail carries the convergence
    /// check, so the pass is recorded again on the next sample.
    /// Pixels Adaptive Sampling has NOT settled, or `None` when it is off
    /// or the device declined the readback.
    ///
    /// Cycles stops a render when every pixel has converged rather than
    /// grinding to the sample count, and this is what lets phiber do the
    /// same. Without it a frame whose floor and background settled at
    /// sample 80 still ran to 2048 — the samples cost almost nothing by
    /// then, but the render did not *end*, which is what a person
    /// actually waits for.
    ///

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
        cam.adapt = self.adapt;
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
                _ => GpuArray::zeroed(gpu, Self::px_len(n as usize)),
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
                    BTail::Adapt => match stage {
                        0 => self.pass.add::<Adapt>(gpu, n, cam, &[px.buffer()]),
                        1 => self.pass.add::<AdaptDilate>(gpu, n, cam, &[px.buffer()]),
                        2 => self.pass.add::<AdaptClear>(gpu, n, cam, &[px.buffer()]),
                        _ => self.pass.add::<AdaptGather>(gpu, n, cam, &[px.buffer()]),
                    },
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
            vis.draw(gpu, &cam, &self.draws);
        }
        // Every sample is timed and waited for: a single slice used to
        // return once submitted, and its caller booked the GPU's work to
        // whatever waited next (the readback). `Gpu::waited` must not set
        // the slice count, nor `wait_for` bound a slice (both tried).
        let clock = self.clock.as_ref();
        if slices == 1 {
            self.time = Clock::run(clock, gpu, &self.pass, 0..steps);
            return Some((self.frame, self.frame + 1 >= self.samples()));
        }
        self.time = Time::default();
        for k in 0..slices as usize {
            // NOTE (unresolved): this drain is `device.poll(wait_indefinitely())`
            // and its own doc says it never returns if the device is lost.
            // On a 0.1 MB scene a 64-sample render was measured at 1830 s of
            // wall against 45 s of CPU — 2.5% busy, nothing written, no
            // error — which is what an indefinite poll on a stalled device
            // looks like.
            //
            // Replacing it with the bounded `Gpu::wait_for` was tried and
            // REVERTED: `wait_for` spawns a thread per call, and per slice
            // that is thread churn (the same render then ran at 40% CPU and
            // still did not finish), while its budget here is the render's
            // ten-minute `PatienceFor` — far too long to bound one dispatch.
            // A fix needs a bounded poll that allocates nothing, and a
            // per-dispatch budget of its own; neither exists yet.
            // The clock's own drain is the one this slice always had.
            self.time += Clock::run(clock, gpu, &self.pass, k..k + 1);
            if k + 1 < slices as usize && !go() {
                // Abandoned: the next sample starts the image over.
                self.last = None;
                return None;
            }
        }
        if steps > slices as usize {
            self.time += Clock::run(clock, gpu, &self.pass, slices as usize..steps);
        }
        Some((self.frame, self.frame + 1 >= self.samples()))
    }
    /// What the last finished sample cost: the time to record and submit
    /// it, and the GPU's time to run it. A host sizes its slices by
    /// [`Time::gpu`]; a readback after the sample is only the copy.
    pub fn time(&self) -> Time {
        self.time
    }
}
