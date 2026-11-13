//! What a tracer gives back: its memory, the image and its planes read off
//! the device, and the passes that work over a finished image (denoise,
//! glow, show).
use super::{Camera, Glow, Show, Tracer};
use crate::bvh::Node;
use crate::gpu::{Gpu, GpuArray, Pass};
use crate::kernel::Dispatch;

impl Tracer {
    /// Every byte this tracer holds on the device: the scene (tree,
    /// faces, vertices, the data table with the texture atlas and the
    /// instance records) and the image buffer, plus the small fixed
    /// tables. This is the figure the render window's `Mem:` should show.
    ///
    /// Blender's `Mem:` is the memory the *renderer* is holding, which on
    /// a scene like this is overwhelmingly geometry and textures on the
    /// card. Reporting only the host-side preview buffers read `8M` on a
    /// 2.4-million-triangle scene, which told the user nothing.
    pub fn bytes(&self) -> u64 {
        let of = |n: usize, each: usize| (n * each) as u64;
        let mut total = 0u64;
        for a in &self.nodes {
            total += of(a.len(), std::mem::size_of::<Node>());
        }
        for a in &self.faces {
            total += of(a.len(), std::mem::size_of::<[u32; 4]>());
        }
        for a in &self.verts {
            total += of(a.len(), std::mem::size_of::<[f32; 8]>());
        }
        total += of(self.data.len(), 4);
        total += self.px.as_ref().map_or(0, |px| of(px.len(), 4));
        total += of(self.blank.len() + self.noise.len() + self.smaa.len() + self.class.len(), 4);
        total
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
    /// 26n..29n is Adaptive Sampling's half image, 29n..30n the raw
    /// converged flag, 30n..31n the flag after the dilation — the one the
    /// tracer obeys — and 31n..32n the samples THIS pixel has taken
    /// (Cycles' `pass_sample_count`, `util.wgsl`'s `adapt_count`).
    ///
    /// 32n..33n is the list of pixels the next sample traces when Adaptive
    /// Sampling is on, and 33n..34n how many of a pixel's samples met
    /// nothing (`emit`'s `missed`), which the light is not averaged over.
    /// The one word after the image is the list's length (`gather.wgsl`) —
    /// which is why the buffer is `PX_LEN · n + 1`.
    pub(crate) const PX_LEN: usize = 34;

    /// The most pixels one image may hold on `gpu`. The image is one
    /// binding of `PX_LEN` words a pixel, and a device caps a binding —
    /// at 128 MiB on many, about 1.4 million pixels — past which creating
    /// the pass is a validation error that ends the render thread.
    pub fn max_pixels(gpu: &Gpu) -> u64 {
        gpu.max_binding().saturating_sub(4) / (Self::PX_LEN as u64 * 4)
    }

    /// Denoise what has been traced so far, replacing the displayed image.
    ///
    /// Five a-trous passes at widening tap spacing (`kernel/denoise.rs`),
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
        use crate::kernel::{Denoise, DenoiseParam};
        let (w, h) = self.size;
        let n = w * h;
        let (Some(px), Some(cam)) = (self.px.as_ref(), self.last) else { return };
        // A deterministic frame has no noise to take out, and wrote no
        // guides for the filter to follow.
        if n == 0 || self.mode.deterministic() {
            return;
        }
        let d = Dispatch::new(n);
        let base = DenoiseParam {
            n,
            groups_x: d.x,
            w,
            h,
            // Tuned on Female-01: a normal has to agree closely, an albedo
            // less so (a noisy albedo is still a texture), and depth is
            // relative so the same figure works at any scale. The last is the
            // stop on the light itself (`denoise.wgsl`): noise still averages
            // out, a real change of brightness stays.
            strength: [64.0, 8.0, 40.0, 7.0],
            view: cam.view,
            keep: u32::from(keep),
            ..Default::default()
        };
        let mut pass = Pass::default();
        // Four passes, a reach of 30 pixels: a fifth reached 62 and blurred across
        // whole walls for noise the first four had already taken out.
        for (i, step) in [1u32, 2, 4, 8].into_iter().enumerate() {
            let params = DenoiseParam { step, round: i as u32, last: u32::from(step == 8), ..base };
            pass.add::<Denoise>(gpu, n, params, &[px.buffer()]);
        }
        pass.run(gpu);
    }

    /// Bloom / Glare over the settled frame (`Glow`, `glow.wgsl`): the
    /// bright parts of the scene-linear image bleed into their
    /// surroundings, as Blender's Eevee Bloom and the compositor's Fog-Glow
    /// Glare do. Run after the samples are in and before a readback, like
    /// [`Self::denoise`] — it rewrites the display pixels in place, so
    /// [`Self::pixels`] and [`Self::display`] come back with the glare.
    ///
    /// `threshold` is the scene-linear luma a pixel must exceed to glow,
    /// `intensity` how strongly the blurred glare is added back, and
    /// `radius` the Gaussian step in texels (wider spreads it further).
    /// A `threshold` at or above the brightest pixel, or `intensity` 0, is
    /// a no-op; the caller decides from the panel whether to run it at all.
    pub fn glow(&mut self, gpu: &Gpu, threshold: f32, intensity: f32, radius: f32) {
        let (w, h) = self.size;
        let n = (w as usize) * (h as usize);
        let (Some(px), Some(cam)) = (self.px.as_ref(), self.last) else { return };
        if n == 0 || intensity <= 0.0 {
            return;
        }
        // The glare scratch: two rgb buffers a pixel (A at 0, B at 3n),
        // ping-ponged by the separable blur. Its own binding, so the
        // packed image buffer (`PX_LEN`) is untouched.
        let scratch = GpuArray::<u32>::zeroed(gpu, 2 * 3 * n);
        let d = Dispatch::new(n as u32);
        let mut pass = Pass::default();
        for stage in 0u32..4 {
            let cam = Camera {
                n: n as u32,
                groups_x: d.x,
                frame: self.frame,
                fx: [threshold, intensity, radius.max(1.0), 1.0],
                slice: [stage, 0, 0, 0],
                ..cam
            };
            pass.add::<Glow>(gpu, n as u32, cam, &[px.buffer(), scratch.buffer()]);
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
        if n == 0 {
            return false;
        }
        // The sums are `k` times the mean, so what goes in is scaled by the
        // sample count the buffer is at — otherwise showing a composited
        // image would darken it by `1 / k`.
        //
        // Per pixel, for the same reason the readbacks are: with Adaptive
        // Sampling two pixels have taken different numbers of samples, so
        // one global `k` would store the wrong sum for most of them. Read
        // BEFORE the image is borrowed mutably below — `counts` reads it.
        let k = self.counts(gpu, n);
        let (Some(px), Some(cam)) = (self.px.as_mut(), self.last) else { return false };
        let mut words = Vec::with_capacity(3 * n);
        for i in 0..n {
            let c = linear.get(i).copied().unwrap_or([0.0; 3]);
            let ki = k.get(i).copied().unwrap_or(1.0);
            words.extend(c.iter().map(|v| (v * ki).to_bits()));
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

    /// Samples each pixel actually took (`util.wgsl`'s `adapt_count`).
    ///
    /// Without Adaptive Sampling every pixel took `frame + 1` of them and
    /// this is uniform; with it, a settled pixel stopped early and the
    /// mean must be taken over ITS count. Dividing by the frame number
    /// instead darkened settled pixels by the fraction of the render they
    /// sat out — the quality regression of 2026-10-19.
    fn counts(&self, gpu: &Gpu, n: usize) -> Vec<f32> {
        let fallback = || vec![(self.frame + 1) as f32; n];
        if self.adapt[3] == 0.0 {
            return fallback();
        }
        let Some(px) = self.px.as_ref() else { return fallback() };
        let raw = px.read_range(gpu, 31 * n, n);
        if raw.len() != n {
            return fallback();
        }
        raw.iter().map(|c| (*c).max(1) as f32).collect()
    }

    /// Linear radiance per pixel (mean of the samples so far), before tone mapping.
    pub fn radiance(&self, gpu: &Gpu) -> Vec<[f32; 3]> {
        let n = (self.size.0 * self.size.1) as usize;
        let words = self.px.as_ref().map(|o| o.read_first(gpu, 4 * n)).unwrap_or_default();
        // Words n..4n: the scene-linear sums (4n..7n hold the display sums).
        let k = self.counts(gpu, n);
        words[n.min(words.len())..]
            .chunks_exact(3)
            .enumerate()
            .map(|(p, c)| [0, 1, 2].map(|i| f32::from_bits(c[i]) / k.get(p).copied().unwrap_or(1.0)))
            .collect()
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
        // The final read, which a peek never takes: where the counters,
        // when built in, go to the render log (`tally.rs`).
        if let Some(t) = self.tally(gpu) {
            print!("{}", t.report());
        }
        let n = (self.size.0 * self.size.1) as usize;
        let words = self.px.as_ref().map(|o| o.read_first(gpu, 14 * n)).unwrap_or_default();
        let k = self.counts(gpu, n);
        words[(13 * n).min(words.len())..]
            .iter()
            .take(n)
            .enumerate()
            .map(|(p, w)| {
                // Misses record -1 a sample, so a pixel that never hit
                // anything sums to -k.
                let mean = f32::from_bits(*w) / k.get(p).copied().unwrap_or(1.0);
                if mean <= 0.0 { 1.0e10 } else { mean }
            })
            .collect()
    }

    /// One three-channel guide buffer, averaged over the samples taken.
    fn guide3(&self, gpu: &Gpu, start: usize) -> Vec<[f32; 3]> {
        let n = (self.size.0 * self.size.1) as usize;
        let words = self.px.as_ref().map(|o| o.read_first(gpu, (start + 3) * n)).unwrap_or_default();
        let k = self.counts(gpu, n);
        words[(start * n).min(words.len())..]
            .chunks_exact(3)
            .take(n)
            .enumerate()
            .map(|(p, c)| [0, 1, 2].map(|i| f32::from_bits(c[i]) / k.get(p).copied().unwrap_or(1.0)))
            .collect()
    }

    /// The current image (RGBA8).
    pub fn pixels(&self, gpu: &Gpu) -> Vec<u8> {
        let n = (self.size.0 * self.size.1) as usize;
        let px = self.px.as_ref().map(|o| o.read_first(gpu, n)).unwrap_or_default();
        bytemuck::cast_slice(&px).to_vec()
    }
}
