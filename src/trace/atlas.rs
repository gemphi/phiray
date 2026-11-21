//! Every texture in one buffer, and the environment's sampling table.

/// Pack linear RGB into Radiance RGBE: three 8-bit mantissas and one shared
/// exponent, biased by 128 (`IMB_radiance_hdr.cc`).
fn rgbe(r: f32, g: f32, b: f32) -> u32 {
    let peak = r.max(g).max(b);
    if !(peak > 1e-32) {
        return 0;
    }
    // `peak / 2^exp` lands in (0.5, 1], so every mantissa fits in one byte.
    let exp = peak.log2().ceil().clamp(-127.0, 127.0);
    let scale = 256.0 / exp.exp2();
    let q = |c: f32| (c * scale).clamp(0.0, 255.0) as u32;
    q(r) | (q(g) << 8) | (q(b) << 16) | (((exp + 128.0) as u32).min(255) << 24)
}

/// All textures in one `u32` buffer: `[count, (offset, w, h) × count, texels…]`,
/// texels packed RGBA8 (sRGB), row 0 at the top.
///
/// Each entry's head is `[first texel, width, height, levels]`: an image
/// added with [`Atlas::add`] is followed by its mip chain, one entry per
/// level, and `levels` counts them (itself included) so a kernel can sample
/// the level a pixel's footprint asks for (`sample_raw_lod`); every other
/// entry — raw words, an environment, a prefiltered copy — has one. On the
/// GPU the head is three words, the level count in the top byte of the width.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Atlas {
    heads: Vec<[u32; 4]>,
    texels: Vec<u32>,
}

/// Grid the environment's sampling table is built on. Coarse on purpose: the
/// table only has to steer rays towards the bright parts, and the estimator
/// divides by the same table's own density, so a coarse grid costs variance
/// and never accuracy.
pub const ENV_CDF_W: u32 = 128;
pub const ENV_CDF_H: u32 = 64;

impl Atlas {
    /// Add the sampling distribution of an equirectangular environment image:
    /// one conditional CDF over columns per row, then a final row holding the
    /// marginal CDF over rows. Cell weights are luminance × sin θ, so a
    /// direction is drawn in proportion to the light it actually carries.
    /// Returns an id like any other entry; read it with `bitcast<f32>`.
    pub fn add_env_cdf(&mut self, w: u32, h: u32, rgb: &[f32]) -> u32 {
        let (cw, ch) = (ENV_CDF_W.min(w.max(1)), ENV_CDF_H.min(h.max(1)));
        let lum = |x: u32, y: u32| {
            let i = ((y.min(h - 1) * w + x.min(w - 1)) * 3) as usize;
            match rgb.get(i..i + 3) {
                Some(c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2],
                None => 0.0,
            }
        };
        // Cell weights: the mean of the source pixels it covers, times the
        // solid angle its latitude band subtends.
        let mut cells = vec![0.0f32; (cw * ch) as usize];
        for gy in 0..ch {
            let sin = (std::f32::consts::PI * (gy as f32 + 0.5) / ch as f32).sin();
            for gx in 0..cw {
                let (x0, x1) = (gx * w / cw, ((gx + 1) * w / cw).max(gx * w / cw + 1));
                let (y0, y1) = (gy * h / ch, ((gy + 1) * h / ch).max(gy * h / ch + 1));
                let mut sum = 0.0f32;
                let mut n = 0u32;
                for y in y0..y1.min(h) {
                    for x in x0..x1.min(w) {
                        sum += lum(x, y);
                        n += 1;
                    }
                }
                cells[(gy * cw + gx) as usize] = sum / n.max(1) as f32 * sin;
            }
        }
        let mut words = vec![0u32; ((ch + 1) * cw) as usize];
        let mut rows = vec![0.0f32; ch as usize];
        for gy in 0..ch {
            let row = &cells[(gy * cw) as usize..((gy + 1) * cw) as usize];
            let total: f32 = row.iter().sum();
            rows[gy as usize] = total;
            let mut acc = 0.0f32;
            for gx in 0..cw {
                // A dark row still needs a usable CDF; its marginal weight is
                // zero, so it is never drawn from.
                acc += if total > 0.0 { row[gx as usize] / total } else { 1.0 / cw as f32 };
                words[(gy * cw + gx) as usize] = acc.min(1.0).to_bits();
            }
            words[(gy * cw + cw - 1) as usize] = 1.0f32.to_bits();
        }
        let total: f32 = rows.iter().sum();
        let mut acc = 0.0f32;
        for gy in 0..ch {
            acc += if total > 0.0 { rows[gy as usize] / total } else { 1.0 / ch as f32 };
            words[(ch * cw + gy) as usize] = acc.min(1.0).to_bits();
        }
        for gx in ch..cw {
            words[(ch * cw + gx) as usize] = 1.0f32.to_bits();
        }
        self.add_bits(cw, ch + 1, &words)
    }

    /// Add raw words as an entry, addressed like an image but read as data.
    pub fn add_bits(&mut self, w: u32, h: u32, words: &[u32]) -> u32 {
        let id = self.heads.len() as u32;
        self.heads.push([self.texels.len() as u32, w, h, 1]);
        self.texels.extend_from_slice(words);
        self.texels.resize(self.heads.last().expect("just pushed")[0] as usize + (w * h) as usize, 0);
        id
    }

    /// Add what Material Preview needs to light a surface from the
    /// environment `rgb` (`w × h`, linear) in one dispatch: its irradiance
    /// coefficients and its prefiltered copies (`ibl.rs`). The image itself
    /// is expected to be in the atlas already, as roughness 0.
    pub fn add_env_lighting(&mut self, w: u32, h: u32, rgb: &[f32]) -> super::ibl::EnvLighting {
        use super::ibl::{Ibl, LEVEL_COUNT, Panorama};
        let env = Panorama::new(w, h, rgb);
        let coefficients = Ibl::irradiance(&env);
        let words: Vec<u32> = coefficients.iter().flatten().map(|v| v.to_bits()).collect();
        let irradiance = self.add_bits(27, 1, &words);
        let chain = Ibl::chain(&env);
        let mut prefiltered = 0;
        for level in 1..LEVEL_COUNT {
            let copy = Ibl::prefiltered(&chain, level as f32 / (LEVEL_COUNT - 1) as f32);
            let id = self.add_hdr(copy.w, copy.h, &copy.rgb);
            if level == 1 {
                prefiltered = id;
            }
        }
        super::ibl::EnvLighting { irradiance, prefiltered, levels: LEVEL_COUNT }
    }

    /// Add a linear float RGB image as RGBE (one shared exponent per texel),
    /// which is how a 4-byte texel can still hold a sun a thousand times
    /// brighter than white. Sampled by `sample_env` in the kernel.
    pub fn add_hdr(&mut self, w: u32, h: u32, rgb: &[f32]) -> u32 {
        let mut texels = Vec::with_capacity((w * h) as usize);
        for px in rgb.chunks_exact(3) {
            texels.push(rgbe(px[0], px[1], px[2]));
        }
        texels.resize((w * h) as usize, 0);
        let id = self.heads.len() as u32;
        self.heads.push([self.texels.len() as u32, w, h, 1]);
        self.texels.extend(texels);
        id
    }

    /// Add an image, box-downsampled by a whole factor until neither side
    /// exceeds `max_side`. Blender's Preferences → Viewport → Textures →
    /// *Limit Size* (`U.glreslimit`) does the same on upload; without it a
    /// character's 4K maps put the whole atlas past one storage binding and
    /// the GPU declined the scene outright.
    pub fn add_bounded(&mut self, w: u32, h: u32, rgba: &[u8], max_side: u32) -> u32 {
        let mut f = 1u32;
        while (w / f).max(h / f) > max_side.max(1) && f < w.max(h) {
            f *= 2;
        }
        if f == 1 {
            return self.add(w, h, rgba);
        }
        let (ow, oh) = ((w / f).max(1), (h / f).max(1));
        let mut out = vec![0u8; (ow * oh * 4) as usize];
        let at = |x: u32, y: u32| ((y * w + x) * 4) as usize;
        for y in 0..oh {
            for x in 0..ow {
                let mut sum = [0u32; 4];
                let mut n = 0u32;
                for dy in 0..f {
                    for dx in 0..f {
                        let (sx, sy) = (x * f + dx, y * f + dy);
                        if sx < w && sy < h && at(sx, sy) + 4 <= rgba.len() {
                            let p = &rgba[at(sx, sy)..at(sx, sy) + 4];
                            for k in 0..4 { sum[k] += p[k] as u32; }
                            n += 1;
                        }
                    }
                }
                let o = ((y * ow + x) * 4) as usize;
                for k in 0..4 { out[o + k] = if n == 0 { 0 } else { (sum[k] / n) as u8 }; }
            }
        }
        self.add(ow, oh, &out)
    }

    /// Bytes this atlas will occupy on the GPU (`words()` × 4).
    pub fn bytes(&self) -> u64 {
        (1 + 3 * self.heads.len() + self.texels.len()) as u64 * 4
    }

    /// Add an image with its mip chain: each level half the one before,
    /// every texel the mean of four, down to one texel. The image's id is
    /// returned; its levels follow it, one id each, and its head counts them.
    pub fn add(&mut self, w: u32, h: u32, rgba: &[u8]) -> u32 {
        let (w, h) = (w.max(1), h.max(1));
        let px = (w * h) as usize;
        let mut level: Vec<u32> = rgba.chunks_exact(4).take(px).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        level.resize(px, 0xff00ffff);
        let id = self.heads.len() as u32;
        let levels = 1 + w.max(h).ilog2();
        let (mut lw, mut lh) = (w, h);
        for k in 0..levels {
            self.heads.push([self.texels.len() as u32, lw, lh, if k == 0 { levels } else { 1 }]);
            self.texels.extend_from_slice(&level);
            if k + 1 < levels {
                level = Self::halved(lw, lh, &level);
                lw = (lw / 2).max(1);
                lh = (lh / 2).max(1);
            }
        }
        id
    }

    /// `texels` (`w × h`, RGBA8 words) at half size, each the mean of the
    /// four it covers — of two or one at an odd edge.
    fn halved(w: u32, h: u32, texels: &[u32]) -> Vec<u32> {
        let (ow, oh) = ((w / 2).max(1), (h / 2).max(1));
        let at = |x: u32, y: u32| texels[(y.min(h - 1) * w + x.min(w - 1)) as usize].to_le_bytes();
        let mut out = Vec::with_capacity((ow * oh) as usize);
        for y in 0..oh {
            for x in 0..ow {
                let mut sum = [0u32; 4];
                let mut n = 0u32;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                    if sx < w && sy < h {
                        let c = at(sx, sy);
                        for k in 0..4 {
                            sum[k] += c[k] as u32;
                        }
                        n += 1;
                    }
                }
                out.push(u32::from_le_bytes([0, 1, 2, 3].map(|k| (sum[k] / n.max(1)) as u8)));
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.heads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    pub fn words(&self) -> Vec<u32> {
        let base = 1 + 3 * self.heads.len() as u32;
        let mut out = vec![self.heads.len() as u32];
        // The level count rides in the width's top byte (`head_levels`).
        out.extend(self.heads.iter().flat_map(|h| [h[0] + base, h[1] | (h[3].min(255) << 24), h[2]]));
        out.extend_from_slice(&self.texels);
        out
    }
}
