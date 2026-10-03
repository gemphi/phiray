//! Image-based lighting, precomputed: what Material Preview needs from an
//! environment image to light a surface in one deterministic dispatch, the
//! way EEVEE lights one from its world probe.
//!
//! Two things are made once per environment, on the CPU, and put in the
//! atlas beside the image:
//! - its **irradiance** as nine spherical-harmonic coefficients (Ramamoorthi
//!   & Hanrahan 2001): the diffuse light a surface facing `n` receives, over
//!   π, so `albedo · Σ cᵢ Yᵢ(n)` is its diffuse radiance;
//! - its **prefiltered** copies, one per roughness step (`LEVELS − 1` of
//!   them; the image itself stands for roughness 0): the GGX lobe of that
//!   roughness convolved over the sphere (Karis 2013, `N = V = R`), sampled
//!   with filtered importance sampling (Křivánek 2008: each sample reads the
//!   image at the level its lobe covers), so the result is smooth from 64
//!   samples where plain sampling would speckle.

mod panorama;

pub use panorama::Panorama;
use rayon::prelude::*;

/// Where an environment's lighting is in the atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvLighting {
    /// The 27 irradiance coefficients (`add_bits`, 27 × 1).
    pub irradiance: u32,
    /// The first prefiltered copy (roughness `1 / (LEVELS − 1)`); the rest
    /// follow it, one id each.
    pub prefiltered: u32,
    /// Roughness steps, the image itself included.
    pub levels: u32,
}

/// Roughness steps: 0 (the image), 0.2, 0.4, 0.6, 0.8, 1.
pub const LEVELS: u32 = 6;
/// A prefiltered copy's size: a blurred sphere needs no more.
pub const PREFILTER_W: u32 = 128;
pub const PREFILTER_H: u32 = 64;
/// GGX samples per prefiltered texel.
const SAMPLES: u32 = 64;

/// The precomputation.
pub struct Ibl;

impl Ibl {
    /// The nine irradiance coefficients, over π, RGB each: `albedo · Σ cᵢ
    /// Yᵢ(n)` is a diffuse surface's radiance under this sky.
    pub fn irradiance(env: &Panorama) -> [[f32; 3]; 9] {
        // Project at a size the sum is quick at; the sky is smooth at this scale.
        let mut small = Panorama { w: env.w, h: env.h, rgb: env.rgb.clone() };
        while small.w > 256 {
            small = small.halved();
        }
        let (w, h) = (small.w, small.h);
        let rows: Vec<[[f32; 3]; 9]> = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut acc = [[0.0f32; 3]; 9];
                let v = (y as f32 + 0.5) / h as f32;
                let theta = (0.5 - v) * std::f32::consts::PI;
                // Solid angle of a texel at this latitude.
                let dw = theta.cos() * (2.0 * std::f32::consts::PI / w as f32) * (std::f32::consts::PI / h as f32);
                for x in 0..w {
                    let d = Panorama::direction((x as f32 + 0.5) / w as f32, v);
                    let c = small.at(x, y);
                    let y_lm = Self::basis(d);
                    for i in 0..9 {
                        for k in 0..3 {
                            acc[i][k] += c[k] * y_lm[i] * dw;
                        }
                    }
                }
                acc
            })
            .collect();
        let mut sum = [[0.0f32; 3]; 9];
        for row in rows {
            for i in 0..9 {
                for k in 0..3 {
                    sum[i][k] += row[i][k];
                }
            }
        }
        // The cosine lobe's band factors, over π: 1, 2/3, 1/4.
        let band = [1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25];
        for i in 0..9 {
            for k in 0..3 {
                sum[i][k] *= band[i];
            }
        }
        sum
    }

    /// The real spherical-harmonic basis to band 2 at `d`, in the kernel's
    /// order (`sh_irradiance`).
    pub fn basis(d: [f32; 3]) -> [f32; 9] {
        let [x, y, z] = d;
        [
            0.282095,
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ]
    }

    /// The image convolved with the GGX lobe of `roughness`, `PREFILTER_W ×
    /// PREFILTER_H`, linear RGB.
    pub fn prefiltered(chain: &[Panorama], roughness: f32) -> Panorama {
        let (w, h) = (PREFILTER_W, PREFILTER_H);
        let base = &chain[0];
        let alpha = (roughness * roughness).max(1e-3);
        // Solid angle of one base texel, for the level each sample reads.
        let texel_sa = 4.0 * std::f32::consts::PI / (base.w as f32 * base.h as f32);
        let rows: Vec<Vec<f32>> = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut row = Vec::with_capacity((w * 3) as usize);
                for x in 0..w {
                    let n = Panorama::direction((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                    let (t, b) = Self::frame(n);
                    let mut sum = [0.0f32; 3];
                    let mut weight = 0.0f32;
                    for i in 0..SAMPLES {
                        let (u1, u2) = Self::hammersley(i, SAMPLES);
                        // A GGX half vector (Walter 2007), about `n`.
                        let cos_h = ((1.0 - u1) / (1.0 + (alpha * alpha - 1.0) * u1)).sqrt();
                        let sin_h = (1.0 - cos_h * cos_h).max(0.0).sqrt();
                        let phi = 2.0 * std::f32::consts::PI * u2;
                        let hv = [0, 1, 2].map(|k| t[k] * sin_h * phi.cos() + b[k] * sin_h * phi.sin() + n[k] * cos_h);
                        // `V = N`: the light direction is `N` mirrored in `H`.
                        let nh = cos_h;
                        let l = [0, 1, 2].map(|k| 2.0 * nh * hv[k] - n[k]);
                        let nl = l[0] * n[0] + l[1] * n[1] + l[2] * n[2];
                        if nl <= 0.0 {
                            continue;
                        }
                        // The lobe's density here, and the level whose texel
                        // covers one sample's share of it.
                        let d = alpha * alpha / (std::f32::consts::PI * (nh * nh * (alpha * alpha - 1.0) + 1.0).powi(2));
                        let pdf = (d * 0.25).max(1e-6);
                        let sample_sa = 1.0 / (SAMPLES as f32 * pdf);
                        let level = (0.5 * (sample_sa / texel_sa).log2() + 1.0).clamp(0.0, (chain.len() - 1) as f32);
                        let (lu, lv) = Panorama::uv(l);
                        let c = Self::sample_level(chain, level, lu, lv);
                        for k in 0..3 {
                            sum[k] += c[k] * nl;
                        }
                        weight += nl;
                    }
                    for k in 0..3 {
                        row.push(if weight > 0.0 { sum[k] / weight } else { 0.0 });
                    }
                }
                row
            })
            .collect();
        Panorama { w, h, rgb: rows.concat() }
    }

    /// The image and its halves down to a few texels.
    pub fn chain(env: &Panorama) -> Vec<Panorama> {
        let mut chain = vec![Panorama { w: env.w, h: env.h, rgb: env.rgb.clone() }];
        while chain.last().is_some_and(|p| p.w > 8 && p.h > 4) {
            let next = chain.last().expect("one").halved();
            chain.push(next);
        }
        chain
    }

    fn sample_level(chain: &[Panorama], level: f32, u: f32, v: f32) -> [f32; 3] {
        let lo = level.floor() as usize;
        let hi = (lo + 1).min(chain.len() - 1);
        let t = level - lo as f32;
        let (a, b) = (chain[lo].sample(u, v), chain[hi].sample(u, v));
        [0, 1, 2].map(|k| a[k] + (b[k] - a[k]) * t)
    }

    /// An orthonormal frame about `n`: `(tangent, bitangent)`.
    fn frame(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let up = if n[2].abs() < 0.999 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let t = cross(up, n);
        let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt().max(1e-6);
        let t = t.map(|v| v / len);
        (t, cross(n, t))
    }

    /// The `i`th of `n` points of the Hammersley set.
    fn hammersley(i: u32, n: u32) -> (f32, f32) {
        let bits = i.reverse_bits();
        ((i as f32 + 0.5) / n as f32, bits as f32 * 2.328_306_4e-10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform white sky: every direction receives 1 (over π: 1), and
    /// every prefiltered copy is that same white.
    #[test]
    fn a_uniform_sky_lights_every_direction_the_same() {
        let env = Panorama::new(64, 32, &vec![1.0; 64 * 32 * 3]);
        let c = Ibl::irradiance(&env);
        for d in [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]] {
            let y = Ibl::basis(d);
            let e: f32 = (0..9).map(|i| c[i][0] * y[i]).sum();
            assert!((e - 1.0).abs() < 0.02, "{d:?}: {e}");
        }
        let chain = Ibl::chain(&env);
        let rough = Ibl::prefiltered(&chain, 0.8);
        assert!(rough.rgb.iter().all(|v| (v - 1.0).abs() < 0.02));
    }

    /// A sky lit from above only: a surface facing up gets the light, one
    /// facing down almost none; a rough reflection of the sun is dimmer and
    /// wider than the sun itself.
    #[test]
    fn light_from_above_reaches_what_faces_up() {
        let (w, h) = (64u32, 32u32);
        let mut rgb = vec![0.0f32; (w * h * 3) as usize];
        for y in 0..h / 4 {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                rgb[i..i + 3].copy_from_slice(&[4.0, 4.0, 4.0]);
            }
        }
        let env = Panorama::new(w, h, &rgb);
        let c = Ibl::irradiance(&env);
        let e = |d: [f32; 3]| -> f32 {
            let y = Ibl::basis(d);
            (0..9).map(|i| c[i][0] * y[i]).sum()
        };
        assert!(e([0.0, 0.0, 1.0]) > 4.0 * e([0.0, 0.0, -1.0]).max(0.01), "up {} down {}", e([0.0, 0.0, 1.0]), e([0.0, 0.0, -1.0]));
        let chain = Ibl::chain(&env);
        let sharp = Ibl::prefiltered(&chain, 0.2);
        let soft = Ibl::prefiltered(&chain, 1.0);
        // The zenith texel: bright in the sharp copy, dimmed by the lobe's
        // reach into the dark in the soft one.
        let zenith = |p: &Panorama| p.rgb[(p.w as usize / 2) * 3];
        assert!(zenith(&sharp) > zenith(&soft) + 0.5, "sharp {} soft {}", zenith(&sharp), zenith(&soft));
        // And the soft copy spreads light to the horizon the sharp one leaves dark.
        let horizon = |p: &Panorama| p.rgb[((p.h as usize / 2) * p.w as usize + p.w as usize / 2) * 3];
        assert!(horizon(&soft) > horizon(&sharp) + 0.05, "soft {} sharp {}", horizon(&soft), horizon(&sharp));
    }
}
