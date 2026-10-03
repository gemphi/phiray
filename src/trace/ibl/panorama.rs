//! A linear RGB equirectangular image: the environment as `ibl` reads it,
//! with the sampling, halving and direction mapping the precomputation
//! needs — the same mapping the kernel's `env_at` uses.

/// A linear RGB equirectangular image.
pub struct Panorama {
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<f32>,
}

impl Panorama {
    pub fn new(w: u32, h: u32, rgb: &[f32]) -> Self {
        let mut rgb = rgb.to_vec();
        rgb.resize((w * h * 3) as usize, 0.0);
        Self { w: w.max(1), h: h.max(1), rgb }
    }

    pub(super) fn at(&self, x: u32, y: u32) -> [f32; 3] {
        let i = ((y.min(self.h - 1) * self.w + x.min(self.w - 1)) * 3) as usize;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }

    /// Bilinear, `u` wrapping and `v` clamped, as the kernel's `sample_env`.
    pub(super) fn sample(&self, u: f32, v: f32) -> [f32; 3] {
        let fx = u.rem_euclid(1.0) * self.w as f32 - 0.5;
        let fy = v.clamp(0.0, 1.0) * self.h as f32 - 0.5;
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let xa = (x0 as i64).rem_euclid(self.w as i64) as u32;
        let xb = (xa + 1) % self.w;
        let ya = (y0 as i64).clamp(0, self.h as i64 - 1) as u32;
        let yb = (y0 as i64 + 1).clamp(0, self.h as i64 - 1) as u32;
        let mix = |a: [f32; 3], b: [f32; 3], t: f32| [0, 1, 2].map(|k| a[k] + (b[k] - a[k]) * t);
        mix(mix(self.at(xa, ya), self.at(xb, ya), tx), mix(self.at(xa, yb), self.at(xb, yb), tx), ty)
    }

    /// Half the size, each texel the mean of four.
    pub(super) fn halved(&self) -> Self {
        let (w, h) = ((self.w / 2).max(1), (self.h / 2).max(1));
        let mut rgb = vec![0.0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = self.at((x * 2 + dx).min(self.w - 1), (y * 2 + dy).min(self.h - 1));
                    for k in 0..3 {
                        sum[k] += c[k] * 0.25;
                    }
                }
                let o = ((y * w + x) * 3) as usize;
                rgb[o..o + 3].copy_from_slice(&sum);
            }
        }
        Self { w, h, rgb }
    }

    /// The direction texel `(x, y)` looks along — `env_at`'s mapping run
    /// backwards: `u = −atan2(y, x) / 2π + 0.5`, `v = 0.5 − asin(z) / π`.
    pub(super) fn direction(u: f32, v: f32) -> [f32; 3] {
        let phi = -(u - 0.5) * 2.0 * std::f32::consts::PI;
        let theta = (0.5 - v) * std::f32::consts::PI;
        [theta.cos() * phi.cos(), theta.cos() * phi.sin(), theta.sin()]
    }

    pub(super) fn uv(d: [f32; 3]) -> (f32, f32) {
        let u = -d[1].atan2(d[0]) / (2.0 * std::f32::consts::PI) + 0.5;
        let v = 0.5 - d[2].atan2((d[0] * d[0] + d[1] * d[1]).sqrt()) / std::f32::consts::PI;
        (u, v)
    }
}
