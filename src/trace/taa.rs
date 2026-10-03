//! Workbench's anti-aliasing (`workbench_effect_antialiasing.cc`): the
//! view is drawn `SAMPLES` times, each sample offset by a jitter from
//! `BLI_jitter_init`'s relaxed table — the sample nearest the centre
//! first, then each the farthest from the one before — and the samples
//! are gathered with a 3 × 3 Blackman-Harris filter of width 2 in
//! `log2(colour + 1)` space, resolved by the accumulated weight. This file
//! is the table and the weights, ported step for step (`BLI_jitter_init`,
//! `BLI_jitterate1/2`, `RandomNumberGenerator`, `filter_blackman_harris`,
//! `setup_taa_weights`); `taa.wgsl` is the gather, `Tracer::sample_in`
//! the sequence.

use std::sync::OnceLock;

/// `U.viewport_aa`'s default: eight samples.
pub const SAMPLES: u32 = 8;
/// `setup_taa_weights`: a 2-pixel-wide filter over the 3 × 3 neighbourhood.
const FILTER_WIDTH: f32 = 2.0;

/// `BLI_rand.hh`'s generator: a 48-bit linear congruence.
struct Rng(u64);

impl Rng {
    fn new(seed: u32) -> Self {
        Self((u64::from(seed) << 16) | 0x330E)
    }

    fn step(&mut self) -> i32 {
        self.0 = (0x5DEE_CE66D_u64.wrapping_mul(self.0).wrapping_add(0xB)) & 0x0000_FFFF_FFFF_FFFF;
        (self.0 >> 17) as u32 as i32
    }

    fn get_double(&mut self) -> f64 {
        f64::from(self.step()) / 2147483648.0
    }
}

/// `BLI_jitterate1`: push each point away from its neighbours (and their
/// wrapped copies) within `radius`, a step of a relaxation.
fn jitterate1(jit: &mut [[f32; 2]], radius: f32) {
    let n = jit.len();
    let mut out = jit.to_vec();
    for i in (0..n).rev() {
        let (mut dx, mut dy) = (0.0f32, 0.0f32);
        let (mut x, mut y) = (jit[i][0], jit[i][1]);
        for j in (0..n).rev() {
            if i == j {
                continue;
            }
            let mut vx = jit[j][0] - x - 1.0;
            let mut vy = jit[j][1] - y - 1.0;
            for _ in 0..3 {
                for _ in 0..3 {
                    if vx.abs() < radius && vy.abs() < radius {
                        let len = (vx * vx + vy * vy).sqrt();
                        if len > 0.0 && len < radius {
                            let l = len / radius;
                            dx += vx / l;
                            dy += vy / l;
                        }
                    }
                    vx += 1.0;
                }
                vx -= 3.0;
                vy += 1.0;
            }
        }
        x -= dx / 18.0;
        y -= dy / 18.0;
        out[i] = [x - x.floor(), y - y.floor()];
    }
    jit.copy_from_slice(&out);
}

/// `BLI_jitterate2`: the same along each axis alone.
fn jitterate2(jit: &mut [[f32; 2]], radius: f32) {
    let n = jit.len();
    let mut out = jit.to_vec();
    for i in (0..n).rev() {
        let (mut dx, mut dy) = (0.0f32, 0.0f32);
        let (mut x, mut y) = (jit[i][0], jit[i][1]);
        for j in (0..n).rev() {
            if i == j {
                continue;
            }
            let mut vx = jit[j][0] - x - 1.0;
            let mut vy = jit[j][1] - y - 1.0;
            for _ in 0..3 {
                if vx.abs() < radius {
                    dx += vx * radius;
                }
                vx += 1.0;
            }
            for _ in 0..3 {
                if vy.abs() < radius {
                    dy += vy * radius;
                }
                vy += 1.0;
            }
        }
        x -= dx / 2.0;
        y -= dy / 2.0;
        out[i] = [x - x.floor(), y - y.floor()];
    }
    jit.copy_from_slice(&out);
}

/// `BLI_jitter_init`: `n` points in `−0.5 ..= 0.5`, seeded and relaxed as
/// Blender's are.
pub fn jitter_init(n: usize) -> Vec<[f32; 2]> {
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f32;
    let (rad1, rad2, rad3) = (1.0 / nf.sqrt(), 1.0 / nf, nf.sqrt() / nf);
    let mut rng = Rng::new(31415926 + n as u32);
    let mut jit = Vec::with_capacity(n);
    let mut x = 0.0f32;
    for i in 0..n {
        let a = x + rad1 * (0.5 - rng.get_double()) as f32;
        let b = i as f32 / nf + rad1 * (0.5 - rng.get_double()) as f32;
        jit.push([a, b]);
        x += rad3;
        x -= x.floor();
    }
    for _ in 0..24 {
        jitterate1(&mut jit, rad1);
        jitterate1(&mut jit, rad1);
        jitterate2(&mut jit, rad2);
    }
    for j in &mut jit {
        j[0] -= 0.5;
        j[1] -= 0.5;
    }
    jit
}

/// `TaaSamples::init_samples`: the table in `−1 ..= 1`, the sample nearest
/// the centre moved onto it and drawn first, the rest each the farthest
/// from the one before.
pub fn table(n: usize) -> Vec<[f32; 2]> {
    let mut s = jitter_init(n);
    if s.is_empty() {
        return s;
    }
    let closest = (0..n).min_by(|&a, &b| (s[a][0].powi(2) + s[a][1].powi(2)).total_cmp(&(s[b][0].powi(2) + s[b][1].powi(2)))).unwrap_or(0);
    let c = s[closest];
    for p in &mut s {
        let w = [(p[0] - c[0] + 0.5).rem_euclid(1.0), (p[1] - c[1] + 0.5).rem_euclid(1.0)];
        *p = [w[0] * 2.0 - 1.0, w[1] * 2.0 - 1.0];
    }
    s.swap(0, closest);
    for i in 0..n.saturating_sub(2) {
        let mut best = (0.0f32, i);
        for j in i + 1..n {
            let d = (s[i][0] - s[j][0]).powi(2) + (s[i][1] - s[j][1]).powi(2);
            if d > best.0 {
                best = (d, j);
            }
        }
        s.swap(i + 1, best.1);
    }
    s
}

/// Sample `k` of `SAMPLES`'s offset, in `−1 ..= 1` (half a pixel either way
/// once `window_translate_m4` divides it by the resolution).
pub fn offset(k: u32) -> [f32; 2] {
    static TABLE: OnceLock<Vec<[f32; 2]>> = OnceLock::new();
    let t = TABLE.get_or_init(|| table(SAMPLES as usize));
    t[(k as usize) % t.len()]
}

/// `filter_blackman_harris`.
pub fn blackman_harris(x: f32, width: f32) -> f32 {
    if x > width * 0.5 {
        return 0.0;
    }
    let x = 2.0 * std::f32::consts::PI * (x / width + 0.5).clamp(0.0, 1.0);
    0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos() - 0.01168 * (3.0 * x).cos()
}

/// `setup_taa_weights`: the 3 × 3 neighbourhood's weights for a sample at
/// `offset`, and their sum — what the resolve divides by, accumulated.
pub fn weights(offset: [f32; 2]) -> ([f32; 9], f32) {
    let mut w = [0.0f32; 9];
    let mut sum = 0.0;
    let mut i = 0;
    for x in -1..=1 {
        for y in -1..=1 {
            let (sx, sy) = (x as f32 - offset[0], y as f32 - offset[1]);
            w[i] = blackman_harris((sx * sx + sy * sy).sqrt(), FILTER_WIDTH);
            sum += w[i];
            i += 1;
        }
    }
    (w, sum)
}

/// The weight the resolve divides by once samples `0 ..= k` are in.
pub fn accumulated_weight(k: u32) -> f32 {
    (0..=k).map(|s| weights(offset(s)).1).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_blenders_eight_centred_and_spread() {
        let t = table(8);
        assert_eq!(t.len(), 8);
        assert_eq!(t[0], [0.0, 0.0], "the sample nearest the centre is moved onto it and drawn first");
        assert!(t.iter().all(|p| p[0].abs() <= 1.0 && p[1].abs() <= 1.0), "{t:?}");
        // Distinct, and relaxed: no two closer than a quarter of the cell.
        for i in 0..8 {
            for j in i + 1..8 {
                let d = ((t[i][0] - t[j][0]).powi(2) + (t[i][1] - t[j][1]).powi(2)).sqrt();
                assert!(d > 0.25, "{i} and {j} sit {d} apart: {t:?}");
            }
        }
        assert_eq!(offset(8), offset(0), "the table wraps");
        // The generator is Blender's: its first draw after seed 31415926 + 8.
        let mut rng = Rng::new(31415926 + 8);
        let first = rng.get_double();
        assert!((0.0..1.0).contains(&first));
    }

    #[test]
    fn the_filter_is_blackman_harris_over_three_by_three() {
        assert!((blackman_harris(0.0, 2.0) - 1.0).abs() < 1e-5, "full weight at the centre");
        assert_eq!(blackman_harris(1.5, 2.0), 0.0, "nothing past the width");
        let (w, sum) = weights([0.0, 0.0]);
        assert!((w[4] - 1.0).abs() < 1e-5, "the pixel itself at an unjittered sample");
        assert!(w[0] < 1e-4 && w[1] > 0.0 && w[1] < 0.1, "corners nothing, edges a little: {w:?}");
        assert!((sum - (w.iter().sum::<f32>())).abs() < 1e-6);
        let shifted = weights([1.0, 0.0]);
        assert!(shifted.0[7] > shifted.0[1], "a sample to the right weighs the right neighbour more");
        assert!(accumulated_weight(7) > accumulated_weight(0));
    }
}
