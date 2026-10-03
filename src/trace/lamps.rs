//! Emitter sampling table (tests and the CPU reference).

use super::kernel::Tri;

/// Emissive triangles chosen in proportion to their power (area × luminance).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lamps {
    pub ids: Vec<u32>,
    /// Normalised cumulative power, `cdf.last() == 1`.
    pub cdf: Vec<f32>,
}

impl Lamps {
    pub fn of(tris: &[Tri]) -> Self {
        let power = |t: &Tri| {
            let (e1, e2) = ([0, 1, 2].map(|k| t.b[k] - t.a[k]), [0, 1, 2].map(|k| t.c[k] - t.a[k]));
            let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let area = 0.5 * cr.iter().map(|v| v * v).sum::<f32>().sqrt();
            area * (0.2126 * t.emit[0] + 0.7152 * t.emit[1] + 0.0722 * t.emit[2])
        };
        let (ids, pw): (Vec<u32>, Vec<f32>) =
            tris.iter().enumerate().map(|(i, t)| (i as u32, power(t))).filter(|(_, p)| *p > 0.0).unzip();
        let total: f32 = pw.iter().sum();
        let mut run = 0.0;
        let cdf = pw.iter().map(|p| { run += p; run / total }).collect();
        Self { ids, cdf }
    }

    /// GPU layout: `[count, ids…, cdf bits…]`.
    pub fn words(&self) -> Vec<u32> {
        let mut w = vec![self.ids.len() as u32];
        w.extend(&self.ids);
        w.extend(self.cdf.iter().map(|c| c.to_bits()));
        w
    }
}

