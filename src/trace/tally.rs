//! The per-pixel counters a measurement reads (`wgsl/tally.wgsl`): rays of
//! each kind and the walk's work, summed over the image. They exist only
//! with the crate's `tally` feature; without it the kernel compiles them
//! away and the image buffer carries no planes for them.
use super::Tracer;
use crate::gpu::Gpu;
/// The planes' names, in `tally.wgsl`'s order.
const NAME_LIST: [&str; Tracer::TALLY_LEN] = [
    "path (samples traced)",
    "camera ray (first hit and every pass through)",
    "cut-out pass (alpha)",
    "glass pass (refraction)",
    "diffuse bounce ray",
    "ambient-occlusion ray",
    "glossy reflection ray",
    "shading point (direct light)",
    "shadow ray: lamp or rig key",
    "shadow ray: rig rim",
    "shadow ray: emitter (glow)",
    "shadow ray: world",
    "BVH node visit",
    "triangle test",
    "instance entry",
    "BVH node visit, shadow rays",
    "triangle test, shadow rays",
    "ray (every walk)",
    "reflection that hit geometry",
    "nodes: camera's first ray",
    "nodes: camera ray on through glass or cut-out",
    "nodes: diffuse bounce",
    "nodes: ambient occlusion",
    "nodes: glossy reflection, opaque surface",
    "nodes: shadow, lamp or rig key",
    "nodes: shadow, rig rim",
    "nodes: shadow, emitter",
    "nodes: shadow, world",
    "nodes: glossy reflection off glass",
    "glossy reflection ray off glass",
];
/// The image's counters, each summed over every pixel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tally {
    pub sum: Vec<u64>,
    pub pixels: u64,
}
impl Tally {
    /// The table the render log takes: each counter's total, its mean a
    /// path, and for the walk's work its mean a ray.
    pub fn report(&self) -> String {
        let at = |k: usize| self.sum.get(k).copied().unwrap_or(0);
        let paths = at(0).max(1) as f64;
        let rays = at(17).max(1) as f64;
        let shadow: u64 = (8..12).map(at).sum();
        let mut out = format!("Tally: {} px, {} paths\n", self.pixels, at(0));
        for (k, name) in NAME_LIST.iter().enumerate() {
            out.push_str(&format!("  {:<48} {:>16}  {:>9.3} a path\n", name, at(k), at(k) as f64 / paths));
        }
        out.push_str(&format!("  shadow rays, all lights {:>16}  {:>9.3} a path\n", shadow, shadow as f64 / paths));
        out.push_str(&format!("  per ray: {:.1} nodes, {:.1} triangles, {:.2} instances\n", at(12) as f64 / rays, at(13) as f64 / rays, at(14) as f64 / rays));
        let shadow_rays = shadow.max(1) as f64;
        let other_rays = (at(17).saturating_sub(shadow)).max(1) as f64;
        out.push_str(&format!(
            "  per shadow ray: {:.1} nodes, {:.1} triangles; per other ray: {:.1} nodes, {:.1} triangles\n",
            at(15) as f64 / shadow_rays,
            at(16) as f64 / shadow_rays,
            (at(12) - at(15)) as f64 / other_rays,
            (at(13) - at(16)) as f64 / other_rays,
        ));
        // Where the walk's work went, by the kind of ray that did it.
        let nodes = at(12).max(1) as f64;
        for k in 19..29 {
            out.push_str(&format!("  {:<48} {:>5.1} % of the nodes\n", NAME_LIST[k], at(k) as f64 / nodes * 100.0));
        }
        out
    }
}
impl Tracer {
    /// Counter planes after the image's own (`tally.wgsl`).
    pub(crate) const TALLY_LEN: usize = 30;
    /// Words the image buffer takes for `n` pixels: its planes, the
    /// gather's count, and the counters when they are built in.
    pub(crate) fn px_len(n: usize) -> usize {
        let tally = if cfg!(feature = "tally") { Self::TALLY_LEN * n } else { 0 };
        Self::PX_LEN * n + 1 + tally
    }
    /// The counters summed over the image; `None` without the feature or
    /// before anything was traced.
    pub fn tally(&self, gpu: &Gpu) -> Option<Tally> {
        if !cfg!(feature = "tally") {
            return None;
        }
        let n = (self.size.0 * self.size.1) as usize;
        let px = self.px.as_ref()?;
        let words = px.read_range(gpu, Self::PX_LEN * n + 1, Self::TALLY_LEN * n);
        if words.len() < Self::TALLY_LEN * n {
            return None;
        }
        let sum = words.chunks_exact(n.max(1)).map(|plane| plane.iter().map(|&w| u64::from(w)).sum()).collect();
        if let Ok(dir) = std::env::var("PHIRAY_TALLY_DIR") {
            save_planes(std::path::Path::new(&dir), &words, self.size.0, self.size.1);
        }
        Some(Tally { sum, pixels: n as u64 })
    }
}
/// Each plane as an 8-bit PGM scaled to its own largest count, so where
/// a kind of ray is spent can be seen (`PHIRAY_TALLY_DIR`).
fn save_planes(dir: &std::path::Path, words: &[u32], w: u32, h: u32) {
    let n = (w * h) as usize;
    let _ = std::fs::create_dir_all(dir);
    for (k, plane) in words.chunks_exact(n.max(1)).enumerate() {
        let most = plane.iter().copied().max().unwrap_or(0).max(1) as f64;
        let mut out = format!("P5\n{w} {h}\n255\n").into_bytes();
        out.extend(plane.iter().map(|&c| (c as f64 / most * 255.0).round() as u8));
        let _ = std::fs::write(dir.join(format!("tally_{k:02}.pgm")), out);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_report_names_every_counter_and_divides_by_paths_and_rays() {
        let mut sum = vec![0u64; Tracer::TALLY_LEN];
        sum[0] = 10;
        sum[1] = 20;
        sum[8] = 30;
        sum[12] = 4000;
        sum[13] = 1000;
        sum[15] = 600;
        sum[16] = 100;
        sum[17] = 50;
        sum[19] = 3000;
        let r = Tally { sum, pixels: 5 }.report();
        assert!(r.contains("Tally: 5 px, 10 paths"), "{r}");
        for name in NAME_LIST {
            assert!(r.contains(name), "{name}: {r}");
        }
        assert!(r.contains("per ray: 80.0 nodes, 20.0 triangles"), "{r}");
        assert!(r.contains("per shadow ray: 20.0 nodes"), "{r}");
        assert!(r.contains("per other ray: 170.0 nodes"), "{r}");
        assert!(r.contains("75.0 % of the nodes"), "{r}");
    }
    #[test]
    fn the_buffer_grows_only_with_the_feature() {
        let extra = if cfg!(feature = "tally") { Tracer::TALLY_LEN * 4 } else { 0 };
        assert_eq!(Tracer::px_len(4), Tracer::PX_LEN * 4 + 1 + extra);
    }
}
