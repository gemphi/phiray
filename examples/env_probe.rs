//! Probe: a white diffuse card under a sky that is 6.0 above the horizon and
//! 0.02 below. Analytically the card (normal horizontal, albedo 0.8) reflects
//! 0.8/π × 6 × π/2 ≈ 2.4. Prints what each estimator half produces.
use phiray::{Atlas, Camera, Ray, Tracer, Tri};

fn sky_atlas(flip: bool) -> Atlas {
    let (w, h) = (16u32, 8u32);
    let mut rgb = Vec::new();
    for y in 0..h {
        let up = (y < h / 2) != flip;
        for _ in 0..w {
            rgb.extend(if up { [6.0, 6.0, 6.0] } else { [0.02, 0.02, 0.02] });
        }
    }
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    atlas
}

fn main() {
    let gpu = Ray::gpu().expect("gpu");
    let n = [0.0, -1.0, 0.0];
    let q = [[-9.0, 2.0, -9.0], [9.0, 2.0, -9.0], [9.0, 2.0, 9.0], [-9.0, 2.0, 9.0]];
    let card = [Tri::new([q[0], q[1], q[2]], [n; 3], [0.8, 0.8, 0.8, 1.0]), Tri::new([q[0], q[2], q[3]], [n; 3], [0.8, 0.8, 0.8, 1.0])];
    let base = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], pivot: [0.0, -1.0, 0.0],
        dist: 1.0, units: 1.0, persp: 1, scale: 1.0, aspect: 1.0, sky: [0.24, 0.24, 0.24, 1.0],
        env: [1.0, 1.0, 1.0, 1.0], envmap: 1, env_cdf: 2, env_hidden: 1, count: 1, sun: [0.0, -0.7, 0.7],
        ..Default::default()
    };
    let w = 16u32;
    for (label, cam, flip) in [
        ("NEE + bounces (env_cdf on)", base, false),
        ("bounces only (env_cdf off → depth 1 + tail)", Camera { env_cdf: 0, ..base }, false),
        ("NEE + bounces, sky flipped (bright below)", base, true),
        ("bounces 1 explicit", Camera { bounces: 1, ..base }, false),
        ("bounces 4 explicit", Camera { bounces: 4, ..base }, false),
    ] {
        let mut tracer = Tracer::new(&gpu, &card, &sky_atlas(flip)).unwrap();
        tracer.set_samples(256);
        loop {
            let (_, done) = tracer.sample(&gpu, cam, w, w);
            if done { break; }
        }
        let rad = tracer.radiance(&gpu);
        let mid = &rad[(w * w / 2) as usize..(w * w / 2 + w) as usize];
        let mean: f32 = mid.iter().map(|c| c[0]).sum::<f32>() / mid.len() as f32;
        let disp = tracer.display(&gpu);
        let dmid = &disp[(w * w / 2) as usize..(w * w / 2 + w) as usize];
        let dmean: f32 = dmid.iter().map(|c| c[0]).sum::<f32>() / dmid.len() as f32;
        let px = tracer.pixels(&gpu);
        let pmid = &px[((w * w / 2) * 4) as usize..((w * w / 2 + w) * 4) as usize];
        let pmean: f32 = pmid.chunks_exact(4).map(|c| f32::from(c[0])).sum::<f32>() / w as f32;
        println!("{label:46} linear {mean:.3}  display-float {dmean:.3}  packed8 {pmean:.0}");
    }
}
