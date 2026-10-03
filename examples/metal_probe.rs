//! Probe: a metal card (base = F0, roughness r) facing the camera under a
//! uniform sky of radiance 1. A single-scattering GGX metal returns at most
//! F(θ) of a uniform environment — F0 at normal incidence, a little less as
//! the lobe loses energy to masking — so the card must read about F0.
use phiray::{Atlas, Camera, Ray, Tracer, Tri};

fn uniform_atlas() -> Atlas {
    let (w, h) = (16u32, 8u32);
    let rgb = vec![1.0f32; (w * h * 3) as usize];
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    atlas
}

fn main() {
    let gpu = Ray::gpu().expect("gpu");
    // The card stands in front of the camera, facing it (normal −back).
    let n = [0.0, -1.0, 0.0];
    let q = [[-9.0, 2.0, -9.0], [9.0, 2.0, -9.0], [9.0, 2.0, 9.0], [-9.0, 2.0, 9.0]];
    let cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], pivot: [0.0, -1.0, 0.0],
        dist: 1.0, units: 1.0, persp: 0, scale: 0.2, aspect: 1.0, sky: [0.24, 0.24, 0.24, 1.0],
        env: [1.0, 1.0, 1.0, 1.0], envmap: 1, env_cdf: 2, env_hidden: 1, count: 0, sun: [0.0, -0.7, 0.7],
        bounces: 4, ..Default::default()
    };
    let w = 16u32;
    for (label, f0, rough, metal) in [
        ("dark gold metal, rough 0.2", [0.058, 0.05, 0.017, 1.0], 0.2, 1.0),
        ("dark gold metal, rough 0.5", [0.058, 0.05, 0.017, 1.0], 0.5, 1.0),
        ("white metal, rough 0.2", [1.0, 1.0, 1.0, 1.0], 0.2, 1.0),
        ("white diffuse 0.8, rough 0.5", [0.8, 0.8, 0.8, 1.0], 0.5, 0.0),
    ] {
        let tri = |p: [[f32; 3]; 3]| {
            let mut t = Tri::new(p, [n; 3], f0);
            t.rough = rough;
            t.metal = metal;
            t
        };
        let card = [tri([q[0], q[1], q[2]]), tri([q[0], q[2], q[3]])];
        let mut tracer = Tracer::new(&gpu, &card, &uniform_atlas()).unwrap();
        tracer.set_samples(512);
        loop {
            let (_, done) = tracer.sample(&gpu, cam, w, w);
            if done { break; }
        }
        let rad = tracer.radiance(&gpu);
        let mid = &rad[(w * w / 2) as usize..(w * w / 2 + w) as usize];
        let mean: [f32; 3] = [0, 1, 2].map(|k| mid.iter().map(|c| c[k]).sum::<f32>() / mid.len() as f32);
        println!("{label:32} linear {:.4} {:.4} {:.4}   (F0 = {:.3} {:.3} {:.3})", mean[0], mean[1], mean[2], f0[0], f0[1], f0[2]);
    }
}
