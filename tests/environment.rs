//! Environment sampling: a world image lights the scene, and the sampling
//! table (`Atlas::add_env_cdf`) must only make the same answer arrive sooner.
//!
//! The estimator divides by the table's own density, so a wrong density shows
//! up as a *bias* — the two paths converge somewhere different. That is what
//! these tests measure: cosine-sampled-only against table-sampled, on the same
//! scene, both run to convergence.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

/// An equirectangular sky: `ground` below the horizon, `sky` above, and a
/// small bright patch standing in for the sun.
fn sky_image(w: u32, h: u32, sky: f32, ground: f32, sun: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; (w * h * 3) as usize];
    for y in 0..h {
        let v = (y as f32 + 0.5) / h as f32;
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            // A patch a few cells across, well above the horizon.
            let hot = (u - 0.3).abs() < 0.02 && (v - 0.25).abs() < 0.02;
            let c = if hot { sun } else if v < 0.5 { sky } else { ground };
            for k in 0..3 {
                out[((y * w + x) * 3 + k) as usize] = c;
            }
        }
    }
    out
}

/// A flat diffuse quad facing +z, seen from straight above, lit only by the
/// world. Returns the mean scene-linear radiance over the image.
fn lit_quad(sampled: bool, samples: u32) -> Option<f32> {
    let gpu = Ray::gpu()?;
    let (w, h) = (64u32, 32u32);
    let rgb = sky_image(w, h, 0.4, 0.05, 40.0);
    let mut atlas = Atlas::default();
    let envmap = atlas.add_hdr(w, h, &rgb);
    let cdf = atlas.add_env_cdf(w, h, &rgb);
    let n = [0.0, 0.0, 1.0];
    let q = [[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]];
    let grey = [0.5, 0.5, 0.5, 1.0];
    let tris = [
        Tri::new([q[0], q[1], q[2]], [n; 3], grey).with_pbr(1.0, 0.0),
        Tri::new([q[0], q[2], q[3]], [n; 3], grey).with_pbr(1.0, 0.0),
    ];
    let cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 1.0, 0.0], back: [0.0, 0.0, 1.0],
        units: 1.0, scale: 0.2, aspect: 1.0, env: [1.0; 4],
        envmap: envmap + 1,
        env_cdf: if sampled { cdf + 1 } else { 0 },
        // One bounce either way, so the comparison is of the sampling alone.
        bounces: 1,
        ..Default::default()
    };
    let mut tr = Tracer::new(&gpu, &tris, &atlas)?;
    tr.set_samples(samples);
    while !tr.sample(&gpu, cam, 24, 24).1 {}
    let px = tr.radiance(&gpu);
    let mean: f32 = px.iter().map(|c| c[0]).sum::<f32>() / px.len().max(1) as f32;
    Some(mean)
}

#[test]
fn the_sampling_table_converges_where_chance_does() {
    let Some(chance) = lit_quad(false, 4096) else { return };
    let Some(sampled) = lit_quad(true, 512) else { return };
    let rel = (sampled - chance).abs() / chance.max(1e-6);
    assert!(
        rel < 0.06,
        "table-sampled {sampled:.4} vs cosine-sampled {chance:.4} ({:.1}% apart) — the density is wrong",
        rel * 100.0
    );
}

#[test]
fn the_sampling_table_finds_the_sun_sooner() {
    // At a low sample count the unguided estimator either misses the bright
    // patch or lands on it far too hard; the guided one should not.
    let Some(a) = lit_quad(true, 64) else { return };
    let Some(b) = lit_quad(true, 512) else { return };
    let settled = (a - b).abs() / b.max(1e-6);
    let Some(c) = lit_quad(false, 64) else { return };
    let Some(d) = lit_quad(false, 512) else { return };
    let noisy = (c - d).abs() / d.max(1e-6);
    assert!(
        settled < noisy,
        "guided sampling settled {:.2}% between 64 and 512 samples, chance {:.2}% — no gain",
        settled * 100.0,
        noisy * 100.0
    );
}
