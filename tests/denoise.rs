//! The a-trous denoiser has to do two things: move a noisy render *towards*
//! the converged one, and leave an edge where it found one. A blur that only
//! did the first would pass a noise test and ruin every picture.
//!
//! The two are guarded separately. Replacing the edge-stopping weight with 1
//! (blur everything) still passes the edge test — albedo demodulation alone
//! keeps a texture edge — but fails the convergence test, because blurring
//! across geometry moves the picture away from the reference. So the second
//! test names what it really guards: demodulation.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

/// A sky with a small, very bright patch — the shape of noise a path tracer
/// actually produces, since most rays miss the bright part.
fn sky(w: u32, h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (w * h * 3) as usize];
    for y in 0..h {
        let v = (y as f32 + 0.5) / h as f32;
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let hot = (u - 0.3).abs() < 0.03 && (v - 0.25).abs() < 0.03;
            let c = if hot { 60.0 } else if v < 0.5 { 0.3 } else { 0.05 };
            for k in 0..3 {
                out[((y * w + x) * 3 + k) as usize] = c;
            }
        }
    }
    out
}

/// Two quads side by side under that sky: the left dark, the right pale, so
/// there is a hard albedo edge down the middle of the frame.
fn scene() -> (Vec<Tri>, Atlas, Camera) {
    let (w, h) = (64u32, 32u32);
    let rgb = sky(w, h);
    let mut atlas = Atlas::default();
    let envmap = atlas.add_hdr(w, h, &rgb);
    let cdf = atlas.add_env_cdf(w, h, &rgb);
    let n = [0.0, 0.0, 1.0];
    let quad = |x0: f32, x1: f32, c: [f32; 4]| {
        let q = [[x0, -4.0, 0.0], [x1, -4.0, 0.0], [x1, 4.0, 0.0], [x0, 4.0, 0.0]];
        [
            Tri::new([q[0], q[1], q[2]], [n; 3], c).with_pbr(1.0, 0.0),
            Tri::new([q[0], q[2], q[3]], [n; 3], c).with_pbr(1.0, 0.0),
        ]
    };
    let mut tris = Vec::new();
    tris.extend(quad(-4.0, 0.0, [0.05, 0.05, 0.05, 1.0]));
    tris.extend(quad(0.0, 4.0, [0.8, 0.8, 0.8, 1.0]));
    let cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 1.0, 0.0], back: [0.0, 0.0, 1.0],
        units: 1.0, scale: 0.12, aspect: 1.0, env: [1.0; 4],
        envmap: envmap + 1, env_cdf: cdf + 1, bounces: 1,
        ..Default::default()
    };
    (tris, atlas, cam)
}

const SIDE: u32 = 48;

fn render(samples: u32, denoise: bool) -> Option<Vec<u8>> {
    let gpu = Ray::gpu()?;
    let (tris, atlas, cam) = scene();
    let mut tr = Tracer::new(&gpu, &tris, &atlas)?;
    tr.set_samples(samples);
    while !tr.sample(&gpu, cam, SIDE, SIDE).1 {}
    if denoise {
        tr.denoise(&gpu);
    }
    Some(tr.pixels(&gpu))
}

/// Mean absolute difference per colour byte.
fn diff(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    (0..n).map(|i| (a[i] as f32 - b[i] as f32).abs()).sum::<f32>() / n.max(1) as f32
}

#[test]
fn denoising_moves_a_noisy_render_towards_the_converged_one() {
    let Some(reference) = render(1024, false) else { return };
    let Some(noisy) = render(12, false) else { return };
    let Some(cleaned) = render(12, true) else { return };
    let before = diff(&noisy, &reference);
    let after = diff(&cleaned, &reference);
    assert!(
        after < before * 0.75,
        "denoised is {after:.2} from the reference, noisy was {before:.2} — no real gain"
    );
}

#[test]
fn demodulation_keeps_the_albedo_edge() {
    let Some(cleaned) = render(12, true) else { return };
    // Read across the middle row: the left half is dark, the right pale, and
    // the step between them must survive the filter.
    let row = (SIDE / 2) * SIDE;
    let at = |x: u32| cleaned[((row + x) * 4) as usize] as f32;
    let left: f32 = (4..SIDE / 2 - 4).map(at).sum::<f32>() / (SIDE / 2 - 8) as f32;
    let right: f32 = (SIDE / 2 + 4..SIDE - 4).map(at).sum::<f32>() / (SIDE / 2 - 8) as f32;
    assert!(right - left > 40.0, "the edge flattened: left {left:.1}, right {right:.1}");
    // And the step itself should still be sharp: two neighbouring pixels
    // either side of the seam, not a long ramp.
    let step = at(SIDE / 2 + 1) - at(SIDE / 2 - 2);
    assert!(step > (right - left) * 0.5, "the seam ramped over {step:.1} of {:.1}", right - left);
}

/// The viewport's filter shows a denoised frame but leaves the samples
/// alone, so the refinement under it is not blurred into itself.
#[test]
fn denoising_what_is_shown_keeps_the_samples() {
    let Some(gpu) = Ray::gpu() else { return };
    let (tris, atlas, cam) = scene();
    let mut tracer = Tracer::new(&gpu, &tris, &atlas).unwrap();
    tracer.set_samples(64);
    for _ in 0..4 {
        tracer.sample(&gpu, cam, SIDE, SIDE);
    }
    let before = tracer.radiance(&gpu);
    let noisy = tracer.pixels(&gpu);
    tracer.denoise_shown(&gpu);
    assert_eq!(tracer.radiance(&gpu), before, "the linear sums are untouched");
    assert_ne!(tracer.pixels(&gpu), noisy, "what is shown is filtered");
    tracer.denoise(&gpu);
    assert_ne!(tracer.radiance(&gpu), before, "a finished render's denoise replaces them");
}

/// On a transparent film a pixel that met nothing must come out of the
/// filter with no colour, as Blender writes it. LOOK-01 (2026-10-27) came
/// out wrapped in a pastel halo at alpha 0: such pixels were lent the light
/// of the surface beside them and divided it by the share of their samples
/// that met something — none (`denoise.wgsl`). This scene does not reach
/// that path (a plain miss's depth keeps it out of a surface's
/// neighbourhood), so it guards the plain case only; the halo itself was
/// checked on LOOK-01's render, alpha-0 pixels all black after the fix.
#[test]
fn a_pixel_that_met_nothing_keeps_no_light_after_denoising() {
    let Some(gpu) = Ray::gpu() else { return };
    let (_, atlas, cam) = scene();
    let n = [0.0, 0.0, 1.0];
    let q = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]];
    let pale = [0.8, 0.8, 0.8, 1.0];
    let tris = vec![
        Tri::new([q[0], q[1], q[2]], [n; 3], pale).with_pbr(1.0, 0.0),
        Tri::new([q[0], q[2], q[3]], [n; 3], pale).with_pbr(1.0, 0.0),
    ];
    let clear = Camera { film: [1, 0, 0, 0], ..cam };
    let mut tracer = Tracer::new(&gpu, &tris, &atlas).unwrap();
    tracer.set_samples(12);
    while !tracer.sample(&gpu, clear, SIDE, SIDE).1 {}
    tracer.denoise(&gpu);
    let px = tracer.pixels(&gpu);
    let centre = (((SIDE / 2) * SIDE + SIDE / 2) * 4) as usize;
    assert_eq!(px[centre + 3], 255, "the quad is in the frame");
    let lit: Vec<usize> = (0..(SIDE * SIDE) as usize)
        .filter(|i| px[i * 4 + 3] == 0 && px[i * 4..i * 4 + 3].iter().any(|c| *c > 2))
        .collect();
    assert!(lit.is_empty(), "{} pixels at alpha 0 carry light, the first {:?}", lit.len(), lit.first().map(|i| &px[i * 4..i * 4 + 4]));
}
