//! Bloom / Glare (`Tracer::glow`, `wgsl/glow.wgsl`): the bright parts of a
//! scene-linear image bleed into the dark around them. Two things are
//! guarded. The glare must REACH OUT — a dark pixel beside a bright patch
//! is brighter with glow than without — and it must be DRIVEN BY
//! BRIGHTNESS, not a flat lift of the whole frame: a threshold above the
//! brightest pixel is a no-op. A blur that lifted everything would pass the
//! first test and fail the second.
//!
//! GPU test: returns early (passes) where there is no adapter, as the other
//! phiray render tests do.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

const SIDE: u32 = 64;

/// Render a dark frame with one small, very bright emissive quad at its
/// centre — the shape a glare spreads from — optionally running the glare
/// pass on the settled frame.
fn render(glow: Option<[f32; 3]>) -> Option<Vec<u8>> {
    let gpu = Ray::gpu()?;
    let n = [0.0, 0.0, 1.0];
    let back = |c: [f32; 4]| {
        let q = [[-8.0, -8.0, -1.0], [8.0, -8.0, -1.0], [8.0, 8.0, -1.0], [-8.0, 8.0, -1.0]];
        [Tri::new([q[0], q[1], q[2]], [n; 3], c), Tri::new([q[0], q[2], q[3]], [n; 3], c)]
    };
    let patch = || {
        let q = [[-0.6, -0.6, 0.0], [0.6, -0.6, 0.0], [0.6, 0.6, 0.0], [-0.6, 0.6, 0.0]];
        // `with_optics(emit, trans, ior)`: a strong emission, no transmission.
        [
            Tri::new([q[0], q[1], q[2]], [n; 3], [0.0, 0.0, 0.0, 1.0]).with_optics([40.0, 40.0, 40.0], 0.0, 1.5),
            Tri::new([q[0], q[2], q[3]], [n; 3], [0.0, 0.0, 0.0, 1.0]).with_optics([40.0, 40.0, 40.0], 0.0, 1.5),
        ]
    };
    let mut tris = Vec::new();
    tris.extend(back([0.01, 0.01, 0.01, 1.0]));
    tris.extend(patch());
    let atlas = Atlas::default();
    let cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 1.0, 0.0], back: [0.0, 0.0, 1.0],
        units: 1.0, scale: 0.18, aspect: 1.0,
        ..Default::default()
    };
    let mut tr = Tracer::new(&gpu, &tris, &atlas)?;
    tr.set_samples(64);
    while !tr.sample(&gpu, cam, SIDE, SIDE).1 {}
    if let Some([t, i, r]) = glow {
        tr.glow(&gpu, t, i, r);
    }
    Some(tr.pixels(&gpu))
}

/// The mean luma of a small box of pixels centred on `(cx, cy)`.
fn box_luma(px: &[u8], cx: u32, cy: u32, half: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    for y in cy.saturating_sub(half)..=(cy + half).min(SIDE - 1) {
        for x in cx.saturating_sub(half)..=(cx + half).min(SIDE - 1) {
            let o = ((y * SIDE + x) * 4) as usize;
            sum += 0.2126 * px[o] as f32 + 0.7152 * px[o + 1] as f32 + 0.0722 * px[o + 2] as f32;
            n += 1.0;
        }
    }
    sum / n.max(1.0)
}

#[test]
fn glare_reaches_out_from_the_bright_patch() {
    let Some(plain) = render(None) else { return };
    // A low threshold so the bright patch glows, a strong intensity and a
    // wide radius so the glare reaches well into the dark surround.
    let Some(bloomed) = render(Some([0.5, 1.0, 8.0])) else { return };
    // A point partway between the patch centre and the corner — dark in the
    // plain render, lit by the glare in the bloomed one.
    let (ox, oy) = (SIDE / 2 + 12, SIDE / 2);
    let before = box_luma(&plain, ox, oy, 2);
    let after = box_luma(&bloomed, ox, oy, 2);
    assert!(after > before + 2.0, "glare did not reach the surround: {before:.1} -> {after:.1}");
}

#[test]
fn a_threshold_above_everything_is_a_no_op() {
    let Some(plain) = render(None) else { return };
    // Threshold far above any scene-linear luma in the frame: nothing is
    // bright enough to glow, so the image must be unchanged.
    let Some(bloomed) = render(Some([1e6, 1.0, 8.0])) else { return };
    let d: f32 = plain.iter().zip(&bloomed).map(|(a, b)| (*a as f32 - *b as f32).abs()).sum::<f32>() / plain.len().max(1) as f32;
    assert!(d < 0.5, "a threshold above the whole frame still changed it by {d:.2}/byte");
}
