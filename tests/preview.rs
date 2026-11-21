//! Material Preview is deterministic: one dispatch is the frame, the same
//! frame every time, smooth where the surface is smooth — and a texture is
//! read at the level the pixel's footprint asks for.

mod card;
use card::{W, camera, card, luma, render, sky_atlas};
use phiray::{Mode, Ray, Tracer};

/// Mean absolute difference between horizontal neighbours over the card's middle.
fn roughness_of(px: &[u8]) -> f32 {
    let mut sum = 0.0;
    let mut n = 0;
    for y in W / 3..2 * W / 3 {
        for x in W / 3..2 * W / 3 - 1 {
            sum += (luma(px, x, y) - luma(px, x + 1, y)).abs();
            n += 1;
        }
    }
    sum / n as f32
}

#[test]
fn one_pass_is_the_frame_and_the_same_frame_every_time() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let scene = card(3.0, 2.0, None);
    let (first, passes) = render(Mode::Preview, camera(), &scene, &atlas);
    assert_eq!(passes, 1, "deterministic: no refinement");
    let (again, _) = render(Mode::Preview, camera(), &scene, &atlas);
    assert_eq!(first, again, "the same frame again");
    assert!(luma(&first, W / 2, W / 2) > 60.0, "the sky lights the card: {}", luma(&first, W / 2, W / 2));
}

#[test]
fn a_flat_surface_is_smooth_where_one_path_traced_sample_is_noisy() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let scene = card(3.0, 2.0, None);
    let (preview, _) = render(Mode::Preview, camera(), &scene, &atlas);
    let gpu = Ray::gpu().unwrap();
    let mut tracer = Tracer::new(&gpu, &scene, &atlas).unwrap();
    tracer.set_mode(Mode::Rendered);
    tracer.set_samples(64);
    let (traced, _) = tracer.render(&gpu, camera(), W, W);
    let (smooth, noisy) = (roughness_of(&preview), roughness_of(&traced));
    assert!(smooth < 1.0, "a flat card is flat: {smooth}");
    assert!(noisy > smooth * 4.0, "the first path-traced sample is the noisy one: {noisy} vs {smooth}");
}

/// No ray falls between the two triangles of a quad. The card's shared
/// diagonal runs from `q[0]` to `q[2]` — the line x = z in the plane
/// y = 2 — and a primary ray meets it exactly where `x + y = W − 1`, so
/// that is the row of pixels to look along. Möller–Trumbore divided by
/// each triangle's own determinant before testing its barycentrics, which
/// let both sides reject a ray that fell on the edge: six of these 48
/// pixels showed the sky instead of the card. The test is watertight now
/// (Woop, Benthin, Benthin and Wald, JCGT 2(1) 2013; `bvh.wgsl`'s
/// `tri_t`, with its CPU twin in `src/trace/tri.rs`), so a ray on the
/// edge hits both triangles and the nearer-wins test gives it to one.
///
/// A hole is read off the frame rather than guessed at: the card is
/// uniform, so each pixel of the diagonal has to match the two beside it
/// in the same row, which are squarely inside one triangle. The card
/// covers the whole frame at this camera (the quad reaches ±3 where the
/// rays reach ±0.75), so there is no silhouette to confuse it with.
#[test]
fn no_ray_falls_between_the_two_triangles_of_a_quad() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let (px, _) = render(Mode::Preview, camera(), &card(3.0, 2.0, None), &atlas);
    let mut worst = (0u32, 0u32, 0.0f32);
    for x in 1..W - 1 {
        let y = W - 1 - x;
        let beside = 0.5 * (luma(&px, x - 1, y) + luma(&px, x + 1, y));
        assert!(beside > 100.0, "the sky lights the card either side of the diagonal at ({x}, {y}): {beside}");
        let apart = (luma(&px, x, y) - beside).abs();
        if apart > worst.2 {
            worst = (x, y, apart);
        }
    }
    assert!(worst.2 < 20.0, "the shared diagonal shows through at ({}, {}): {} luma off the card", worst.0, worst.1, worst.2);
}

#[test]
fn a_distant_checker_reads_as_grey_through_its_mip_levels() {
    if Ray::gpu().is_none() {
        return;
    }
    let mut atlas = sky_atlas();
    // A 64 × 64 checker of single texels.
    let mut rgba = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64 {
        for x in 0..64 {
            let on = (x + y) % 2 == 0;
            rgba.extend(if on { [255, 255, 255, 255] } else { [0, 0, 0, 255] });
        }
    }
    let tex = atlas.add(64, 64, &rgba);
    // Tiled eight times over a card that fills the frame: 512 texels over
    // 48 pixels — read at its base level this would be a moiré of black
    // and white; at the level the footprint asks for it is grey.
    let scene = card(3.0, 2.0, Some((tex, 8.0)));
    let (px, _) = render(Mode::Preview, camera(), &scene, &atlas);
    let plain = render(Mode::Preview, camera(), &card(3.0, 2.0, None), &atlas).0;
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for y in W / 3..2 * W / 3 {
        for x in W / 3..2 * W / 3 {
            let ratio = luma(&px, x, y) / luma(&plain, x, y).max(1.0);
            lo = lo.min(ratio);
            hi = hi.max(ratio);
        }
    }
    assert!(hi - lo < 0.25, "no moiré: the checker is one grey across the card ({lo:.2}..{hi:.2})");
    assert!(lo > 0.2 && hi < 0.8, "grey, not black or white ({lo:.2}..{hi:.2})");
}
