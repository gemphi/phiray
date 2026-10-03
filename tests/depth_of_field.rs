//! A camera with an aperture blurs what is not on the plane of focus.
//!
//! The measurement is edge sharpness, and the claim is *conditional*: a bug
//! that jitters the ray origin without moving the aim point blurs everything,
//! including what should be sharp, and would pass a test that only asked
//! "did the image soften". So the same edge is rendered three ways — pinhole,
//! open aperture focused on it, and open aperture focused elsewhere — and
//! only the third may be soft.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

const W: u32 = 128;
/// Where the card sits, and so where the camera must focus to see it sharply.
const CARD_DEPTH: f32 = 12.0;

/// An emissive half-plane facing −Y at depth `y`, filling the left of the
/// frame. Its right-hand edge lands on the middle column.
fn card(y: f32) -> Vec<Tri> {
    let n = [0.0, -1.0, 0.0];
    let q = [[-50.0, y, -50.0], [0.0, y, -50.0], [0.0, y, 50.0], [-50.0, y, 50.0]];
    [Tri::new([q[0], q[1], q[2]], [n; 3], [0.0, 0.0, 0.0, 1.0]), Tri::new([q[0], q[2], q[3]], [n; 3], [0.0, 0.0, 0.0, 1.0])]
        .iter()
        .map(|t| t.with_optics([8.0, 8.0, 8.0], 0.0, 1.5))
        .collect()
}

/// The camera, `focus` metres ahead, with `aperture` metres of lens radius.
/// The eye ends up at y = −2, so the card at y = 12 is 14 away.
fn camera(aperture: f32, focus: f32) -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        pivot: [0.0, -1.0, 0.0],
        dist: 1.0,
        units: 1.0,
        persp: 1,
        scale: 1.0,
        aspect: 1.0,
        // One light, so the studio rig stays out of it.
        count: 1,
        aperture,
        focus,
        ..Default::default()
    }
}

/// The scene-linear radiance, not the 8-bit pixel. Depth of field is a
/// geometric effect and the card emits 8.0, well over white: measured after
/// the view transform, a blurred edge would look hard because the *mean*
/// clips (as it does in Cycles), and the blur would be invisible to the test.
fn render(gpu: &phiray::Gpu, scene: &[Tri], cam: Camera) -> Vec<[f32; 3]> {
    let mut tracer = Tracer::new(gpu, scene, &Atlas::default()).unwrap();
    tracer.set_samples(512);
    let (mut done, mut _taken) = (false, 0);
    while !done {
        (_taken, done) = tracer.sample(gpu, cam, W, W);
    }
    tracer.radiance(gpu)
}

/// How many columns the middle row takes to fall from bright to dark: the
/// width of the card's edge, in pixels.
fn edge_width(px: &[[f32; 3]]) -> f32 {
    let row = W / 2;
    let at = |x: u32| px[(row * W + x) as usize][0];
    let bright = at(4);
    let dark = at(W - 5);
    assert!(bright > dark + 1.0, "the card does not fill the left of the frame: {bright} against {dark}");
    let level = |f: f32| dark + (bright - dark) * f;
    // Columns where the row is still above 85% and where it has fallen below
    // 15% — the distance between them is the edge.
    let last_bright = (0..W).filter(|x| at(*x) >= level(0.85)).next_back().unwrap_or(0);
    let first_dark = (0..W).find(|x| at(*x) <= level(0.15)).unwrap_or(W - 1);
    first_dark.saturating_sub(last_bright) as f32
}

#[test]
fn only_what_is_off_the_focus_plane_is_blurred() {
    let Some(gpu) = Ray::gpu() else { return };
    let scene = card(CARD_DEPTH);
    let eye_to_card = CARD_DEPTH + 2.0;

    let pinhole = edge_width(&render(&gpu, &scene, camera(0.0, 0.0)));
    assert!(pinhole <= 3.0, "a pinhole camera should leave a hard edge, not {pinhole} pixels");

    // Open the lens but focus on the card: it must stay as sharp as the
    // pinhole. This is the assertion a "blur everything" bug fails.
    let focused = edge_width(&render(&gpu, &scene, camera(0.35, eye_to_card)));
    assert!(focused <= pinhole + 2.0, "the card is on the focus plane and should still be sharp: {focused} against {pinhole}");

    // Focus well in front of it and the same edge spreads. The circle of
    // confusion at the card is `aperture × |d − f| / f` in world units —
    // here 0.35 × 11 / 3 ≈ 1.3, which the frame maps to several pixels.
    let defocused = edge_width(&render(&gpu, &scene, camera(0.35, 3.0)));
    assert!(defocused >= pinhole + 5.0, "the card is off the focus plane and should be blurred: {defocused} against {pinhole}");

    // And the blur grows with the aperture rather than being a fixed
    // softening — which is what makes the f-stop mean anything.
    let wider = edge_width(&render(&gpu, &scene, camera(0.7, 3.0)));
    assert!(wider > defocused, "a wider aperture should blur more: {wider} against {defocused}");
}
