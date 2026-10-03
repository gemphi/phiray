//! A hidden environment lights the scene without being seen.
//!
//! Material Preview's studio HDRI has to do two things at once: reflect in a
//! metal and light a diffuse surface, while the background behind the model
//! stays the flat viewport grey. `Camera::env_hidden` is the switch; this
//! checks both halves, because a switch that only changed the background
//! would pass a screenshot and silently unlight the model.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

const W: u32 = 32;

/// A tiny equirectangular sky: bright above, dark below.
fn sky_atlas() -> Atlas {
    let (w, h) = (16u32, 8u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        let up = y < h / 2;
        for _ in 0..w {
            rgb.extend(if up { [6.0, 6.0, 6.0] } else { [0.02, 0.02, 0.02] });
        }
    }
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    atlas
}

fn render(cam: Camera, scene: &[Tri]) -> Vec<u8> {
    let gpu = Ray::gpu().unwrap();
    let mut tracer = Tracer::new(&gpu, scene, &sky_atlas()).unwrap();
    tracer.set_samples(64);
    let (mut px, mut done) = tracer.render(&gpu, cam, W, W);
    while !done {
        (px, done) = tracer.render(&gpu, cam, W, W);
    }
    px
}

fn luma(px: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * f32::from(px[i]) + 0.7152 * f32::from(px[i + 1]) + 0.0722 * f32::from(px[i + 2])
}

#[test]
fn a_hidden_environment_lights_the_model_and_shows_the_sky_colour_behind_it() {
    if Ray::gpu().is_none() {
        return;
    }
    // A white diffuse card in the lower half of the frame; the upper half is
    // empty and looks straight into the environment.
    let n = [0.0, -1.0, 0.0];
    // Wide and tall enough that the lower half of the frame is all card.
    let q = [[-9.0, 2.0, -9.0], [9.0, 2.0, -9.0], [9.0, 2.0, 0.0], [-9.0, 2.0, 0.0]];
    let card = [Tri::new([q[0], q[1], q[2]], [n; 3], [0.8, 0.8, 0.8, 1.0]), Tri::new([q[0], q[2], q[3]], [n; 3], [0.8, 0.8, 0.8, 1.0])];
    let base = Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        pivot: [0.0, -1.0, 0.0],
        dist: 1.0,
        units: 1.0,
        persp: 1,
        scale: 0.3,
        aspect: 1.0,
        sky: [0.24, 0.24, 0.24, 1.0],
        // The fallback rig keys off ; a zero vector would make it NaN.
        sun: [0.0, -0.7, 0.7],
        // `env` scales the image (the world's colour × strength); at zero the
        // environment is black whatever the image says.
        env: [1.0, 1.0, 1.0, 1.0],
        envmap: 1,
        env_cdf: 2,
        count: 0,
        ..Default::default()
    };
    let shown = render(Camera { env_hidden: 0, ..base }, &card);
    let hidden = render(Camera { env_hidden: 1, ..base }, &card);

    // Behind the model: the shown case sees the bright sky, the hidden one
    // the flat viewport grey.
    let sky_shown = luma(&shown, W / 2, 2);
    let sky_hidden = luma(&hidden, W / 2, 2);
    assert!(sky_shown > 200.0, "an unhidden environment should be seen: {sky_shown}");
    assert!((sky_hidden - 61.0).abs() < 12.0, "a hidden environment should show the sky colour (0.24 → ~61): {sky_hidden}");

    // On the model: both are lit by the same environment.
    let card_shown = luma(&shown, W / 2, W - 6);
    let card_hidden = luma(&hidden, W / 2, W - 6);
    assert!(card_hidden > 40.0, "the hidden environment should still light the card: {card_hidden} (shown {card_shown})");
    assert!((card_shown - card_hidden).abs() < 25.0, "hiding the environment should not change the lighting: {card_shown} against {card_hidden}");

    // And the studio light is the *only* light: with no lamps, the kernel's
    // three-point fallback rig must stay off under a hidden environment,
    // exactly as Blender switches the scene's lamps off in Material Preview.
    // `count: 1` with a zero lamp is how the rig is disabled by hand.
    let no_rig = render(Camera { env_hidden: 1, count: 1, ..base }, &card);
    let card_no_rig = luma(&no_rig, W / 2, W - 6);
    assert!((card_hidden - card_no_rig).abs() < 6.0, "the three-point rig is still adding light under the studio HDRI: {card_hidden} against {card_no_rig}");
}
