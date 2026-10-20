//! `Tracer::show`: scene-linear pixels a host has changed, put back and
//! run through the display step alone.
//!
//! This is the way a composited render gets shown without a second
//! implementation of the view transform (`show.wgsl`). The test is that the
//! arithmetic is the *same* arithmetic `emit` does: show the image its own
//! radiance back and nothing may move; show it something else and the
//! display follows, under each view transform.

use phiray::{Atlas, Camera, Gpu, Mode, Ray, Tracer, Tri, View};

const W: u32 = 24;

fn camera(view: View) -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        units: 1.0,
        scale: 0.1,
        aspect: 1.0,
        dist: 10.0,
        sky: [0.2, 0.22, 0.25, 0.0],
        env: [0.4, 0.4, 0.4, 1.0],
        view: view.words(0.0, 0.0),
        ..Default::default()
    }
}

/// A wall facing the camera, lit by the world alone.
fn wall() -> Vec<Tri> {
    let q = [[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]];
    let n = [0.0, -1.0, 0.0];
    vec![
        Tri::new([q[0], q[1], q[2]], [n; 3], [0.6, 0.4, 0.2, 1.0]),
        Tri::new([q[0], q[2], q[3]], [n; 3], [0.6, 0.4, 0.2, 1.0]),
    ]
}

fn settled(tr: &mut Tracer, gpu: &Gpu, view: View) {
    let (_, mut done) = tr.render(gpu, camera(view), W, W);
    let mut guard = 0;
    while !done && guard < 200 {
        guard += 1;
        (_, done) = tr.render(gpu, camera(view), W, W);
    }
}

#[test]
fn showing_the_image_its_own_radiance_changes_nothing() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Rendered);
    for view in [View::Standard, View::Filmic, View::AgX] {
        settled(&mut tr, &gpu, view);
        let (was_px, was_display, was_linear) = (tr.pixels(&gpu), tr.display(&gpu), tr.radiance(&gpu));
        assert!(was_linear.iter().any(|c| c[0] > 0.0), "{view:?}: the wall is lit");
        // The sums are `k` times the mean; `show` has to scale by the same
        // `k` on the way in or the image would come back `1 / k` as bright.
        assert!(tr.show(&gpu, &was_linear), "{view:?}: shown");
        let now_linear = tr.radiance(&gpu);
        for (a, b) in was_linear.iter().zip(&now_linear) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() <= 1e-4 * a[k].abs().max(1.0), "{view:?}: the radiance is its own: {a:?} vs {b:?}");
            }
        }
        // And the display step is `emit`'s: the same bytes come back.
        let now_display = tr.display(&gpu);
        for (a, b) in was_display.iter().zip(&now_display) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() <= 1e-4, "{view:?}: {a:?} vs {b:?}");
            }
        }
        assert_eq!(was_px, tr.pixels(&gpu), "{view:?}: and the same displayed bytes");
    }
}

/// A grade applied to the linear pixels shows through the view transform —
/// which is the whole point: half the light is not half the displayed
/// value under Filmic or AgX, so the transform has to run after the change.
#[test]
fn a_change_to_the_linear_pixels_shows_through_the_transform() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Rendered);
    for view in [View::Standard, View::Filmic, View::AgX] {
        settled(&mut tr, &gpu, view);
        let before = tr.display(&gpu);
        let linear = tr.radiance(&gpu);
        // One stop down, which is what a compositor Exposure of −1 does.
        let darker: Vec<[f32; 3]> = linear.iter().map(|c| c.map(|v| v * 0.5)).collect();
        assert!(tr.show(&gpu, &darker));
        let after = tr.display(&gpu);
        let lit: Vec<usize> = (0..linear.len()).filter(|&i| linear[i][0] > 0.05).collect();
        assert!(!lit.is_empty(), "{view:?}: something is lit");
        for &i in &lit {
            assert!(after[i][0] < before[i][0], "{view:?} pixel {i}: half the light is dimmer");
        }
        // Under Standard the display is the linear value gamma-encoded, so
        // halving the light is NOT halving the display; under Filmic and
        // AgX it is further still. Either way the ratio is not 0.5, which
        // is exactly why the grade cannot be applied to the display.
        let ratio = after[lit[0]][0] / before[lit[0]][0].max(1e-6);
        assert!(ratio > 0.5, "{view:?}: the transform is concave, so the drop is less than half: {ratio}");
    }
}

/// A Filmic look swaps the contrast curve: Very High Contrast pushes a dim
/// wall darker and Very Low Contrast lifts it, against the base Filmic —
/// the direction Blender's `filmic_to_*.spi1d` curves move the midtones.
#[test]
fn a_filmic_look_bends_the_midtones_its_way() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Rendered);
    let shown = |tr: &mut Tracer, look: phiray::Look| {
        let mut cam = camera(View::Filmic);
        cam.view = View::Filmic.words_with(0.0, 0.0, look);
        let (_, mut done) = tr.render(&gpu, cam, W, W);
        let mut guard = 0;
        while !done && guard < 200 {
            guard += 1;
            (_, done) = tr.render(&gpu, cam, W, W);
        }
        let px = tr.pixels(&gpu);
        // 4 bytes a pixel; the centre pixel's red channel.
        px[(4 * (W * (W / 2) + W / 2)) as usize] as f32
    };
    let base = shown(&mut tr, phiray::Look::None);
    let high = shown(&mut tr, phiray::Look::VeryHighContrast);
    let low = shown(&mut tr, phiray::Look::VeryLowContrast);
    // The wall shows above middle grey here (base ≈ 143/255), so more
    // contrast pushes it further from the middle — brighter — and less
    // contrast pulls it back towards it.
    assert!(high > base && base > low, "high {high} base {base} low {low}");
    assert!(high - base > 2.0 && base - low > 2.0, "the looks actually move it: high {high} base {base} low {low}");
}
