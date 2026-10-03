//! A sample dispatched in slices is the sample, and one abandoned between
//! slices leaves nothing behind.

use phiray::{Atlas, Camera, Mode, Ray, Tracer, Tri};

const W: u32 = 40;

fn scene() -> Vec<Tri> {
    let n = [0.0, -1.0, 0.0];
    let q = [[-2.0, 2.0, -2.0], [2.0, 2.0, -2.0], [2.0, 2.0, 2.0], [-2.0, 2.0, 2.0]];
    vec![Tri::new([q[0], q[1], q[2]], [n; 3], [0.8, 0.3, 0.2, 1.0]), Tri::new([q[0], q[2], q[3]], [n; 3], [0.8, 0.3, 0.2, 1.0])]
}

fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        dist: 4.0,
        units: 1.0,
        persp: 1,
        scale: 1.0,
        aspect: 1.0,
        sky: [0.2, 0.2, 0.2, 0.0],
        sun: [0.0, -0.7, 0.7],
        env: [0.3, 0.3, 0.3, 1.0],
        ..Default::default()
    }
}

#[test]
fn a_sample_in_slices_is_the_whole_sample() {
    let Some(gpu) = Ray::gpu() else { return };
    let scene = scene();
    let mut whole = Tracer::new(&gpu, &scene, &Atlas::default()).unwrap();
    whole.set_mode(Mode::Solid);
    let (a, _) = whole.render(&gpu, camera(), W, W);
    let mut sliced = Tracer::new(&gpu, &scene, &Atlas::default()).unwrap();
    sliced.set_mode(Mode::Solid);
    assert_eq!(sliced.sample_in(&gpu, camera(), W, W, 7, || true), Some((0, false)), "the first of Solid's eight jittered samples");
    let b = sliced.pixels(&gpu);
    assert_eq!(a, b, "seven slices: the same pixels");
    assert!(a.chunks_exact(4).any(|p| p[0] > 100), "and the card is in them");
}

#[test]
fn a_sample_abandoned_between_slices_starts_the_image_over() {
    let Some(gpu) = Ray::gpu() else { return };
    let mut tracer = Tracer::new(&gpu, &scene(), &Atlas::default()).unwrap();
    tracer.set_mode(Mode::Rendered);
    tracer.set_samples(4);
    assert_eq!(tracer.sample_in(&gpu, camera(), W, W, 4, || true), Some((0, false)));
    assert_eq!(tracer.sample_in(&gpu, camera(), W, W, 4, || true), Some((1, false)));
    let mut asked = 0;
    let stopped = tracer.sample_in(&gpu, camera(), W, W, 4, || {
        asked += 1;
        false
    });
    assert_eq!((stopped, asked), (None, 1), "abandoned after the first slice");
    // The same view again: sample 0, not 3 — the half-written image is not kept.
    assert_eq!(tracer.sample_in(&gpu, camera(), W, W, 4, || true), Some((0, false)));
}
