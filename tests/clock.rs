//! The GPU clock (`gpu/clock.rs`): a single-slice sample used to return as
//! soon as it was submitted, so its "time" was the submit and the GPU's work
//! was booked to the readback after it. `Tracer::time` now holds both, apart.
use phiray::{Atlas, Camera, Ray, Tracer, Tri};
/// A grid of small quads in front of the camera: enough triangles and
/// pixels that tracing one sample costs the GPU far more than recording it.
fn scene() -> Vec<Tri> {
    let mut t = Vec::new();
    let k = 96;
    let s = 10.0 / k as f32;
    for i in 0..k {
        for j in 0..k {
            let (x, y) = (-5.0 + i as f32 * s, -5.0 + j as f32 * s);
            let z = ((i * 7 + j * 3) % 5) as f32 * 0.05;
            let p = [[x, y, z], [x + s, y, z], [x + s, y + s, z], [x, y + s, z]];
            let c = [0.6, 0.5, 0.4, 1.0];
            t.push(Tri::new([p[0], p[1], p[2]], [[0.0, 0.0, 1.0]; 3], c));
            t.push(Tri::new([p[0], p[2], p[3]], [[0.0, 0.0, 1.0]; 3], c));
        }
    }
    t
}
fn camera() -> Camera {
    let (sp, cp) = (45f32.to_radians().sin(), 45f32.to_radians().cos());
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, sp, cp],
        back: [0.0, -cp, sp],
        units: 1.0,
        scale: 0.1,
        aspect: 16.0 / 9.0,
        persp: 1,
        dist: 8.0,
        sun: [0.0, -0.6, 0.8],
        sky: [0.25, 0.27, 0.3, 1.0],
        env: [0.3, 0.32, 0.36, 1.0],
        ..Default::default()
    }
}
#[test]
fn a_single_slice_sample_reports_the_gpus_time_apart_from_its_submit() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &scene(), &Atlas::default()).expect("scene");
    tr.set_samples(64);
    let (w, h) = (1920, 1080);
    // The first sample records the pass; the ones after replay it.
    for _ in 0..3 {
        tr.sample_in(&gpu, camera(), w, h, 1, || true).expect("one slice");
    }
    let time = tr.time();
    eprintln!("submit {:?}, GPU {:?} (timestamps: {})", time.submit, time.gpu, gpu.timed);
    assert!(time.gpu > time.submit, "GPU {:?} should exceed submit {:?}", time.gpu, time.submit);
}
#[test]
fn sliced_samples_sum_the_gpus_time_over_their_slices() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &scene(), &Atlas::default()).expect("scene");
    tr.set_samples(64);
    for _ in 0..2 {
        tr.sample_in(&gpu, camera(), 640, 360, 4, || true).expect("never told to stop");
    }
    let time = tr.time();
    assert!(time.gpu > std::time::Duration::ZERO && time.gpu >= time.submit, "{time:?}");
}
