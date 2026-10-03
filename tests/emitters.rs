//! Glowing meshes: emitter sampling and diffuse rays combined with MIS, checked
//! against closed-form irradiance in linear light.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

/// Mean linear radiance (red) of the centre 8 × 8 pixels after one sample and
/// after all samples, looking at the wall (y = 0, facing +y) from 0.1 in front.
fn wall_view(gpu: &phiray::Gpu, tris: &[Tri]) -> (f32, f32) {
    let mut cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, 1.0, 0.0],
        units: 1.0, scale: 4.0, aspect: 1.0, persp: 1, dist: 0.1, count: 1, ..Default::default()
    };
    cam.ao = 1e-3;
    let mut tr = Tracer::new(gpu, tris, &Atlas::default()).unwrap();
    let mean = |r: &[[f32; 3]]| (4..12).flat_map(|y| (4..12).map(move |x| r[y * 16 + x][0])).sum::<f32>() / 64.0;
    let (_, mut done) = tr.render(gpu, cam, 16, 16);
    let first = mean(&tr.radiance(gpu));
    while !done {
        (_, done) = tr.render(gpu, cam, 16, 16);
    }
    (mean(&tr.radiance(gpu)), first)
}

fn panel(y: f32, s: f32, c: [f32; 4]) -> [Tri; 2] {
    let n = [0.0, 1.0, 0.0];
    let q = [[-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s]];
    [Tri::new([q[0], q[1], q[2]], [n; 3], c), Tri::new([q[0], q[2], q[3]], [n; 3], c)]
}

#[test]
fn emitters_are_counted_once_near_and_far() {
    // Glowing panels light a grey wall (albedo 0.5, rough); emitter sampling and
    // diffuse rays are combined with MIS, so neither case may be doubled or lost.
    let Some(gpu) = Ray::gpu() else { return };
    let wall = || panel(0.0, 40.0, [0.5, 0.5, 0.5, 1.0]).map(|t| t.with_pbr(1.0, 0.0)).to_vec();
    // Near: a huge panel 0.2 away fills the hemisphere → 0.5 × L.
    let mut near = wall();
    near.extend(panel(0.2, 40.0, [0.0; 4]).map(|t| t.with_optics([0.4; 3], 0.0, 1.5)));
    let (all, one) = wall_view(&gpu, &near);
    let want = 0.5 * 0.4;
    assert!((all - want).abs() < want * 0.1, "near: converged {all} vs expected {want}");
    assert!((one - all).abs() < all * 0.15, "near: one sample {one} vs converged {all}");
    // Far: a 0.2 × 0.2 panel 2 away, facing the wall → (0.5 / π) · L · A / d².
    let mut far = wall();
    far.extend(panel(2.0, 0.1, [0.0; 4]).map(|t| t.with_optics([50.0; 3], 0.0, 1.5)));
    let (all, _) = wall_view(&gpu, &far);
    let want = 0.5 / std::f32::consts::PI * 50.0 * 0.04 / 4.0;
    assert!((all - want).abs() < want * 0.1, "far: converged {all} vs expected {want}");
    // Middle: a 1 × 1 panel 1 away, where both strategies matter. Irradiance at
    // the centre of a square of half-size a at distance d: 4 L s atan(s), s = a / √(a² + d²).
    let mut mid = wall();
    mid.extend(panel(1.0, 0.5, [0.0; 4]).map(|t| t.with_optics([2.0; 3], 0.0, 1.5)));
    let (all, _) = wall_view(&gpu, &mid);
    let s = 0.5f32 / 1.25f32.sqrt();
    let want = 0.5 / std::f32::consts::PI * 4.0 * 2.0 * s * s.atan();
    assert!((all - want).abs() < want * 0.08, "middle: converged {all} vs expected {want}");
}
