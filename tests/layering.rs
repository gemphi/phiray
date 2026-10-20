//! The Principled's dielectric layer over its diffuse (`svm/closure.h`):
//! F0 from the IOR, and the diffuse weighted by what the layer did not
//! reflect (`closure_layering_weight`). A white surface under a white
//! world gives back what arrives — the reflection is taken out of the
//! diffuse, not added on top of it.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

/// Mean scene-linear radiance of a large quad facing the camera, of
/// `albedo` and roughness `rough`, reflecting at `ior`, under a uniform
/// world of 1.
fn quad(albedo: f32, rough: f32, ior: f32) -> Option<f32> {
    let gpu = Ray::gpu()?;
    let n = [0.0, 0.0, 1.0];
    let q = [[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]];
    let c = [albedo, albedo, albedo, 1.0];
    let tris = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]].map(|t| Tri::new(t, [n; 3], c).with_pbr(rough, 0.0).with_optics([0.0; 3], 0.0, ior));
    let cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 1.0, 0.0], back: [0.0, 0.0, 1.0],
        units: 1.0, scale: 0.2, aspect: 1.0, env: [1.0; 4], count: 1,
        ..Default::default()
    };
    let mut tr = Tracer::new(&gpu, &tris, &Atlas::default())?;
    tr.set_samples(256);
    while !tr.sample(&gpu, cam, 16, 16).1 {}
    let px = tr.radiance(&gpu);
    Some(px.iter().map(|c| c[0]).sum::<f32>() / px.len().max(1) as f32)
}

#[test]
fn a_white_dielectric_returns_the_light_it_gets_not_more() {
    let Some(plain) = quad(1.0, 0.3, 1.0) else { return };
    let Some(glossy) = quad(1.0, 0.3, 1.5) else { return };
    // Before the layering this came back a reflection's worth brighter
    // (about 1.045 at roughness 0.3), which every pose render showed as a
    // pale cast.
    assert!((glossy - plain).abs() / plain < 0.02, "IOR 1.5 {glossy:.4} against no reflection {plain:.4}");
    // And the layer is there: a black surface still reflects.
    let Some(black) = quad(0.0, 0.3, 1.5) else { return };
    let Some(none) = quad(0.0, 0.3, 1.0) else { return };
    assert!(black > 0.03 && none < 0.002, "the reflection of a black dielectric {black:.4}, of one at IOR 1 {none:.4}");
}
