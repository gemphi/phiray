//! The rig both power tests stand on: a 20 m Lambertian plane, a black
//! world, and an orthographic camera straight down on it, as
//! `truth/blender_lamp_power.py` sets it up (`lamp_power.rs`, `world_power.rs`).
use phiray::{Atlas, Camera, LIGHTS, Light, Tri};
pub const W: u32 = 32;
/// A black world: the lamp is the only light.
pub fn dark() -> Atlas {
    let (w, h) = (8u32, 4u32);
    let rgb = vec![0.0f32; (w * h * 3) as usize];
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    atlas.add_env_lighting(w, h, &rgb);
    atlas
}
/// A 20 x 20 m plane at z = 0, albedo 0.8, as rough as a surface gets so
/// its lobe is Lambert and nothing else.
pub fn plane() -> Vec<Tri> {
    let n = [0.0, 0.0, 1.0];
    let q = [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0]];
    let c = [0.8, 0.8, 0.8, 1.0];
    vec![
        Tri::new([q[0], q[1], q[2]], [n; 3], c).with_pbr(1.0, 0.0),
        Tri::new([q[0], q[2], q[3]], [n; 3], c).with_pbr(1.0, 0.0),
    ]
}
/// Straight down on the origin, orthographic, 2 m across — the frame
/// `blender_lamp_power.py` sets up.
pub fn overhead(light: Light, count: u32) -> Camera {
    let mut lights = [Light::default(); LIGHTS];
    lights[0] = light;
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        back: [0.0, 0.0, 1.0],
        pivot: [0.0, 0.0, 0.0],
        dist: 10.0,
        units: 1.0,
        persp: 0,
        // The frame spans 2 m, as the reference's `ortho_scale = 2.0` does:
        // the kernel's extent is `1 / (aspect × scale)`. It stood at 1.0 —
        // a 1 m frame — which left every flat-field number here right (the
        // centre of a flat field is the centre either way) and every
        // *profile* twice as wide as Blender's in pixels (§58).
        scale: 0.5,
        aspect: 1.0,
        // A black sky and no environment: the lamp is all there is.
        sky: [0.0; 4],
        env: [0.0, 0.0, 0.0, 1.0],
        envmap: 0,
        env_cdf: 0,
        env_hidden: 0,
        preview: [0; 4],
        count,
        lights,
        ..Default::default()
    }
}
