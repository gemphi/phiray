//! Material Preview's lamps: a shadow as soft as the lamp's radius, an
//! area light's penumbra following its shape, a spot's blend and its
//! scaled cone. The scene is `card/`; the frame itself is `preview.rs`.
mod card;
use card::{W, camera, card, luma, render, sky_atlas};
use phiray::{Camera, Light, Mode, Ray, Tri};
#[test]
fn a_lamp_casts_a_shadow_as_soft_as_its_radius() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let mut scene = card(3.0, 2.0, None);
    // A small card between a lamp just off the axis and the big one: its
    // shadow falls on the big card to the left of it, and the camera sees
    // that shadow's outer edge past the small card's silhouette.
    scene.extend(card(0.3, 1.0, None));
    let lit_by = |li: Light, blocked: bool| {
        let mut lights = [Light::default(); phiray::MAX_LIGHT];
        lights[0] = li;
        let cam = Camera { count: 1, lights, envmap: 0, env_cdf: 0, preview: [0; 4], env: [0.05, 0.05, 0.05, 1.0], ..camera() };
        let geo: Vec<Tri> = if blocked { scene.clone() } else { card(3.0, 2.0, None) };
        render(Mode::Preview, cam, &geo, &atlas).0
    };
    let lamp = |size: f32| Light { pos: [0.6, -2.0, 0.0], kind: 1, color: [800.0, 800.0, 800.0], size, dir: [0.0, 1.0, 0.0], cone: 0.0, ..Light::default() };
    let shot = |size: f32, blocked: bool| lit_by(lamp(size), blocked);
    // The shadow lies to the left of the small card's silhouette; the row
    // through the middle of the frame crosses it.
    const EDGE: u32 = 12;
    let band = |px: &[u8]| (0..EDGE).map(|x| luma(px, x, W / 2)).collect::<Vec<f32>>();

    let hard = shot(0.0, true);
    let open = band(&shot(0.0, false));
    let umbra = band(&hard).into_iter().fold(f32::MAX, f32::min);
    assert!(open[8] > umbra + 80.0, "the blocker shadows the row: open {} umbra {umbra}", open[8]);

    // The penumbra: pixels along the shadow's edge that are neither the
    // umbra nor the open light. A lamp the size of a point has none — its
    // edge steps from one to the other, as EEVEE's does with Shadow Soft
    // Size at 0 — and the wider the lamp, the more of them, because the
    // shadow ray is aimed over its shape (`shadow.wgsl`).
    let lit = open[0];
    let (lo, hi) = (umbra + 0.15 * (lit - umbra), lit - 0.15 * (lit - umbra));
    let between = |px: &[u8]| band(px).into_iter().filter(|&v| v > lo && v < hi).count();
    let (none, some, more) = (between(&hard), between(&shot(0.3, true)), between(&shot(0.6, true)));
    assert_eq!(none, 0, "a point-sized lamp steps from lit to shaded");
    assert!(some >= 2 && more > some, "the penumbra widens with the radius: {none} {some} {more}");

    // A Sun's shape is an Angle, and its ray goes inside that cone
    // (`shadow_ray_generate_directional`), so its shadow softens the same
    // way — at the angle, not at a radius.
    let sun = |size: f32| Light { kind: 0, color: [2.0, 2.0, 2.0], size, dir: [-0.3, 1.0, 0.0], ..Light::default() };
    let sharp = lit_by(sun(0.0), true);
    let sun_open = band(&lit_by(sun(0.0), false));
    let sun_umbra = band(&sharp).into_iter().fold(f32::MAX, f32::min);
    let (slo, shi) = (sun_umbra + 0.15 * (sun_open[0] - sun_umbra), sun_open[0] - 0.15 * (sun_open[0] - sun_umbra));
    let sun_between = |px: &[u8]| band(px).into_iter().filter(|&v| v > slo && v < shi).count();
    assert!(sun_open[8] > sun_umbra + 40.0, "the blocker shadows the row under a sun too: {} {sun_umbra}", sun_open[8]);
    assert!(sun_between(&lit_by(sun(0.2), true)) > sun_between(&sharp), "a sun's angle softens its shadow");
}

/// An area light's penumbra is as wide as the light is *in that
/// direction*: a strip along X blurs its shadow's vertical edges and
/// barely its horizontal ones, and turning the same strip into the plane's
/// other axis swaps that. The taps spread over the light's own shape in
/// its own plane (`shadow.wgsl`), so the shape has to reach the record
/// (`AREA_RECT`, `right`, `size_y`) for this to hold.
#[test]
fn an_area_lights_penumbra_follows_its_own_shape() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let mut scene = card(3.0, 2.0, None);
    scene.extend(card(0.3, 1.0, None));
    // The light shines along +y, so its plane spans world X and Z and its
    // `right` is X; `size` lies along X and `size_y` across it.
    let strip = |size: f32, size_y: f32| Light {
        pos: [0.6, -2.0, 0.0],
        kind: 3,
        color: [800.0, 800.0, 800.0],
        size,
        dir: [0.0, 1.0, 0.0],
        cone: 0.0,
        right: [1.0, 0.0, 0.0],
        size_y,
        shape: phiray::AREA_RECT,
        ..Light::default()
    };
    let shot = |li: Light, blocked: bool| {
        let mut lights = [Light::default(); phiray::MAX_LIGHT];
        lights[0] = li;
        let cam = Camera { count: 1, lights, envmap: 0, env_cdf: 0, preview: [0; 4], env: [0.05, 0.05, 0.05, 1.0], ..camera() };
        let geo: Vec<Tri> = if blocked { scene.clone() } else { card(3.0, 2.0, None) };
        render(Mode::Preview, cam, &geo, &atlas).0
    };
    const EDGE: u32 = 12;
    let band = |px: &[u8]| (0..EDGE).map(|x| luma(px, x, W / 2)).collect::<Vec<f32>>();
    let point = shot(strip(0.0, 0.0), true);
    let open = band(&shot(strip(0.0, 0.0), false));
    let umbra = band(&point).into_iter().fold(f32::MAX, f32::min);
    let (lo, hi) = (umbra + 0.15 * (open[0] - umbra), open[0] - 0.15 * (open[0] - umbra));
    let between = |px: &[u8]| band(px).into_iter().filter(|&v| v > lo && v < hi).count();
    // The scan is horizontal, so the X edge is what blurs it.
    let wide = between(&shot(strip(1.2, 0.02), true));
    let tall = between(&shot(strip(0.02, 1.2), true));
    assert!(wide > tall, "a strip across the scan softens it, one along it does not: {wide} vs {tall}");
    assert!(tall <= 1, "the narrow direction stays a step: {tall}");
}

/// A spot's Blend is how much of its cone fades: the kernels smoothstep
/// from the cone's cosine to `cone + blend`, as EEVEE does
/// (`eevee_light.cc`'s `spot_blend = (1 − spot_size) × spotblend`, then
/// `light_spot_attenuation`). Blend 0 gives a hard-edged circle of light
/// and a wide Blend a soft one, where before the file's value was ignored
/// for Blender's default 0.15.
#[test]
fn a_spots_blend_fades_the_edge_of_its_cone() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let scene = card(3.0, 2.0, None);
    // A spot on the axis, pointing at the card. The frame sees about a
    // twentieth of the card, so the cone has to be narrow — 6° at four
    // units puts the pool's edge well inside it.
    let spot = |blend: f32| {
        let cone = 6.0_f32.to_radians().cos();
        Light { pos: [0.0, -2.0, 0.0], kind: 2, color: [600.0, 600.0, 600.0], size: 0.0, dir: [0.0, 1.0, 0.0], cone, blend: (1.0 - cone) * blend, ..Light::default() }
    };
    let shot = |blend: f32| {
        let mut lights = [Light::default(); phiray::MAX_LIGHT];
        lights[0] = spot(blend);
        let cam = Camera { count: 1, lights, envmap: 0, env_cdf: 0, preview: [0; 4], env: [0.02, 0.02, 0.02, 1.0], ..camera() };
        render(Mode::Preview, cam, &scene, &atlas).0
    };
    // Along the middle row the pool is bright in the centre and dark
    // outside; the edge's width is what Blend sets.
    let row = |px: &[u8]| (0..W).map(|x| luma(px, x, W / 2)).collect::<Vec<f32>>();
    let hard = row(&shot(0.0));
    let (dark, bright) = (hard.iter().cloned().fold(f32::MAX, f32::min), hard.iter().cloned().fold(0.0, f32::max));
    assert!(bright > dark + 60.0, "the spot lights a pool: {dark} ..= {bright}");
    let (lo, hi) = (dark + 0.2 * (bright - dark), bright - 0.2 * (bright - dark));
    let edge = |r: &[f32]| r.iter().filter(|&&v| v > lo && v < hi).count();
    let (none, some, wide) = (edge(&hard), edge(&row(&shot(0.4))), edge(&row(&shot(1.0))));
    assert!(none <= 2, "Blend 0 is a hard edge: {none} pixels across it");
    assert!(some > none && wide > some, "the edge widens with Blend: {none} {some} {wide}");
}

/// A spot's cone follows the object's scale: squash the object and the pool
/// of light is an ellipse, not a circle (EEVEE's `spot_size_inv` through
/// `light_spot_attenuation`'s `ellipse` term; `spot_mask` in `light.wgsl`).
/// `spot_scale` is `scale.z / scale.xy`, and the pool's reach along an axis
/// is exactly `tan(half angle) / spot_scale` — so 0.75 reaches a third
/// further, and the axis with the smaller number is the long one.
#[test]
fn a_scaled_spots_pool_of_light_is_an_ellipse() {
    if Ray::gpu().is_none() {
        return;
    }
    let atlas = sky_atlas();
    let scene = card(3.0, 2.0, None);
    // The same 6° spot as the Blend test, on the axis. Its own X is the
    // world's, so its Y lies along the world's Z — the image's vertical.
    let shot = |right: [f32; 3], spot_scale: [f32; 2]| {
        let cone = 6.0_f32.to_radians().cos();
        let mut lights = [Light::default(); phiray::MAX_LIGHT];
        lights[0] = Light {
            pos: [0.0, -2.0, 0.0],
            kind: 2,
            color: [600.0, 600.0, 600.0],
            size: 0.0,
            dir: [0.0, 1.0, 0.0],
            cone,
            blend: (1.0 - cone) * 0.2,
            right,
            spot_scale,
            ..Light::default()
        };
        let cam = Camera { count: 1, lights, envmap: 0, env_cdf: 0, preview: [0; 4], env: [0.02, 0.02, 0.02, 1.0], ..camera() };
        render(Mode::Preview, cam, &scene, &atlas).0
    };
    // Half the pool's brightest, measured on the frame itself rather than
    // assumed: the lit run across the middle row and down the middle
    // column is the pool's width and height in pixels.
    let extent = |px: &[u8]| {
        let peak = (0..W).map(|x| luma(px, x, W / 2)).fold(0.0, f32::max);
        let lit = |v: f32| v > peak * 0.5;
        (
            (0..W).filter(|&x| lit(luma(px, x, W / 2))).count(),
            (0..W).filter(|&y| lit(luma(px, W / 2, y))).count(),
        )
    };
    let round = shot([1.0, 0.0, 0.0], [1.0, 1.0]);
    let (rw, rh) = extent(&round);
    assert!(rw > 2 && rw == rh, "an unscaled spot is round: {rw} x {rh}");
    // `[1, 1]` is the ellipse term's own round case, so it has to give the
    // same frame as a record with no axes and no scale at all — the guard
    // in `spot_mask`, and batch 147's plain smoothstep. The two agree to
    // the last level and no further: `inverseSqrt(1 + |xy|²/z²)` is `z`
    // only up to rounding.
    let apart = round.iter().zip(&shot([0.0; 3], [0.0; 2])).map(|(a, b)| a.abs_diff(*b)).max();
    assert!(apart <= Some(1), "[1, 1] is the round case: {apart:?} levels apart");
    // A third further along the squashed axis, to within a pixel of
    // `rw / 0.75`, and the other axis untouched.
    let far = (rw as f32 / 0.75).round() as usize;
    let (tw, th) = extent(&shot([1.0, 0.0, 0.0], [1.0, 0.75]));
    assert_eq!(tw, rw, "scaling Y leaves the width alone");
    assert!(th.abs_diff(far) <= 1, "a spot scaled in Y reaches 1/0.75 as far in Y: {th}, not {far}");
    let (ww, wh) = extent(&shot([1.0, 0.0, 0.0], [0.75, 1.0]));
    assert_eq!(wh, rh, "and in X, the height alone");
    assert!(ww.abs_diff(far) <= 1, "reaching as far in X: {ww}, not {far}");
}
