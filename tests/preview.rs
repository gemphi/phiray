//! Material Preview is deterministic: one dispatch is the frame, the same
//! frame every time, smooth where the surface is smooth — and a texture is
//! read at the level the pixel's footprint asks for.

use phiray::{Atlas, Camera, Light, Mode, Ray, Tracer, Tri};

const W: u32 = 48;

/// A sky bright above and dark below, with Material Preview's lighting.
fn sky_atlas() -> Atlas {
    let (w, h) = (32u32, 16u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for _ in 0..w {
            rgb.extend(if y < h / 2 { [3.0, 3.0, 3.0] } else { [0.05, 0.05, 0.05] });
        }
    }
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    let lighting = atlas.add_env_lighting(w, h, &rgb);
    assert_eq!((lighting.irradiance, lighting.prefiltered), (2, 3));
    atlas
}

fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        pivot: [0.0, 0.0, 0.0],
        dist: 4.0,
        units: 1.0,
        persp: 1,
        scale: 1.0,
        aspect: 1.0,
        sky: [0.24, 0.24, 0.24, 0.0],
        sun: [0.0, -0.7, 0.7],
        env: [1.0, 1.0, 1.0, 1.0],
        envmap: 1,
        env_cdf: 2,
        env_hidden: 1,
        preview: [3, 4, phiray::trace::ibl::LEVELS, 0],
        ..Default::default()
    }
}

/// A card facing the camera, `half` wide, at `y` (further from the camera
/// as `y` grows), its texture `tex` tiled `tiles` times across.
fn card(half: f32, y: f32, tex: Option<(u32, f32)>) -> Vec<Tri> {
    let n = [0.0, -1.0, 0.0];
    let q = [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]];
    let color = [0.8, 0.8, 0.8, 1.0];
    let (a, b) = (Tri::new([q[0], q[1], q[2]], [n; 3], color), Tri::new([q[0], q[2], q[3]], [n; 3], color));
    match tex {
        Some((id, tiles)) => vec![
            a.with_texture(id, [[0.0, 0.0], [tiles, 0.0], [tiles, tiles]]),
            b.with_texture(id, [[0.0, 0.0], [tiles, tiles], [0.0, tiles]]),
        ],
        None => vec![a, b],
    }
}

fn render(mode: Mode, cam: Camera, scene: &[Tri], atlas: &Atlas) -> (Vec<u8>, u32) {
    let gpu = Ray::gpu().unwrap();
    let mut tracer = Tracer::new(&gpu, scene, atlas).unwrap();
    tracer.set_mode(mode);
    tracer.set_samples(8);
    let mut passes = 0;
    let (mut px, mut done) = tracer.render(&gpu, cam, W, W);
    passes += 1;
    while !done {
        (px, done) = tracer.render(&gpu, cam, W, W);
        passes += 1;
    }
    (px, passes)
}

fn luma(px: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * f32::from(px[i]) + 0.7152 * f32::from(px[i + 1]) + 0.0722 * f32::from(px[i + 2])
}

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
        let mut lights = [Light::default(); phiray::LIGHTS];
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
        let mut lights = [Light::default(); phiray::LIGHTS];
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
        let mut lights = [Light::default(); phiray::LIGHTS];
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
        let mut lights = [Light::default(); phiray::LIGHTS];
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
