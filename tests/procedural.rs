//! Procedural textures per pixel (`procedural.wgsl`, Cycles' SVM): a
//! Noise or Voronoi texture on a material's base colour — through a colour
//! ramp or not — on a Mix Shader's weight, or on the normal through a
//! Bump node. A flat wall is no longer flat where one is set.

use phiray::{Atlas, Camera, Mode, Proc, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

/// A dim sky above, dark below, with Material Preview's lighting.
fn sky_atlas() -> Atlas {
    let (w, h) = (32u32, 16u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for _ in 0..w {
            rgb.extend(if y < h / 2 { [0.9, 0.9, 0.9] } else { [0.05, 0.05, 0.05] });
        }
    }
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    atlas.add_env_lighting(w, h, &rgb);
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

/// A texture at a scale that shows: Generated coordinates run 0 ..= 1 over
/// the wall, and the frame sees about a twentieth of it, so Blender's default scale
/// of 5 is one slow cell across the frame — as it would be in Blender.
fn fine(proc: Proc) -> Proc {
    Proc { scale: 40.0, ..proc }
}

/// A wall filling the frame, facing the camera.
fn wall(color: [f32; 4]) -> Tri {
    let n = [0.0, -1.0, 0.0];
    Tri::new([[-9.0, 0.0, -9.0], [9.0, 0.0, -9.0], [0.0, 0.0, 12.0]], [n; 3], color).with_pbr(0.8, 0.0)
}

fn frame(gpu: &phiray::Gpu, scene: Scene, mode: Mode) -> Vec<u8> {
    let mut tr = Tracer::from_scene(gpu, scene).expect("scene");
    tr.set_mode(mode);
    // Material Preview settles at once; the path tracer's noise has to
    // fall below the textures' variation.
    tr.set_samples(if mode == Mode::Preview { 1 } else { 512 });
    let (mut px, mut done) = tr.render(gpu, camera(), W, W);
    while !done {
        (px, done) = tr.render(gpu, camera(), W, W);
    }
    px
}

/// Mean and standard deviation of the frame's luma.
fn spread(px: &[u8]) -> (f32, f32) {
    let l: Vec<f32> = px.chunks(4).map(|p| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32).collect();
    let mean = l.iter().sum::<f32>() / l.len() as f32;
    let var = l.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l.len() as f32;
    (mean, var.sqrt())
}

#[test]
fn noise_and_voronoi_vary_a_flat_wall_and_a_ramp_colours_it() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    for mode in [Mode::Preview, Mode::Rendered] {
        let plain = frame(&gpu, Scene::from_tris(&[wall([0.6, 0.6, 0.6, 1.0])], &sky_atlas()), mode);
        let (_, flat) = spread(&plain);
        assert!(flat < 4.0, "{mode:?}: a plain wall is flat: {flat}");
        // Noise on the base colour: the wall varies across the frame, grey.
        let noisy = wall([0.6, 0.6, 0.6, 1.0]).with_procs(Some(fine(Proc::noise(1))), None);
        let px = frame(&gpu, Scene::from_tris(&[noisy], &sky_atlas()), mode);
        let (mean, dev) = spread(&px);
        assert!(dev > flat + 4.0, "{mode:?}: the noise shows: deviation {dev} over a flat {flat}");
        assert!(mean > 30.0 && mean < 230.0, "{mode:?}: mean {mean}");
        assert!(px.chunks(4).all(|p| (p[0] as i32 - p[1] as i32).abs() <= 3), "{mode:?}: Fac is grey");
        // Noise Color: three channels, not grey.
        let colored = wall([0.6, 0.6, 0.6, 1.0]).with_procs(Some(fine(Proc { kind: 2, ..Proc::noise(1) })), None);
        let px = frame(&gpu, Scene::from_tris(&[colored], &sky_atlas()), mode);
        assert!(px.chunks(4).any(|p| (p[0] as i32 - p[1] as i32).abs() > 20), "{mode:?}: Color is coloured");
        // Voronoi Distance: cells, varying too.
        let cells = wall([0.6, 0.6, 0.6, 1.0]).with_procs(Some(fine(Proc::voronoi(1))), None);
        let (_, dev) = spread(&frame(&gpu, Scene::from_tris(&[cells], &sky_atlas()), mode));
        assert!(dev > flat + 4.0, "{mode:?}: the cells show: deviation {dev} over a flat {flat}");
        // A ramp from red to blue over the noise: every pixel on that line.
        let mut atlas = sky_atlas();
        let ramp: Vec<u8> = (0..256).flat_map(|i| [255 - i as u8, 0, i as u8, 255]).collect();
        let ramp = atlas.add(256, 1, &ramp);
        let ramped = wall([1.0; 4]).with_procs(Some(fine(Proc { ramp, ..Proc::noise(1) })), None);
        let px = frame(&gpu, Scene::from_tris(&[ramped], &sky_atlas_with(atlas)), mode);
        // The lighting's own green — the white sky off a dielectric's
        // specular — is what a plain red wall shows; the ramp adds none.
        let lit = frame(&gpu, Scene::from_tris(&[wall([1.0, 0.0, 0.0, 1.0])], &sky_atlas()), mode);
        let green = lit.chunks(4).map(|p| p[1]).max().unwrap();
        assert!(px.chunks(4).all(|p| p[1] <= green + 6), "{mode:?}: no green on a red–blue ramp beyond the lighting's {green}");
        // A Noise's Fac keeps to the middle of 0 ..= 1, so the wall is
        // purple leaning either way, not red here and blue there.
        assert!(px.chunks(4).any(|p| p[0] > p[2] + 10) && px.chunks(4).any(|p| p[2] > p[0] + 10), "{mode:?}: both sides of the ramp's middle show");
    }
}

/// `sky_atlas` with more images added.
fn sky_atlas_with(atlas: Atlas) -> Atlas {
    atlas
}

#[test]
fn a_procedural_mix_weight_and_a_bump_vary_by_pixel() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    for mode in [Mode::Preview, Mode::Rendered] {
        // Mix(red, white) by a Noise: between the two, and varying where a
        // constant weight is flat.
        let red = wall([0.8, 0.1, 0.1, 1.0]);
        let white = wall([0.9, 0.9, 0.9, 1.0]);
        let mut constant = Scene::new(sky_atlas());
        let other = constant.material(phiray::Mat::of(&white));
        constant.add(&red.with_second(other, 0.5));
        let (mean_c, dev_c) = spread(&frame(&gpu, constant, mode));
        let mut varying = Scene::new(sky_atlas());
        let other = varying.material(phiray::Mat::of(&white));
        varying.add(&red.with_second(other, 0.5).with_procs(Some(fine(Proc::noise(2))), None));
        let (mean_v, dev_v) = spread(&frame(&gpu, varying, mode));
        assert!(dev_v > dev_c + 3.0, "{mode:?}: the weight varies by pixel: {dev_v} vs {dev_c}");
        assert!((mean_v - mean_c).abs() < 40.0, "{mode:?}: about the same mix overall: {mean_v} vs {mean_c}");

        // A Bump from a Noise on a flat wall: the shading varies — gently
        // under Material Preview's soft sky, which hardly minds a tilt.
        let (_, flat) = spread(&frame(&gpu, Scene::from_tris(&[wall([0.6, 0.6, 0.6, 1.0])], &sky_atlas()), mode));
        let bumped = wall([0.6, 0.6, 0.6, 1.0]).with_procs(None, Some(fine(Proc::noise(0)).bump(1.0, 1.0)));
        let (_, dev) = spread(&frame(&gpu, Scene::from_tris(&[bumped], &sky_atlas()), mode));
        assert!(dev > flat + 1.0, "{mode:?}: the bump shows: deviation {dev} over a flat {flat}");
    }
}

/// A wide camera: the frame spans about ±46°, so the angle between the
/// wall and the ray runs from head on at the centre to well past 40° in
/// the corners — which is what a view-dependent slot answers to.
fn wide_camera() -> Camera {
    Camera { scale: 0.12, ..camera() }
}

/// A wall big enough to fill the wide frame, as two triangles.
fn big_wall(color: [f32; 4]) -> [Tri; 2] {
    let n = [0.0, -1.0, 0.0];
    let q = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| Tri::new([a, b, c], [n; 3], color).with_pbr(0.8, 0.0);
    let (lo, hi) = (-20.0f32, 20.0f32);
    [
        q([lo, 0.0, lo], [hi, 0.0, lo], [hi, 0.0, hi]),
        q([lo, 0.0, lo], [hi, 0.0, hi], [lo, 0.0, hi]),
    ]
}

/// The mean luma of the frame's middle `n` × `n` block and of its four
/// corner blocks — the head-on pixels against the grazing ones.
fn centre_and_corners(px: &[u8], n: u32) -> (f32, f32) {
    let luma = |x: u32, y: u32| {
        let p = &px[((y * W + x) * 4) as usize..];
        0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
    };
    let block = |x0: u32, y0: u32| {
        let mut sum = 0.0;
        for y in y0..y0 + n {
            for x in x0..x0 + n {
                sum += luma(x, y);
            }
        }
        sum / (n * n) as f32
    };
    let mid = (W - n) / 2;
    let corners = [block(0, 0), block(W - n, 0), block(0, W - n), block(W - n, W - n)];
    (block(mid, mid), corners.iter().sum::<f32>() / 4.0)
}

/// A Fresnel and a Layer Weight on the mix weight are the surface's angle
/// to the ray, not a texture: the wall takes the second closure at its
/// grazing corners and the first where it faces the camera
/// (`svm_node_fresnel`, `svm_node_layer_weight`; the CPU twin and the
/// hemispherical means are `phiray::fresnel`). A constant weight has no
/// such gradient, which is what tells this apart from vignetting.
#[test]
fn a_fresnel_and_a_layer_weight_mix_by_the_angle_to_the_camera() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let dark = [0.05, 0.02, 0.02, 1.0];
    let light = [0.95, 0.95, 0.95, 1.0];
    let scene = |slot: Option<Proc>, mix: f32| {
        let mut scene = Scene::new(sky_atlas());
        let other = scene.material(phiray::Mat::of(&big_wall(light)[0]));
        for t in big_wall(dark) {
            scene.add(&t.with_second(other, mix).with_procs(slot, None));
        }
        scene
    };
    let shot = |slot: Option<Proc>, mix: f32| {
        let mut tr = Tracer::from_scene(&gpu, scene(slot, mix)).expect("scene");
        tr.set_mode(Mode::Preview);
        tr.set_samples(1);
        let (px, _) = tr.render(&gpu, wide_camera(), W, W);
        centre_and_corners(&px, 8)
    };
    // A constant weight of a half is the same mix everywhere.
    let (flat_mid, flat_corner) = shot(None, 0.5);
    assert!((flat_corner - flat_mid).abs() < 12.0, "a constant weight is flat: centre {flat_mid}, corners {flat_corner}");
    // Facing at Blend 0.9 raises the cosine to the fifth: nothing head on,
    // most of the second closure at the corners.
    let (mid, corner) = shot(Some(Proc::layer_weight(2, 0.9, true)), 0.0);
    assert!(corner > mid + 25.0, "Facing brightens the corners: centre {mid}, corners {corner}");
    assert!(mid < flat_mid, "head on it is the first closure: {mid} vs {flat_mid}");
    // A Fresnel at IOR 1.8 is the same shape, gentler.
    let (f_mid, f_corner) = shot(Some(Proc::fresnel(2, 1.8)), 0.0);
    assert!(f_corner > f_mid + 3.0, "Fresnel brightens the corners: centre {f_mid}, corners {f_corner}");
    // And the Fresnel output of a Layer Weight runs the same way.
    let (l_mid, l_corner) = shot(Some(Proc::layer_weight(2, 0.5, false)), 0.0);
    assert!(l_corner > l_mid + 3.0, "Layer Weight Fresnel brightens the corners: centre {l_mid}, corners {l_corner}");
}
