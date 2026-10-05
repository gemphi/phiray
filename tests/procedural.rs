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
