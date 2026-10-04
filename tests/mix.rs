//! A Mix Shader's two closures are two lobes (`Mat::second`, `mix`): the
//! path tracer shades one of them per hit, picked by the weight, so the
//! converged frame is the weighted mean of the two shaded alone; Material
//! Preview shades both and lerps, so its frame is that mean exactly.

use phiray::{Atlas, Camera, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 32;

/// A sky light above and dark below — dim enough that a mirror of it does
/// not clip — with Material Preview's lighting.
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

/// A wall filling the frame, tilted so the mirror sees the bright sky.
fn wall(color: [f32; 4], rough: f32, metal: f32) -> Tri {
    let n = [0.0, -0.8, 0.6];
    Tri::new([[-9.0, 0.0, -9.0], [9.0, 0.0, -9.0], [0.0, 0.0, 12.0]], [n; 3], color).with_pbr(rough, metal)
}

/// The wall's red diffuse and white mirror, as one material of two lobes
/// at `mix`, or either alone.
fn scene(mix: Option<f32>, which: usize) -> Scene {
    let red = wall([0.8, 0.1, 0.1, 1.0], 1.0, 0.0);
    let mirror = wall([1.0, 1.0, 1.0, 1.0], 0.0, 1.0);
    let mut s = Scene::new(sky_atlas());
    match mix {
        Some(w) => {
            let other = s.material(phiray::Mat::of(&mirror));
            s.add(&red.with_second(other, w));
        }
        None => s.add(&[red, mirror][which]),
    }
    s
}

/// The frame's mean colour in scene-linear units (the display transform
/// undone — the default view is sRGB), scaled to 0 ..= 255: the lobes mix
/// in radiance, before the transform, so that is where the mean holds.
fn mean(px: &[u8]) -> [f32; 3] {
    let linear = |b: u8| {
        let c = b as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let n = (px.len() / 4) as f32;
    let mut m = [0.0; 3];
    for p in px.chunks(4) {
        for k in 0..3 {
            m[k] += 255.0 * linear(p[k]) / n;
        }
    }
    m
}

fn settled(gpu: &phiray::Gpu, s: Scene, mode: Mode, samples: u32) -> Vec<u8> {
    let mut tr = Tracer::from_scene(gpu, s).expect("scene");
    tr.set_mode(mode);
    tr.set_samples(samples);
    let (mut px, mut done) = tr.render(gpu, camera(), W, W);
    while !done {
        (px, done) = tr.render(gpu, camera(), W, W);
    }
    px
}

#[test]
fn a_mix_of_two_lobes_is_the_weighted_mean_of_the_two() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    for mode in [Mode::Preview, Mode::Rendered] {
        let samples = if mode == Mode::Preview { 1 } else { 256 };
        let red = mean(&settled(&gpu, scene(None, 0), mode, samples));
        let mirror = mean(&settled(&gpu, scene(None, 1), mode, samples));
        assert!((red[0] - mirror[0]).abs() > 20.0 || (red[1] - mirror[1]).abs() > 20.0, "{mode:?}: the two lobes look different: {red:?} {mirror:?}");
        let mixed = mean(&settled(&gpu, scene(Some(0.5), 0), mode, samples));
        // Preview lerps the radiance exactly; the path tracer's pick is a
        // coin per hit, so its mean carries the noise of 256 samples.
        let tolerance = if mode == Mode::Preview { 6.0 } else { 16.0 };
        for k in 0..3 {
            let expect = 0.5 * (red[k] + mirror[k]);
            assert!((mixed[k] - expect).abs() < tolerance, "{mode:?} channel {k}: mixed {} vs mean {} of {} and {}", mixed[k], expect, red[k], mirror[k]);
        }
        // All one lobe at either end of the weight.
        let none = mean(&settled(&gpu, scene(Some(0.0), 0), mode, samples));
        let all = mean(&settled(&gpu, scene(Some(1.0), 0), mode, samples));
        for k in 0..3 {
            assert!((none[k] - red[k]).abs() < 6.0 && (all[k] - mirror[k]).abs() < 6.0, "{mode:?}: weight 0 is the first lobe, 1 the second: {none:?} {all:?}");
        }
    }
}
