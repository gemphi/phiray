//! Rendered mode smoke test: a lit box over a floor, seen from the front-top.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

fn quad(p: [[f32; 3]; 4], n: [f32; 3], c: [f32; 4]) -> [Tri; 2] {
    [Tri::new([p[0], p[1], p[2]], [n; 3], c), Tri::new([p[0], p[2], p[3]], [n; 3], c)]
}

fn scene() -> Vec<Tri> {
    let red = [0.8, 0.2, 0.2, 1.0];
    let mut t = quad([[-5., -5., 0.], [5., -5., 0.], [5., 5., 0.], [-5., 5., 0.]], [0., 0., 1.], [0.7, 0.7, 0.7, 1.0]).to_vec();
    let (l, h) = (-1.0f32, 1.0f32);
    let z = |a: f32| a + 1.0; // box sits on the floor
    let faces = [
        ([[l, l, z(h)], [h, l, z(h)], [h, h, z(h)], [l, h, z(h)]], [0., 0., 1.]),
        ([[l, l, z(l)], [h, l, z(l)], [h, l, z(h)], [l, l, z(h)]], [0., -1., 0.]),
        ([[h, l, z(l)], [h, h, z(l)], [h, h, z(h)], [h, l, z(h)]], [1., 0., 0.]),
        ([[l, h, z(l)], [l, l, z(l)], [l, l, z(h)], [l, h, z(h)]], [-1., 0., 0.]),
        ([[h, h, z(l)], [l, h, z(l)], [l, h, z(h)], [h, h, z(h)]], [0., 1., 0.]),
    ];
    for (p, n) in faces {
        t.extend(quad(p, n, red));
    }
    t
}

/// Blender Front view pitched down 30°: right = +X, back = towards −Y and up.
fn camera(persp: bool) -> Camera {
    let (sp, cp) = (30f32.to_radians().sin(), 30f32.to_radians().cos());
    let sun = [0.4f32, -0.5, 0.77];
    let l = (sun.iter().map(|v| v * v).sum::<f32>()).sqrt();
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, sp, cp],
        back: [0.0, -cp, sp],
        pivot: [0.0, 0.0, 1.0],
        units: 1.0,
        scale: 0.08,
        aspect: 1.0,
        persp: persp as u32,
        dist: 1.0 / (0.08 * 1.44),
        sun: sun.map(|v| v / l),
        sky: [0.25, 0.27, 0.3, 1.0],
        env: [0.3, 0.32, 0.36, 1.0],
        grid: 1.0,
        grid_color: [0.5, 0.5, 0.5, 0.4],
        ..Default::default()
    }
}

#[test]
fn renders_box_floor_and_sky() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &scene(), &Atlas::default()).expect("scene");
    for persp in [false, true] {
        let (w, h) = (160u32, 120u32);
        let mut frames = 1;
        let (mut px, mut done) = tr.render(&gpu, camera(persp), w, h);
        while !done {
            (px, done) = tr.render(&gpu, camera(persp), w, h);
            frames += 1;
        }
        assert_eq!(frames, phiray::Tracer::SAMPLES);
        assert_eq!(px.len(), (w * h * 4) as usize);
        let at = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2]]
        };
        let centre = at(w / 2, h / 2);
        assert!(centre[0] > centre[1] + 30, "box is red: {centre:?}");
        let top = at(2, 2);
        assert!(top[0].abs_diff(top[2]) < 40 && top[2] > 40, "sky at the top: {top:?}");
        if let Ok(dir) = std::env::var("PHIRAY_DUMP") {
            let mut ppm = format!("P6 {w} {h} 255\n").into_bytes();
            ppm.extend(px.chunks(4).flat_map(|c| [c[0], c[1], c[2]]));
            std::fs::write(format!("{dir}/trace_{persp}.ppm"), ppm).expect("dump");
        }
    }
}

#[test]
fn texture_is_sampled_per_hit() {
    let Some(gpu) = Ray::gpu() else { return };
    // 2×2 texture: top-left red, top-right green, bottom-left blue, bottom-right white.
    let mut atlas = Atlas::default();
    let id = atlas.add(2, 2, &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
    let n = [0.0, -1.0, 0.0];
    let q = [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let white = [1.0; 4];
    let tris = [
        Tri::new([q[0], q[1], q[2]], [n; 3], white).with_texture(id, [uv[0], uv[1], uv[2]]),
        Tri::new([q[0], q[2], q[3]], [n; 3], white).with_texture(id, [uv[0], uv[2], uv[3]]),
    ];
    let mut tr = Tracer::new(&gpu, &tris, &atlas).expect("tris");
    let cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0],
        units: 1.0, scale: 0.4, aspect: 1.0, sun: [0.0, -1.0, 0.0],
        env: [0.5; 4], count: 0, ..Default::default()
    };
    let (px, _) = tr.render(&gpu, cam, 64, 64);
    let at = |x: u32, y: u32| { let i = ((y * 64 + x) * 4) as usize; [px[i], px[i + 1], px[i + 2]] };
    let (tl, tr_, bl) = (at(22, 22), at(42, 22), at(22, 42));
    assert!(tl[0] > tl[1] + 40 && tl[0] > tl[2] + 40, "top-left red {tl:?}");
    assert!(tr_[1] > tr_[0] + 40, "top-right green {tr_:?}");
    assert!(bl[2] > bl[0] + 40, "bottom-left blue {bl:?}");
}

#[test]
fn many_file_lights_light_the_scene() {
    use phiray::Light;
    let Some(gpu) = Ray::gpu() else { return };
    let mut tr = Tracer::new(&gpu, &scene(), &Atlas::default()).expect("scene");
    let mut cam = camera(false);
    cam.env = [0.0; 4];
    // Six coloured point lights above the box; only the red ones are strong.
    for k in 0..6 {
        let red = k % 2 == 0;
        cam.lights[k] = Light {
            pos: [(k as f32 - 2.5) * 1.5, -3.0, 4.0],
            kind: 1,
            color: if red { [4000.0, 0.0, 0.0] } else { [0.0, 0.0, 30.0] },
            size: 0.2,
            ..Default::default()
        };
    }
    cam.count = 6;
    let (w, h) = (80u32, 60u32);
    let (mut px, mut done) = tr.render(&gpu, cam, w, h);
    while !done {
        (px, done) = tr.render(&gpu, cam, w, h);
    }
    let i = ((h / 2 * w + w / 2) * 4) as usize;
    let floor = ((h * 3 / 4 * w + w / 8) * 4) as usize;
    assert!(px[i] > 60, "lit box {:?}", &px[i..i + 3]);
    assert!(px[floor] > px[floor + 1] + 20, "floor lit red {:?}", &px[floor..floor + 3]);
}

#[test]
fn split_buffers_render_the_same() {
    use phiray::Scene;
    let Some(gpu) = Ray::gpu() else { return };
    // A 5 × 5 grid seen from above: 50 faces, 36 shared vertices.
    let mut tris = Vec::new();
    for (i, j) in (0..5).flat_map(|i| (0..5).map(move |j| (i as f32, j as f32))) {
        let q = [[i, j, 0.0], [i + 1.0, j, 0.0], [i + 1.0, j + 1.0, 0.0], [i, j + 1.0, 0.0]];
        let c = if ((i + j) as u32).is_multiple_of(2) { [0.8, 0.3, 0.2, 1.0] } else { [0.2, 0.3, 0.8, 1.0] };
        tris.extend(quad(q, [0.0, 0.0, 1.0], c));
    }
    let cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 1.0, 0.0], back: [0.0, 0.0, 1.0], pivot: [2.5, 2.5, 0.0],
        units: 1.0, scale: 0.2, aspect: 1.0, sun: [0.3, 0.2, 0.93], env: [0.3; 4], ..Default::default()
    };
    let render = |limit: u64, chunks: usize| {
        let scene = Scene::from_tris(&tris, &Atlas::default());
        let mut tr = Tracer::with_chunks(&gpu, scene, limit, chunks).expect("fits in the chunks");
        assert_eq!(tr.chunks(), chunks);
        if limit == 1088 {
            assert_eq!(tr.split()[1..3], [50, 34], "the faces whole, the vertices split 34 to a binding");
        }
        let (mut px, mut done) = tr.render(&gpu, cam, 40, 40);
        while !done {
            (px, done) = tr.render(&gpu, cam, 40, 40);
        }
        px
    };
    // 1088 bytes — over the data table's floor (1068): the 36 vertices (34
    // per binding) span two bindings; the 50 faces fit one.
    let whole = render(1 << 20, 2);
    assert_eq!(whole, render(1088, 2));
    assert!(whole.chunks(4).any(|p| p[0] > 100), "grid is visible");
    // Beyond twice the limit two bindings decline (the painter falls back) …
    let refused = |limit, chunks| Tracer::with_chunks(&gpu, Scene::from_tris(&tris, &Atlas::default()), limit, chunks).is_none();
    assert!(refused(200, 2));
    // … while four bindings (when the device has them) still fit, down to a quarter.
    if Tracer::chunks_for(&gpu) == 4 {
        assert_eq!(whole, render(1 << 20, 4));
        // 1068 bytes: the data table — the two materials (112 words each),
        // the empty atlas's word and the one instance's record (41 words) —
        // is the tightest binding.
        assert_eq!(whole, render(1068, 4));
        assert!(refused(1064, 4), "the data table does not fit");
    }
}
