//! Emission, glass, emitter sampling and normal maps.

use phiray::{Atlas, Camera, Ray, Tracer, Tri};

#[test]
fn emission_glows_and_glass_shows_what_is_behind() {
    let Some(gpu) = Ray::gpu() else { return };
    let n = [0.0, -1.0, 0.0];
    let quad = |y: f32, c: [f32; 4]| {
        let q = [[-1.0, y, -1.0], [1.0, y, -1.0], [1.0, y, 1.0], [-1.0, y, 1.0]];
        [Tri::new([q[0], q[1], q[2]], [n; 3], c), Tri::new([q[0], q[2], q[3]], [n; 3], c)]
    };
    let mut cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0],
        units: 1.0, scale: 0.4, aspect: 1.0, sun: [0.0, -1.0, 0.0], ..Default::default()
    };
    let centre = |tr: &mut Tracer, cam: Camera| {
        let (mut px, mut done) = tr.render(&gpu, cam, 32, 32);
        while !done {
            (px, done) = tr.render(&gpu, cam, 32, 32);
        }
        let i = (16 * 32 + 16) * 4;
        [px[i], px[i + 1], px[i + 2]]
    };
    // A black, emissive orange card in a black world, no lights.
    cam.count = 1; // one zero-power light: no studio rig
    let glow: Vec<Tri> = quad(0.0, [0.0, 0.0, 0.0, 1.0]).iter().map(|t| t.with_optics([4.0, 1.5, 0.2], 0.0, 1.5)).collect();
    let c = centre(&mut Tracer::new(&gpu, &glow, &Atlas::default()).unwrap(), cam);
    assert!(c[0] > 200 && c[0] > c[2] + 60, "emission glows orange {c:?}");
    // A clear glass pane in front of a green emitter: the green shows through.
    let mut scene: Vec<Tri> = quad(0.0, [1.0; 4]).iter().map(|t| t.with_optics(vec3(0.0), 1.0, 1.0).with_pbr(0.02, 0.0)).collect();
    scene.extend(quad(1.0, [0.0, 0.0, 0.0, 1.0]).iter().map(|t| t.with_optics([0.2, 3.0, 0.2], 0.0, 1.5)));
    let c = centre(&mut Tracer::new(&gpu, &scene, &Atlas::default()).unwrap(), cam);
    assert!(c[1] > 150 && c[1] > c[0] + 60, "green through glass {c:?}");
}

fn vec3(v: f32) -> [f32; 3] {
    [v; 3]
}

#[test]
fn glowing_mesh_lights_its_surroundings() {
    let Some(gpu) = Ray::gpu() else { return };
    let n = [0.0, -1.0, 0.0];
    let quad = |q: [[f32; 3]; 4], c: [f32; 4]| [Tri::new([q[0], q[1], q[2]], [n; 3], c), Tri::new([q[0], q[2], q[3]], [n; 3], c)];
    // A white wall seen head-on, and a blue emitter panel beside it, in front
    // of the wall but outside the view.
    let wall = quad([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], [0.8, 0.8, 0.8, 1.0]);
    let lamp: Vec<Tri> = quad([[2.5, -4.0, -2.0], [2.5, 0.0, -2.0], [2.5, 0.0, 2.0], [2.5, -4.0, 2.0]], [0.0, 0.0, 0.0, 1.0])
        .iter()
        .map(|t| t.with_optics([0.2, 1.0, 12.0], 0.0, 1.5))
        .collect();
    let mut cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0],
        units: 1.0, scale: 0.4, aspect: 1.0, sun: [0.0, -1.0, 0.0], ..Default::default()
    };
    cam.count = 1; // one zero-power light: no studio rig, black world
    let centre = |tris: &[Tri]| {
        let mut tr = Tracer::new(&gpu, tris, &Atlas::default()).unwrap();
        let (mut px, mut done) = tr.render(&gpu, cam, 32, 32);
        while !done {
            (px, done) = tr.render(&gpu, cam, 32, 32);
        }
        let i = (16 * 32 + 16) * 4;
        [px[i], px[i + 1], px[i + 2]]
    };
    let dark = centre(&wall);
    let mut lit_scene = wall.to_vec();
    lit_scene.extend(lamp);
    let lit = centre(&lit_scene);
    assert!(dark[2] < 10, "no light, black wall {dark:?}");
    assert!(lit[2] > dark[2] + 25 && lit[2] > lit[0] + 15, "blue glow lights the wall {lit:?}");
    // Direct sampling of the emitter: one sample is already close to the result.
    let mut tr = Tracer::new(&gpu, &lit_scene, &Atlas::default()).unwrap();
    let (first, _) = tr.render(&gpu, cam, 32, 32);
    let mean = |px: &[u8]| {
        let v: Vec<f32> = (12..20).flat_map(|y| (12..20).map(move |x| ((y * 32 + x) * 4 + 2) as usize)).map(|i| px[i] as f32).collect();
        v.iter().sum::<f32>() / v.len() as f32
    };
    let one = mean(&first);
    assert!((one - lit[2] as f32).abs() < lit[2] as f32 * 0.3, "one sample {one} vs converged {}", lit[2]);
}

#[test]
fn bright_lamps_are_picked_more_often() {
    use phiray::Lamps;
    let tri = |s: f32, e: f32| Tri::new([[0.0; 3], [s, 0.0, 0.0], [0.0, s, 0.0]], [[0.0, 0.0, 1.0]; 3], [0.0; 4]).with_optics([e; 3], 0.0, 1.5);
    let lamps = Lamps::of(&[tri(1.0, 0.0), tri(2.0, 1.0), tri(1.0, 1.0), tri(1.0, 0.0)]);
    assert_eq!(lamps.ids, vec![1, 2]);
    assert!((lamps.cdf[0] - 0.8).abs() < 1e-5 && (lamps.cdf[1] - 1.0).abs() < 1e-6, "{:?}", lamps.cdf);

    // One strong panel plus many faint specks: still clean after one sample.
    let Some(gpu) = Ray::gpu() else { return };
    let n = [0.0, -1.0, 0.0];
    let quad = |q: [[f32; 3]; 4], c: [f32; 4]| [Tri::new([q[0], q[1], q[2]], [n; 3], c), Tri::new([q[0], q[2], q[3]], [n; 3], c)];
    let mut scene = quad([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], [0.8, 0.8, 0.8, 1.0]).to_vec();
    scene.extend(quad([[2.5, -4.0, -2.0], [2.5, 0.0, -2.0], [2.5, 0.0, 2.0], [2.5, -4.0, 2.0]], [0.0; 4]).iter().map(|t| t.with_optics([6.0; 3], 0.0, 1.5)));
    for k in 0..300 {
        let x = -30.0 - k as f32 * 0.1;
        scene.push(Tri::new([[x, 20.0, 0.0], [x + 0.01, 20.0, 0.0], [x, 20.0, 0.01]], [n; 3], [0.0; 4]).with_optics([0.5; 3], 0.0, 1.5));
    }
    let cam = Camera {
        right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0],
        units: 1.0, scale: 0.4, aspect: 1.0, sun: [0.0, -1.0, 0.0], count: 1, ..Default::default()
    };
    let mut tr = Tracer::new(&gpu, &scene, &Atlas::default()).unwrap();
    let (first, _) = tr.render(&gpu, cam, 32, 32);
    let (mut px, mut done) = (first.clone(), false);
    while !done {
        (px, done) = tr.render(&gpu, cam, 32, 32);
    }
    let mean = |px: &[u8]| (12..20).flat_map(|y| (12..20).map(move |x| px[((y * 32 + x) * 4) as usize] as f32)).sum::<f32>() / 64.0;
    let (one, all) = (mean(&first), mean(&px));
    assert!(all > 30.0 && (one - all).abs() < all * 0.3, "one sample {one} vs converged {all}");
}

#[test]
fn normal_map_bends_the_shading() {
    let Some(gpu) = Ray::gpu() else { return };
    // Map tilted towards +u (x): R=230 → x≈0.8, G=128 → y=0, B=160 → z≈0.25.
    let mut atlas = Atlas::default();
    let id = atlas.add(1, 1, &[230, 128, 160, 255]);
    let n = [0.0, -1.0, 0.0];
    let q = [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let grey = [0.6, 0.6, 0.6, 1.0];
    let plain = [Tri::new([q[0], q[1], q[2]], [n; 3], grey), Tri::new([q[0], q[2], q[3]], [n; 3], grey)];
    let bumped = [
        plain[0].with_normal_map(id, 1.0, [uv[0], uv[1], uv[2]]),
        plain[1].with_normal_map(id, 1.0, [uv[0], uv[2], uv[3]]),
    ];
    // Front view, sun from the right (+x): the tilted normal faces it more.
    let cam = |tris: &[Tri]| {
        let c = Camera {
            right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0],
            units: 1.0, scale: 0.4, aspect: 1.0, count: 1,
            ..Default::default()
        };
        let mut c = c;
        c.lights[0] = phiray::Light { kind: 0, color: [3.0; 3], dir: [-0.94, 0.34, 0.0], ..Default::default() };
        let mut tr = Tracer::new(&gpu, tris, &atlas).unwrap();
        let (mut px, mut done) = tr.render(&gpu, c, 16, 16);
        while !done {
            (px, done) = tr.render(&gpu, c, 16, 16);
        }
        px[(8 * 16 + 8) * 4]
    };
    let (flat, bent) = (cam(&plain), cam(&bumped));
    assert!(bent > flat + 30, "normal map turns the surface to the light: flat {flat}, bent {bent}");
}
