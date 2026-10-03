//! X-ray (Alt Z): Workbench sees through every surface at the X-ray alpha,
//! blended as Blender blends them (weighted blended order-independent
//! transparency), and the wireframe shows the edges behind the faces.
//!
//! The Solid frames are held to Blender's own: `truth/xray.txt` is Blender
//! 5.0's Workbench render of the same two walls from the same camera
//! (`truth/blender_xray.py`) — a red wall in front of a blue one, X-ray off,
//! at 0.5 and at 0.2 — read where the front wall is seen alone, where both
//! are, where the back one is and where neither is.

use phiray::{Atlas, Camera, Mode, Ray, Tracer, Tri};

const W: u32 = 80;
const H: u32 = 60;
const TRUTH: &str = include_str!("truth/xray.txt");

/// The spots `blender_xray.py` reads, `(x, y)` from the top.
fn spot(name: &str) -> (u32, u32) {
    match name {
        "front" => (17, 30),
        "both" => (40, 30),
        "back" => (60, 30),
        _ => (2, 2),
    }
}

/// A wall at depth `y` over `x` × `z`, facing the camera (−Y).
/// Roughness 0.4, as Workbench gives an object's colour (`Material(color)`).
fn wall(x: (f32, f32), z: (f32, f32), y: f32, color: [f32; 4]) -> [Tri; 2] {
    let p = [[x.0, y, z.0], [x.1, y, z.0], [x.1, y, z.1], [x.0, y, z.1]];
    let n = [[0.0, -1.0, 0.0]; 3];
    [Tri::new([p[0], p[1], p[2]], n, color).with_pbr(0.4, 0.0), Tri::new([p[0], p[2], p[3]], n, color).with_pbr(0.4, 0.0)]
}

fn walls() -> Vec<Tri> {
    let mut scene = wall((-2.5, 1.0), (-1.5, 1.5), 0.0, [0.8, 0.2, 0.2, 1.0]).to_vec();
    scene.extend(wall((-1.0, 3.0), (-1.2, 1.2), 2.0, [0.2, 0.3, 0.8, 1.0]));
    scene
}

/// Blender's camera: at (0, −10, 0) looking along +Y, 50 mm on a 36 mm
/// sensor (`2 tan(fov/2) = 0.72`), the world (0.05) behind the walls.
fn camera(see_through: f32) -> Camera {
    let (scale, fov) = (0.1, 36.0 / 50.0);
    let dist = 1.0 / (scale * fov);
    let world = 1.055 * 0.05f32.powf(1.0 / 2.4) - 0.055;
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        pivot: [0.0, -10.0 + dist, 0.0],
        units: 1.0,
        scale,
        aspect: W as f32 / H as f32,
        persp: 1,
        dist,
        sky: [world, world, world, 0.0],
        env: [world, world, world, 1.0],
        grid: 0.0,
        see_through,
        ..Default::default()
    }
}

fn pixel(px: &[u8], (x, y): (u32, u32)) -> [i32; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i] as i32, px[i + 1] as i32, px[i + 2] as i32]
}

#[test]
fn xray_blends_the_walls_as_blender_does() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &walls(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Solid);
    let mut wrong = Vec::new();
    let mut frames = std::collections::HashMap::new();
    for line in TRUTH.lines().filter(|l| !l.starts_with('#')) {
        let w: Vec<&str> = line.split_whitespace().collect();
        let alpha: f32 = w[0].parse().unwrap();
        let want = [w[2], w[3], w[4]].map(|v| v.parse::<i32>().unwrap());
        let px = frames.entry(w[0].to_string()).or_insert_with(|| tr.render(&gpu, camera(1.0 - alpha), W, H).0);
        let ours = pixel(px, spot(w[1]));
        if ours.iter().zip(want).any(|(a, b)| (a - b).abs() > 2) {
            wrong.push(format!("alpha {alpha} {}: ours {ours:?}, Blender's {want:?}", w[1]));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// White wires on the sky, looking straight at a band in front of a wall.
fn wire_scene() -> (Vec<Tri>, Camera) {
    let mut scene = wall((-3.0, 3.0), (-3.0, 3.0), 0.0, [0.6, 0.6, 0.6, 1.0]).to_vec();
    scene.extend(wall((-3.0, 3.0), (-2.0, 2.0), -2.0, [0.6, 0.6, 0.6, 1.0]));
    let cam = Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        units: 1.0,
        scale: 0.1,
        aspect: 1.0,
        dist: 10.0,
        sky: [0.2, 0.22, 0.25, 0.0],
        env: [1.0, 1.0, 1.0, 1.0],
        ..Default::default()
    };
    (scene, cam)
}

#[test]
fn xray_wireframe_shows_the_edge_behind_a_face_and_fades_it() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let (scene, cam) = wire_scene();
    let mut tr = Tracer::new(&gpu, &scene, &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Wire);
    const WS: u32 = 64;
    let luma = |px: &[u8], x: u32, y: u32| {
        let i = ((y * WS + x) * 4) as usize;
        0.2126 * px[i] as f32 + 0.7152 * px[i + 1] as f32 + 0.0722 * px[i + 2] as f32
    };
    // World (1.5, 1.5) is on the wall's diagonal, behind the band.
    let (x, y) = (WS / 2 + 10, WS / 2 - 10);
    let sky = |px: &[u8]| luma(px, 1, y);
    let (hidden, _) = tr.render(&gpu, cam, WS, WS);
    assert!(luma(&hidden, x, y) < sky(&hidden) + 10.0, "X-ray off: the edge behind the band is hidden");
    // Blender's default: X-ray on at alpha 0 — every edge, whole.
    let (whole, _) = tr.render(&gpu, Camera { see_through: 1.0, ..cam }, WS, WS);
    let full = luma(&whole, x, y) - sky(&whole);
    assert!(full > 80.0, "X-ray: the edge behind the band draws ({full})");
    // Alpha 0.5: behind the band it draws at half.
    let (half, _) = tr.render(&gpu, Camera { see_through: 0.5, ..cam }, WS, WS);
    let faded = luma(&half, x, y) - sky(&half);
    assert!((faded - full * 0.5).abs() < full * 0.15, "faded to half: {faded} of {full}");
}
