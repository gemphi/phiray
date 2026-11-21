//! What the BVH walk costs on a scene heavy enough for the walk to be the
//! cost: the GPU's own time for a sample, and a hash of the image, so two
//! versions of `bvh.wgsl` can be timed against each other and shown to draw
//! the same picture.
//!
//! The walk is read from disk (`phiray::hot`), so no rebuild stands between
//! two versions: point `PHIRAY_SHADERS` at a folder holding the kernel's
//! files and run again with another folder.
//!
//! Run: `PHIRAY_SHADERS=crates/phiray/src/trace/wgsl cargo run -p phiray --example walk_cost`

use phiray::{Atlas, Camera, Gpu, Mode, Ray, Tracer, Tri};

const W: u32 = 1280;
const H: u32 = 720;
/// Cells along a side of the height field: two triangles a cell.
const SIDE: usize = 700;
/// Samples before the clock starts (the pass recorded, the kernel compiled).
const WARM_COUNT: u32 = 2;
const SAMPLE_COUNT: u32 = 16;

/// The field's height at `(x, y)`: waves a ray can be shadowed by and
/// bounce between, so closest-hit and any-hit rays both have work.
fn height(x: f32, y: f32) -> f32 {
    0.12 * (9.0 * x).sin() * (7.0 * y).cos() + 0.03 * (41.0 * x + 3.0).sin() * (37.0 * y).sin()
}

fn point(i: usize, j: usize) -> [f32; 3] {
    let x = i as f32 / SIDE as f32 * 2.0 - 1.0;
    let y = j as f32 / SIDE as f32 * 2.0 - 1.0;
    [x, y, height(x, y)]
}

fn flat(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
    [n[0] / len, n[1] / len, n[2] / len]
}

fn field() -> Vec<Tri> {
    let albedo = [0.8, 0.6, 0.4, 1.0];
    let mut tris = Vec::with_capacity(SIDE * SIDE * 2);
    for j in 0..SIDE {
        for i in 0..SIDE {
            let (a, b, c, d) = (point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1));
            tris.push(Tri::new([a, b, c], [flat(a, b, c); 3], albedo));
            tris.push(Tri::new([a, c, d], [flat(a, c, d); 3], albedo));
        }
    }
    tris
}

/// Looking down on the field at 45 degrees.
fn camera() -> Camera {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, s, s],
        back: [0.0, -s, s],
        dist: 2.6,
        units: 1.0,
        persp: 1,
        scale: 1.0,
        aspect: W as f32 / H as f32,
        sky: [0.2, 0.2, 0.2, 0.0],
        sun: [0.3, -0.5, 0.6],
        env: [0.3, 0.3, 0.3, 1.0],
        bounces: 4,
        ..Default::default()
    }
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))
}

fn sample(tracer: &mut Tracer, gpu: &Gpu) -> f64 {
    tracer.sample(gpu, camera(), W, H);
    tracer.time().gpu.as_secs_f64() * 1e3
}

fn main() {
    let read = phiray::hot::reload();
    println!("walk from: {}", match (phiray::hot::live(), read.is_empty()) {
        (true, false) => std::env::var("PHIRAY_SHADERS").unwrap_or_default(),
        _ => "the binary's baked text".to_string(),
    });
    let Some(gpu) = Ray::gpu() else {
        println!("no GPU: nothing to measure");
        return;
    };
    let scene = field();
    println!("{} x {}, {} triangles, {} on {:?}, timestamps: {}", W, H, scene.len(), gpu.info.name, gpu.info.backend, gpu.timed);
    let mut tracer = Tracer::new(&gpu, &scene, &Atlas::default()).expect("a tracer for the field");
    tracer.set_mode(Mode::Rendered);
    tracer.set_samples(WARM_COUNT + SAMPLE_COUNT);
    for _ in 0..WARM_COUNT {
        sample(&mut tracer, &gpu);
    }
    let mut list: Vec<f64> = (0..SAMPLE_COUNT).map(|_| sample(&mut tracer, &gpu)).collect();
    list.sort_by(f64::total_cmp);
    let mean = list.iter().sum::<f64>() / list.len() as f64;
    println!("gpu ms a sample: median {:.2}, mean {:.2}, least {:.2}, most {:.2}", list[list.len() / 2], mean, list[0], list[list.len() - 1]);
    println!("image hash after {} samples: {:016x}", WARM_COUNT + SAMPLE_COUNT, fnv(&tracer.pixels(&gpu)));
}
