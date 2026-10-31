//! What a sample of a trivial scene costs, as a function of how many
//! slices it is cut into.
//!
//! `Tracer::sample_in` records one command buffer a slice (`Pass::run_step`)
//! and then drains the whole device (`Gpu::wait`) before the next, so a
//! sample's cost has a part that is tracing and a part that is `slices` ×
//! (submit + drain). The second part does not care what is in the scene.
//! A twelve-triangle cube makes the tracing part as small as it goes, so
//! whatever is left over at 32 or 64 slices is the overhead — and if that
//! is most of it, a cube and a heavy scene cost about the same, which is
//! exactly the complaint this example exists to settle.
//!
//! Run: `cargo run --release -p phiray --example slice_cost`

use phiray::{Atlas, Camera, Gpu, Mode, Ray, Tracer, Tri};
use std::time::Instant;

/// 1920 × 1080 — what a final render actually runs at, so the per-slice
/// cost is weighed against a real sample's worth of tracing.
const W: u32 = 1920;
const H: u32 = 1080;
/// Samples timed for each slice count, after a warm-up sample that pays
/// for recording the pass and compiling the pipeline.
const SAMPLES: u32 = 8;
const CUTS: [u32; 6] = [1, 2, 4, 16, 32, 64];

/// A unit cube at the origin: twelve triangles, flat normals, one colour.
fn cube() -> Vec<Tri> {
    let v = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    // Each face as two triangles, wound outwards, with the face's normal.
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0, 3, 2, 1], [0.0, 0.0, -1.0]),
        ([4, 5, 6, 7], [0.0, 0.0, 1.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ([3, 7, 6, 2], [0.0, 1.0, 0.0]),
        ([0, 4, 7, 3], [-1.0, 0.0, 0.0]),
        ([1, 2, 6, 5], [1.0, 0.0, 0.0]),
    ];
    let albedo = [0.8, 0.3, 0.2, 1.0];
    let mut tris = Vec::with_capacity(12);
    for (q, n) in faces {
        tris.push(Tri::new([v[q[0]], v[q[1]], v[q[2]]], [n; 3], albedo));
        tris.push(Tri::new([v[q[0]], v[q[2]], v[q[3]]], [n; 3], albedo));
    }
    tris
}

fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        dist: 6.0,
        units: 1.0,
        persp: 1,
        scale: 1.0,
        aspect: W as f32 / H as f32,
        sky: [0.2, 0.2, 0.2, 0.0],
        sun: [0.0, -0.7, 0.7],
        env: [0.3, 0.3, 0.3, 1.0],
        ..Default::default()
    }
}

/// `SAMPLES` samples at `cuts` slices each: the wall time a sample took,
/// and how much of it the process spent inside `Gpu::wait`.
fn time(gpu: &Gpu, cuts: u32) -> (f64, f64, u32) {
    let scene = cube();
    let mut tracer = Tracer::new(gpu, &scene, &Atlas::default()).expect("a tracer for the cube");
    tracer.set_mode(Mode::Rendered);
    tracer.set_samples(SAMPLES + 8);
    // Warm-up: records the pass for this slice count and compiles the
    // kernel, neither of which a later sample pays for again.
    tracer.sample_in(gpu, camera(), W, H, cuts, || true);
    gpu.wait();

    let (ns0, n0) = Gpu::waited();
    let at = Instant::now();
    for _ in 0..SAMPLES {
        tracer.sample_in(gpu, camera(), W, H, cuts, || true);
        gpu.wait();
    }
    let wall = at.elapsed().as_secs_f64() * 1e3;
    let (ns1, n1) = Gpu::waited();
    let waited = (ns1 - ns0) as f64 / 1e6;
    (wall / f64::from(SAMPLES), waited / f64::from(SAMPLES), (n1 - n0) / SAMPLES)
}

fn main() {
    let Some(gpu) = Ray::gpu() else {
        println!("no GPU: nothing to measure");
        return;
    };
    println!("{} x {}, a 12-triangle cube, {} samples timed per row\n", W, H, SAMPLES);
    println!("{:>6}  {:>12}  {:>12}  {:>7}  {:>6}", "slices", "ms/sample", "ms in wait", "waits", "wait%");
    let mut first = 0.0;
    for cuts in CUTS {
        let (ms, waited, waits) = time(&gpu, cuts);
        if cuts == 1 {
            first = ms;
        }
        let pct = if ms > 0.0 { waited / ms * 100.0 } else { 0.0 };
        println!("{:>6}  {:>12.1}  {:>12.1}  {:>7}  {:>5.1}%", cuts, ms, waited, waits, pct);
        if cuts == CUTS[CUTS.len() - 1] && first > 0.0 {
            println!("\n64 slices costs {:.1}x one slice on the same scene", ms / first);
        }
    }
}
