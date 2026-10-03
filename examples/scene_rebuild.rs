//! What a scene change costs on this machine's GPU — a moved object, a
//! played frame — two ways: a new tracer over the same device (its kernels
//! compiled once, `Gpu::pipeline`), and the same tracer given the new pose
//! (`Tracer::repose`: vertices in place, the BVH refitted). Each is timed
//! alone (the CPU's part), then to its first frame read back. Then the
//! motion stops: a frame over the refitted tree, the tree built again
//! (`Tracer::retree`, the image kept), and a frame over the new tree.
//! `cargo run --release -p phiray --example scene_rebuild`

use phiray::{Atlas, Camera, Gpu, Mode, Scene, Tracer, Tri};
use std::time::Instant;

/// A `side × side` grid of quads, its right half moved `dx` along X.
fn tiles(side: usize, dx: f32) -> Vec<Tri> {
    let n = [0.0, -1.0, 0.0];
    let step = 6.4 / side as f32;
    let mut out = Vec::with_capacity(side * side * 2);
    for i in 0..side {
        for j in 0..side {
            let dx = if i >= side / 2 { dx } else { 0.0 };
            let (x, z) = (i as f32 * step - 3.2 + dx, j as f32 * step - 3.2);
            let p = [[x, 0.0, z], [x + step * 0.9, 0.0, z], [x + step * 0.9, 0.0, z + step * 0.9], [x, 0.0, z + step * 0.9]];
            out.push(Tri::new([p[0], p[1], p[2]], [n; 3], [0.6, 0.6, 0.6, 1.0]));
            out.push(Tri::new([p[0], p[2], p[3]], [n; 3], [0.6, 0.6, 0.6, 1.0]));
        }
    }
    out
}

fn ms(at: Instant) -> f64 {
    at.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let Some(gpu) = Gpu::shared() else {
        return println!("no GPU adapter");
    };
    println!("GPU: {} ({}, {})\n", gpu.adapter.name, gpu.adapter.kind_label(), gpu.adapter.backend_label());
    let cam = Camera { right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], units: 1.0, scale: 0.15, aspect: 1.0, dist: 10.0, ..Default::default() };
    for side in [64, 320] {
        println!("{} triangles", side * side * 2);
        for mode in [Mode::Solid, Mode::Rendered] {
            let at = Instant::now();
            let mut posed = Tracer::new(&gpu, &tiles(side, 0.0), &Atlas::default()).expect("a scene");
            posed.set_mode(mode);
            let _ = posed.render(&gpu, cam, 512, 512);
            println!("  {mode:?}: first build and frame {:>7.1} ms", ms(at));
            for step in 1..=3 {
                let moved = tiles(side, step as f32 * 0.5);
                let (scene, again) = (Scene::from_tris(&moved, &Atlas::default()), Scene::from_tris(&moved, &Atlas::default()));
                let at = Instant::now();
                let mut fresh = Tracer::from_scene(&gpu, again).expect("a scene");
                fresh.set_mode(mode);
                let built = ms(at);
                let _ = fresh.render(&gpu, cam, 512, 512);
                let built_frame = ms(at);
                let at = Instant::now();
                assert!(posed.repose(&gpu, &scene));
                let reposed = ms(at);
                let _ = posed.render(&gpu, cam, 512, 512);
                println!("  {mode:?} moved {step}: built again {built:>6.1} ms (+ frame {built_frame:>6.1}), reposed {reposed:>6.1} ms (+ frame {:>6.1})", ms(at));
            }
            let (mut aside, mut again) = (cam, cam);
            (aside.pivot, again.pivot) = ([0.1, 0.0, 0.0], [0.2, 0.0, 0.0]);
            let at = Instant::now();
            let _ = posed.render(&gpu, aside, 512, 512);
            let loose = ms(at);
            let at = Instant::now();
            assert!(posed.retree(&gpu));
            let retreed = ms(at);
            let at = Instant::now();
            let _ = posed.render(&gpu, again, 512, 512);
            println!("  {mode:?} still: a frame over the refitted tree {loose:>6.1} ms, retree {retreed:>6.1} ms, a frame over the new tree {:>6.1} ms", ms(at));
        }
    }
}
