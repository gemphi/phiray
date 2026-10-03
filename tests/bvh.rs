//! GPU BVH == CPU BVH: same topology, same bounds, refit after motion.

use phiray::{Aabb, Bvh, GpuBvh, Ray};

fn prims(n: usize, shift: f32) -> Vec<Aabb> {
    let mut x = 0x2545_F491u32;
    let mut r = move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        (x % 100_000) as f32 * 0.001
    };
    (0..n)
        .map(|i| {
            let a = if i % 13 == 0 { [5.0, 5.0, 5.0] } else { [r(), r() + shift, r()] };
            Aabb::of_tri(a, [a[0] + 0.3, a[1], a[2]], [a[0], a[1] + 0.3, a[2] + 0.05])
        })
        .collect()
}

fn same(gpu: &Bvh, cpu: &Bvh) {
    assert_eq!(gpu.order, cpu.order);
    assert_eq!(gpu.depth, cpu.depth);
    for i in cpu.reachable() {
        let (g, c) = (gpu.nodes[i], cpu.nodes[i]);
        assert_eq!((g.lo.i, g.hi.i), (c.lo.i, c.hi.i), "node {i} links");
        assert_eq!(g.bounds(), c.bounds(), "node {i} bounds");
    }
}

#[test]
fn gpu_build_and_refit_match_cpu() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    for n in [1, 2, 3, 257, 100_003] {
        let p = prims(n, 0.0);
        let cpu = Bvh::build(&p, 4);
        let t = std::time::Instant::now();
        let mut g = GpuBvh::build(&gpu, &p, 4);
        if n > 1000 {
            eprintln!("{n} prims: GPU build {:?}, depth {}", t.elapsed(), g.depth);
        }
        same(&g.download(&gpu), &cpu);

        // Move everything: refit on the GPU == CPU refit on the old topology.
        let moved = prims(n, 2.5);
        assert!(g.update(&gpu, &moved));
        let mut want = cpu.clone();
        want.refit(&moved);
        same(&g.download(&gpu), &want);
    }
}
