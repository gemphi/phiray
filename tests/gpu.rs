//! GPU tests run on any adapter (lavapipe in CI); skipped when none exists.

use phiray::kernels::{Pair, Skin, SkinParams};
use phiray::{GpuArray, Pass, Ray};

fn translate(x: f32, y: f32, z: f32) -> [f32; 16] {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m
}

struct Rig {
    rest: Vec<[f32; 4]>,
    offsets: Vec<u32>,
    pairs: Vec<Pair>,
}

/// Vertex i gets i % 4 influences (0 = unweighted, 3 = three bones).
fn rig(n: usize) -> Rig {
    let rest = (0..n).map(|i| [i as f32 * 0.01, (i % 7) as f32, -(i as f32).sqrt(), 1.0]).collect();
    let (mut offsets, mut pairs) = (vec![0u32], Vec::new());
    for i in 0..n {
        for k in 0..i % 4 {
            pairs.push(Pair { bone: (k % 2) as u32, weight: 0.25 + k as f32 * 0.5, next: (k % 2) as u32, blend: 0.0 });
        }
        offsets.push(pairs.len() as u32);
    }
    Rig { rest, offsets, pairs }
}

fn close(a: &[[f32; 4]], b: &[[f32; 4]]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!(x.iter().zip(y).all(|(p, q)| (p - q).abs() <= 1e-4), "{x:?} vs {y:?}");
    }
}

#[test]
fn skin_matches_cpu_and_replays() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    // > one row of workgroups would need 16.7M items; 70k still spans many blocks.
    let n = 70_001;
    let Rig { rest, offsets, pairs } = rig(n);
    let mut bones = vec![translate(1.0, 0.0, 0.0), translate(0.0, 2.0, -4.0)];
    let min = 0.0001;
    let r = GpuArray::from(&gpu, &rest);
    let o = GpuArray::from(&gpu, &offsets);
    let q = GpuArray::from(&gpu, &pairs);
    let mut b = GpuArray::from(&gpu, &bones);
    let out = GpuArray::<[f32; 4]>::from(&gpu, &vec![[0.0; 4]; n]);

    let mut pass = Pass::default();
    let params = SkinParams { min, ..Default::default() };
    pass.add::<Skin>(&gpu, n as u32, params, &[r.buffer(), o.buffer(), q.buffer(), b.buffer(), out.buffer()]);
    pass.run(&gpu);
    close(&out.read(&gpu), &Skin::cpu(&rest, &offsets, &pairs, &bones, min));

    // Next frame: only the bone buffer changes; the recorded pass is replayed as-is.
    bones[1] = translate(-3.0, 0.5, 0.0);
    assert!(!b.write(&gpu, &bones));
    pass.run(&gpu);
    close(&out.read(&gpu), &Skin::cpu(&rest, &offsets, &pairs, &bones, min));
    assert_eq!(pass.compiled(), 1);
}
