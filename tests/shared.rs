//! One device per process (`Gpu::shared`): the viewport's tracer and GPU
//! skinning get the same device — the same queue, the same compiled
//! kernels — until the Preferences pick another (`Gpu::switch`), which
//! opens a new one and says so (`Gpu::generation`). Its own test binary:
//! the shared device is process-wide.

use phiray::{Atlas, Camera, Gpu, GpuAdapter, GpuAdapters, Mode, Tracer, Tri};

#[test]
fn the_process_shares_one_device_until_another_is_picked() {
    let Some(first) = Gpu::shared() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    assert_eq!(Gpu::in_use(), Some(first.adapter.clone()));
    assert_eq!(GpuAdapter::pick(GpuAdapters::all(), None), Some(&first.adapter), "Auto: the best ranked");
    // A kernel compiled through one handle is compiled for every holder.
    let n = [0.0, -1.0, 0.0];
    let tris = [Tri::new([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], [n; 3], [0.6, 0.6, 0.6, 1.0])];
    let mut tracer = Tracer::new(&first, &tris, &Atlas::default()).expect("a scene");
    tracer.set_mode(Mode::Solid);
    let cam = Camera { right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], units: 1.0, scale: 0.2, aspect: 1.0, dist: 10.0, ..Default::default() };
    tracer.render(&first, cam, 16, 16);
    assert!(Gpu::shared().expect("the same device").compiled() >= 1);

    let before = Gpu::generation();
    let picked = Gpu::switch(Some(first.adapter.name.clone()));
    assert_eq!((Gpu::generation(), picked), (before + 1, Some(first.adapter.clone())), "a new device on the card named");
    assert_eq!(Gpu::shared().expect("the new device").compiled(), 0, "which compiles its kernels afresh");
    GpuAdapters::prefer(None);
}
