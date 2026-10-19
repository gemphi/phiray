//! `GpuBvh`: build on the GPU (codes, sort, links, bounds), refit per frame
//! by replay. **Leaf packing** is the one step still on the CPU, and the one
//! readback the build makes is the order it needs (see `jobs.md`).

use super::build::{Bvh, Link};
use super::kernels::{Karras, KarrasParams, Morton, MortonParams, Refit, RefitParams};
use super::sort::GpuSort;
use super::node::{Aabb, Node};
use crate::gpu::{Gpu, GpuArray, Pass};

pub struct GpuBvh {
    pub prims: GpuArray<Aabb>,
    pub order: GpuArray<u32>,
    /// Final bounds live here (the refit ping-pong always ends in `nodes`).
    pub nodes: GpuArray<Node>,
    scratch: GpuArray<Node>,
    refit: Pass,
    pub depth: u32,
    pub leaf: u32,
}

impl GpuBvh {
    pub fn build(gpu: &Gpu, prims: &[Aabb], leaf: u32) -> Self {
        let n = prims.len();
        let gprims = GpuArray::from(gpu, prims);
        let mut pass = Pass::default();

        // 1. Morton codes on the GPU.
        let s = Bvh::scene(prims);
        let inv = std::array::from_fn(|k| {
            let e = s.hi[k] - s.lo[k];
            if e > 0.0 { 1.0 / e } else { 0.0 }
        });
        let codes = GpuArray::<u32>::from(gpu, &vec![0; n]);
        let mp = MortonParams { lo: s.lo, inv, ..Default::default() };
        pass.add::<Morton>(gpu, n as u32, mp, &[gprims.buffer(), codes.buffer()]);
        pass.run(gpu);

        // 2. Sort on the GPU (`radix.rs`), where it used to come back to
        //    the CPU: read the codes, `par_sort_unstable_by_key`, upload the
        //    result. The 64-bit key makes the radix order identical to that
        //    sort's, value for value, so this is a move and not a change.
        let sort = GpuSort::record(gpu, codes.buffer(), n as u32);
        sort.run(gpu);

        // 3. Karras links on the GPU, reading the sorted codes where the
        //    sort left them (`GpuSort::code`) rather than from an upload.
        let links = if n > 1 {
            let out = GpuArray::<Link>::from(gpu, &vec![Link::default(); n - 1]);
            pass.clear();
            let kp = KarrasParams { leaves: n as u32, ..Default::default() };
            pass.add::<Karras>(gpu, n as u32 - 1, kp, &[sort.code().buffer(), out.buffer()]);
            pass.run(gpu);
            out.read(gpu)
        } else {
            Vec::new()
        };
        let order = sort.order(gpu);

        // 4. Pack (CPU) and upload; 5. bounds by refit passes.
        let packed = Bvh::assemble(order, &links, leaf);
        let mut bvh = Self {
            order: GpuArray::from(gpu, &packed.order),
            nodes: GpuArray::from(gpu, &packed.nodes),
            scratch: GpuArray::from(gpu, &packed.nodes),
            prims: gprims,
            refit: Pass::default(),
            depth: packed.depth,
            leaf,
        };
        bvh.record_refit(gpu);
        bvh.refit(gpu);
        bvh
    }

    /// Record `depth + 1` Jacobi steps (rounded up to even so the result lands in `nodes`).
    fn record_refit(&mut self, gpu: &Gpu) {
        let total = self.nodes.len() as u32;
        let steps = (self.depth + 1).next_multiple_of(2);
        for k in 0..steps {
            let (src, dst) = if k % 2 == 0 { (&self.nodes, &self.scratch) } else { (&self.scratch, &self.nodes) };
            let bufs = [self.prims.buffer(), self.order.buffer(), src.buffer(), dst.buffer()];
            self.refit.add::<Refit>(gpu, total, RefitParams::default(), &bufs);
        }
    }

    /// Recompute bounds from the current prims (topology unchanged).
    pub fn refit(&self, gpu: &Gpu) {
        self.refit.run(gpu);
    }

    /// New prim boxes (same count) → refit. Returns `false` if the count changed (rebuild instead).
    pub fn update(&mut self, gpu: &Gpu, prims: &[Aabb]) -> bool {
        if prims.len() != self.prims.len() || self.prims.write(gpu, prims) {
            return false;
        }
        self.refit(gpu);
        true
    }

    /// Read back as a CPU `Bvh` (tests, debugging).
    pub fn download(&self, gpu: &Gpu) -> Bvh {
        let mut bvh = Bvh { nodes: self.nodes.read(gpu), order: self.order.read(gpu), depth: self.depth, ..Default::default() };
        bvh.widen();
        bvh
    }
}
