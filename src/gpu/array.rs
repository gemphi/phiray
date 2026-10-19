//! `GpuArray<T>`: one storage buffer, grown by doubling, never reallocated per frame.

use super::Gpu;
use bytemuck::Pod;
use std::marker::PhantomData;

pub struct GpuArray<T: Pod> {
    buf: wgpu::Buffer,
    len: usize,
    cap: usize,
    _t: PhantomData<T>,
}

const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::VERTEX);

impl<T: Pod> GpuArray<T> {
    pub fn new(gpu: &Gpu, cap: usize) -> Self {
        let cap = cap.max(1);
        Self { buf: Self::alloc(gpu, cap), len: 0, cap, _t: PhantomData }
    }

    /// `len` items of zero: wgpu clears every new buffer, so nothing is
    /// uploaded (an image buffer is ~100 bytes a pixel).
    pub fn zeroed(gpu: &Gpu, len: usize) -> Self {
        Self { len, ..Self::new(gpu, len) }
    }

    pub fn from(gpu: &Gpu, items: &[T]) -> Self {
        let mut a = Self::new(gpu, items.len());
        a.write(gpu, items);
        a
    }

    fn alloc(gpu: &Gpu, cap: usize) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phiray.array"),
            size: (cap * size_of::<T>()).max(4) as u64,
            usage: USAGE,
            mapped_at_creation: false,
        })
    }

    /// Replace contents. Returns `true` when the buffer was reallocated (rebind needed).
    pub fn write(&mut self, gpu: &Gpu, items: &[T]) -> bool {
        let grown = items.len() > self.cap;
        if grown {
            self.cap = items.len().next_power_of_two();
            self.buf = Self::alloc(gpu, self.cap);
        }
        self.len = items.len();
        gpu.queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(items));
        grown
    }

    /// `items` over `chunks` arrays of at most `limit` bytes each; `None` if that
    /// is not enough. Returns the arrays and the items per chunk (index `i` is in
    /// chunk `i / per`); unused chunks hold one placeholder item.
    /// Split `items` over `chunks` bindings, each holding `cut` items.
    ///
    /// The kernels turn an index into (binding, offset) with a division on
    /// every node, face and vertex they touch. Making `cut` a power of two
    /// so that becomes a shift was tried and **measured 21% slower** on
    /// pose 26 (254s -> 308s at 64 samples), so it was reverted: the divide
    /// is not what this traversal is spending its time on, and rounding the
    /// capacity down would have cost a large scene a binding's worth of
    /// room for nothing. `docs/render-fix-report.md` has the numbers.
    pub fn split(gpu: &Gpu, items: &[T], limit: u64, chunks: usize) -> Option<(Vec<Self>, u32)> {
        let cap = (limit / size_of::<T>() as u64).max(1) as usize;
        if items.len() > chunks * cap {
            return None;
        }
        let cut = items.len().clamp(1, cap);
        let parts = (0..chunks)
            .map(|k| match items.get(k * cut..((k + 1) * cut).min(items.len())) {
                Some(part) if !part.is_empty() => Self::from(gpu, part),
                _ => Self::new(gpu, 1),
            })
            .collect();
        Some((parts, cut as u32))
    }

    /// Overwrite the first `items.len()` items, keeping the rest (the light
    /// table in front of an atlas that did not change). `false` when they do
    /// not fit.
    pub fn write_front(&mut self, gpu: &Gpu, items: &[T]) -> bool {
        if items.len() > self.len {
            return false;
        }
        gpu.queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(items));
        true
    }

    /// Overwrite `items.len()` items from item `at`, keeping the rest (the
    /// top tree at the front of the nodes, the instance table behind the
    /// atlas). `false` when they reach past the end.
    pub fn write_at(&mut self, gpu: &Gpu, at: usize, items: &[T]) -> bool {
        if at + items.len() > self.len {
            return false;
        }
        if !items.is_empty() {
            gpu.queue.write_buffer(&self.buf, (at * size_of::<T>()) as u64, bytemuck::cast_slice(items));
        }
        true
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    /// Copy back to the CPU (tests, readback). Blocks.
    pub fn read(&self, gpu: &Gpu) -> Vec<T> {
        self.read_first(gpu, self.len)
    }

    /// The most one staging buffer is asked for at a time.
    ///
    /// A full frame's readback used to be a single allocation the size of
    /// the whole buffer, and `BStill::read` makes **six** of them back to
    /// back (pixels, display, radiance, albedo, normal, depth). On a
    /// 1440 × 1440 frame that is hundreds of megabytes of host-visible
    /// memory asked for at once, on top of the scene itself; the driver
    /// refused one, wgpu handed back an invalid buffer, and `map_async`
    /// panicked with *"Buffer with 'phiray.read' label is invalid"*.
    /// Reading in bounded pieces asks for a constant amount instead.
    const CHUNK: u64 = 32 << 20;

    /// Copy back the first `count` items only. Empty when the device did
    /// not drain within [`Gpu::patience`] — the caller falls back.
    pub fn read_first(&self, gpu: &Gpu, count: usize) -> Vec<T> {
        self.read_range(gpu, 0, count)
    }

    /// Copy back `count` items from `from`. The image buffer holds its
    /// planes end to end — 31 of them (`Tracer::PX_WORDS`) — so a caller
    /// that wants one plane near the end must not be made to read the
    /// 248 MB in front of it (`Tracer::unconverged`).
    pub fn read_range(&self, gpu: &Gpu, from: usize, count: usize) -> Vec<T> {
        let from = from.min(self.len);
        let count = count.min(self.len - from);
        let skip = (from * size_of::<T>()) as u64;
        let bytes = (count * size_of::<T>()) as u64;
        if bytes == 0 {
            return Vec::new();
        }
        // wgpu maps on 4-byte boundaries, and `T` is a multiple of that in
        // every array phiray reads, so a chunk is cut to whole items.
        let item = size_of::<T>() as u64;
        let step = (Self::CHUNK / item).max(1) * item;
        let mut out: Vec<u8> = Vec::with_capacity(bytes as usize);
        let mut at = 0u64;
        while at < bytes {
            let take = step.min(bytes - at);
            let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("phiray.read"),
                size: take,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&self.buf, skip + at, &staging, 0, take);
            gpu.queue.submit([enc.finish()]);
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            if !gpu.wait_for(gpu.patience()) {
                // The device did not drain in time. Return nothing rather
                // than read an unmapped range (which panics) or wait for
                // ever: every caller treats a short read as "the GPU
                // declined" and computes the same result on the CPU.
                return Vec::new();
            }
            out.extend_from_slice(&slice.get_mapped_range());
            staging.unmap();
            at += take;
        }
        bytemuck::cast_slice(&out).to_vec()
    }
}
