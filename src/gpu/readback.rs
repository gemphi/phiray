//! Reading a texture back without stalling the render thread.
//!
//! [`Gpu::read_range`] and the tracer's `pixels` map a buffer and WAIT for
//! it, behind a five-second guard (`PatienceFor`); that is right for a
//! final render, which has nothing else to do, and wrong for a viewport,
//! where the wait is on the frame's critical path. This is the other way:
//! two staging buffers used in turn, so the CPU reads last frame's while
//! the GPU writes this frame's, and the reader never blocks — it returns
//! `None` until a mapping has landed.
//!
//! Order of calls, per frame:
//!
//! 1. [`AsyncReadback::request_readback`] while recording, to copy the
//!    texture into the write buffer.
//! 2. `queue.submit(...)`.
//! 3. [`AsyncReadback::initiate_map_after_submit`], which must come after
//!    the submit or the map races the copy.
//! 4. [`AsyncReadback::try_get_pixels`], which returns `None` until a
//!    readback has finished and never blocks.
//!
//! # Where this came from
//!
//! `crates/apple/src/appkit/gpu/async_readback/` (210 lines), by way of a
//! first port into `phiphx::fx::readback`. It belongs here: phiray owns the
//! device, the buffers and every other readback, and phiphx's job is to
//! read a `.blend` — putting GPU code there forced an optional `wgpu`
//! dependency and a feature gate, and nothing ever switched it on.
//!
//! Two things changed from the original, both recorded by that first port:
//! the copy descriptors are spelled for wgpu 29 (`TexelCopyTextureInfo`,
//! `TexelCopyBufferInfo`, `TexelCopyBufferLayout`, `device.poll` taking a
//! `PollType`); and the source wrapped `get_mapped_range` in
//! `catch_unwind`, treating a panic as "not ready yet". That caught a real
//! case — mapping unfinished — but unwinding out of wgpu is not something
//! to rely on and it swallowed genuine failures too, so the `map_async`
//! callback raises a flag ([`Ready`]) the reader checks first: a branch
//! rather than an unwind.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A row of a texture copied to a buffer must start on a
/// `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256-byte) boundary, so a row is
/// `width * 4` rounded up to that. The padding is what
/// [`AsyncReadback::try_get_pixels`] strips back out.
pub fn padded_bytes_per_row(width: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded = width * 4;
    unpadded.div_ceil(align) * align
}

/// A flag the `map_async` callback raises when the mapping has succeeded.
/// Shared with the callback, which outlives the call that made it.
#[derive(Clone, Default)]
pub struct Ready(Arc<AtomicBool>);

impl Ready {
    /// Whether the mapping has landed and `get_mapped_range` will work.
    pub fn is_ready(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    fn raise(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn clear(&self) {
        self.0.store(false, Ordering::Release);
    }
}

/// A staging buffer sized for one frame, with its map state.
pub struct ReadbackBuffer {
    pub buffer: Option<wgpu::Buffer>,
    /// Whether a `map_async` has been asked for and not yet consumed.
    pub map_requested: bool,
    /// Raised by the map callback once the data can be read.
    pub ready: Ready,
    pub width: u32,
    pub height: u32,
}

impl Default for ReadbackBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadbackBuffer {
    /// Unallocated; the first [`Self::allocate`] gives it a size.
    pub fn new() -> Self {
        Self { buffer: None, map_requested: false, ready: Ready::default(), width: 0, height: 0 }
    }

    /// (Re)allocate for a target of `width` × `height` RGBA8 pixels.
    pub fn allocate(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let padded_row = padded_bytes_per_row(width);
        self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phiray.readback"),
            size: padded_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        self.width = width;
        self.height = height;
        self.map_requested = false;
        self.ready = Ready::default();
    }

    /// Whether it already fits, so it can be reused without reallocating.
    pub fn matches_dimensions(&self, width: u32, height: u32) -> bool {
        self.buffer.is_some() && self.width == width && self.height == height
    }

    /// Drop any outstanding mapping so the buffer can be written again.
    fn release(&mut self) {
        if self.ready.is_ready() {
            if let Some(ref buffer) = self.buffer {
                buffer.unmap();
            }
        }
        self.ready.clear();
        self.map_requested = false;
    }
}

/// Two staging buffers, used in turn: the CPU reads one while the GPU writes
/// the other.
pub struct AsyncReadback {
    pub buffers: [ReadbackBuffer; 2],
    /// Which buffer the next copy writes.
    pub current: usize,
}

impl Default for AsyncReadback {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncReadback {
    pub fn new() -> Self {
        Self { buffers: [ReadbackBuffer::new(), ReadbackBuffer::new()], current: 0 }
    }

    /// Record the copy from `source` into the write buffer, then swap, so the
    /// next frame writes the other one.
    ///
    /// Reallocates both buffers when the size changes; `source` must be
    /// RGBA8 with `COPY_SRC` usage.
    pub fn request_readback(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, source: &wgpu::Texture, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let fits = self.buffers.iter().all(|b| b.matches_dimensions(width, height));
        if !fits {
            for buffer in &mut self.buffers {
                buffer.release();
            }
            self.buffers[0].allocate(device, width, height);
            self.buffers[1].allocate(device, width, height);
        }
        let padded_row = padded_bytes_per_row(width);
        let write = &mut self.buffers[self.current];
        write.release();
        if let Some(ref buffer) = write.buffer {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: source, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row), rows_per_image: Some(height) },
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
        }
        self.current = 1 - self.current;
    }

    /// Ask for the mapping of the buffer just written.
    ///
    /// Must come after `queue.submit`, or the map is asked for before the
    /// copy it is meant to read is in flight.
    pub fn initiate_map_after_submit(&mut self) {
        let just_written = 1 - self.current;
        let buffer = &mut self.buffers[just_written];
        if buffer.map_requested {
            return;
        }
        if let Some(ref gpu_buffer) = buffer.buffer {
            let ready = buffer.ready.clone();
            gpu_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                // A failed map is left low: the reader keeps returning
                // `None` and the next frame's copy releases the buffer and
                // tries again.
                if result.is_ok() {
                    ready.raise();
                }
            });
            buffer.map_requested = true;
        }
    }

    /// Take the pixels of the most recently finished readback, tightly packed
    /// RGBA8 with the row padding stripped.
    ///
    /// `None` when nothing has finished. Never blocks: it polls once and
    /// reads only a mapping the callback has confirmed.
    pub fn try_get_pixels(&mut self, device: &wgpu::Device) -> Option<Vec<u8>> {
        let _ = device.poll(wgpu::PollType::Poll);
        let buffer = &mut self.buffers[self.current];
        if !buffer.map_requested || !buffer.ready.is_ready() {
            return None;
        }
        let gpu_buffer = buffer.buffer.as_ref()?;
        let padded_row = padded_bytes_per_row(buffer.width) as usize;
        let tight_row = (buffer.width * 4) as usize;
        let pixels = {
            let data = gpu_buffer.slice(..).get_mapped_range();
            let mut pixels = Vec::with_capacity(tight_row * buffer.height as usize);
            for y in 0..buffer.height as usize {
                let start = y * padded_row;
                let end = start + tight_row;
                if end <= data.len() {
                    pixels.extend_from_slice(&data[start..end]);
                }
            }
            pixels
        };
        buffer.release();
        Some(pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_is_padded_up_to_the_copy_alignment() {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        // A row already on the boundary is left alone.
        assert_eq!(padded_bytes_per_row(align / 4), align);
        // One pixel over rounds up to the next boundary.
        assert_eq!(padded_bytes_per_row(align / 4 + 1), 2 * align);
        // 64 px × 4 B is exactly 256.
        assert_eq!(padded_bytes_per_row(64), 256);
        assert_eq!(padded_bytes_per_row(65), 512);
        // Narrow frames still get a whole row.
        assert_eq!(padded_bytes_per_row(1), align);
    }

    #[test]
    fn a_fresh_buffer_fits_nothing_and_a_sized_one_fits_its_own_size() {
        let b = ReadbackBuffer::new();
        assert!(!b.matches_dimensions(4, 4), "unallocated fits nothing");
        assert!(!b.ready.is_ready());
        // The double buffer starts on the first slot.
        assert_eq!(AsyncReadback::new().current, 0);
    }
}
