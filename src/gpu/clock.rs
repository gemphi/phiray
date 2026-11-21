//! The GPU's own time for a pass. `queue.submit` returns once the work is
//! queued, not once it has run, so a clock around it reads the cost of
//! recording and calls it the cost of tracing: a heavy single-slice sample
//! "took" a few milliseconds and the readback after it paid the rest.
//! Apple names the fix — report the GPU's time for a dispatch, not the time
//! to submit it (`MTLCounters.h:43,166`, `MTLCommandBuffer.h:289,294`).
//!
//! With `TIMESTAMP_QUERY` the pass stamps its start and end and the
//! difference is read back. Without it one hard sync a sample makes the wall
//! time include the GPU's. Either way the two costs come back apart:
//! [`Time::submit`] and [`Time::gpu`].
use super::{Gpu, Pass};
use std::time::Duration;
use web_time::Instant;
/// One sample's two costs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Time {
    /// Recording the pass and handing it to the queue.
    pub submit: Duration,
    /// Running it: the stamped span, or submit to drained without stamps.
    pub gpu: Duration,
}
impl std::ops::AddAssign for Time {
    fn add_assign(&mut self, other: Self) {
        self.submit += other.submit;
        self.gpu += other.gpu;
    }
}
/// A pass's two timestamps and the buffers that carry them home.
pub struct Clock {
    stamps: wgpu::QuerySet,
    /// Resolved ticks (a `QUERY_RESOLVE` buffer cannot be mapped).
    ticks: wgpu::Buffer,
    /// What the host maps.
    staging: wgpu::Buffer,
}
impl Clock {
    /// Two 8-byte ticks.
    const SIZE: u64 = 16;
    /// `None` when the device was not opened with timestamp queries.
    pub fn new(gpu: &Gpu) -> Option<Self> {
        if !gpu.timed {
            return None;
        }
        let stamps = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("phiray.clock"), ty: wgpu::QueryType::Timestamp, count: 2 });
        let buffer = |label, usage| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: Self::SIZE, usage, mapped_at_creation: false });
        let ticks = buffer("phiray.clock.ticks", wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
        let staging = buffer("phiray.clock.read", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        Some(Self { stamps, ticks, staging })
    }
    /// Run `steps` of `pass` as one compute pass and time it, blocking
    /// until it has finished: the stamps cannot be read before then, and
    /// without them that one sync is the clock.
    pub fn run(clock: Option<&Self>, gpu: &Gpu, pass: &Pass, steps: std::ops::Range<usize>) -> Time {
        let at = Instant::now();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let writes = clock.map(|c| wgpu::ComputePassTimestampWrites { query_set: &c.stamps, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
        pass.record(&mut enc, steps, writes);
        if let Some(c) = clock {
            enc.resolve_query_set(&c.stamps, 0..2, &c.ticks, 0);
            enc.copy_buffer_to_buffer(&c.ticks, 0, &c.staging, 0, Self::SIZE);
        }
        gpu.queue.submit([enc.finish()]);
        let submit = at.elapsed();
        if let Some(c) = clock {
            c.staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        gpu.wait();
        // No stamps: the wall time to a drained device is the GPU's time.
        let ran = clock.and_then(|c| c.read(gpu)).unwrap_or_else(|| at.elapsed());
        Time { submit, gpu: ran }
    }
    /// The span between the two ticks, the staging buffer mapped by the
    /// drain in [`Self::run`].
    fn read(&self, gpu: &Gpu) -> Option<Duration> {
        let slice = self.staging.slice(..);
        let ticks: [u64; 2] = {
            let view = slice.get_mapped_range();
            let t: &[u64] = bytemuck::cast_slice(&view);
            [t[0], t[1]]
        };
        self.staging.unmap();
        // A counter that went backwards (a reset between the stamps) is no reading.
        let span = ticks[1].checked_sub(ticks[0])?;
        Some(Duration::from_nanos((span as f64 * f64::from(gpu.queue.get_timestamp_period())) as u64))
    }
}
