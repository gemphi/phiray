//! Device ownership, buffers and recorded passes. One device per process
//! ([`Gpu::shared`]), on the adapter `adapter.rs` ranks first, shared by
//! everything phiray runs — the viewport's tracer, GPU skinning — with the
//! compute pipelines it compiled, so a kernel compiles once per device
//! rather than once per scene.

mod adapter;
mod array;
mod pass;
mod raster;
/// Where the process's one device lives — a static on a desktop, a
/// thread-local in a browser (`shared.rs` says why).
mod shared;

pub use adapter::{GpuAdapter, GpuAdapters};
pub use array::GpuArray;
pub use pass::Pass;
pub use raster::Visibility;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// How long a readback may block before the caller gives up and takes the
/// CPU path. wgpu's poll is `wait_indefinitely` by design; on a driver reset
/// (Windows TDR) it never returns, and a reader thread parked inside it is
/// what "Phiber hangs on import" looked like — no progress, and the Cancel
/// key could not reach the thread to stop it.
pub const WAIT_LIMIT: Duration = Duration::from_secs(5);

/// How long a readback waits for the device, when the caller has not said.
///
/// [`WAIT_LIMIT`] is a *hang* guard, and five seconds is right for the
/// viewport, where a frame that is not back in five seconds is no longer
/// worth drawing. It is the wrong budget for a **final render**: a
/// 2048-sample 1440 × 1440 frame of a seven-million-face scene keeps the
/// GPU busy for the best part of a minute, the readback gave up, and
/// `read_first` returned an empty buffer — which every caller reads as
/// "the GPU declined". That is how `phiber -b … -f 1` on pose 26 produced
/// a blank Render Result while the GPU was working perfectly.
///
/// So the budget is the caller's: [`Gpu::patience`] holds the current one
/// and [`PatienceFor`] raises it for as long as a render needs.
static PATIENCE_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(5_000);

/// Raise the readback budget for as long as this value lives, then put the
/// previous one back — so a final render may wait minutes without the
/// viewport's own reads losing their five-second guard.
pub struct PatienceFor(u64);

impl PatienceFor {
    pub fn render(limit: Duration) -> Self {
        let was = PATIENCE_MS.swap(limit.as_millis() as u64, std::sync::atomic::Ordering::Relaxed);
        Self(was)
    }
}

impl Drop for PatienceFor {
    fn drop(&mut self) {
        PATIENCE_MS.store(self.0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// How many times the shared device was replaced. A plain counter, so it
/// stays a `static` on every target — only the device itself has to move to
/// a thread-local in a browser (`shared.rs`).
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// The GPU: one device + queue, shared by every renderer mode.
#[derive(Clone)]
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Which GPU this is (name, backend, driver) — reported by `examples/limits`.
    pub info: wgpu::AdapterInfo,
    /// The adapter as the Preferences list it.
    pub adapter: GpuAdapter,
    /// Compute pipelines compiled on this device, by kernel: a slot each,
    /// so compiling one kernel holds up only the callers that want that
    /// kernel ([`Gpu::pipeline`]).
    pipelines: Arc<Mutex<HashMap<u64, Arc<OnceLock<wgpu::ComputePipeline>>>>>,
    /// Render pipelines compiled on this device (the visibility prepass).
    rasters: Arc<Mutex<HashMap<u64, wgpu::RenderPipeline>>>,
}

impl Gpu {
    /// A device of its own on the best adapter (`adapter.rs`); `None` when
    /// there is none. The app uses [`Gpu::shared`].
    ///
    /// **Panics in a browser**, because `block_on` does on a page's one
    /// thread. Nothing on the web path may call this: use
    /// [`Gpu::open_shared`].
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::open())
    }

    /// The process's device, opened on first use — `None` when the machine
    /// has no adapter. Every caller gets the same device until
    /// [`Gpu::switch`] replaces it.
    #[cfg(not(target_family = "wasm"))]
    pub fn shared() -> Option<Self> {
        shared::with(|slot| slot.get_or_insert_with(Self::new).clone())?
    }

    /// The device [`Gpu::open_shared`] opened, and never an attempt to open
    /// one here.
    ///
    /// Opening needs `await`, and the only way to reach an async result from
    /// a synchronous function is to block — which on the one thread a page
    /// has does not wait, it panics (see [`Gpu::open`]). So a browser build
    /// opens the device once at startup and every later caller reads the
    /// result from here: the viewport's refine pass, Preferences ▸ System,
    /// the skinner. Until that future lands this is `None`, which those
    /// callers already handle, because it is what a machine with no adapter
    /// returns.
    #[cfg(target_family = "wasm")]
    pub fn shared() -> Option<Self> {
        Self::opened()
    }

    /// Open the shared device without blocking, and keep it — what
    /// [`Gpu::shared`] cannot do in a browser.
    ///
    /// Idempotent: a device already open is returned as it is, and so is an
    /// earlier "there is no adapter", so a second caller does not ask the
    /// browser for a second device.
    pub async fn open_shared() -> Option<Self> {
        if let Some(known) = shared::try_with(|slot| slot.clone()).flatten() {
            return known;
        }
        let gpu = Self::open().await;
        shared::with(|slot| *slot = Some(gpu.clone()));
        gpu
    }

    /// Open the device again on the adapter `preferred` names (`None`: the
    /// best), and make it the shared one; what it opened on.
    #[cfg(not(target_family = "wasm"))]
    pub fn switch(preferred: Option<String>) -> Option<GpuAdapter> {
        GpuAdapters::prefer(preferred);
        let gpu = Self::new();
        let picked = gpu.as_ref().map(|g| g.adapter.clone());
        shared::with(|slot| {
            *slot = Some(gpu);
            GENERATION.fetch_add(1, Ordering::SeqCst);
        });
        picked
    }

    /// A browser exposes one device and no choice of adapter, and reopening
    /// it would need `await` ([`Gpu::open_shared`]). The preference is
    /// recorded so the panel still reads back what was asked for, and the
    /// open device stays.
    #[cfg(target_family = "wasm")]
    pub fn switch(preferred: Option<String>) -> Option<GpuAdapter> {
        GpuAdapters::prefer(preferred);
        Self::in_use()
    }

    /// Bumped each time [`Gpu::switch`] replaces the shared device, so a
    /// holder of the old one can tell.
    pub fn generation() -> u64 {
        GENERATION.load(Ordering::SeqCst)
    }

    /// The shared device when it is open — not while it is being opened;
    /// it does not open it.
    pub fn opened() -> Option<Self> {
        shared::try_with(|slot| slot.as_ref()?.clone())?
    }

    /// The shared device's adapter, when it is open.
    pub fn in_use() -> Option<GpuAdapter> {
        Self::opened().map(|g| g.adapter)
    }

    /// Kernels compiled on this device so far.
    pub fn compiled(&self) -> usize {
        self.pipelines.lock().map_or(0, |cache| cache.values().filter(|slot| slot.get().is_some()).count())
    }

    /// The pipeline for `key`, compiled by `make` the first time this
    /// device is asked for it.
    ///
    /// The map is locked only to find the kernel's slot; the compile runs
    /// in the slot. It used to run under the map's lock, and Material
    /// Preview takes four and a half minutes to compile on a GTX 1660 Ti:
    /// while `Tracer::warm` compiled it after the first frame, every
    /// dispatch of every other kernel — the viewport's Solid, already
    /// compiled, an F12 render — queued behind it, and the viewport stood
    /// still for those minutes whatever the user did.
    pub(crate) fn pipeline(&self, key: u64, make: impl FnOnce() -> wgpu::ComputePipeline) -> wgpu::ComputePipeline {
        let slot = {
            let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            Arc::clone(cache.entry(key).or_default())
        };
        slot.get_or_init(make).clone()
    }

    /// The render pipeline for `key`, compiled by `make` the first time.
    pub(crate) fn render_pipeline(&self, key: u64, make: impl FnOnce() -> wgpu::RenderPipeline) -> wgpu::RenderPipeline {
        let mut cache = self.rasters.lock().unwrap_or_else(|e| e.into_inner());
        cache.entry(key).or_insert_with(make).clone()
    }

    /// Open the device without blocking — the browser's only option.
    ///
    /// [`Gpu::new`] wraps this in `pollster::block_on`, which is right on a
    /// desktop and impossible on the web: blocking the page's one thread
    /// is what WebGPU's async API exists to prevent, and `block_on` there
    /// does not merely stall, it panics. So this is the door a wasm
    /// front end comes through, and it is `pub` for that reason alone —
    /// native callers should keep using `Gpu::shared`, which opens once
    /// per process and hands the same device to every renderer mode.
    pub async fn open() -> Option<Self> {
        let (adapter, picked) = GpuAdapters::chosen().await?;
        // Everything the adapter offers (kernels need > 4 storage buffers).
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("phiray"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Self { device, queue, info: adapter.get_info(), adapter: picked, pipelines: Arc::default(), rasters: Arc::default() })
    }

    /// Largest storage buffer one binding may cover on this device.
    pub fn max_binding(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size
    }

    /// Block until queued work is done. Never returns if the device is
    /// lost — call [`Gpu::wait_for`] from anything that must not hang.
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// [`Gpu::wait`] with a deadline: `false` when the device did not drain
    /// within `limit`. wgpu has no cancellable poll, so the wait still runs —
    /// on a helper thread — but the caller is released and can fall back.
    /// The readback budget in force (`PatienceFor`).
    pub fn patience(&self) -> Duration {
        Duration::from_millis(PATIENCE_MS.load(std::sync::atomic::Ordering::Relaxed))
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn wait_for(&self, limit: Duration) -> bool {
        let (tx, rx) = std::sync::mpsc::channel();
        let gpu = self.clone();
        let spawned = std::thread::Builder::new()
            .name("phiray.wait".to_string())
            .spawn(move || {
                gpu.wait();
                let _ = tx.send(());
            })
            .is_ok();
        spawned && rx.recv_timeout(limit).is_ok()
    }

    /// In a browser: **always `false`**, and that is the honest answer.
    ///
    /// The desktop version above hands the blocking poll to a helper thread.
    /// Neither half of that is available here. The device cannot cross a
    /// thread (`shared.rs`), so there is no helper to hand it to; and
    /// blocking the one thread a page has would freeze the tab rather than
    /// wait, so `wgpu`'s WebGPU backend makes `poll` a no-op — completion is
    /// driven by the browser's event loop, which only runs once this
    /// function has returned.
    ///
    /// So it polls once to make progress where it can and reports that the
    /// device did not drain. Callers treat `false` as "fall back" — the
    /// readback declines and the CPU path runs — which is slow but correct.
    /// Returning `true` would be a lie that has readers copy out buffers the
    /// GPU has not finished writing, and wrong pixels are harder to find than
    /// a slow frame.
    ///
    /// The real fix is an `async` readback that awaits
    /// `map_async`'s callback instead of polling for it, which is the same
    /// change `Gpu::open` already made for device creation. Until then a
    /// browser build renders through whatever path does not need a readback.
    #[cfg(target_family = "wasm")]
    pub fn wait_for(&self, _limit: Duration) -> bool {
        let _ = self.device.poll(wgpu::PollType::Poll);
        false
    }
}
