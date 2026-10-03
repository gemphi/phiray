//! Which GPU phiray runs on. Every adapter the drivers report, across
//! backends, ranked: a discrete GPU before an integrated one, before a
//! virtual one, before anything else, and a software (CPU) adapter last —
//! then Vulkan or Metal before DirectX 12 (whose shaders compile slowly
//! through FXC) before OpenGL — then the one that binds the most. The
//! first wins unless a name is preferred (Preferences ▸ System ▸ Render
//! Devices, or `PHIRAY_GPU=<part of the name>`).
//!
//! wgpu's own `request_adapter` with no power preference does not rank at
//! all: it takes the first adapter the first backend lists — on a laptop
//! often the integrated GPU, on some machines a software one.

use std::cmp::Reverse;
use std::sync::{Mutex, OnceLock};

/// One adapter as the Preferences list it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuAdapter {
    pub name: String,
    pub kind: wgpu::DeviceType,
    pub backend: wgpu::Backend,
    pub driver: String,
    /// The largest storage buffer one binding may cover, in bytes.
    pub binding: u64,
}

impl GpuAdapter {
    fn of(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let driver = [info.driver, info.driver_info].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        Self { name: info.name, kind: info.device_type, backend: info.backend, driver, binding: adapter.limits().max_storage_buffer_binding_size as u64 }
    }

    /// How it ranks: lower is better.
    pub fn rank(&self) -> (u8, u8, Reverse<u64>) {
        let kind = match self.kind {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        };
        let backend = match self.backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal => 0,
            wgpu::Backend::Dx12 => 1,
            wgpu::Backend::Gl => 2,
            _ => 3,
        };
        (kind, backend, Reverse(self.binding))
    }

    /// A software adapter (lavapipe, WARP, SwiftShader): the CPU tracing,
    /// at a fraction of a card's speed.
    pub fn is_software(&self) -> bool {
        self.kind == wgpu::DeviceType::Cpu
    }

    /// `Discrete`, `Integrated`, `Virtual`, `Software`, `Other`.
    pub fn kind_label(&self) -> &'static str {
        match self.kind {
            wgpu::DeviceType::DiscreteGpu => "Discrete",
            wgpu::DeviceType::IntegratedGpu => "Integrated",
            wgpu::DeviceType::VirtualGpu => "Virtual",
            wgpu::DeviceType::Cpu => "Software",
            wgpu::DeviceType::Other => "Other",
        }
    }

    /// `Vulkan`, `Metal`, `DirectX 12`, `OpenGL`.
    pub fn backend_label(&self) -> &'static str {
        match self.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DirectX 12",
            wgpu::Backend::Gl => "OpenGL",
            _ => "WebGPU",
        }
    }

    /// Does `wanted` (a whole name, or part of one, any case) name it?
    fn named(&self, wanted: &str) -> bool {
        let (name, wanted) = (self.name.to_lowercase(), wanted.trim().to_lowercase());
        !wanted.is_empty() && (name == wanted || name.contains(&wanted))
    }

    /// Of `list`, the one phiray takes: the first `preferred` names, else
    /// the best ranked.
    pub fn pick<'a>(list: &'a [Self], preferred: Option<&str>) -> Option<&'a Self> {
        let named = preferred.and_then(|wanted| list.iter().filter(|a| a.named(wanted)).min_by_key(|a| a.rank()));
        named.or_else(|| list.iter().min_by_key(|a| a.rank()))
    }

    /// One row per GPU, best first: an adapter a better backend already
    /// lists (the same card through Vulkan and DirectX 12) is left out.
    pub fn distinct(list: &[Self]) -> Vec<Self> {
        let mut sorted = list.to_vec();
        sorted.sort_by_key(Self::rank);
        let mut out: Vec<Self> = Vec::new();
        for a in sorted {
            if !out.iter().any(|b| b.name == a.name) {
                out.push(a);
            }
        }
        out
    }
}

/// The adapters this machine has, and the one to prefer.
pub struct GpuAdapters;

/// What `PHIRAY_GPU` or the Preferences asked for.
static PREFERRED: Mutex<Option<String>> = Mutex::new(None);
/// Every adapter the drivers report, read once.
static FOUND: OnceLock<Vec<GpuAdapter>> = OnceLock::new();

impl GpuAdapters {
    fn instance() -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
    }

    /// Every adapter, as the drivers list them.
    pub fn all() -> &'static [GpuAdapter] {
        FOUND.get_or_init(|| pollster::block_on(Self::instance().enumerate_adapters(wgpu::Backends::all())).iter().map(GpuAdapter::of).collect())
    }

    /// Prefer the adapter `name` names (`None`: the best ranked).
    pub fn prefer(name: Option<String>) {
        if let Ok(mut slot) = PREFERRED.lock() {
            *slot = name.filter(|n| !n.trim().is_empty());
        }
    }

    /// The name preferred: the Preferences' choice, else `PHIRAY_GPU`.
    pub fn preferred() -> Option<String> {
        let chosen = PREFERRED.lock().ok().and_then(|slot| slot.clone());
        chosen.or_else(|| std::env::var("PHIRAY_GPU").ok().filter(|n| !n.trim().is_empty()))
    }

    /// The adapter phiray opens: the preferred one if the machine has it,
    /// else the best ranked.
    pub(super) async fn chosen() -> Option<(wgpu::Adapter, GpuAdapter)> {
        let adapters = Self::instance().enumerate_adapters(wgpu::Backends::all()).await;
        let found: Vec<GpuAdapter> = adapters.iter().map(GpuAdapter::of).collect();
        let _ = FOUND.set(found.clone());
        let pick = GpuAdapter::pick(&found, Self::preferred().as_deref())?.clone();
        let at = found.iter().position(|a| *a == pick)?;
        adapters.into_iter().nth(at).map(|a| (a, pick))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Backend, DeviceType};

    fn gpu(name: &str, kind: DeviceType, backend: Backend, mb: u64) -> GpuAdapter {
        GpuAdapter { name: name.into(), kind, backend, driver: String::new(), binding: mb << 20 }
    }

    /// A laptop as its drivers list it, integrated first — what wgpu's
    /// unranked request took.
    fn laptop() -> Vec<GpuAdapter> {
        vec![
            gpu("Intel(R) UHD Graphics 630", DeviceType::IntegratedGpu, Backend::Vulkan, 2048),
            gpu("NVIDIA GeForce RTX 3060 Laptop GPU", DeviceType::DiscreteGpu, Backend::Dx12, 2048),
            gpu("NVIDIA GeForce RTX 3060 Laptop GPU", DeviceType::DiscreteGpu, Backend::Vulkan, 2048),
            gpu("Microsoft Basic Render Driver", DeviceType::Cpu, Backend::Dx12, 128),
            gpu("NVIDIA GeForce RTX 3060 Laptop GPU/PCIe/SSE2", DeviceType::Other, Backend::Gl, 2048),
        ]
    }

    #[test]
    fn the_discrete_gpu_wins_through_its_best_backend_unless_another_is_named() {
        let list = laptop();
        let best = GpuAdapter::pick(&list, None).unwrap();
        assert_eq!((best.kind, best.backend), (DeviceType::DiscreteGpu, Backend::Vulkan));
        assert_eq!(GpuAdapter::pick(&list, Some("intel")).unwrap().name, "Intel(R) UHD Graphics 630");
        assert_eq!(GpuAdapter::pick(&list, Some("no such card")).unwrap().kind, DeviceType::DiscreteGpu, "an absent name falls back");
        let rows: Vec<(&str, &str)> = GpuAdapter::distinct(&list).iter().map(|a| (a.kind_label(), a.backend_label())).collect();
        assert_eq!(rows, [("Discrete", "Vulkan"), ("Integrated", "Vulkan"), ("Other", "OpenGL"), ("Software", "DirectX 12")], "one row per card");
        let only_software = [gpu("llvmpipe (LLVM 19.1.7, 256 bits)", DeviceType::Cpu, Backend::Vulkan, 128)];
        assert_eq!(GpuAdapter::pick(&only_software, None).unwrap().kind_label(), "Software", "a software one when there is nothing else");
    }
}
