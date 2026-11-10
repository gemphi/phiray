//! Recorded compute passes: build steps once, replay each frame (Warp graph capture).

use super::Gpu;
use crate::kernel::{Dispatch, Kernel};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

struct Step {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    dispatch: Dispatch,
}

/// A list of kernel launches, and the pipelines they use (compiled on the
/// device, shared with every other pass there).
#[derive(Default)]
pub struct Pass {
    cache: HashMap<u64, wgpu::ComputePipeline>,
    steps: Vec<Step>,
}

impl Pass {
    /// The kernel's pipeline: compiled once per device (`Gpu::pipeline`) —
    /// a new scene, a new pass, reuses it.
    fn pipeline<K: Kernel>(&mut self, gpu: &Gpu) -> wgpu::ComputePipeline {
        let compiled = gpu.pipeline(K::key(), || {
            let desc = wgpu::ShaderModuleDescriptor {
                label: Some(K::NAME),
                source: wgpu::ShaderSource::Wgsl(K::source().into()),
            };
            // Our own kernels, so no checks woven into them: naga otherwise
            // guards every buffer read and counts every loop, the BVH walk
            // included. `PHIRAY_CHECKED=1` puts them back to chase a kernel
            // that reads past its buffer or never leaves a loop.
            let module = match std::env::var_os("PHIRAY_CHECKED") {
                Some(_) => gpu.device.create_shader_module(desc),
                // SAFETY: the kernels are this crate's, every one sizes its
                // buffers from the dispatch it is given and bounds its loops
                // (the traversal by its stack). The tests run them this way;
                // with `PHIRAY_CHECKED=1` they run them checked.
                None => unsafe { gpu.device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) },
            };
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(K::NAME),
                layout: None,
                module: &module,
                entry_point: Some(K::ENTRY),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        self.cache.entry(K::key()).or_insert(compiled).clone()
    }

    /// Compile `K` on the device now, so the first pass to use it does not
    /// pay for it (`Tracer::warm`).
    pub fn compile<K: Kernel>(gpu: &Gpu) {
        Pass::default().pipeline::<K>(gpu);
    }

    /// Record a launch over `n` items. `params` must start with `[n, groups_x]`;
    /// those two fields are filled in here. Returns the step index.
    pub fn add<K: Kernel>(&mut self, gpu: &Gpu, n: u32, params: K::Params, buffers: &[&wgpu::Buffer]) -> usize {
        self.add_over::<K>(gpu, n, n, params, buffers)
    }

    /// Record a launch of `items` invocations for a kernel whose params say
    /// `n` items in all — one slice of a larger job (`Tracer::sample_in`).
    pub fn add_over<K: Kernel>(
        &mut self,
        gpu: &Gpu,
        n: u32,
        items: u32,
        mut params: K::Params,
        buffers: &[&wgpu::Buffer],
    ) -> usize {
        assert_eq!(buffers.len(), K::BINDS.len(), "{}: binding count", K::NAME);
        let dispatch = Dispatch::new(items);
        let head: &mut [u32] = bytemuck::cast_slice_mut(bytemuck::bytes_of_mut(&mut params));
        head[0] = n;
        head[1] = dispatch.x;
        let pipeline = self.pipeline::<K>(gpu);
        let params = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("phiray.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }];
        entries.extend(buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: b.as_entire_binding(),
        }));
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(K::NAME),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        self.steps.push(Step { pipeline, group, params, dispatch });
        self.steps.len() - 1
    }

    /// Update a step's params (per-frame data) without rebuilding anything.
    pub fn set<P: bytemuck::Pod>(&self, gpu: &Gpu, step: usize, offset: u64, data: &P) {
        gpu.queue.write_buffer(&self.steps[step].params, offset, bytemuck::bytes_of(data));
    }

    /// Replay every step in one command buffer.
    pub fn run(&self, gpu: &Gpu) {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.record(&mut enc, 0..self.steps.len(), None);
        gpu.queue.submit([enc.finish()]);
    }
    /// Record `steps` into `enc` as one compute pass, stamped by `writes`
    /// when given (`Clock::run`).
    pub(super) fn record(&self, enc: &mut wgpu::CommandEncoder, steps: std::ops::Range<usize>, writes: Option<wgpu::ComputePassTimestampWrites>) {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: writes });
        for s in &self.steps[steps] {
            cp.set_pipeline(&s.pipeline);
            cp.set_bind_group(0, &s.group, &[]);
            cp.dispatch_workgroups(s.dispatch.x, s.dispatch.y, 1);
        }
    }

    /// Replay one step in its own command buffer.
    pub fn run_step(&self, gpu: &Gpu, step: usize) {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            let s = &self.steps[step];
            cp.set_pipeline(&s.pipeline);
            cp.set_bind_group(0, &s.group, &[]);
            cp.dispatch_workgroups(s.dispatch.x, s.dispatch.y, 1);
        }
        gpu.queue.submit([enc.finish()]);
    }

    /// Drop recorded steps, keep compiled pipelines.
    pub fn clear(&mut self) {
        self.steps.clear();
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Pipelines this pass uses (one per kernel type).
    pub fn compiled(&self) -> usize {
        self.cache.len()
    }
}
