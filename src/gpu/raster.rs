//! The visibility prepass: a raster pipeline that draws every face of the
//! scene into a render target of hits — face, instance, barycentrics,
//! distance — with a depth test, the way Workbench's prepass finds what
//! each pixel sees, then copies the target into a buffer the compute
//! kernels read (`vis2.wgsl`). The shader is `trace/wgsl/raster.wgsl`.
//! A mesh shown by many instances is one instanced draw (roadmap §4.4,
//! Step 1), as `DRW_shgroup_call_instances` batches Workbench's.

use super::{Gpu, GpuArray};
use wgpu::util::DeviceExt;

/// Bytes a row of the copied target is padded to (`COPY_BYTES_PER_ROW_ALIGNMENT`).
const ROW_ALIGN: u32 = 256;
/// Bytes per pixel of the target: four 32-bit words.
const TEXEL: u32 = 16;

pub struct Visibility {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    target: wgpu::TextureView,
    depth: wgpu::TextureView,
    /// The copy of the target, `stride × h` pixels of four words.
    hits: GpuArray<u32>,
    size: (u32, u32),
    /// Pixels a row of `hits` spans (the row padded to the copy alignment).
    stride: u32,
}

impl Visibility {
    /// A pass over `w × h` pixels drawing with `source` (keyed `key` in the
    /// device's cache) over `buffers` — the faces, vertices and data the
    /// shader's bindings 1.. name — with `params` as its uniform.
    pub fn new<P: bytemuck::Pod>(gpu: &Gpu, key: u64, name: &str, source: &str, buffers: &[&wgpu::Buffer], params: &P, w: u32, h: u32) -> Self {
        let pipeline = gpu.render_pipeline(key, || {
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(source.into()) });
            gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: None,
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba32Uint, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let params = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("phiray.raster.params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }];
        entries.extend(buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() }));
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(name), layout: &pipeline.get_bind_group_layout(0), entries: &entries });
        let (w, h) = (w.max(1), h.max(1));
        let texture = |format, usage| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("phiray.raster"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let target = texture(wgpu::TextureFormat::Rgba32Uint, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let depth = texture(wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let stride = Self::stride_for(w);
        let hits = GpuArray::zeroed(gpu, (stride * h * 4) as usize);
        Self { pipeline, group, params, target, depth, hits, size: (w, h), stride }
    }

    /// Pixels a copied row spans for a target `w` wide: the row padded to
    /// the copy alignment.
    pub fn stride_for(w: u32) -> u32 {
        (w.max(1) * TEXEL).div_ceil(ROW_ALIGN) * ROW_ALIGN / TEXEL
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// The hits, four words a pixel, rows `stride` pixels apart.
    pub fn hits(&self) -> &GpuArray<u32> {
        &self.hits
    }

    /// Draw each of `draws` — `[first vertex, vertex count, first
    /// instance slot, instance count]`, one instanced draw a mesh
    /// (`Trees::draws`) — with `params` and copy the hits out. A pixel
    /// nothing covers reads face `u32::MAX`.
    pub fn draw<P: bytemuck::Pod>(&self, gpu: &Gpu, params: &P, draws: &[[u32; 4]]) {
        gpu.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(params));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("phiray.raster"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: f64::from(u32::MAX), g: 0.0, b: 0.0, a: 0.0 }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            for &[v0, nv, i0, ni] in draws {
                pass.draw(v0..v0 + nv, i0..i0 + ni);
            }
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: self.target_texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: self.hits.buffer(),
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.stride * TEXEL), rows_per_image: Some(self.size.1) },
            },
            wgpu::Extent3d { width: self.size.0, height: self.size.1, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([enc.finish()]);
    }

    fn target_texture(&self) -> &wgpu::Texture {
        self.target.texture()
    }
}
