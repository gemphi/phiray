# Warp → phiray: how it works on the GPU, and what we take

Source: `SOURCE.txt` (NVIDIA Warp, Apache-2.0, see `LICENSE.md` and `licenses/`).
This folder is a **read-only reference subset** (`native/`, `_src/`, `design/`). Nothing here is compiled.
Warp is a *simulation / compute* framework, not a renderer. Its value to phiray is the
GPU compute core: kernel model, BVH, ray queries, textures, zero-copy buffers.

phiray renders with **wgpu** (Vulkan / Metal / DX12 / WebGPU), not CUDA. So we port *ideas and
data layouts*, not code paths. Every row below says the Warp mechanism and our wgpu equivalent.

## 1. The whole pipeline in one picture

```
Python @wp.kernel ──codegen.py──► C++ source ──NVRTC──► PTX ──cuModuleLoad──► kernel
                                             └─clang/LLVM ORC JIT──► CPU fn (same source)
wp.launch(kernel, dim, inputs) ──► wp_cuda_launch_kernel(block_dim=256, grid, stream)
arrays: one pointer + dtype + shape; mempool per stream; DLPack / GL-buffer zero-copy
```

phiray equivalent:

```
Rust trait Kernel { const WGSL; type Params: Pod }  ──► naga/wgpu ShaderModule (cached by hash)
Pass::dispatch(kernel, n) ──► workgroups = ceil(n / 256) split over x/y (65535 cap)
buffers: GpuArray<T: Pod> (one wgpu::Buffer) ; mapped staging ring for uploads
```

## 2. Kernel model (`_src/codegen.py`, `native/warp.cu`)

| Warp | Detail | phiray (wgpu) |
|---|---|---|
| One source, two targets | Same C++ text compiled by NVRTC (GPU) and embedded clang (CPU) — `codegen.py:7156` header, `clang/clang.cpp:497` | One WGSL source; CPU fallback = same Rust `fn` over `rayon` (we already do this for modifiers) |
| Thread index | `_idx = (blockIdx.z*gridDim.y + blockIdx.y)*(gridDim.x*blockDim.x) + blockIdx.x*blockDim.x + threadIdx.x; if (_idx >= dim.size) return;` (`codegen.py:7359`) | `let i = gid.x + gid.y * (groups_x * 256u); if (i >= p.n) { return; }` |
| Block size | `block_dim = 256`; lean grid: `gx ≤ 2^24/256`, spill to `gy`,`gz` ≤ 65535 (`warp.cu:5567`) | `@workgroup_size(256)`; `groups_x ≤ 65535`, spill to `y` |
| Grid-stride | One thread loops `_idx += blockDim*gridDim` to cover overflow instead of a 3-D grid (`codegen.py:7391`) | Not needed: wgpu gives 3-D dispatch; keep the lean form |
| Cache | SHA-256 over kernel source + options → `wp_{name}_{hash7}` in user cache dir, `.meta` gate (`context.py:2935, 4385`) | `blake3(wgsl + defines)` → `PipelineCache` on disk (`wgpu::PipelineCache` on Vulkan) |
| Launch struct | Args packed once per type tuple (`_invoke_cache`, `context.py:10734`) | One `BindGroupLayout` per `Kernel` type; bind groups cached per buffer set |
| Graph capture | `ScopedCapture` records launches, replays with no Python (`warp.cu:3434`) | Record a `Vec<PassStep>` once; per frame only `queue.write_buffer(params)` + replay |

Lesson: Warp's speed is *not* exotic. It is (1) no per-call allocation, (2) no per-call argument
marshalling, (3) replaying a recorded launch list. All three map 1:1 to wgpu.

## 3. Memory (`_src/types.py`, `native/warp.cu:736–866`)

* Allocators: host, pinned (`cudaMallocHost`), device, **async mempool on the stream**, managed.
* Frees during capture are deferred so the recorded graph keeps valid pointers.
* Zero-copy out: `__cuda_array_interface__`, `__dlpack__` (waits on the producer stream first).
* GL interop: `RegisteredGLBuffer` maps a VBO/PBO into CUDA; kernels write instance matrices and
  read framebuffers without a CPU copy (`render_opengl.py:2605, 2716, 2788`).

phiray:
* `GpuArray<T>` owns one `wgpu::Buffer` with `STORAGE | COPY_DST`; grow by doubling, never per frame.
* Upload through a **mapped staging ring** (`MAP_WRITE` buffers reused) — the wgpu analogue of pinned memory.
* Our viewport buffers *are* the compute buffers (same device): skinning / instance kernels write
  straight into vertex/instance buffers — this is the RegisteredGLBuffer trick for free.
* File → GPU: `mmap` the `.blend`/`.phx`, slice straight into the staging ring. On Linux the page
  cache makes this near memcpy speed; no server needed for single-scene work. (GPUDirect Storage
  is not used by Warp either.)

## 4. BVH (`native/bvh.h`, `native/bvh.cu`, `native/mesh.h`)

* Node = two halves, 16 bytes each: `{x, y, z: f32, i: u31, leaf: u1}` (`bvh.h:161`).
  Inner: `lower.i` = left, `upper.i` = right. Leaf: `[lower.i, upper.i)` into `primitive_indices`.
* Builders: SAH (16 buckets), median, **LBVH on GPU**, cuBQL.
* LBVH (`bvh.cu:184–577`):
  1. scene bounds by `cub::BlockReduce` (256);
  2. 30-bit Morton code of each centroid, **group id in the high 32 bits** (instances stay contiguous);
  3. radix sort key/index pairs;
  4. neighbour key deltas → leaves → bottom-up hierarchy (Apetrei-style); a thread climbs to the parent and stops when the sibling is not ready;
  5. `mark_packed_leaf_nodes`: any subtree with ≤ `leaf_size` prims **or depth ≥ 32** becomes a leaf.
* Refit: parents + counts arrays, bottom-up kernel (`bvh_refit_kernel`) — used every frame for skinning.
* Traversal: fixed **32-entry stack** (`BVH_QUERY_STACK_SIZE`); closest-point queries visit the near child first; slab test with
  precomputed reciprocal direction (`intersect.h:128`), Möller–Trumbore triangles (`intersect.h:208`).
* Mesh handle is a `u64` id → struct lookup (`mesh_get(id)`); queries: ray, closest point, winding sign.

phiray: same node layout in WGSL (`struct Half { p: vec3f, i: u32 }`, leaf bit in `i`'s top bit),
32-slot `array<u32, 32>` stack, depth capped at build time so the stack never overflows. Build LBVH
in compute (Morton + radix sort passes), refit per frame after GPU skinning.

## 5. Textures (`_src/texture.py`, `native/texture.cpp`, `native/texture.h`)

* GPU: `cuMipmappedArrayCreate` + `cuTexObjectCreate`; hardware filtering via `tex2D`/`tex2DLod`.
* Mips: full chain by 2× box filter; filter/address per axis; normalized coords required for mips.
* CPU fallback: explicit bilinear/trilinear + mip blend.
* Gap in Warp: clang-CUDA path returns zeros for textures.

phiray: `wgpu::Texture` with mips + `Sampler`; compute uses `textureSampleLevel` (no derivatives in
compute). For path tracing keep one **texture atlas array** (`texture_2d_array`) so a single bind
group covers every material.

## 6. Rendering in Warp (`_src/render/render_opengl.py`)

Not a path tracer — a debug rasteriser: GLSL 330, instanced shapes (mat4 split over attribs 3–6,
colours 7–8, divisor 1), Phong sun + fill + 30 % ambient, derivative-faded checker, sky gradient,
grid, FBO with depth32, PBO readback, **tiled rendering** (many cameras, one pass per tile).

Worth taking: the tiled pass (thumbnails, multi-view, asset browser previews in one draw) and the
derivative-faded checker (no moiré on the floor grid).

## 7. What this means for phiray's three renderers

| Renderer | Uses from Warp |
|---|---|
| Solid (phiber canvas today) | nothing on GPU yet; later: GPU skin + instance kernels writing the vertex buffers |
| Material Preview (wgpu raster) | instancing layout, tiled previews, checker, mipmapped textures |
| Rendered (wgpu path tracer) | kernel model, LBVH + refit, 32-stack traversal, Möller, texture array, launch replay |

See `jobs.md` for the ordered work.
