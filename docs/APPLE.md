# Apple's rendering standards against phiray, and how to render faster

Read 2026-11-10 against the working tree. **Nothing was built, run or
measured.** Every "evidence" cell is a line that was read; every "expected
effect" is inference. Rows marked (checked) were re-opened by a second
reader. `PORT_LEDGER.md` says what was ported out of the Apple tree and
`ROADMAP.md` what phiray lacks in breadth; this file is the standards
checklist and the speed list.

## Where the standards are written down

The Rust files named `metal` and `metalfx` are transcriptions with nothing
inside (`PORT_LEDGER.md` section 1). The standards themselves are Apple's SDK
headers, which are on disk. Every Apple citation below is from them.

| Short | Path (from `gemphi/`) |
|---|---|
| **HM** | `phi/apple-source/ruskit/Interfaces/MacOSX/Metal/Headers/` |
| **HF** | `phi/apple-source/ruskit/Interfaces/MacOSX/MetalFX/Headers/` |
| **HK, HQ, HD** | same root: `MetalKit`, `QuartzCore`, `ModelIO` |
| **MP** | `phi/apple-source/ruskit/MetalPerformanceShaders.framework/Versions/A/Frameworks/MPSRayIntersector.framework/Versions/A/Headers/` |
| **R** | `phikit/crates/phiray/src/` |
| **S** | `phikit/crates/phiray/src/trace/wgsl/` |
| **B** | `phikit/crates/phiber/src/view/draw/canvas/view3d/rendered/` |
| **W** | the cargo registry copy of wgpu 29.0.4 |

## The answer in five lines

- On a heavy scene the time is **inside the Trace dispatch, not the
  readback.** The code's own figures: 18 ms a sample for a 12-triangle cube
  at 1920x1080 (B `slices.rs:48-54`) against about 4 s a sample for pose 26
  (B `stepwise.rs:155-156`).
- Two fixes in the inner loop of `bvh.wgsl`, one day each, touch every ray.
- The viewport never turns adaptive sampling on. The kernels exist.
- Material Preview's kernel takes four minutes to compile; the path tracer
  had the same disease and was cured.
- The viewport's timing is wrong in a way that hides all of the above
  (row 6), so fix the clock before trusting any number.

## Measure first

1. **GPU time per dispatch** (Trace, tail passes, readback copy) on the
   cube and on pose 26, with a hard sync after each sample as
   `examples/slice_cost.rs:83-90` does, or with timestamp queries.
2. **Per-pixel counters** in a spare image plane: node visits, triangle
   tests, instance entries, closest-hit and any-hit rays a sample. This
   says whether traversal or shading dominates.

## Ranked speedups

Most gain for the effort first. All work on Windows wgpu today unless said.

| # | Change | Where the time goes today | Expected effect | Days | Risk |
|---|---|---|---|---|---|
| 1 | **Shadow rays test the triangle before reading its material**, and an "all opaque" instance bit skips the read outright | For an any-hit ray each leaf candidate does `mat_at(id)` and two more reads before `tri_t` (S `bvh.wgsl:135-140`, checked). Up to 3 shadow rays a shading point (S `integrate.wgsl:69-82`) and 4 diffuse bounces by default, so most rays on a path are any-hit | Fewer dependent loads on most rays; same image. This is Apple's opaque flag (HM `MTLAccelerationStructure.h:144`) | 1 | Low |
| 2 | **Stop loading each node twice** | Three `node()` loads a step: the node, then both children for their box tests, then the chosen child again at the top of the loop (S `bvh.wgsl:131,147-148`, and `:176,183-184`, checked). Each goes through the chunk-select chain (S `bind4.wgsl:24-30`) | About a third fewer node loads on descent | 1-2 | Low |
| 3 | **Adaptive sampling in the viewport** | The render thread never calls `set_adaptive`; the only caller is the final render (B `stepwise.rs:100`, checked by grep). So every pixel takes all `preview_samples`. `adapt.wgsl` and `gather.wgsl` already exist | Settled pixels stop costing; the view ends when none are open | 1-2 | Low to medium: gate `dilate` and `gather`, which run every sample though flags change only at step ends (S `adapt.wgsl:83-96`) |
| 4 | **Warm start for Material Preview** | `visible()` has six call sites in `shadow.wgsl` (`:68,73,77,93,105,117`, checked), four inside constant 8-tap loops, reached from four calls inside a constant 6-layer loop in `preview.wgsl`. Recorded compile: Preview 247,997 ms, Trace 145 ms (R `trace/build.rs:48-51`, checked). The path tracer names this pattern as the cause of its own 3-5 minute compiles and fixed it (S `integrate.wgsl:53-65`). `cache: None` (R `gpu/pass.rs:50`, checked) | Preview's first use drops from minutes to seconds. This is Apple's binary archive (HM `MTLBinaryArchive.h:61`); wgpu's pipeline cache is Vulkan only | 2-3 | Low |
| 5 | **Display path, the cheap half** | A shown frame costs a new staging buffer, a helper thread and a blocking poll (R `gpu/array.rs:164-175`), three full-frame CPU copies, a CPU channel swap (B `thread.rs:160`, checked) and a new image uploaded by gpui. A final render reads 49n words where 14n are used (B `still.rs:158`) | Removes about three passes over the frame and a thread spawn a frame; matters most in Solid, Wire and Preview, where a frame is one dispatch | 2-4 | Low |
| 6 | **Fix slice timing; add GPU timestamps** (done: `gpu/clock.rs`, `Tracer::time`; the slicer takes the GPU time, `tests/clock.rs`) | With one slice `sample_in` submitted and returned without waiting (R `trace/mod.rs:639-642` at `938afbf`, checked), so the `traced` the slicer stores (B `thread.rs:117-124`, checked) is submit time. The prediction falls to about zero and that mode stays at one slice. GPU work is then booked as "readback", which is why two comments conclude readback is the slow part | Right numbers, right-sized slices, and no single multi-second dispatch after a light scene (the Windows watchdog). `TIMESTAMP_QUERY` is in wgpu-hal for Vulkan and DX12 | 2 | Low |
| 7 | **Leaf locality** | One 16-byte face and three 32-byte `vert()` loads a candidate (S `bvh.wgsl:139-140`, checked). Only faces are in leaf order (R `trace/scene.rs:187-189`) | Better memory locality in leaves, most on multi-million-face meshes | 2-6 | Medium: touches `repose` and `retree` |
| 8 | **A blue-noise or low-discrepancy sampler** | Every dimension is one PCG hash chain (S `shade.wgsl:68`). The blue-noise table is bound to the effects pass only | Less noise a sample at low counts, so fewer samples with row 3 | 2-4 | Medium: tests pin estimator values |
| 9 | **Node format** | The 32-byte node holds its own box; the right child is always left + 1 (R `bvh/sah.rs:101-103`), so that word is free. Two 32-word stacks a walk | Both child boxes in the parent is one load a step; then wide quantised nodes | 5-10 | Medium to high: naga's local-memory lowering already cost 40% once (S `bvh.wgsl:23-27`) |
| 10 | **Temporal reuse on navigation** | Any camera change zeroes the accumulation (R `trace/sample.rs:126-132`) and drops to 1/8 size (B `coarse.rs:17-19`). The denoiser is spatial only | Navigation shows a reprojected, filtered image where it now restarts as noise. This is MetalFX's temporal scaler (HF `MTLFXTemporalScaler.h:337`) and MPS's SVGF (MP `MPSSVGF.h:109`) | 8-15 | Medium to high: ghosting |
| 11 | **Hardware ray queries, behind a feature** | wgpu 29.0.4 sets `EXPERIMENTAL_RAY_QUERY` on Vulkan, DX12 and Metal and has `create_blas` / `create_tlas`. phiray requests no features (R `gpu/mod.rs:231-235`). Every walk goes through `trace` and `trace_blas`, so there is one seam | Large on ray-tracing hardware; unknown on the GTX 1660 Ti named in R `gpu/mod.rs:198` | 10-20 | High: experimental API, a second kernel family |
| 12 | **GPU-resident present** | Two devices force the CPU hop. wgpu-hal can import a D3D11 shared handle as a Vulkan texture (W `wgpu-hal/src/vulkan/device.rs:544`) | Removes readback and upload for every shown frame | 5-10 plus a gpui primitive | High, untested |

## Apple's standards, row by row

| Apple standard | What it standardises | phiray today | Verdict |
|---|---|---|---|
| `MTLAccelerationStructure` (HM `MTLAccelerationStructure.h:201,1026,1199`) | Driver-built two-level BVH, hardware traversal | CPU binned-SAH tree a mesh plus a SAH top tree (R `bvh/sah.rs:85`, R `trace/forest.rs:44-121`); WGSL stack walk | Speedup 11 |
| Build, refit, compact; fast-build against fast-trace (HM `MTLAccelerationStructureCommandEncoder.h:44,75,180`) | GPU build, in-place refit | Refit on the CPU for every mesh, then whole arrays re-uploaded (R `trace/pose.rs:86-95`). The GPU LBVH and refit exist and the tracer does not use them (R `bvh/gpu.rs:11-99`) | Dirty-mesh refit and ranged upload, 2-3 d, when playback is the complaint |
| Opaque flags (HM `MTLAccelerationStructure.h:101,106,144`) | Opaque geometry skips any-hit shading | A shadow ray reads the material first | Speedup 1 |
| Intersection and visible function tables (HM `MTLIntersectionFunctionTable.h:126`) | Per-geometry callable shaders | Fixed material record, tests inline | Do not adopt (`PORT_LEDGER.md` 3.5) |
| Motion and curve geometry (HM `MTLAccelerationStructure.h:347,368,548`) | Motion blur, hair | Triangles only | A feature gap, not speed |
| `MTLBinaryArchive` (HM `MTLBinaryArchive.h:61,82,128`) | Compiled pipelines kept on disk | In-process map only | Speedup 4 |
| Async pipeline creation (HM `MTLDevice.h:864,1313,1439`) | Compile off the frame path | A helper thread and a slot a kernel; a caller that wants a kernel still compiling blocks | Mostly met |
| Indirect dispatch (HM `MTLComputeCommandEncoder.h:236`) | GPU-decided work size | Full-size dispatch every time | 1 d, after speedup 3 |
| GPU counters and timestamps (HM `MTLCounters.h:43,166`, `MTLCommandBuffer.h:289,294`) | GPU-side timing | CPU wall clocks only | Speedup 6 |
| Completion handlers and present (HM `MTLCommandBuffer.h:319,348`) | Results without a blocking wait | Blocking map and poll; `AsyncReadback` is ported and has no caller | Speedup 5 |
| Hardware textures: arrays, anisotropy, GPU mips, BC (HM `MTLTexture.h:28`, `MTLSampler.h:135`) | Filtered, compressed, bindless | RGBA8 words in a storage buffer, bilinear by hand (S `texture.wgsl:76-96`), CPU mips | Memory more than speed; replace the six runtime `%` with selects now (0.5 d) |
| Heaps, residency, sparse (HM `MTLHeap.h:35`, `MTLResidencySet.h:47`) | Sub-allocation, streaming | One buffer an array, split over 2 or 4 bindings; oversize scenes refused | `ROADMAP.md`; not speed |
| `MTLRasterizationRateMap` (HM `MTLRasterizationRate.h:244`) | Variable shading rate | Per-pixel adaptive sampling, final render only | Speedup 3 |
| Shared storage (HM `MTLResource.h:84`) | Zero copy between CPU, GPU and APIs | The image crosses two devices through the CPU | Speedup 12 |
| `MTLFXSpatialScaler` (HF `MTLFXSpatialScaler.h:185`) | Render small, upscale | A navigated view starts at 1/8 and doubles; gpui stretches it | Low value |
| `MTLFXTemporalScaler`, `MTLFXTemporalDenoisedScaler` (HF `MTLFXTemporalScaler.h:214-337`, `MTLFXTemporalDenoisedScaler.h:278-476`) | History reuse from motion, depth and jitter; denoise plus upscale for ray-traced input | A new view restarts at sample 0. Depth and instance planes exist; no motion vectors, no specular albedo or roughness planes | Speedup 10 |
| `MTLFXFrameInterpolator` (HF `MTLFXFrameInterpolator.h:314`) | Generated in-between frames | Absent | No use to a progressive tracer |
| `MPSRayIntersector` (MP `MPSRayIntersector.h:645`; Apple deprecated it for Metal ray tracing) | Wavefront ray buffers, any-hit type | Megakernel; `first` is any-hit; a watertight test | Not now |
| `MPSSVGF` (MP `MPSSVGF.h:109,284,430,541`) | Reprojection plus variance-guided a-trous | A-trous with a luminance stop in place of variance (R `kernel/denoise.wgsl:98-99`) | Speedup 10 |
| `MTKView` draw on demand (HK `MTKView.h:185,221`) | Redraw only on change | The window is woken when a frame lands (B `wake.rs:4-9`) | Met |
| `CAMetalLayer`, `CAMetalDisplayLink` (HQ `CAMetalLayer.h:54`, `CAMetalDisplayLink.h:33`) | GPU-resident swapchain, paced present | RGBA8 read back, swapped on the CPU, re-uploaded by gpui on DirectX 11 | Speedups 5 and 12 |
| `MDLMesh` tangents and baking, `MDLMaterial`, `MDLCamera`, `MDLLightProbe` (HD `MDLMesh.h:235-682`, `MDLMaterial.h:130-153`) | PBR vocabulary, physical lens, SH irradiance | All present; tangents derived a hit; no baking | Met |

## The rest of the list

| Topic | Evidence | Verdict |
|---|---|---|
| Megakernel against wavefront | One dispatch a sample (R `trace/shader.rs:276-290`) | Not now. 15-25 d, and the megakernel is why the tracer is small. Let measurement 2 decide |
| Raster prepass for Preview | Only Solid and Wire use it (R `trace/visible.rs:20-22`); Preview casts a primary ray a layer | Yes for Preview, 1-2 d, after speedup 4. No for Rendered: it needs lens and filter jitter a pixel |
| Light sampling | One lamp by reservoir, up to 64 (S `light.wgsl:127-139`) | A light tree helps noise, not time. Not a speed item |
| Russian roulette | Present, capped at 0.95 (S `integrate.wgsl:209-213`) | Leave |
| 2 or 4 bindings | A scene that fits one binding takes the first branch. A power-of-two shift was measured 21% slower and reverted (R `gpu/array.rs:63-69`) | Leave |
| Pixel filter | 20 bisection steps of three `sin`, an axis, a sample (S `brdf.wgsl:144-154`) | A table, 0.5 d, optional |

What the Meta references offer for speed:

- **DRTK** (`reference/DRTK/src/rasterize/rasterize_kernel.cu:115-161`): one
  thread a triangle over its bounding box with a 64-bit atomic minimum of
  packed depth and index. A compute-only visibility buffer for adapters
  with no hardware raster. Niche: on a software adapter phiray already
  found tracing faster (R `trace/visible.rs:6-10`).
- **pytorch3d** (`reference/pytorch3d/pytorch3d/csrc/rasterize_coarse/rasterize_coarse.cu:76-211`):
  faces binned into screen tiles, then each pixel tests its tile's list.
  The reusable part is the binning, for per-tile lamp lists at primary hits.
- **vkraygs, CLOD-3DGS, HVVR** (added 2026-11-10, not yet read): a
  hardware-rasterised ray-based splatter, continuous level of detail on
  Vulkan, and a hybrid CPU/GPU ray caster with hierarchical visibility.
  Read these three first for rows 9 and 11.

## What not to do

- Port from the `metal` or `metalfx` transcriptions.
- Take the "optimisations" in `appkit/gpu`: Hi-Z, culling, `uniform_cache`
  and `gpu_fallback` were read as stubs or without a call site, and they
  need a raster depth buffer. The five performance write-ups in `phi/`
  hold one measured number; every "after" figure is labelled expected.
- Property-bag materials or per-geometry function tables.
- A power-of-two chunk split (measured 21% slower).
- Drive the slice count from `Gpu::waited` (tried, lost the device,
  B `slices.rs:65-73`) or bound each slice with `wait_for` (tried,
  reverted, R `trace/sample.rs:223-236`, checked).
- Runtime-indexed vectors in the walk (cost 40% a sample).
- A wavefront rewrite, or rasterised primary hits in Rendered, before the
  two measurements.

## Not verified

- No measurements. The numbers are from comments in the code; that the
  4 s figure was a 1440x1440 frame is an assumption.
- Which backend is live on this machine, its binding limit, and whether it
  reports ray query, pipeline cache or D3D11 import.
- That the driver inlines and unrolls Preview as it did Trace is inferred
  from the code's own account of Trace.
- Not read line by line: the procedural WGSL, the Solid, Wire, fx, dof and
  smaa kernels, `bvh/build.rs`, `radix.rs`, `sort.rs`, `sim/`.
- Drift noticed: `MAX_LIGHT` is 64 in code (R `trace/camera.rs:332`) against
  32 in `ROADMAP.md` 3.1; `docs/render-fix-report.md`, cited at
  R `gpu/array.rs:69`, is not in `docs/`.
