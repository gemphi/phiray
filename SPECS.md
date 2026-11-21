# phiray — specs

What each part must do. One row is one requirement, with the Apple
standard it answers and the reference it was read from. "Today" says
whether the crate already meets it. Apple's standards row by row, with the
evidence, are in [docs/APPLE.md](docs/APPLE.md).

## Sources

| Short | What | Where (from `gemphi/`) |
|---|---|---|
| **HM** | Metal headers | `phi/apple-source/ruskit/Interfaces/MacOSX/Metal/Headers/` |
| **HF** | MetalFX headers | `phi/apple-source/ruskit/Interfaces/MacOSX/MetalFX/Headers/` |
| **HK, HQ, HD** | MetalKit, QuartzCore, Model I/O headers | same root: `MetalKit`, `QuartzCore`, `ModelIO` |
| **MP** | Metal Performance Shaders ray intersector and SVGF | `phi/apple-source/ruskit/MetalPerformanceShaders.framework/Versions/A/Frameworks/MPSRayIntersector.framework/Versions/A/Headers/` |
| **REF** | the Meta renderers, shallow submodules | `phikit/crates/phiray/reference/<repo>/` |
| index | every reference, one line each | [reference/README.md](reference/README.md) |
| index | what was ported from the Apple tree, and what is only a transcription | [docs/PORT_LEDGER.md](docs/PORT_LEDGER.md) |

Rule for every row: a part that changes pixels ships with a fixture that
pins them, and a part that claims speed ships with the GPU time before and
after.

## Measure

| Part | Must | Apple | Reference | Today |
|---|---|---|---|---|
| GPU clock | report GPU time for each dispatch of a sample, not the time to submit it | HM `MTLCounters.h:43,166`, `MTLCommandBuffer.h:289,294` | - | met: `gpu/clock.rs`, `Tracer::time`; a hard sync where the adapter has no timestamps |
| counters | count node visits, triangle tests and rays a pixel into a spare plane, off by default | - | - | no |

## Acceleration structure

| Part | Must | Apple | Reference | Today |
|---|---|---|---|---|
| two levels | hold one lower tree a mesh and one top tree over instances | HM `MTLAccelerationStructure.h:201,1026,1199` | - | met, on the CPU |
| opaque | let a shadow ray skip the material read on geometry marked opaque, and test the triangle before any material read otherwise | HM `MTLAccelerationStructure.h:101,106,144` | - | no |
| node load | load each node once a step of the walk | - | - | no: three loads |
| refit | refit only the meshes whose vertices moved and upload only their ranges | HM `MTLAccelerationStructureCommandEncoder.h:44,75,180` | mvp `extensions/mvpraymarch/bvh.cu:156-199` | no: every mesh, whole arrays |
| top tree on the GPU | rebuild the top tree a frame with the Morton, sort, Karras and Refit kernels | same | mvp `extensions/mvpraymarch/bvh.cu:30-153` | kernels exist, the tracer does not call them |
| wide node | hold four children's boxes in one node as arrays | - | HVVR `libraries/hvvr/raycaster/bvh_node.h:17-37` | no: 32-byte binary node |
| ray query | walk through hardware ray queries behind a feature, under the same `trace` and `trace_blas` | HM `MTLAccelerationStructure.h:201` | - | no |

## Primitive kinds

| Part | Must | Apple | Reference | Today |
|---|---|---|---|---|
| the seam | choose the test by the instance's kind in one place, `trace_blas` | intersection function tables HM `MTLIntersectionFunctionTable.h:126` (the idea; not the mechanism) | - | triangles only |
| triangle | be watertight | - | - | met |
| splat | bound a Gaussian by its opacity radius; return alpha = exp(-D) x opacity from a transform made once at compile | - | vkraygs `src/shader/raygs_projection.comp:208-250`, `src/shader/raygs_splat.frag:15` | no |
| ellipsoid medium | store 10 floats; give entry and exit analytically; integrate density along the ray in closed form | - | volumetric_primitives `volprim/integrators/common.py:47-65, 193-239, 346-367` | no |
| brick | test an oriented box by slabs in its frame; sample RGBA trilinearly inside | - | mvp `extensions/mvpraymarch/utils.h:728-824`, `extensions/mvpraymarch/primsampler.h:44-66` | no |
| sparse voxels | return ordered `t_min, t_max` intervals a leaf | - | NSVF `fairnr/clib/src/intersect_gpu.cu:170-237` | no |
| curve | test a cubic segment with radius; give the tangent for shading | curve geometry HM `MTLAccelerationStructure.h:548` | HairMSNN `headers/model.h:86-91` | no |
| motion | move geometry within the shutter | motion geometry HM `MTLAccelerationStructure.h:347,368` | - | no |

## Queries and media

| Part | Must | Apple | Reference | Today |
|---|---|---|---|---|
| closest, any | return the nearest hit, or stop at the first | MP `MPSRayIntersector.h:645` | - | met |
| collect all | return every overlapped primitive up to a cap, with the union range | - | mvp `extensions/mvpraymarch/utils.h:728-824` | no |
| overlap stack | keep the media a ray is inside, 32 deep, pushed on entry and popped on exit | - | volumetric_primitives `volprim/integrators/stack.py:12-69` | no |
| transmittance | make a shadow ray through media a product of closed-form integrals, one a primitive | - | volumetric_primitives `volprim/integrators/volprim_prb.py:304-432` | no |
| marching | sample only inside intervals, in proportion to chord, and stop on optical depth | - | NSVF `fairnr/clib/src/sample_gpu.cu:108-202`, `fairnr/modules/renderer.py:193-218` | no |

## Sampling

| Part | Must | Apple | Reference | Today |
|---|---|---|---|---|
| adaptive | stop sampling a settled pixel in the viewport as in a final render; dispatch over the open list | HM `MTLRasterizationRate.h:244`, `MTLComputeCommandEncoder.h:236` | - | final render only |
| numbers | draw each dimension from a blue-noise or low-discrepancy sequence | - | - | one hash chain |
| footprint | carry a footprint with each primary ray | - | vkraygs `src/shader/raygs_projection.comp:210-220` | no |
| foveation | cast nested layers at their own resolution around a focus point | HM `MTLRasterizationRate.h:244` | CLOD-3DGS `src/foveation/foveated_layers.h:25-33`; HVVR `libraries/hvvr/raycaster/foveated.h:32` | no |
| budget | pick detail from the median GPU time of recent frames | - | CLOD-3DGS `src/vkgs/engine/engine.cc:1366-1394` | slices, from CPU time |
| beam | start a tile's primary rays from the tree cut its frustum touches | - | HVVR `libraries/hvvr/raycaster/traverse_avx.h:316, 412-468` | no |

## Visibility, reconstruction, present

| Part | Must | Apple | Reference | Today |
|---|---|---|---|---|
| vis | keep face, instance and barycentrics of the first hit a pixel for every mode | - | DRTK `drtk/rasterize.py:64`; pytorch3d `pytorch3d/renderer/mesh/rasterizer.py:20-63` | Solid and Wire only |
| temporal | on a camera move, reproject the last image by motion and depth and clamp it to the new neighbourhood | HF `MTLFXTemporalScaler.h:214-337` | HVVR `libraries/hvvr/raycaster/gpu_foveated.cu:269-307` | no: the sums are zeroed |
| denoise | steer the a-trous filter by variance | MP `MPSSVGF.h:109,284,430,541`; HF `MTLFXTemporalDenoisedScaler.h:278-476` | - | a luminance stop |
| compositor | blend foveation layers with a smooth seam | - | CLOD-3DGS `src/shader/compositor.frag:39-76` | no |
| cache | cut a path short and add a cached remainder at the first hit, in its own sums | - | HairMSNN `cuda/hair_msnn.cu:135-185, 273-356` | no |
| warm start | keep compiled kernels on disk across runs | HM `MTLBinaryArchive.h:61,82,128` | - | no: `cache: None` |
| present | show a frame without a blocking wait or a CPU copy | HM `MTLCommandBuffer.h:319,348`; HQ `CAMetalLayer.h:54` | - | blocking readback, CPU swap, upload |

## Surfaces

| Part | Must | Apple | Reference | Today |
|---|---|---|---|---|
| materials | cover the PBR vocabulary, physical lens, tangents | HD `MDLMesh.h:235-682`, `MDLMaterial.h:130-153` | - | met |
| textures | sample filtered, mipped, compressed textures | HM `MTLTexture.h:28`, `MTLSampler.h:135` | - | RGBA8 words, bilinear by hand |
| per-face texture | texture a face without a UV map | - | Replica-Dataset `ReplicaSDK/shaders/atlas.glsl:10-189` | no |
| vector texture | sample text and paths with exact coverage at any zoom | - | AnalyticTextXR `al/vg/shaders/Vg.hlsli:119` | no |
| hair | scatter with four lobes on the strand's tangent frame | - | HairMSNN `cuda_headers/disney_hair.cuh:186, 277` | no |

## Gradients

| Part | Must | Apple | Reference | Today |
|---|---|---|---|---|
| edge gradient | give gradients to vertices through visibility, read from `vis` and the image alone | - | DRTK `drtk/edge_grad_estimator.py:20-180` | no |
| soft blend | blend K hits a pixel by distance and depth | - | pytorch3d `pytorch3d/renderer/blending.py:117-239` | no |

## Not taken

Per-geometry function tables, property-bag materials, a wavefront rewrite
before measurement, frame interpolation, anything from the `metal` or
`metalfx` Rust transcriptions, and every training loop in the references.
