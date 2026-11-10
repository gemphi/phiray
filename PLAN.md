# phiray — plan

The folder tree, today and suggested, and the order to build it in. What
each part must do is in [SPECS.md](SPECS.md); the diagrams are in
[ARCHITECTURE.md](ARCHITECTURE.md). Nothing here is measured: the first
step is to measure.

## The folder tree today

| Path | Holds | Files |
|---|---|---|
| `src/lib.rs` | modules and re-exports | 1 |
| `src/hot.rs` | live WGSL reload | 1 |
| `src/kernel/` | `Kernel`, `Bind`, `Dispatch`, `BLOCK`; Denoise, Convolve, Bilateral, MatMul, Skin | 4 + 2 WGSL |
| `src/mode.rs` | `Mode`: Wire, Solid, Preview, Rendered | 1 |
| `src/gpu/` | `Gpu`, `GpuArray`, `Pass`, `Clock`, `Visibility`, readback, the adapter list | 8 |
| `src/bvh/` | the SAH tree; the GPU tree: Morton, radix sort, Karras, Refit | 11 |
| `src/trace/` | `Scene`, `Tracer` (built, sampled, read: `build.rs`, `sample.rs`, `read.rs`), camera, forest, pose, lamp, light, atlas, materials, procedurals | 26 |
| `src/trace/wgsl/` | every tracing and display kernel | 40 WGSL |
| `src/trace/{forest,ibl,scene,smaa}/` | parts of the above | 5 |
| `src/sim/` | rigid bodies: broad phase, GJK, EPA, SAT, contact | 16 |
| `docs/` | `APPLE.md`, `PORT_LEDGER.md`, `ROADMAP.md`, one page a subject | - |
| `reference/` | 32 Meta repositories, Warp, the index | - |

## The folders to add (suggested)

| Path | Holds | Read from | Order |
|---|---|---|---|
| `src/gpu/clock.rs` | GPU time a dispatch by timestamp query; a hard sync when queries are missing (done) | `MTLCounters.h:43,166` | 1 |
| `src/trace/wgsl/count.wgsl` | counters a pixel: node visits, triangle tests, rays | - | 1 |
| `src/trace/wgsl/bvh.wgsl` | the two inner-loop fixes | `MTLAccelerationStructure.h:144` | 2 |
| `src/trace/adapt.rs` | adaptive sampling in the viewport, dispatch over open pixels | `MTLRasterizationRate.h:244`, `MTLComputeCommandEncoder.h:236` | 3 |
| `src/gpu/cache.rs` | compiled kernels kept on disk | `MTLBinaryArchive.h:61` | 3 |
| `src/vis/mod.rs`, `src/trace/wgsl/walk.wgsl` | the visibility record for Preview and Rendered: the four words a pixel `gpu::Visibility` already holds for Solid and Wire, written by a walk-only kernel | DRTK, pytorch3d, HVVR | 4 |
| `src/recon/temporal.rs`, `wgsl/reproject.wgsl` | motion vectors, reprojection, neighbourhood clamp, variance | HVVR, `MTLFXTemporalScaler.h:337`, `MPSSVGF.h:109` | 5 |
| `src/beam/mod.rs`, `wgsl/beam.wgsl` | tile frusta, the cut, the starting stack a tile | HVVR | 5 |
| `src/prim/mod.rs` | the kind of an instance, the layout of each kind's arrays | - | 6 |
| `src/prim/splat.rs`, `wgsl/splat.wgsl` | Gaussian splats: compile, bound, closed-form response | vkraygs, volumetric_primitives | 6 |
| `src/lod/mod.rs` | prefix order, the frame-time budget | CLOD-3DGS | 6 |
| `src/prim/ellipsoid.rs`, `src/medium/`, `wgsl/medium.wgsl` | ellipsoid media, the overlap stack, transmittance | volumetric_primitives | 7 |
| `src/prim/brick.rs`, `src/prim/voxel.rs` | bricks in oriented boxes, sparse voxels | mvp, NSVF, lightplane | 8 |
| `src/prim/curve.rs`, `wgsl/curve.wgsl`, `wgsl/hair.wgsl` | cubic strands, the hair BSDF | HairMSNN, `MTLAccelerationStructure.h:548` | 9 |
| `src/sample/` | foveation layers, blue noise, ray footprints | CLOD-3DGS, HVVR, DONERF | 9 |
| `src/surface/` | per-face textures, analytic vector textures | Replica-Dataset, AnalyticTextXR | 10 |
| `src/grad/` | edge gradients over `vis` | DRTK, pytorch3d | 11 |
| `src/hw/` | hardware ray query behind a feature | `MTLAccelerationStructure.h:201` | 12 |
| `src/gpu/mod.rs` | for the web: the device opened through the async door that has no caller today; a constructor that adopts a device and queue another crate already holds | phikit `docs/web.md`, blockers 1 and 3 | with 3 |
| `src/gpu/present.rs` | the traced image handed on as a texture, with no readback: what the web needs and what the desktop's present path wants | `CAMetalLayer.h:54`; phikit `docs/web.md` | with 3 |

Names are singular: files, folders and types. A name for many things is a
collective noun (`Forest`, `Histogram`), a `...List` (`LampList`,
`GpuAdapterList`, `BIND_LIST`), or failing both a `P` prefix; counts end in
`_COUNT`, caps start `MAX_`, lengths in words end in `_LEN`. `tests/`,
`examples/` and `benches/` keep Cargo's names.

Suggested, not decided: `src/trace/wgsl/` holds 38 files in one folder. As
kernels are added, group them by stage (`walk/`, `shade/`, `display/`,
`solid/`) so a stage's kernels sit together.

## Order of work

```mermaid
graph LR
    w1["1 measure<br/>GPU time, counters"]
    w2["2 inner loop<br/>shadow rays, node loads"]
    w3["3 viewport<br/>adaptive, Preview warm start, display path"]
    w4["4 vis buffer"]
    w5["5 temporal reuse · beam"]
    w6["6 splats · lod"]
    w7["7 ellipsoid media"]
    w8["8 bricks · sparse voxels"]
    w9["9 curves and hair · sample sets"]
    w10["10 surface kinds"]
    w11["11 gradients"]
    w12["12 ray query"]
    w1 --> w2 --> w3 --> w4
    w4 --> w5
    w4 --> w6 --> w7 --> w8
    w6 --> w9 --> w10
    w4 --> w11
    w2 --> w12
```

## Why this order

```mermaid
graph TB
    a["1 to 3 change no structure and touch every frame"]
    b["4 is the hinge: temporal reuse, the beam prepass and gradients all read vis,<br/>and it is the planes Denoise and Edge already use, kept on purpose"]
    c["6 first among the kinds: most of the Meta renderers draw splats,<br/>and the radix sort they need is already in bvh/radix.rs"]
    d["7 before 8: an ellipsoid's integral is closed-form, no marching,<br/>so fog and smoke arrive without a sampler"]
    e["12 last: an experimental API and a second kernel family,<br/>behind the same trace and trace_blas seam"]
    a --> b --> c --> d --> e
```

## What each new kind needs, end to end

```mermaid
graph TB
    k1["a compile step<br/>prim/kind.rs: file to flat arrays"]
    k2["a bound<br/>one box a primitive, in the kind's own lower tree"]
    k3["a test<br/>wgsl: ray against one primitive"]
    k4["a query<br/>closest · any · collect all · ordered"]
    k5["a shade branch<br/>surface, emission, or a medium"]
    k6["a pose path<br/>repose: the arrays change, the tree refits"]
    k7["a truth test<br/>one fixture, checked against the reference renderer's output"]
    k1 --> k2 --> k3 --> k4 --> k5 --> k6 --> k7
```
