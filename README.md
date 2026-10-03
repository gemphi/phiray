# phiray

Phiber's own renderer, next to Cycles. Entry point: `phiray::Ray`.

| Mode | Blender name | Engine |
|---|---|---|
| `Mode::Solid` | SOLID | phiber canvas (today) → GPU skin/instance kernels |
| `Mode::Preview` | MATERIAL | wgpu raster, mipmapped textures, tiled previews |
| `Mode::Rendered` | RENDERED | wgpu compute path tracer (LBVH, 32-slot stack) |

```
src/
├── lib.rs     Ray, re-exports
├── mode.rs    Mode ↔ Blender shading names
├── kernel.rs  Kernel trait, Bind, Dispatch (256 blocks, x→y spill), WGSL prelude
├── bvh/       Aabb, Half/Node (16-byte halves, leaf bit), Bvh (CPU LBVH), GpuBvh + Morton/Karras/Refit kernels
├── gpu/       Gpu (device+queue), GpuArray<T>, Pass (cache + record + replay)
├── trace/     Tracer, Camera, Scene (compact faces/vertices/materials), Tri + wgsl/ (Rendered mode)
└── kernels/   Skin (linear blend skinning, CSR influences) + CPU twin
tests/         gpu.rs (skinning), bvh.rs (build + refit), trace.rs (image) — GPU == CPU; skip without an adapter
reference/
└── warp/      NVIDIA Warp subset (Apache-2.0, read-only) + ANALYSIS.md
```

The GPU design follows `reference/warp/ANALYSIS.md`. The old app shell and docs live in
`crates/apple/apps/phiray` (BGE-replacement UI); this crate is the renderer core.
Status: `jobs.md`.

Writing a kernel: implement `Kernel` (`NAME`, `WGSL`, `BINDS`, `Params` starting with
`n, groups_x`), use `item(gid, p.groups_x)` for the index, add a CPU twin, then
`pass.add::<K>(&gpu, n, params, &[buffers])` once and `pass.run(&gpu)` every frame.

Used by Phiber: `phiber::v2::scene::RaySkin` implements phiphx's `Skinner` with `Skin`,
so Armature deform of opened files runs here.
Phiber's Rendered shading (`view3d/rendered.rs`) builds a `Tracer` per scene and paints its image.
