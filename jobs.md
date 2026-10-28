# Jobs — `phiray`

| State | Open | Done |
|---|---|---|
| Material Preview + Rendered live (GPU ray tracing) | 8 | 30 |

## Todo

- [ ] Scenes beyond twice the per-binding limit: stream geometry per object (Cycles-style device memory management)

- [ ] Mapped staging ring + mmap file → ring (`GpuArray` itself is done)
- [ ] GPU radix sort + GPU leaf packing (build is fully on the GPU then; today sort/pack run on the CPU)
- [ ] Textures on roughness / metallic / emission inputs
- [ ] Keep the image on the GPU: gpui on Windows draws with DirectX 11, so this needs a shared D3D texture handle between wgpu (DX12) and gpui — sizeable; readback is throttled meanwhile
- [ ] Instancing (one BVH per mesh + top level) so object moves don't rebuild
- [ ] Preview raster: instancing, texture array, tiled thumbnails
- [ ] Keep skinned vertices on the GPU for Preview/Rendered (no readback once those modes draw)

## Done

- [x] **A third material slot: the other colour of a Base Color Mix** (`Mat::color_proc2`, `MAT_WORDS` 120 → 164; `mix.wgsl`): two procedural textures on one Base Color, blended per pixel by Blender's `ramp_blend` — the slot carries the node's mode where a first slot carries its target and the Factor where a bump slot carries its Strength, so the record grew by one slot and no more
- [x] **Fresnel and Layer Weight as procedural slots** (`procedural.wgsl`'s `PROC_FRESNEL`, `PROC_LAYER_FRESNEL`, `PROC_LAYER_FACING`; `trace/fresnel.rs` the CPU twin): a slot whose value is the hit's angle to the ray rather than a texture — Cycles' `svm_node_fresnel` and `svm_node_layer_weight` over `fresnel_dielectric_cos`. `shade.wgsl` and `preview.wgsl` set `proc_n` / `proc_wi` per hit before any slot is read, so a Mix Shader keyed on either shades by pixel instead of by one folded number. The baker's stand-in is the same function's hemispherical mean (`fresnel_dielectric_Fss`, `1 − 2/(p + 2)`), and `trace/fresnel.rs`'s test holds the two together by quadrature; `tests/procedural.rs` renders a wide-angle wall and finds the grazing corners taking the second closure
- [x] Crate created, workspace member, entry `Ray`
- [x] `Mode` ↔ Blender shading names
- [x] `Kernel` trait + `Dispatch` sizing (tested to 20 M items)
- [x] `Half`/`Node` layout (16 bytes) + `morton` with group bits (tested)
- [x] Warp reference moved here with `ANALYSIS.md`
- [x] `Gpu { device, queue }` (headless open, `Ray::gpu()`)
- [x] `GpuArray<T>`: grow by doubling, `write` reports rebind, `read` for readback
- [x] `Pass`: pipeline cache by `Kernel::key()`, recorded steps, `set` params, `run` replay
- [x] `kernels::Skin` (any number of influences, CSR) + CPU twin; GPU == CPU on 70 001 verts, replay after bone change
- [x] phiphx armature data runs through `Skin` in Phiber (`RaySkin`); rigged fixtures match the CPU path
- [x] LBVH: `Bvh` CPU reference (Karras links, packed leaves, depth < 32) + `GpuBvh` (Morton, Karras, Jacobi refit kernels); GPU == CPU bit-exact up to 100 003 prims (`tests/bvh.rs`)
- [x] Per-frame refit: `GpuBvh::update` replays `depth + 1` recorded steps
- [x] `Tracer` + `trace.wgsl`: 32-slot stack traversal, Möller, smooth normals, sun + shadow, 6-ray AO, sky (`tests/trace.rs`)
- [x] Realistic shading: file lights (sun/point/spot/area, soft shadows), world environment, roughness/metallic GGX, glossy reflection, ACES + sRGB, 32-sample progressive refinement with AA, floor grid with X/Y axes
- [x] Base-colour textures sampled per hit: `Atlas` (one buffer), bilinear, sRGB→linear; packed textures reach the render (`tests/trace.rs`, TEXTURED-PACKED fixture)
- [x] Render thread in Phiber (`rendered/worker.rs`): UI posts scene/view, thread refines the newest view, UI polls frames — orbiting never waits on the GPU
- [x] Up to 16 file lights (4 sampled per pixel, reweighted) (`tests/trace.rs`)
- [x] `Tracer::sample` / `pixels`: the render thread reads back samples 1–4, then every 4th, then the final one
- [x] Material Preview also ray traced (studio rig, file lights/world ignored, like Blender's default preview)
- [x] Principled optics: emission, glass (refraction, IOR Fresnel, tinted, ≤ 8 events), alpha cut-outs; shadows pass glass; reflections shade what they hit (`tests/trace.rs`, MATERIALS-BSDF fixture)
- [x] Glowing meshes light their surroundings: indirect diffuse rays run to the first hit when the scene has emitters (`tests/trace.rs`)
- [x] Next-event estimation for glowing meshes: one emitter triangle + point per sample, shadowed; one sample is within 30% of the converged result (`tests/trace.rs`)
- [x] Emitters picked by power (area × luminance, `Lamps` cdf); one sample stays within 30% even with 300 faint specks (`tests/trace.rs`)
- [x] Tangent-space normal maps (per-triangle tangents from UVs, Normal Map strength); raw bytes, colour textures stay sRGB (`tests/trace.rs`, NORMAL-MAP fixture)
- [x] `Tracer::fits` / `Gpu::max_binding`: scenes over the device's binding limit are refused and the render thread tells the painter to fall back (no crash)
- [x] Compact scene (`trace/scene.rs`): shared vertices (32 B), faces (16 B), material table; CPU-built compacted BVH (half the nodes). LAND (7.2 M tris) and Mannequins (6.9 M) now render within a 128 MB binding limit (~340 MB total instead of 1.6 GB)
- [x] Smooth shading: per-corner normals (split/custom, or smooth faces) from `SceneMesh::corners`; flat faces stay flat
- [x] Split bindings: nodes, faces and vertices each span two bindings, data (table + atlas) and pixels (packed + accumulation) share one each — 8 storage bindings, up to 2× the device limit per array; split render == whole render (`tests/trace.rs`)
- [x] MIS (power heuristic) between emitter sampling and diffuse rays; samples average in linear light (display values are mapped back through inverse ACES), `Tracer::radiance`; near / middle / far emitters match closed-form irradiance and each weight mutation fails the test (`tests/emitters.rs`)
- [x] Lower build memory: `Bvh::build_compact` (breadth-first, drops sort/link arrays early), `Scene::seal` frees the dedup maps; FLOWER-CITY (7.6 M tris) opens + renders. Heavy-file check: `PHIBER_TEST_FILES="a.blend;b.blend" cargo test --release -p phiber --lib fixture_renders -- --nocapture`
- [x] Build without a global vertex map: `Scene::{vertex, material, face}`; phiber shares vertices per mesh (`rendered/verts.rs`, key = vertex + corner-normal id)
- [x] Phiber stops copying meshes every frame (`Arc` meshes shared by the model and `BObject3D::Loaded`)
- [x] Textures on Roughness, Metallic and Emission Color (`Tri::with_maps`, 3 ids per material, `Mat` is 17 words); each multiplies its value, emitters sample the emission image too (`tests/maps.rs`)
- [ ] ~20 M-triangle files (HEXA-COMBINED 20.8 M, Smile&Driveaway 20.2 M): faces (330 MB) exceed 2× a 128 MB binding on lavapipe — more face bindings when the device allows, or per-object streaming
- [x] Phiber Rendered shading draws the phiray image (`view3d/rendered.rs`); FLORIS/THYMUS render from their saved views

> Format: status table, `- [ ]` open, `- [x]` done. Move lines, don't delete them.
