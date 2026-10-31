# Ports out of the Apple tree

What phikit took from the `apple` reference tree and the `gemphi/phi` tree,
what it left, and what should follow. Surveyed 2026-10-28; destinations
settled 2026-10-29.

**The ports land in the crate that already owns that work.** This ledger
lives in `phiray/docs/` because the GPU ports land in phiray — it owns the
device, the buffers, the `Kernel`/`Pass` model and every image-space pass.
An earlier pass put them in `phiphx::fx` behind a feature; that is the asset
reader, so it needed an optional `wgpu` and a feature gate, and nothing ever
switched it on. The lesson is the rule above, and §2 records where each piece
actually went.

The live trees are **phinet, phikit and puikit**. `apple`, the Blender source
and the `phi/` crates are reference points under `ext/`: they are read and
cited, never built and never depended on.

---

## 1. The survey

Lines are `wc -l` over `*.rs`. "Real" means code that computes something;
"declared" means a trait method with no body, or a body that assigns a field
and returns.

| Area | Lines | What it actually is | Verdict |
|---|---|---|---|
| `apple/src/appkit/gpu/` | 22,770 | **A wgpu renderer.** 97 files call `wgpu::`, **zero** call `msg_send!` or name an `MTL*` type. 20 WGSL shaders, no `.metal`. One `todo!` in the whole tree. Deferred lighting, shadows, SSAO, FXAA, tone map, bloom, Hi-Z occlusion, async readback, decals, glass, EXR loading. | **Take.** Mis-filed, not Apple-specific, and it was excluded from the workspace so none of it was compiling. |
| `apple/src/macos/metal/` | 7,218 | A transcription of Metal's headers into Rust types. 190 structs, 36 traits, 163 bodyless trait methods, 451 bodies — and the bodies are getters, setters and `Default`. No FFI of any kind. `acceleration_structure_encoder.rs` (46 lines) is the clearest case: `build_acceleration_structure` increments `self.build_count` and returns. | **Leave.** It cannot build an acceleration structure, on Metal or anywhere. |
| `apple/src/macos/metalfx/` | 763 | 4 traits, 70 bodyless methods, 32 bodies. Every scaler's `encode_to_command_buffer` is `// Stub: would encode spatial upscaling commands` over an empty body. | **Leave.** There is no upscaler in it to port. |
| `apple/src/accelerate/` | 480 | Pure-Rust reimplementations under Apple's names, read line by line 2026-10-29. `vdsp.rs` (162): one real radix-2 Cooley–Tukey FFT and eight scalar loops (`vadd`, `vsub`, `vmul`, `dotpr`, `sve`, `meanv`, `measqv`, `rmsqv`). `linear_algebra.rs` (71): `cblas_sgemm` as a triple loop. `bnns.rs` (105): one dense layer. `vimage.rs` (123): scaling and spatial convolution. | **Leave — permanently, not "for now".** Three reasons, in order. (1) **The workspace already has the GPU versions of the same things, in the right crate**: `phiray::kernels::image` is `MPSImageConvolution`, `MPSImageBilateralFilter` and `MPSMatrixMultiplication` as WGSL *with CPU twins* — so vimage's convolution and `cblas_sgemm` are already here and already run on a GPU. (2) **phinet has a better FFT**: `phistd`'s is unitary and sits **on an autodiff tape** (`t.fft(h)` / `t.ifft(hm)`), which is what `phifno`'s Fourier Neural Operators need; a bare in-place transform on `&mut [f32]` cannot serve that. (3) **Two correctness traps.** `vdsp_fft_zrip` is named for vDSP's *real-to-complex packed* transform (DC and Nyquist packed into element 0) but its body is a plain complex-to-complex FFT, so a caller following Apple's docs gets silently wrong data; and `vdsp_create_fftsetup` validates and stores `radix ∈ {2,3,5}` while the transform is hardcoded radix-2 and never reads it (`_setup`), so a radix-3 or radix-5 setup computes the wrong thing with no error. The twiddles also run by iterative recurrence, which drifts in `f32` over large `n`. |
| `apple/src/coreml/` | 2,743 | 87 structs, 143 bodies, 10 bodyless. Model descriptors, feature types, a prediction path over `HashMap`s. No `.mlmodel` parser, no inference. | **Leave.** API shapes with nothing behind them. |
| `phi/crates/phi-oml` (`phi-ml`) | 19,193 | A separate project, not a phikit crate. Only 5 source files touch wgpu; 8 WGSL shaders. Depends on `../../lib/phical`, outside the crates tree. | **Leave.** Porting it means resolving its own dependency graph first; it is its own job, not a salvage. |
| `phi/crates/macos`, `appkit`, `kitpro`, `uikit`, `iosar`, `xrosar`, `puikit`, `realitykit`, `reality-foundation` | ~124,000 | Same character as `apple/src/macos/`: header transcriptions. `kitpro` alone is 41,262 lines of Reality Composer Pro's shapes. | **Leave.** Volume is not value; 124,000 lines of declarations is worth less than the 631 lines of bloom that run. |
| `phi/sys/philog` | 12 | A `println!` macro. | **Leave.** `crates/phicore/zlog` is a real structured logger and phikit already has it. |

The headline: the only thing in either tree that *runs* on a GPU is
`appkit/gpu`, and it is wgpu. Everything filed under `metal`, `metalfx`,
`coreml`, `realitykit` and `kitpro` is the shape of an API with nothing
inside it, so there is nothing there to port even on a Mac.

---

## 2. What came, and where it is now

| From | Landed in | Why |
|---|---|---|
| `appkit/gpu/async_readback/` (210) | **`phiray::gpu::readback`** — `AsyncReadback`, `ReadbackBuffer`, `padded_bytes_per_row` | ROADMAP §3.1 *Display*: "Read back to the CPU every frame". phiray's own `read_range` maps and WAITS behind a five-second guard, which is right for a final render and wrong for a viewport. This is the non-blocking counterpart and sits beside it. Two tests. |
| `appkit/gpu/post_process/bloom/` + `shaders/bloom.wgsl` (631 + 130) | **superseded** by `phiray`'s `Glow` kernel (`trace/wgsl/glow.wgsl`, `Tracer::glow`) | ROADMAP §3.1 *Post* wanted bloom; it exists now, wired to Properties ▸ Render ▸ Bloom and GPU-tested. The apple version is the better *quality* — threshold → mip chain → blur → tent upsample, against `Glow`'s single separable Gaussian — and it carries a real bug fix (`ParamsRing`: the original had every mip read one shared uniform slot, so the whole chain ran at a single `texel_size`). **If the wider glare is wanted, merge that mip chain into `Glow`**, from `_temp/phifx-folded-into-phiphx/`. Do not revive a second bloom pass. |
| `appkit/gpu/post_process/pass_trait.rs` (68) | **not carried** | A render-pass contract. phiray's post passes are compute kernels over one pixel buffer (`Fx`, `Smaa`, `Taa`, `Show`, `Edge`, `Glow`), so it would have no implementors. Take it only if phiray grows a render-pass chain. |

The intermediate `phifx` crate, and the `phiphx::fx` module it became, are
both in `_temp/phifx-folded-into-phiphx/` with the reasoning in
`_temp/INVENTORY.md`. Nothing was deleted; the original is still in the
reference tree.

### What changed on the way over

Fixed, not carried:

1. **A real bug in bloom's uniforms.** Every stage wrote its
   `BloomParamsGpu` to offset 0 of one uniform buffer and then recorded a
   render pass into the shared encoder. `queue.write_buffer` is applied at
   submit, ahead of that encoder's commands — so all the writes landed before
   any pass ran and every pass read the last one. The whole chain ran at a
   single mip's `texel_size`. Now each stage takes its own slot of
   `ParamsRing` and binds it by dynamic offset (`src/bloom/params.rs`).
   `params_slots()` and a test keep the ring as long as the chain is busy.
2. **`catch_unwind` around `get_mapped_range`.** The readback treated a panic
   as "not ready yet". The `map_async` callback now raises a flag and the
   reader checks it, so not-ready is a branch.
3. **wgpu 29.** The source predates it: `ImageCopyTexture`/`ImageCopyBuffer`/
   `ImageDataLayout` → `TexelCopy*`; `Maintain::Poll` → `PollType::Poll`
   returning a `Result`; `entry_point` takes an `Option`;
   `compilation_options`, `cache`, `depth_slice` and `multiview_mask` added;
   `multiview` renamed; `bind_group_layouts` takes `Option<&_>`;
   `push_constant_ranges` → `immediate_size`; `mipmap_filter` takes
   `MipmapFilterMode`.
4. **`HDR_FORMAT` severed.** It was `appkit::gpu::deferred_lighting`'s; it is
   `phifx::HDR_FORMAT` now, so a post-process pass does not drag a deferred
   renderer behind it.
5. Two dead `let` bindings gone from the vertex stage; `PostProcessPass::execute`
   takes `&mut self`, which the ring needs and no caller minded.

---

## 3. What should follow, in order

Each row is a module of `apple/src/appkit/gpu/`, with the dependencies that
have to be severed first. "Coupling" is what `grep -o 'crate::[a-z_:]*'`
finds in it.

### 3.1 Next — each checked against what phiray already has (2026-10-29)

1. **`gpu_fallback/`** (189 + 166 tests, zero GPU coupling) — the best of
   these, but it must be **re-expressed, not copied**. Read its interface:
   the shape is right (a state struct, a retry ladder of at most three, a
   `last_good_*` pair restored on exhaustion, an outcome enum), and it takes
   a `try_allocate` closure so it touches no GPU at all. But its knobs are a
   rasterizer's — `shadow_resolution`, `bloom_enabled`,
   `GpuRecoveryAction::UseForwardPass`, `ShaderFallbackOutcome::RevertedToForward`
   — and phiray has no shadow maps and no forward pass, so copying it would
   import a vocabulary that means nothing here.
   phiray's real failure is the one ROADMAP §3.1 *Geometry* records: a scene
   over the device's per-binding limit is refused outright by
   `Tracer::why_not` / `max_pixels` and the render just stops with "the GPU
   declined this scene" — **there is no ladder at all**. The port is the
   structure over phiray's own knobs: bindings per array
   (`Tracer::with_chunks`, 2 or 4), the traced pixel count
   (`Tracer::max_pixels`, which already bands a frame), and the optional post
   passes (`Glow`, denoise) — retry, keep the last size that worked, and
   report the reason rather than a blank frame.
2. **Tone map and exposure** — **duplicate, mostly.** phiray already applies
   Blender's colour management in `view.wgsl`: `view_transform_of` does
   exposure (`exp2(view.y)`), the Standard/Filmic/AgX transforms and the
   looks, and `Show` runs it as its own pass. Take **only auto-exposure**
   (metering the frame to choose the stop), which phiray has no equivalent
   for; leave the operators.
3. **FXAA** — **duplicate.** phiray has SMAA (`smaa.wgsl` + the area/search
   tables) and Workbench's jittered TAA gather. Skip.
4. **`uniform_cache/`** (213, zero coupling) — a size-classed uniform slot
   pool. Still no caller: `Glow` binds one scratch `GpuArray` and the other
   kernels take a `Camera` uniform, so nothing needs slots yet. Take it with
   the bloom mip chain, if that lands, and fold `ParamsRing` into it.

### 3.2 After that — coupled, but worth severing

5. **Hi-Z occlusion culling** — `occlusion/` (666 lines + `hi_z_downsample.wgsl`).
   Depth-pyramid occlusion queries. Maps to ROADMAP §3.1 *Geometry*. Couples
   to `appkit::math` (`SCNBoundingBox`, `SCNVector3`) and `appkit::gpu::renderer`.
   Sever by taking `glam` types instead — simkit already does this, so the
   workspace has one vocabulary for a bounding box.
6. **`culling.rs`** — frustum culling against a BVH, 6 planes, sign-selected
   corner. Small and correct; same `appkit::math` severance. Check against
   phiray's own top-tree first.
7. **SSAO** — **duplicate for Solid.** phiray already has Workbench's cavity
   and curvature in `fx.wgsl` (`Fx` kernel, `Camera::cavity` / `ssao`), and
   it path-traces occlusion in Rendered. There is no gap to fill.
8. **`shadows/`** (1,447 lines) — cascade allocator, spot shadows, a
   prioritisation heuristic, PCF sampling. ROADMAP §3.1 *Shadows* says phiray
   needs **nothing** here in quality — it ray-traces shadows in every mode —
   but it also says "cost grows per lamp per pixel". The allocator and the
   prioritiser are the interesting parts, not the depth passes.

### 3.3 Not coming

- **`macos/metal/`, `macos/metalfx/`, `coreml/`** — §1. Nothing executes.
- **`mesh_sync/`** — couples to `appkit::ecs`, `macos::scenekit` and `ns_log`.
  phikit has its own ECS (`phicore/phiecs`) and its own logger (`phicore/zlog`);
  this would be a third vocabulary for the same thing.
- **`decals/`, `glass_pipeline/`, `deferred_lighting/`, `gbuffer/`,
  `scene_pass/`, `viewport/`, `renderer/`** — a complete second renderer.
  phiray is the renderer. Taking passes out of this tree is salvage; taking
  the renderer is a fork.
- **`loader_exr.rs`, `assets/blend_loader.rs`** — phiphx already reads `.blend`,
  and the rule is that `.blend` files stay where they are.
- **`accelerate/`** — §1. Revisit only if something needs an FFT.
- **Hardware ray tracing.** The one thing phiray's ROADMAP wants most
  (§3.1, "10-20 d, driver risk") is **not in either tree**. The Metal files
  named after it are a struct with a counter. It has to be written against
  wgpu's experimental ray queries; there is nothing to port.

---

## 3.3b Glass — `glass_pipeline/` (768 + 120 WGSL), surveyed 2026-10-29

**The best-matched port left in the tree, and it does not belong to phiray.**

phiber already has a Liquid Glass theme (`theme/style/glass.rs`): per-layer
alphas for ground / chrome / pane / control / floating, a rim hairline, glow
and halo, tinted tiles, and Reduce Transparency. Its own module note says
exactly what is missing — *"gpui has no backdrop filter and no way to sample
what it drew underneath"* — and that is precisely what this module is.

Apple's recipe is two halves:

1. **Separable two-pass Gaussian blur** (`glass_blur.wgsl`, 64 lines): a
   compute pass, 16×16 workgroups, CPU-precomputed weights uploaded as a
   uniform, edge-clamped and weight-normalised. `GlassEffect` defaults to
   `blur_radius: 12.0` clamped to 8–20, `sigma = radius / 3`.
2. **Composite** (`glass_composite.wgsl`, 56 lines): a **saturation boost**
   (`mix(gray, rgb, saturation)`, default **1.4**, clamped 1.0–1.8), then the
   tint mix, then opacity (default 0.7, clamped 0.4–0.8).

Cross-check: phiber's `LIQUID_GLASS.ground` is 0.72 against Apple's
`opacity` 0.7 — the alphas were already in the right family. The parameter
phiber lacked was the **saturation boost**, and that half is now in
(`BGlass::saturation`, Apple's 1.4, lifted onto the glazed surface's own
colour since there is no backdrop to sample; pinned by
`glass_lifts_a_surface_s_saturation_the_way_apple_lifts_its_backdrop_s` and
dropped by Reduce Transparency).

**The blur half is blocked, and the gate is real.** It needs the renderer
that draws the UI to render to an offscreen target, blur it, then composite
the glass surfaces over it. That is `crates/phigpu/gpui_wgpu`
(`wgpu_renderer.rs`, 1,913 lines) — **not phiray**, which draws the 3D canvas,
and not phiphx. And phiber on Windows currently runs gpui's **DirectX**
renderer (`gpui_windows/directx_renderer.rs`), as phikit's own `Cargo.toml`
notes beside the `gpui_wgpu` member, so a wgpu glass pass would not be live
on this machine until phiber renders through `gpui_wgpu`. Take the blur when
that switch happens; the shader is 64 lines and the weights are CPU-side.

## 3.4 Fyrox (`ext/REFS/Fyrox-master`) — surveyed 2026-10-29

A mature Rust game engine, MIT licensed (so it *may* be taken into
GPL-3.0 phiber, but its copyright notice has to travel with any code that
comes across). 260k lines. Closer in kind to phiber than the apple tree —
and yet **almost nothing is portable**, for one reason that governs every
row below: **phiber's specification is Blender, and Blender's source is
already vendored at `ext/REFS/blender`.** Fyrox is a different engine with
different semantics, so porting from it would import a model that looks
plausible and is wrong — the worst kind of error for parity work.

| Area | Lines | What it is | Verdict |
|---|---:|---|---|
| `fyrox-ui` | 55,713 | Its own retained-mode UI toolkit | **Leave.** phiber is GPUI/buikit, and one app uses one UI kit (`.kiro/steering/ui-kit-per-app.md`). |
| `fyrox-animation/machine/` | ~2,700 | An **ABSM**: blend trees, blend spaces, transitions, states, masks, parameters | **Leave.** A runtime gameplay concept (blend walk→run on a speed parameter). Blender's analogue is the NLA — a track/strip stack with extend modes and influence — which phiphx already has (`nla.rs`, `nla_eval.rs`, `nla_stack.rs`, `nla_edit.rs`). Different shape, not a missing piece. |
| `fyrox-animation/{track,value,container,pose}.rs`, `fyrox-math/curve.rs` | ~1,600 | A generic track / curve / keyframe model | **Leave.** phiphx's is Blender-exact and better for this job: `BKey` carries `BezTriple`'s `ipo`, `easing`, both handle types, `back`/`amplitude`/`period`, and now `keytype`; `easing.rs` has all ten easing kinds; `handles.rs` and `smooth.rs` do Continuous Acceleration. It is checked against Blender truth data. |
| `fyrox-animation/spritesheet/` | 844 | 2D sprite-sheet animation | **Leave.** Not a Blender feature. |
| `fyrox-impl/src/scene/joint.rs` | 775 | Joint definitions mapped onto **Rapier** | **Reference only.** There is no solver here to take — Rapier is a dependency. For phiray's missing XPBD joints the authoritative model is Blender's own `RigidBodyCon` (`DNA_rigidbody_types.h`: 8 live `eRigidBodyCon_Type`s, per-axis linear/angular limits, springs, motors, `breaking_threshold`), which phiphx already reads. Read Fyrox's for a second opinion on limit handling, nothing more. |
| `editor/src/plugins/curve_editor.rs` | 586 | A curve editor widget | **Reference only — the best thing here.** The widget is `fyrox-ui` and unusable, but the *interaction logic* is worth reading for phiber's largest menu gap (Graph Editor, 109 entries not yet): key drag, handle drag, box select, zoom-to-fit. phiber already has the geometry (`time/curve/handle/`, `time/view/cur.rs`'s Frame All). |
| `editor/src/plugins/ragdoll.rs` | 1,277 | A wizard that builds a ragdoll from a skeleton | **Leave.** Blender has no such feature; it is not parity work. |
| `fyrox-graphics-gl`, `fyrox-graphics` | 8,175 | An **OpenGL** renderer | **Leave.** phiray is wgpu compute. The apple tree was worth reading because it was wgpu; this is not. |
| `editor/` (rest), `fyrox-impl/` (rest) | ~150,000 | A complete engine and editor | **Leave.** Taking an editor is a fork, not a salvage. |

Neither of the two things phiray's ROADMAP wants most is in here: **no
hardware ray tracing** (OpenGL) and **no physics solver** (Rapier). Its real
value is architectural — `ARCHITECTURE.md`, and how it separates scene,
renderer and editor — plus the curve-editor gestures above.

## 3.5 Ideas, not code — `appkit/gpu/material/` and `macos/realitykit/`

Read 2026-10-29 for *design* rather than for lines to copy. Four findings,
and only one of them is a gap.

### Take: shader hot reload (`material/hot_reload/manager.rs`, 141)

phiray bakes every WGSL with `include_str!`, which is right for shipping —
the shaders travel in the binary and the wasm build has no filesystem — but
it means a one-character shader edit costs a full rebuild of a crate pinned
at `opt-level = 3`, about two minutes. That tax is paid on every iteration of
shader work.

Three specific ideas in this file are worth having, beyond "watch the file":

1. **Compare file *contents*, not mtime.** `check_file_changed` reads the
   source and diffs it against the last compiled text, so a save that
   changed nothing — or a `touch` — does not trigger a recompile.
2. **A failed compile keeps the previous module** and pushes the error onto
   `last_errors` for the UI to show. The viewport keeps drawing the last good
   shader instead of going black or panicking. Same instinct as
   `gpu_fallback`'s `last_good_*`.
3. **Budget the recompile** and warn when it is exceeded (theirs is 2 s).

Shape for phiray: a dev-only path (feature or `debug_assertions`) where
`Kernel::source()` prefers a file under a `PHIRAY_SHADERS` directory when
one is set and falls back to the baked `include_str!` otherwise. The
shipping and wasm story is unchanged; only a developer with the env var set
reads from disk.

### Already have: shader variants (`material/shader_variant.rs`, 163)

`ShaderVariant { shader, vertex_entry, fragment_entry }` and
`RenderPassDefinition { name, variant, draw_params }` — one WGSL source with
several entry points, picked per pass. **phiray's `Kernel` trait is this
already, and tighter**: `NAME` + `ENTRY` + `WGSL`, with pipelines cached by an
FNV-1a hash of (name, entry, source). `Adapt` and `AdaptDilate` are one
source with two entries; `Trace`/`Trace4`, `Solid`/`Solid4`, `Wire`/`Wire4`
and `Preview`/`Preview4` are variants over the binding count. Nothing to take.

### Do NOT take: the property bag (`material/property_bag/`, ~190)

Per-category `HashMap<String, _>` for textures, scalars, vectors and
matrices, 64 entries each, with a `CategoryFull` error. It reads as a
flexible material system and it is the **opposite of phiray's design on
purpose**: phiray's materials are a *fixed* ~120-word record that phiphx
bakes a Blender node graph down to (`trace/mat.rs`, `proc.rs`), so there is
no string lookup, no per-draw hashing and no node interpreter on the GPU —
which is a large part of why §1.3 can say the tracer is small and fast.
Recorded here because it is attractive at first glance; adopting it would
trade the thing that makes the renderer quick for flexibility nothing is
asking for.

### Confirms: phiray's material model is a superset

`RealityKit`'s `PhysicallyBasedMaterial` is base colour, roughness, metallic,
specular, emissive colour + intensity, normal, **occlusion**, clearcoat and
clearcoat roughness. phiray's `Tri`/`Mat` carries all of those (emission
folds colour × strength, as Blender does) **plus** transmission, IOR, bump, a
normal map, three texture maps, subsurface weight and radius, sheen with
roughness and tint, anisotropy with rotation, a Mix Shader partner and two
procedural slots.

The one field RealityKit has that phiray does not is an **occlusion map**.
It is not a parity gap: Blender's Principled BSDF has no AO input either —
ambient occlusion is a separate node, or baked into base colour — so there is
nothing for phiber to mirror. Noted so the absence is a decision and not an
oversight.

## 4. Rules a continuing session should keep

- No file over ~250 lines; `else if` is never written, `match` instead.
- Every doc comment says which file it came from.
- Nothing is deleted. Anything that must go moves to
  `C:\Users\phiac\Workspace\gemphi\_temp\` with a line in its `INVENTORY.md`.
- `crates/apple` stays excluded from the workspace and stays on disk: it is
  the source these ports cite, and more is still to come out of it.
- Build with `CARGO_TARGET_DIR=target/mods`. The machine is Windows, and
  everything ported has to keep building here — which is the whole reason the
  wgpu half of the Apple tree is worth more than the Metal half.
