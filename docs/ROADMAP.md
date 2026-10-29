# phiray: capability audit and roadmap

Read against the working tree on 2026-10-26 (phiray at `f71cd2d` plus other
sessions' uncommitted edits, which move a few counts by tens of lines). Every
line count below was taken with `wc -l` on that tree, and every claim names the file it
comes from. Where phiray's own `README.md`, `jobs.md` or comments disagree with
the code, the code wins; the disagreements are listed in section 7.

Effort is given in focused engineer-days for one person who already knows this
codebase, including the CPU twin and tests. Add about half again for review and
integration churn.

---

## Answers in one screen

**1. Why is phiray so small?** It is smaller than it looks from the Rust count
alone, but not by much. `src/` is 7,574 lines of Rust plus 5,782 lines of WGSL
(13,356 lines of source), with 4,741 lines of tests. The shading lives in the
WGSL, which is 43% of the source. It is small because it does one thing in one
way: it ray-casts every viewport mode in compute over a BVH, with the material
model fixed and baked down by phiphx before it arrives. wgpu does all the
per-API backend work, so there is no backend code at all. It also has no asset
pipeline (phiphx, 51.7k lines), no physics, no volumes, no hair, no motion blur
and no general material graph. Section 1 goes through it module by module.

**2. Where is the physics engine?** There is none, anywhere. Not in phiray, not
in phiphx, not in phiber, not in `../phinet/crates`, and no physics crate in
any `Cargo.toml` or `Cargo.lock`. phiphx does not even read Blender's physics
settings: `blend/modifier/report.rs` is a table of modifier *names* used to
tell the user which modifiers were dropped, and a grep of `phiphx/src` for
rigid body, cloth, soft body, collision, point cache or force field structs
finds nothing. The Physics properties panel in phiber
(`part/props/physics.rs`) is UI spec with factory defaults, not the file's
values. Section 2 designs one: XPBD with substeps, rigid bodies plus joints
plus cloth plus soft bodies in one solver, on phiray's `Kernel` trait with a
CPU twin per kernel, reusing phiray's LBVH and radix sort for collision.

**3. Is it as good as Unreal 5, RealityKit or Blender?** On offline image
quality for the scenes it accepts, the Rendered mode is a real path tracer
(multi-bounce, GGX, glass, MIS, environment importance sampling, adaptive
sampling, Filmic/AgX), and Solid is a close Workbench port. It is nowhere near
Unreal or Blender in breadth: no physics, no real-time GI, no
volumes, hair, motion blur, true displacement, light tree, general material
graphs or hardware ray tracing, and the image goes through the CPU every frame.
Section 3 has the row-by-row table with effort.

**4. The Warp OpenGL viewer.** It is a simulation *visualiser*: instanced
primitive shapes whose transforms a GPU kernel writes straight from simulation
body poses, deformable meshes written in place, points and lines as instanced
spheres and capsules, and tiled multi-camera rendering with a GPU-side tile
split. Its shading is plain Phong with a checker; it has **no shadows**, no
textures, no anti-aliasing and no colour management. phiray already shades
far better. What phiray lacks, and should port, is the scene structure:
shared-geometry instancing, GPU-written instance transforms and vertices,
primitive shapes, and tiled cameras. Warp's simulation module (`warp.sim`) is
not in the reference subset, so physics has to be designed, not ported.
Section 4 has the mapping onto Solid, Preview and Rendered and a port plan.

---

## 1. What phiray is, module by module

### 1.1 Line counts

| Module | Rust | WGSL | What it is |
|---|---:|---:|---|
| `lib.rs`, `mode.rs`, `kernel.rs` | 177 | - | Entry `Ray`, `Mode`, the `Kernel` trait and `Dispatch` |
| `bvh/` | 1,415 | (inline) | CPU LBVH and SAH builders, GPU LBVH build (Morton, radix sort, Karras) and refit |
| `gpu/` | 1,013 | - | The device, buffers, recorded passes, adapter ranking, one raster prepass |
| `kernels/` | 549 | 108 | Skinning, MPS-style image/matrix kernels, a-trous denoiser |
| `trace/` | 4,420 | 5,674 | The `Tracer`: every viewport mode, scene upload, instancing, lights, textures, effects |
| **Total `src/`** | **7,574** | **5,782** | plus `smaa/area.bin` (a 104 KB binary table, not code) |
| `tests/` (33 files) | 4,741 | - | GPU against CPU and against Blender-derived truths; skip with no adapter |

For scale, the renderer-adjacent code that lives outside phiray: phiber's
`view/draw/canvas/view3d/rendered/` (render thread, scene flattening, view
placement) is 3,914 lines, and phiphx's `blend/shade/` (shader node graphs
baked down to phiray's fixed material record) is 4,309 lines.

### 1.2 What each module does (from the code)

**Core (`lib.rs` 35, `mode.rs` 40, `kernel.rs` 102).** `Mode` has four
values, not three: `Wire`, `Solid`, `Preview`, `Rendered`, mapped to Blender's
`View3DShading.type` names. `Kernel` is the compute model taken from Warp:
`NAME`, `WGSL`, `BINDS` (read/write per storage binding after the params
uniform), `Params: Pod` starting with `n, groups_x`; 256-wide workgroups,
x-then-y spill past 65,535 groups, a WGSL prelude giving `item(gid,
groups_x)`. Pipelines are cached by an FNV-1a hash of name, entry and source.

**`bvh/` (1,415).**
- `node.rs` (125): the 32-byte `Node` of two 16-byte `Half`s (corner plus
  index, leaf bit in the top bit), Warp's layout; `Aabb`; 30-bit Morton codes
  with a group id in the high word; `STACK = 32`.
- `build.rs` (195): CPU Karras LBVH, the reference the GPU kernels must match;
  leaf packing of small or deep subtrees so depth stays under 32;
  `build_compact` (breadth-first, drops the sort arrays early); CPU refit.
- `sah.rs` (189): binned SAH builder (16 bins, Wald 2007). This is the tree the
  tracer actually walks; the Morton tree is the fast GPU-mirrored one.
- `kernels.rs` (170): WGSL `Morton`, `Karras`, `Refit` (Jacobi refit: `depth + 1`
  ping-pong passes, no atomics).
- `radix.rs` (202), `sort.rs` (183), `spread.rs` (32): a GPU LSD radix sort on
  a 64-bit `(code, index)` key, so the atomic scatter can race and the result
  is still deterministic and identical to the CPU sort. Recorded once,
  replayed.
- `gpu.rs` (105): `GpuBvh`: codes, sort and Karras links on the GPU; leaf
  packing still on the CPU (one readback of the order per build); refit per
  frame by replay; `update` refits when the primitive count is unchanged.
- `shape.rs` (52) compaction, `tests.rs` (116), `mod.rs` (46).

**`gpu/` (1,013).**
- `mod.rs` (277): one shared `Gpu` per process (device, queue, adapter info),
  compute and render pipeline caches shared by every pass, readback with a
  5 s hang guard (Windows TDR) that a final render can raise (`PatienceFor`).
- `adapter.rs` (199): ranks every adapter (discrete before integrated before
  software; Vulkan/Metal before DX12 before GL), user-overridable.
- `array.rs` (188): `GpuArray<T>`: one storage buffer grown by doubling,
  `split` across several bindings when an array exceeds the per-binding limit.
- `pass.rs` (140): record kernel launches once, `set` params, replay (Warp's
  graph capture).
- `raster.rs` (148): the only render pipeline in phiray: a visibility prepass
  that rasterizes every face into a face/barycentric/depth target for Solid and
  Wire on hardware adapters.
- `shared.rs` (61): static device on desktop, thread-local on wasm.

**`kernels/` (549 Rust, 108 WGSL).**
- `skin.rs` (91): linear blend skinning with any number of influences (CSR),
  B-Bone segment blending, plus its CPU twin. phiber uses it through
  `phiber/src/scene/skin.rs`, with the CPU taking meshes under 20,000 vertices.
- `image.rs` (385): WGSL bodies for `MPSImageConvolution`,
  `MPSImageBilateralFilter` and `MPSMatrixMultiplication`, each with a CPU twin.
- `denoise.rs` (64) + `denoise.wgsl` (100): edge-avoiding a-trous wavelet
  denoiser (Dammertz 2010) guided by first-hit albedo, normal and depth.

**`trace/` (4,420 Rust, 5,674 WGSL).** One `Tracer` serves all four modes.
- `mod.rs` (925): build from a compact scene, split big arrays over 2 or 4
  bindings, pick the mode's kernel, the sampling loop (`sample_in` slices a
  sample so a new view never waits), passes read back (radiance, display,
  albedo, normal, depth), denoise, outline, adaptive sampling control.
- `kernel.rs` (784): the data types: `Tri`, the large `Camera` uniform, up to
  `LIGHTS = 32` file lights, `Atlas` (every texture in one storage buffer with
  a CPU-built mip chain, sampled by hand), environment CDF.
- `scene.rs` (341): compact scene: 32-byte shared vertices, 16-byte faces,
  material table. LAND (7.2 M triangles) and FLOWER-CITY (7.6 M) render;
  20 M-triangle files are refused on a 128 MB binding limit.
- `mesh.rs`, `instance.rs`, `trees.rs`, `pose.rs`: one SAH tree per mesh
  in its own space plus a top tree over the instances' world bounds; a
  mesh is placed by any number of instances (`Scene::place`), so a
  collection instanced a thousand times holds its faces, vertices and
  tree once — Cycles' `Object`s over one `Geometry` (§4.4, Step 1, done).
  The face word holds the material alone; the traversal carries the
  instance from the top-tree leaf into `Hit.inst`, so there is no cap on
  the instance count. Moving an object re-uploads the instance table and
  top tree only (`reinstance`); moved vertices refit that mesh's tree
  (`repose`); `retree` rebuilds once motion stops. The raster prepass is
  one instanced draw a mesh (`Trees::draws`, `@builtin(instance_index)`).
- `mat.rs` (83), `proc.rs` (242): the material record (base colour,
  roughness, metallic, transmission, IOR, emission, bump, normal map, three
  texture maps, SSS weight and radius, sheen, coat, anisotropy, one Mix Shader
  partner) and two procedural slots (40 words each).
- `ibl/` (322): environment irradiance as 9 spherical-harmonic coefficients and
  GGX-prefiltered levels, for Material Preview.
- `taa.rs` (277), `smaa.rs` (46), `outline.rs` (115), `blue_noise.rs` (114),
  `hash.rs` (105), `srgb.rs` (47), `visible.rs` (64), `lamps.rs` (37):
  Workbench's jittered AA and Blackman-Harris gather, SMAA tables, selection
  outline, EEVEE's blue noise, Blender's object-colour hash, sRGB, prepass
  wiring, emitter CDF.
- `shader.rs` (288): the kernel declarations (`Trace`, `Solid`, `Preview`,
  `Wire`, each in 2- and 4-binding forms, plus `Fx`, `Taa`, `Smaa`, `Adapt*`,
  `Edge`, `Show`).

The WGSL, grouped:

| Group | Lines | Files |
|---|---:|---|
| Path tracer | 1,096 | `shade.wgsl` (main loop, DOF with bladed bokeh, sample clamp), `light.wgsl` (GGX VNDF, lamps, mesh emitters, env importance sampling, MIS, indirect bounces) |
| Cycles texture nodes | 1,294 | `noise`, `voronoi`, `pattern` (checker, brick, gradient, magic, wave), `gabor`, `math` (all 41 ops), `hash`, `procedural` (slots, mapping, bump) |
| Workbench (Solid, Wire) | 1,017 | `solid`, `workbench` (studio rig, MatCap), `wire`, `fx` (cavity SSAO, curvature, object outline), `dof` |
| Anti-aliasing | 491 | `taa`, `smaa`, `smaa_lib` |
| Material Preview | 406 | `preview` (one primary ray, IBL, glass, mip-level textures), `shadow` (EEVEE's soft shadow rays) |
| Adaptive sampling | 140 | `adapt` (Cycles' convergence test), `gather` (active-pixel compaction) |
| Plumbing | 982 | bindings, layout, camera, BVH traversal (32-slot stack, Moller-Trumbore, instance transforms), textures, raster prepass |
| Colour, overlays | 248 | `view` (Standard/Filmic/AgX tables sampled from Blender 5.0, looks), `outline`, `show` |

### 1.3 Why it is this size

1. **wgpu is the backend.** One WGSL source runs on Vulkan, Metal, DX12 and
   WebGPU through naga. There is no RHI, no per-API code, no shader
   cross-compiler in the tree.
2. **One kernel per mode.** The path tracer is a single megakernel per sample,
   not a wavefront integrator with queues and per-material shaders. Workbench,
   Preview and Wire are one dispatch each plus small effect passes.
3. **The material model is fixed.** phiphx bakes Blender's node graphs down to
   one 120-word record per material. There is no material compiler and no
   node interpreter on the GPU.
4. **The asset side is elsewhere.** Reading .blend, meshes, textures and
   materials is phiphx (51.7k lines); flattening the scene and running the
   render thread is phiber (about 4k lines in `rendered/`).
5. **Algorithms are ported, not frameworks.** Each feature cites the Blender
   function it ports, so a feature costs 100 to 300 lines rather than a
   subsystem.
6. **Whole categories are absent**: physics, particles, volumes, hair, motion
   blur, displacement, light tree, baking, render layers beyond four passes.

Cycles' own source tree is more than an order of magnitude larger, and
Unreal's rendering modules about two. Line count is the wrong yardstick,
though; the feature table in section 3 is the right one.

### 1.4 phiphx, briefly (51,700 lines)

`blend/` 34,833 (file and SDNA 1,723; object 2,202; mesh 2,005; curve 1,155;
modifier 5,185 evaluating nine modifier types: Armature, Subsurf, Mirror,
Array, Triangulate, Solidify, Displace, Weld, Lattice; rig 6,616 with pose,
constraints, drivers, IK; anim 4,941 with fcurves, NLA and motion; shade
4,309; scene 2,388; write 3,314 writing an edited .blend back; asset 297).
`scene/` 4,080 (`SceneModel`, `SceneTimeline`, `SceneMotion::posed_at(frame)`
and friends). `export/` 3,651 (PHX, glTF, USD/USDA, PLY, PCD, LAS, XYZ, JSON).
`import/` 2,554 (glTF behind a feature, USD/USDA/USDC/USDZ, PLY, PHX, a simple
OBJ; FBX and STL are enum entries only). Plus `spatial` 2,150, `geom` 858,
`uv` 886 (LSCM), `comp` 882 (compositor tree), `format` 466, `sky` 424,
`cloud` 414.

It contains **no simulation**. The only time-stepped integrator is
`SpatialSpring::update(dt)` (`spatial/spatial_animator.rs:27`), a damped UI
spring. The other solvers are kinematic or numeric (SDLS IK, constraint stack,
LSCM, Bezier tridiagonal).

---

## 2. Physics

### 2.1 What exists today

Checked by grep over `phikit/crates` and `phinet/crates` for broadphase,
narrowphase, sweep and prune, GJK, EPA, SAT, contact manifold, constraint
solver, sequential impulse, PGS, XPBD, PBD, Verlet, rigid body structs, and
for physics crates (rapier, parry, physx, jolt, bullet, avian, nphysics) in
every `Cargo.toml`.

| Where | What is there | Simulation? |
|---|---|---|
| phiray, phiphx, phiber, phinet | Nothing. "PBD" in `phiphx/src/blend/anim/tests.rs` is a pose-bone tag. | No |
| `phiphx/src/blend/modifier/report.rs` | `MODIFIER_NAMES`: Soft Body (10), Particle System (19), Cloth (22), Collision (23), Fluid, Dynamic Paint. `skipped_modifiers()` returns names of dropped modifiers. No settings read. | No |
| `phiphx/src/blend/rig/constraint/mod.rs` | Object constraint types; Rigid Body Joint is not among them. | No |
| `phiber/src/part/props/physics.rs` (60), `props/scene.rs:53` | Property specs for Rigid Body, Collision, Cloth and Rigid Body World with factory defaults. | No (UI only) |
| `phiber/src/modal/reader/store/whats_new.rs:15` | Text describing Blender's XPBD Solver node. | No |
| `phiplay/src/play/ball.rs` (152) | Closed-form ballistic flight and an Euler step for a loose ball with floor bounce and rolling decay. No body-body collision. | Game motion only |
| `crates/apple` (excluded from the workspace, `Cargo.toml:9`, not built) | RealityKit-shaped structs; an explicit-Euler gravity step; a 60-line Verlet grid cloth (`physics_advanced.rs:639`) with Y-up gravity, no collision, unused; a CPU AABB tree query (`services/ecs/bvh.rs:95`) with no physics caller. | Toy, unbuilt |
| `phiray/reference/warp` | Warp's BVH, mesh queries, hash grid, graph colouring, sort/scan. **No `warp.sim`.** | Building blocks only |

What does exist and matters for physics: phiphx can already evaluate
animation at any frame (`BFCurve::eval`, NLA, `SceneMotion::posed_at`),
phiber has a playhead (`app/playback.rs`) and a posing worker
(`scene/pose/mod.rs`), and the tracer already accepts moved objects
(`reinstance`) and moved vertices (`repose`) without rebuilding. The render
side of an animated simulation is in place; the simulation is not.

### 2.2 Design

#### Where it lives

A `sim` module in phiray (`phiray/src/sim/`), next to `bvh/` and `kernels/`,
and a `physics` reader in phiphx (`phiphx/src/blend/physics/`). Reasons:

- The solver needs exactly what phiray already has: `Kernel`, `Pass` replay,
  `GpuArray`, the radix sort, `GpuBvh` build and refit.
- It must share the device with the tracer. Body poses and cloth vertices
  written by a kernel can then be the instance table and vertex buffer the
  tracer reads, with no copy. This is what Warp's `RegisteredGLBuffer` buys
  Warp; one wgpu device gives it for free.
- The planned restructure moves phiphx into the phiray repo, so settings,
  solver and renderer end up in one repository anyway.

If `sim/` outgrows the renderer, split it into its own crate in the same repo
later; the boundary (it uses `gpu`, `bvh`, `kernel`; nothing in `trace` uses it)
makes that mechanical.

#### Solver: XPBD with substeps, not sequential impulses

| | Sequential impulses (Box2D, Bullet, PhysX PGS/TGS) | XPBD (Macklin et al. 2016; Muller et al. 2020; Macklin "Small Steps" 2019) |
|---|---|---|
| Level | Velocity impulses, Gauss-Seidel, warm starting | Position constraints with compliance, velocities derived |
| Covers | Rigid bodies and joints; cloth needs a separate solver | Rigid bodies, joints, cloth, soft bodies, ropes in one loop |
| Stiffness | Iteration- and timestep-dependent | Compliance is physical (1/stiffness), independent of iteration count |
| Stability | Good for stacks with warm starting; stiff joints drift | Unconditionally stable; substeps replace iterations |
| GPU | Inherently sequential; needs colouring or Jacobi | Each constraint projects independently; colouring or Jacobi averaging |
| Blender parity | Closest to Bullet (Blender's rigid body engine) | Not Bullet; cloth stiffness maps onto compliance directly |

XPBD is the choice because the user wants rigid bodies, cloth and soft bodies,
and only XPBD does all three in one solver with one set of data structures,
one test harness and one GPU path. Blender's cloth settings (tension,
compression, shear, bending stiffness) become compliances; Blender's
"Quality" and "Substeps per Frame" become substeps. Muller 2020 shows XPBD
rigid bodies with joints and stacking competitive with impulse solvers when
substepped.

The honest costs: friction is approximate (static friction at position level,
dynamic friction in a velocity pass); very stiff contact needs more substeps;
and a scene tuned in Blender will not reproduce Blender's bake, because
Blender runs Bullet for rigid bodies and an implicit mass-spring solver for
cloth. Exact parity would mean porting Bullet; that is not proposed.

#### State layout

All arrays are `GpuArray<T: Pod>` with a CPU `Vec<T>` twin of the same layout.

```rust
/// One rigid body, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Body {
    pub x: [f32; 3],     // centre of mass, world
    pub inv_mass: f32,   // 0 for passive and kinematic bodies
    pub q: [f32; 4],     // orientation, xyzw
    pub v: [f32; 3],     // linear velocity
    pub shape: u32,      // index into `shapes`
    pub w: [f32; 3],     // angular velocity, world
    pub flags: u32,      // Kind in bits 0..2, sleeping, collision group mask index
}

/// Per-body constants, 48 bytes.
pub struct BodyConst {
    pub inv_inertia: [f32; 3], // principal, body frame
    pub friction: f32,
    pub com_local: [f32; 3],   // centre of mass in object space (object origin != COM)
    pub restitution: f32,
    pub lin_damp: f32,
    pub ang_damp: f32,
    pub margin: f32,
    pub groups: u32,           // Blender's 20 collision collections as a bit mask
}

/// The pose at the start of a substep (XPBD derives velocity from it).
pub struct BodyPrev { pub x: [f32; 3], pub _a: f32, pub q: [f32; 4] }

#[repr(u32)]
pub enum Kind { Active = 0, Passive = 1, Kinematic = 2 }

/// Blender's `RigidBodyOb.shape`.
#[repr(u32)]
pub enum ShapeKind { Box, Sphere, Capsule, Cylinder, Cone, Hull, Mesh }

/// 32 bytes; `data` points into `hull_points` or `mesh_faces` for Hull and Mesh.
pub struct Shape { pub kind: u32, pub data: u32, pub count: u32, pub radius: f32, pub half: [f32; 3], pub _a: f32 }

/// A contact found by the narrowphase, re-evaluated every substep.
pub struct Contact {
    pub a: u32, pub b: u32,
    pub anchor_a: [f32; 3], pub lambda_n: f32, // body-local anchors (Muller 2020)
    pub anchor_b: [f32; 3], pub lambda_t: f32,
    pub normal: [f32; 3],   pub depth0: f32,
}
```

Shapes are convex through one support function, so one GJK/EPA path serves
box, sphere, capsule, cylinder, cone and hull. It is a `match`, never a chain
of conditions, and the WGSL twin is a `switch` on the same values:

```rust
fn support(s: &Shape, hull: &[[f32; 4]], d: [f32; 3]) -> [f32; 3] {
    match ShapeKind::from(s.kind) {
        ShapeKind::Sphere => scale(norm(d), s.radius),
        ShapeKind::Box => map3(|k| s.half[k].copysign(d[k])),
        ShapeKind::Capsule => add(scale(norm(d), s.radius), [0.0, 0.0, s.half[2].copysign(d[2])]),
        ShapeKind::Cylinder => cylinder_support(s, d),
        ShapeKind::Cone => cone_support(s, d),
        ShapeKind::Hull => hull_support(&hull[s.data as usize..][..s.count as usize], d),
        ShapeKind::Mesh => unreachable!("mesh shapes never take the convex path"),
    }
}
```

Cloth and soft bodies use particles:

```rust
pub struct Particle { pub x: [f32; 3], pub inv_mass: f32, pub v: [f32; 3], pub body: u32 } // 32 B
pub struct Distance { pub i: u32, pub j: u32, pub rest: f32, pub compliance: f32 }      // stretch, shear, pull/push
pub struct Bend { pub i: [u32; 4], pub rest: f32, pub compliance: f32 }                  // dihedral across an edge
pub struct Goal { pub i: u32, pub stiffness: f32, pub damping: f32, pub _a: u32 }        // pin / soft body goal
```

Constraints are stored per colour: `colours: Vec<(u32, u32)>` ranges into the
constraint arrays, computed once from the static topology.

#### One step

Per frame, `h = frame_dt * time_scale / substeps`. Collision detection runs
once per frame with bounds swept by `v * frame_dt` plus margin (Muller 2020);
contact constraints are re-evaluated every substep from their body-local
anchors.

1. **Bounds**: per body and per cloth triangle, swept AABB (`Bounds` kernel).
2. **Broadphase**: LBVH over body AABBs, pairs out (below).
3. **Narrowphase**: contacts out, sorted (below).
4. For each of `substeps`:
   1. **Integrate**: `prev = pose`; `v += h * g * effector weight`;
      `x += h * v`; `q += 0.5 * h * [w, 0] * q`, normalised. Kinematic and
      animated passive bodies take their pose from `SceneMotion::posed_at` at
      the substep's time instead.
   2. **Project**: contacts (normal plus static friction), joints, cloth and
      soft constraints, each with its compliance and `lambda` reset at the
      start of the substep. `solver_iterations` from the file sets the
      iterations per substep (default 1 when substeps are high).
   3. **Velocities**: `v = (x - prev.x) / h`; `w` from `q * prev.q^-1`.
   4. **Velocity pass**: dynamic friction, restitution (using the
      pre-solve normal velocity), linear and angular damping.
5. **Sleep**: bodies under Blender's linear and angular sleep thresholds for a
   while stop integrating until a contact with an awake body wakes them.
6. **Write out**: body poses to instance matrices, particles to vertices
   (section 4 port plan).

#### Collision: broad, mid and narrow phase, reusing phiray's BVH

- **Broadphase** reuses `GpuBvh::build` (Morton codes, radix sort, Karras
  links on the GPU) over the body AABBs, rebuilt each frame. One new kernel,
  `Overlap`, walks the tree for each body's box with the existing 32-slot
  stack and Warp's `bvh_query_aabb` order (`reference/warp/native/bvh.h:547`),
  emitting pairs `i < j` whose collision group masks intersect. Slots come
  from an `atomic<u32>` counter, so the pair order is not deterministic;
  a radix sort on the 64-bit key `(min id, max id)` restores it, which is the
  same trick `bvh/radix.rs` already uses. The pair buffer is a `GpuArray`
  with an overflow flag; on overflow it doubles and the frame reruns.
- **Known cost**: `GpuBvh::build` still reads the order back to pack leaves
  (`bvh/gpu.rs`, open job in `jobs.md`). For the broadphase that is one
  readback per frame. Either finish GPU leaf packing (one kernel) or keep the
  topology and only refit for several frames.
- **Midphase** for Mesh colliders: a `Bvh::build_sah` over the collider's
  triangles in object space, the same builder the tracer uses for its object
  trees. Query triangles under the other shape's box in that space.
- **Narrowphase**, one thread per pair, dispatched by a `match` on the
  ordered pair of shape kinds:
  - sphere-sphere, sphere-capsule, capsule-capsule: closed form
    (segment-segment closest points);
  - box-box and hull-hull: SAT over face normals and edge-edge axes, then
    Sutherland-Hodgman clipping of the reference face for up to 4 contact
    points in one frame (a full manifold without frame-to-frame matching,
    which suits the GPU);
  - anything with a cylinder or cone: GJK for distance, EPA for penetration,
    plus 3 rotated support queries around the normal for extra points;
  - convex against Mesh: triangles from the midphase, triangle-convex SAT.
  - Mesh against Mesh between two moving bodies is **not supported**: an
    active body with a Mesh shape is simulated as its convex hull, and the
    open reports it the way `skipped_modifiers()` reports dropped modifiers.
    Blender allows it (Bullet GImpact), so this is a visible difference.

#### Contacts and friction without float atomics

WGSL has `atomicAdd` on `i32`/`u32` only: no float atomics and no `f64`. A
body touched by many contacts cannot accumulate float corrections with
atomics. The design avoids them:

- **Contacts (dynamic topology): Jacobi with averaging** (Macklin et al.
  2014, "Unified Particle Physics"). After the narrowphase, each contact
  contributes two keys `(body << 32) | (contact * 2 + side)`; the radix sort
  orders them and a segment pass gives each body its range. That CSR is built
  once per frame and reused by every substep. Per substep: kernel `Contacts`
  (one thread per contact) writes the correction for each side into that
  side's slot; kernel `Apply` (one thread per body) sums its slots in sorted
  order, divides by the count, scales by a relaxation factor around 1.5, and
  applies. Fixed order means a deterministic result on a given device.
- **Joints, cloth, soft (static topology): coloured Gauss-Seidel.** The
  constraint graph is coloured once at setup on the CPU (greedy, the approach
  of Warp's `native/coloring.cpp`, from Gaia), and each colour is one dispatch
  in which no two constraints share a particle or body. The CPU twin walks the
  colours in the same order, so it computes the same thing.
- **Risk acknowledged**: Jacobi converges more slowly than Gauss-Seidel, and
  tall stacks of boxes are the classic failure. Substeps and averaging help;
  if not enough, colour contacts per frame (Jones-Plassmann on the GPU) for
  the bodies in contact islands above a size.

#### Joints (Blender's rigid body constraint types)

Blender stores them as `RigidBodyCon` on an object (usually an empty) in the
world's constraints collection, linking `ob1` and `ob2`. XPBD positional and
angular constraints after Muller 2020:

| Blender type | XPBD constraints |
|---|---|
| Fixed | attach anchors (positional, compliance 0) + align full frames (angular) |
| Point | attach anchors |
| Hinge | attach anchors + align hinge axes; optional angle limits about the axis |
| Slider | align full frames + positional constraint on the two off-axis directions; optional linear limits |
| Piston | as Slider but rotation about the axis left free |
| Generic | per-axis linear and angular limits (6 DOF) |
| Generic Spring | Generic plus compliant targets per axis from spring stiffness and damping |
| Motor | target angular or linear velocity applied in the velocity pass, clamped by max impulse |

`breaking_threshold` breaks a joint when `|lambda| / h^2` exceeds it.
`disable_collisions` adds the pair to a skip list read by `Overlap`.
Per-constraint solver iterations override the world's.

#### Cloth

Blender's cloth is a modifier (`ClothModifierData` with `sim_parms` and
`coll_parms`) evaluated where it sits in the stack, after any Armature above
it. XPBD mapping:

- **Stretch**: a `Distance` per structural edge. Tension and compression get
  separate compliances (choose by the sign of the constraint), from
  `tension` and `compression` stiffness; `vgroup_struct` scales them.
- **Shear**: a `Distance` per quad diagonal (Blender's shear springs) from
  `shear` stiffness.
- **Bending**: a dihedral `Bend` per interior edge from `bending` stiffness;
  Blender's linear bending model maps onto a distance across the edge instead.
- **Pins**: `vgroup_mass` (the pin group): weight 1 means `inv_mass = 0`,
  partial weights become `Goal` constraints with `pin_stiffness`.
- **Air damping** (`air_damping`) and per-type damping in the velocity pass;
  **pressure** as one volume constraint over the closed mesh;
  **quality** as substeps.
- **Collision with Collision-modifier objects**: per particle, closest point
  on the collider's triangle tree within `distance_min` plus the collider's
  outer thickness, friction from the collision settings. This is a port of
  Warp's `mesh_query_point` traversal (`native/mesh.h:977`, near child first,
  distance bound shrinking) into WGSL over phiray's node layout.
- **Self collision**: particle against triangle within the self distance,
  using a tree over the cloth's own triangles refit every substep with
  `GpuBvh::update` (the refit replay that exists). Edge-edge self collision is
  left out of the first version; thin features can pass through each other.
- No continuous collision detection in the first version; substeps and
  thickness stand in.

#### Soft bodies

Blender's soft body (`Object.soft`) is surface-based: mass-spring edges with
pull, push and bending stiffness, a goal (vertex group, goal stiffness,
damping, min/max), collision against Collision objects, optional self
collision. The XPBD version uses the same particles and constraints as cloth:
`Distance` with separate pull and push compliances, `Bend`, `Goal` towards the
animated (posed) vertex position, and an optional volume constraint. That
covers Blender's soft body settings faithfully enough. Volumetric
(tetrahedral) bodies would need a tetrahedraliser and are a later, separate
project.

Not designed here: particles, fluids (Mantaflow), Dynamic Paint, force fields
beyond gravity and effector weights, destruction.

#### Kernels and CPU twins

Every kernel follows `kernels/skin.rs`: a `Kernel` impl with its WGSL, and a
`cpu` function over the same `Pod` slices that the tests compare against.

| Kernel | Items | Reads | Writes |
|---|---|---|---|
| `Bounds` | bodies, cloth triangles | bodies, shapes, particles | AABBs |
| (reuse) `Morton`, radix sort, `Karras`, `Refit` | AABBs | | tree |
| `Overlap` | bodies | tree, AABBs, groups | pairs, counter |
| (reuse) radix sort | pairs | | sorted pairs |
| `Narrow` | pairs | bodies, shapes, hulls, mesh trees | contacts, counter |
| `ContactKeys` + radix sort + `Segments` | contacts | contacts | per-body CSR |
| `Integrate` | bodies, particles | | poses, prev |
| `Contacts` | contacts | bodies, contacts | per-side corrections |
| `Apply` | bodies | CSR, corrections | poses |
| `Joints` | joints of one colour | bodies | poses, lambdas |
| `Distances`, `Bends`, `Goals` | constraints of one colour | particles | particles |
| `ClothCollide` | particles | collider trees | particles |
| `Velocities` | bodies, particles | poses, prev | velocities |
| `Friction` | contacts | | velocities (via CSR, same as `Apply`) |
| `Place` | instances | poses | instance table (section 4) |

Everything is recorded once into a `Pass` and replayed; per frame only the
params change. Small scenes run the CPU twin with rayon instead of the GPU,
the policy `GPU_MIN_VERTS` already uses for skinning. Results are compared
within a tolerance, not bit for bit: GPU compilers contract to FMA and the
CPU does not.

#### Fitting phiber

1. **Read the settings** (phiphx, new `blend/physics/`). The SDNA reader
   resolves fields by name from the file itself, with the accessors that
   exist (`off`, `ptr`, `int_field` in `blend/object/layer.rs`, `float_field`
   in `blend/object/data.rs`, `list` in `blend/modifier/mod.rs`), so a wrong
   field name fails at read time, not silently. Structs to read:
   `Scene.rigidbody_world` (`RigidBodyWorld`: collection, constraints
   collection, effector weights, `substeps_per_frame`,
   `num_solver_iterations`, `time_scale`, point cache frame range),
   `Object.rigidbody_object` (`RigidBodyOb`: type, shape, flags such as
   kinematic and deactivation, mass, friction, restitution, margin, damping,
   sleep thresholds, collision collections), `Object.rigidbody_constraint`
   (`RigidBodyCon`), the Cloth modifier's `ClothSimSettings` and
   `ClothCollSettings`, `Object.pd` for colliders, `Object.soft`. Confirm each
   field name against `DNA_rigidbody_types.h`, `DNA_cloth_types.h`,
   `DNA_object_force_types.h` for the Blender version the fixture was saved
   with. Add `SceneModel.physics: ScenePhysics`.
2. **Show the real values** in the Physics and Scene property panels that
   already exist (`part/props/physics.rs`, `props/scene.rs`).
3. **Evaluate in Blender's order**: animation and drivers
   (`SceneMotion::posed_at`), then the rigid body world, then each object's
   modifier stack where Cloth and Soft Body sit, then rendering.
4. **Cache, because simulation is sequential.** Animation is a pure function
   of the frame; simulation is not. A `SimCache` per world keeps one state per
   frame from the start frame (rigid: 28 bytes per body; cloth: 12 bytes per
   particle) and a "valid up to" frame. Scrubbing back reads the cache;
   jumping forward steps through the gap; editing a setting or an animated
   passive body invalidates from the current frame. Baking writes the cache
   to disk later. phiber's playhead (`app/playback.rs`) and posing worker
   (`scene/pose/mod.rs`) drive it.
5. **Hand off to the renderer**: rigid bodies to `Tracer::reinstance` (instance
   table and top tree only); cloth and soft bodies to `Tracer::repose`
   (vertices and that object's tree refit). Both exist today. The
   GPU-to-GPU version of both is the Warp port (section 4).

#### Tests

Same style as `tests/bvh.rs` and `tests/emitters.rs`: GPU against CPU twin,
and both against closed forms or Blender.

- free fall against `x0 + v0 t + g t^2 / 2`; pendulum period; spring
  frequency against `sqrt(k / m)`;
- restitution: bounce height ratio equals `e^2`; friction: a box on a slope
  starts sliding at `atan(mu)`;
- a stack of 10 boxes stays standing for 1,000 frames; energy does not grow;
- determinism: two runs on one device are identical;
- each joint type holds its degrees of freedom within tolerance;
- cloth: a pinned sheet's sag against the analytic catenary for the given
  stiffness; no particle inside a collider by more than the thickness;
- parity: Blender rigid body and cloth fixtures baked in Blender, compared
  frame by frame with a documented tolerance (Bullet and XPBD will differ).

---

## 3. Gap table

Effort is for the missing part only. "Not realistic" means a multi-person-year
system; trying it would stall everything else.

### 3.1 Against Unreal Engine 5

| Area | Unreal 5 | phiray today | Missing | Effort |
|---|---|---|---|---|
| Global illumination | Lumen: dynamic diffuse GI and reflections, software (SDF) and hardware RT, surface cache | Rendered: path tracing with multi-bounce diffuse, GGX glossy, glass, MIS, env importance. Preview: environment light only (SH9 + prefiltered GGX), no bounce light | Real-time GI for Preview: probes, temporal reuse | Probe grid updated by the existing tracer (DDGI-style): 15-25 d. Lumen-class: not realistic |
| Geometry | Nanite: clustered virtual geometry, per-cluster LOD, streaming; instanced static meshes | Compact scene, SAH tree per object plus a top tree; 7.6 M triangles fine; no shared geometry; 65,536 objects max; 20 M triangles refused on a 128 MB binding limit | Shared-geometry instancing, streaming, LOD | Instancing 5-8 d; per-object streaming 10-15 d; Nanite-class: not realistic |
| Physics | Chaos: rigid bodies, joints, cloth, destruction, vehicles, ragdolls, fields | None | All | Rigid, joints, cloth, soft, GPU path and bake: 51-72 d (section 5). Destruction, vehicles: not planned |
| Materials | Node graph compiled to HLSL, Substrate layered BSDFs, material instances, virtual textures | Fixed Principled subset (base colour, roughness, metallic, transmission, IOR, emission, alpha, normal map, bump, 3 texture maps, sheen, coat, anisotropy, SSS approximation), one Mix of two materials, two procedural slots from Cycles' texture nodes | General node graphs, layering, virtual texturing; textures are a storage-buffer atlas sampled by hand | SVM-style node interpreter in WGSL: 20-30 d; hardware texture arrays: 4-6 d |
| Lights | Thousands of dynamic lights (clustered shading, MegaLights), IES, light functions, channels | Up to 32 file lights (sun, point, spot, area square/rect/disk/ellipse), one picked by power per shading point (`lamp_query`); emissive meshes with power-weighted NEE and MIS; HDRI importance sampling | Many-light sampling, IES, light linking | Light tree 8-12 d; IES 2-3 d; linking 3-5 d |
| Shadows | Virtual Shadow Maps, ray-traced and contact shadows | Ray-traced in every mode (Rendered area-correct; Preview one soft shadow ray per lamp; Solid one directional ray) | Nothing in quality; cost grows per lamp per pixel | - |
| Post | TSR upscaling, bloom, auto exposure, motion blur, cinematic DOF, grain, LUTs | Blender colour management (Standard, Filmic, AgX, looks, exposure, gamma), Workbench DOF, path-traced DOF with bladed bokeh, sample clamp, a-trous denoiser | Temporal upscaling, bloom, auto exposure, motion blur, grain | Bloom and exposure 3-5 d; post motion blur from a velocity pass 6-10 d; temporal upscaler 10-15 d |
| Ray tracing hardware | DXR / Vulkan RT | Software traversal in compute | Hardware ray queries (experimental in wgpu) | 10-20 d, driver risk |
| Display | Straight to the swapchain | Read back to the CPU every frame, handed to gpui (DirectX 11 on Windows) | GPU-resident present | 5-10 d |

### 3.2 Against Apple RealityKit and Metal

| Area | RealityKit / Metal | phiray today | Missing | Effort |
|---|---|---|---|---|
| PBR | `PhysicallyBasedMaterial` (base colour, metallic, roughness, normal, emissive, ambient occlusion, clearcoat, sheen, anisotropy, opacity), `ShaderGraphMaterial` (MaterialX) | All of those inputs except an AO map | AO map input; MaterialX graphs | AO map 1-2 d; MaterialX rides on the node interpreter above |
| Image-based lighting | `ImageBasedLightComponent`, environment resources, AR environment probes | HDRI world with importance sampling (Rendered), SH9 + prefiltered levels (Preview), studio rig and MatCaps (Solid) | Local reflection probes | Optional, 5-8 d |
| Physics | `PhysicsBodyComponent` (dynamic, static, kinematic), `CollisionComponent` shapes (box, sphere, capsule, convex, mesh), joints (fixed, spherical, revolute, prismatic), collision events | None. `crates/apple` (not built) has API-shaped structs and a toy cloth | All | As section 5 |
| Anchors | `AnchorEntity` (plane, image, face, body, world) on ARKit tracking | None; `crates/apple/.../anchor_entity.rs` is structs only | Camera tracking stack | Out of scope for a desktop Blender-style app |
| Shadows | Grounding and dynamic shadows | Ray-traced | - | - |
| Compute libraries | MPS (convolution, bilateral, matrix multiply), MetalFX upscaling, Metal ray tracing | `kernels/image.rs` has the three MPS kernels in WGSL with CPU twins; own BVH traversal | Upscaling; hardware RT | Upscaler 10-15 d; hardware RT as above |
| Platform | Metal only, tuned per Apple GPU | wgpu across Vulkan, Metal, DX12, WebGPU; `examples/metal_probe.rs` | Per-platform tuning | As needed |

### 3.3 Against Blender (the thing phiber imitates)

| Blender | Feature | phiray today | Missing | Effort |
|---|---|---|---|---|
| Workbench (Solid) | Studio, MatCap, Flat lighting; colour by material, object, random, single, attribute, texture | Ported (`workbench.wgsl`, `solid.wgsl`, MatCap at `solid.wgsl:69`, Texture at `solid.wgsl:34`) | - | - |
| Workbench | Cavity, curvature, shadow, outline, X-ray, DOF, AA (TAA + SMAA), backface culling, specular toggle | Ported (`fx.wgsl`, `dof.wgsl`, `taa.wgsl`, `smaa*.wgsl`, X-ray up to 32 layers) | - | - |
| EEVEE (Material Preview) | World probe lighting, light probes (sphere, volume), screen-space and ray-traced reflections, virtual shadow maps, volumetrics, SSS, motion blur | Deterministic one-ray-per-pixel kernel: IBL, one soft shadow ray per lamp, glass, mip-level textures | Bounce light, scene reflections (reflections show the environment, not other objects), volumetrics, SSS | Probes 15-25 d (shared with the Lumen row); scene reflections by a second ray 3-5 d |
| Cycles (Rendered) | Path tracing, light paths and bounce limits, MIS, adaptive sampling, sample clamp, caustics and filter glossy, transparent film | All present (`shade.wgsl`, `light.wgsl`, `adapt.wgsl`) | - | - |
| Cycles | Light tree | Power CDF over up to 32 lamps and emissive faces | Light tree | 8-12 d |
| Cycles | Volumes (world and object, VDB) | None | Homogeneous volumes; VDB grids | 8-12 d; VDB 20+ d |
| Cycles | Random-walk SSS | Wrap-lighting approximation (`scatter_bleed`, `sss_tint` in `light.wgsl`) | Real subsurface transport | 8-12 d |
| Cycles | Hair and curves | None (triangles only) | Curve primitive and intersection | 10-15 d |
| Cycles | Motion blur | None | Time-sampled transforms and refit per time step | 15+ d |
| Cycles | True displacement, adaptive subdivision | Bump only (procedural bump slot, normal maps) | Displacement at load | 10-15 d |
| Cycles | Shader nodes, OSL | Subset baked by phiphx (`blend/shade`) into a fixed record | General graphs | 20-30 d (node interpreter) |
| Cycles | Denoising (OpenImageDenoise, OptiX) | a-trous guided by albedo, normal, depth | ML denoiser | OIDN through its C API on the CPU: 3-5 d |
| Cycles | Passes, AOVs, Cryptomatte, baking | Radiance, albedo, normal, depth | The rest | Passes 1-2 d each; Cryptomatte 4-6 d; baking 10+ d |
| Cycles | Particle and collection instancing | None shared; one object, one copy | Shared geometry | 5-8 d (section 4) |
| Physics | Bullet rigid bodies, cloth, soft body, fluid, particles | None | Rigid, joints, cloth, soft | 51-72 d (section 5); fluids and particles not planned |

---

## 4. Warp's OpenGL viewer (`reference/warp/_src/render/render_opengl.py`)

3,889 lines of Python on pyglet and OpenGL 3.3, Apache-2.0, pinned commit in
`reference/warp/SOURCE.txt`. `ANALYSIS.md` section 6 is right that it is a
debug rasteriser, not a path tracer.

### 4.1 What it does

- **Shape registry** (`register_shape`, line 2394): each geometry is
  registered once under a hash (`geo_hash`) and deduplicated; one VAO, VBO
  and index buffer per shape; vertices are 8 floats (position, normal, UV).
  Procedural generators for sphere, capsule, cylinder, cone, box, plane and
  arrow at 32 segments (lines 3637-3878).
- **Instancing** (`allocate_shape_instances`, line 2577): one instance buffer
  for all shapes; each instance is a `mat4` passed as four `vec4` attributes
  (locations 3-6, divisor 1) plus two checker colours (7, 8). Each shape is
  drawn with `glDrawElementsInstancedBaseInstance` at its offset.
- **Simulation to render on the GPU** (`update_vbo_transforms`, line 265;
  `update_body_transforms`, line 2716): an instance can be attached to a
  simulation body; a Warp kernel computes `body_q[body] * local`, converts the
  quaternion and scale to a matrix and writes it straight into the GL buffer
  through CUDA-GL interop (`RegisteredGLBuffer`). The CPU never touches it.
- **Deformable meshes** (`update_shape_vertices`, line 3610): sim particle
  positions written into the VBO in place; smooth normals by
  `compute_average_normals` with float `atomic_add` (line 410).
- **Points and lines** (`render_points`, line 3481; `_render_lines`): points
  as instanced spheres sized per point (`update_points_positions`), segments
  as instanced capsules oriented along the segment (`update_line_transforms`).
- **Shading** (lines 24-117): Phong with ambient 0.3, the sun plus a fixed
  fill light at 0.3, specular powers 32 and 64, and a UV checker faded to grey
  by screen-space derivatives so it does not moire. **No shadows, no
  textures** (the texture loaders are commented out, line 2919 onwards), no MSAA,
  no tone mapping or sRGB handling.
- **Background**: a sky sphere with a tanh gradient and sun glow; a grey
  `GL_LINES` grid.
- **Offscreen**: an FBO with RGB8 colour and 32-bit depth, shown as a
  full-screen quad; `render_depth` shows depth through a Bourke colour map.
- **Tiled multi-camera** (`setup_tiled_rendering`, line 1529;
  `_render_scene_tiled`, line 2199): N tiles, each with its own viewport
  rectangle, view and projection matrix and its own list of instances. The
  draw loops over tiles: set the viewport, draw grid and sky, then one draw
  call per instance in the tile. `get_pixels` (line 2788) reads the whole FBO
  into a PBO, maps it into CUDA, and `copy_rgb_frame_tiles` (line 513)
  splits it into a `(tiles, h, w, 3)` array, flipped, as float or u8; depth
  is linearised from NDC. Used for batch camera rendering of many simulation
  environments into image tensors with no CPU copy.
- **Loop**: fly camera with keys and mouse, `up_axis` handled by a model
  matrix, `begin_frame`/`end_frame`, pause, FPS counters, 2D and 3D callbacks.

### 4.2 What phiray already does better

Shading in every mode (Workbench studio rig and MatCaps, ray-traced shadows,
cavity, AA, colour management), textures with mips, the floor grid (filtered
by the pixel footprint, with axes), headless rendering (the tracer is
headless already), depth readback (`Tracer::depth`). None of Warp's GLSL is
worth porting. The derivative-faded checker idea is already how phiray filters
its grid and procedural textures (compute has no derivatives, so it uses
`pix * t`, the pixel footprint at the hit).

### 4.3 What phiray lacks, mapped onto its modes

| Warp feature | Solid | Preview | Rendered | Port? |
|---|---|---|---|---|
| Shared-geometry instancing (one mesh, N transforms) | yes | yes | yes | Yes, first: prerequisite for the rest |
| Primitive shapes (sphere, capsule, box, cylinder, cone, plane, arrow) | debug draw, empties | yes | yes | Yes |
| Instances attached to simulation bodies, matrices written on the GPU | yes | yes | yes | Yes |
| Deformable vertices written on the GPU | yes | yes | yes | Yes |
| Points as instanced spheres | particles | particles | particles | Yes |
| Lines as instanced capsules | overlays | - | - | Yes, as an overlay |
| Tiled multi-camera with GPU tile split | thumbnails, quad view | material balls, asset browser | camera arrays | Yes |
| Depth view through a colour map | debug | - | - | Small, optional |
| Phong and checker shading, sky sphere, line grid, fly camera, info label | - | - | - | No |
| CUDA-GL interop, FBO/PBO, legacy GL path | - | - | - | No: one wgpu device makes it unnecessary |

Why it matters now: Warp's viewer exists so that a simulation can be drawn
without the CPU touching its data. phiray needs exactly that the day physics
lands: thousands of rigid bodies sharing a few hull meshes, cloth vertices
changing every substep, and contact and joint overlays.

### 4.4 Port plan to wgpu (ideas and data layouts)

**Step 1. Shared-geometry instancing (5-8 d) — done.** `Instance` is
split in two (`mesh.rs`, `instance.rs`); the GPU record keeps its 40 words
with the mesh's root, face range and texture space copied in and the mesh
id in word 38, so a hit reads one record. The light table lists an
emissive face once an instance (`lamps.rs`), the raster target carries the
instance in its second word with the barycentrics packed into the third
(`raster.wgsl`), and phiphx's collection instances share their member's
`Arc<SceneMesh>` (`scene/share.rs`), which phiber's scene build keys on
(`rendered/scene.rs`). The plan as written:

```rust
/// Geometry, uploaded once: a range of faces and vertices and its tree.
pub struct Mesh { pub face_lo: u32, pub face_hi: u32, pub vert_lo: u32, pub vert_hi: u32, pub root: u32, pub bounds: [[f32; 3]; 2] }

/// A placement of a mesh. Many instances may name one mesh.
pub struct Instance { pub world: [[f32; 4]; 4], pub mesh: u32, pub body: i32, pub color: [f32; 3], pub hash: u32, pub flags: u32 }
```

The face word stops carrying the instance id (today its top 16 bits, read as
`face(h.tri).w >> 16`); the traversal carries the instance from the top-tree
leaf into the hit (`Hit { t, u, v, tri, inst }`), and `outline_store`,
`inst_at` and the procedural object coordinates read `h.inst`. The raster
prepass draws one instanced draw per mesh with `@builtin(instance_index)`
naming the instance, and its target gains a word for the instance. The
65,536 cap goes. Unlocks collection instances, particle-instanced objects and
rigid bodies that share hulls.

**Step 2. Primitive registry (3-5 d).** A `Shapes` table from a geometry key
(kind plus dimensions, as Warp's `geo_hash`) to a `Mesh` slot, generated once
in object space with exact normals. Capsules and cylinders are generated per
radius and half-height, not scaled non-uniformly, as Warp does. Later option:
analytic sphere and capsule leaves in the tree (exact silhouettes, a few bytes
each) flagged in the leaf, for large particle counts.

**Step 3. Instance streams written on the GPU (3-5 d).**
- `Place` kernel (Warp's `update_vbo_transforms`): reads `Body` poses and a
  per-instance `Local { offset: [[f32; 4]; 3], scale: [f32; 3], body: i32 }`,
  writes the instance record (world matrix and the inverse the traversal
  needs, `INSTANCE_WORDS = 40`) and the instance's world AABB.
- Top tree: refit on the GPU from those AABBs with the existing `Refit`
  kernel, instead of `Trees::retlas` on the CPU plus an upload; rebuild on the
  GPU every few frames (it is small).
- `Points` kernel (Warp's `update_points_positions`) and `Segments` kernel
  (Warp's `update_line_transforms`). For segments, build the basis with the
  branchless orthonormal basis `light.wgsl` already has, not Warp's
  `acos` plus axis-angle, which degenerates when a segment is parallel to up.

**Step 4. Deformable vertices written on the GPU (4-6 d).** A kernel writes
particle positions into the tracer's vertex buffer range of that mesh (32-byte
vertices: position plus packed normal and UV), addressed through the same
2- or 4-binding split as `binds2.wgsl`/`binds4.wgsl`. Normals: Warp uses float
`atomic_add`; WGSL has none, so use the mesh's static vertex-to-face CSR and
sum adjacent face normals per vertex in a fixed order. Then refit that mesh's
tree on the GPU (today `repose` refits on the CPU with `Bvh::refit_slots` and
uploads); the tree is the same `Node` layout, shifted into the shared node
array, so the `Refit` kernel needs a sub-range and a precomputed level order.

**Step 5. Tiled multi-camera (5-8 d).** Compute makes this simpler than Warp's
redraw per viewport:

```rust
/// One tile of a larger image.
pub struct Tile { pub rect: [u32; 4], pub view: u32, pub root: u32 } // x, y, w, h; camera; top-tree root (u32::MAX = the scene's)

/// The per-view part of today's `Camera` uniform, moved to a storage array.
pub struct ViewCam { pub origin: [f32; 3], pub ortho: f32, pub right: [f32; 3], pub fov: f32, pub up: [f32; 3], pub clip_start: f32, pub fwd: [f32; 3], pub clip_end: f32 }
```

One dispatch over the whole atlas image: each pixel finds its tile (uniform
grid: a divide; free layout: one `Pass::add_over` slice per tile), takes
`views[tile.view]`, and traverses from `tile.root`. Lights and settings stay
in the shared uniform. Per-tile scenes (an asset browser where each tile shows
one asset) get their own tiny top tree each, which is what `root` is for;
Warp's per-tile instance lists become that. One readback, then a `Split`
kernel (Warp's `copy_rgb_frame_tiles`) packs each tile contiguously so a
thumbnail is one slice. Works for all three modes; Rendered tiles accumulate
independently because accumulation is already per pixel. The tile seams are
already handled (`Tracer::seam_rows`).

**Step 6. Physics overlay (3-5 d).** Collision shapes, contact points and
normals, joint frames, velocity arrows, cloth pins: instanced primitives in a
small overlay top tree traced after the scene hit, flat-shaded, depth-tested
against it (or drawn through, X-ray style), composited into the display pixel
the way `outline.wgsl` composites the selection outline.

### 4.5 What Warp's subset gives physics, and what it does not

In the subset, usable as reference: `native/bvh.h` (`bvh_query_aabb`, the
overlap traversal for the broadphase), `native/mesh.h` (`mesh_query_point*`:
closest point with inside/outside by parity, normal or winding number;
`mesh_query_aabb` for the midphase), `native/hashgrid.*` (particle neighbour
search, for particle self-collision later), `native/coloring.cpp` (graph
colouring for parallel Gauss-Seidel), `native/deterministic.h` (deterministic
atomics by sorting scatter records, the same idea as phiray's 64-bit radix
key), `sort`, `scan`, `reduce`, `runlength_encode` (segmented reductions for
the contact CSR), `svd.h` (3x3 SVD for shape matching), `tile_cholesky.h`
(small dense solves).

Not in the subset: `warp.sim` (its integrators, collision kernels, model
builder and importers) and `warp.fem`. Upstream, Warp's simulation work has
since moved to a separate project (Newton); if a reference solver is wanted,
check its licence and vendor it read-only next to Warp. The design above does
not depend on it.

---

## 5. Milestones

Physics first, because that is the gap the user feels; the Warp port steps
are slotted where physics needs them.

| Phase | Work | Days | What it unlocks in phiber |
|---|---|---:|---|
| **1** | **M0** Physics settings reader in phiphx (`blend/physics/`, `ScenePhysics`); Physics and Scene panels show the file's values; `SimCache` driven by the playhead | 4-6 | Opening a Blender physics file shows its real settings; frames step a (still empty) simulation clock |
| 1 | **M1** Rigid bodies, CPU twin first: integrator, sphere, box, capsule, cylinder, cone, hull; broadphase over phiray's `Bvh`; narrowphase (closed form, SAT and clipping, GJK/EPA); XPBD contacts, friction, restitution, damping, sleeping; passive animated bodies; output through `reinstance` | 10-14 | Blender rigid body scenes (falling, stacking, dominoes) play in every shading mode |
| 1 | **R-overlay** Primitive registry and physics overlay (port steps 2 and 6, CPU-placed) | 5-7 | You can see collision shapes, contacts and joint frames while debugging M1 and M2 |
| **2** | **M2** Joints: the eight Blender constraint types, limits, springs, motors, breaking, disabled collisions | 6-8 | Hinges, chains, ragdolls and machines from files |
| 2 | **M4** Cloth: stretch, shear, bending, pins, damping, pressure, collision with Collision objects, particle-triangle self collision; coloured kernels and CPU twin | 12-16 | Blender cloth files simulate (flags, drapes, garments on posed characters) |
| **3** | **R1** Shared-geometry instancing (port step 1) | 5-8 | Collection instances; thousands of bodies sharing meshes; the 65,536 cap is gone |
| 3 | **R3** GPU-written instances and vertices (port steps 3 and 4) | 7-10 | Physics playback with no CPU round trip per frame |
| 3 | **M3** GPU rigid path: `Overlap`, pair sort, `Narrow`, contact CSR, Jacobi `Contacts`/`Apply`, with twins; size switch like `GPU_MIN_VERTS` | 10-14 | 10k+ bodies (debris, Cell Fracture output) at interactive rates |
| **4** | **M5** Soft bodies (surface XPBD with goal) | 5-8 | Blender soft body files |
| 4 | **M6** Bake to disk; parity fixtures against Blender bakes | 4-6 | Reproducible renders of simulated shots; known divergence from Blender, measured |
| 4 | **R4** Tiled multi-camera (port step 5) | 5-8 | Asset browser thumbnails, material previews, quad view in one dispatch |
| 4 | **R5** GPU-resident display (shared D3D handle, or gpui's wgpu renderer `phigpu/gpui_wgpu`) | 5-10 | Full-rate viewport; removes the per-frame readback that physics playback would double |
| **5** | Renderer gaps by value per day: bloom and exposure (3-5), OIDN (3-5), light tree (8-12), volumes (8-12), random-walk SSS (8-12), hair (10-15), node interpreter (20-30), Preview probes (15-25), streaming past 20 M triangles (10-15) | as listed | Closer to Cycles and EEVEE in what files can show |

Totals: physics M0 to M6, 51-72 d; Warp port R-overlay, R1, R3, R4, 22-33 d;
GPU display 5-10 d. Phases 1 and 2 (about 37-51 d) are the minimum for
"phiber has physics".

---

## M1 status (2026-10-29)

`src/sim/` (1,704 lines over ten files) and `tests/sim.rs` are in: the
`Pod` state arrays, one support function per convex kind, the broadphase
over `crate::bvh::Bvh`, the narrowphase (closed forms, SAT with
clipping, GJK/EPA, a passive mesh's triangle tree), XPBD substeps with
positional friction, dynamic friction and restitution on the
velocities, sleeping, `from_blend` and `poses`. CPU only; no kernels yet
(that is M3).

**Passing** (`cargo test -p phiray --test sim`, 7 of 8): free fall
against the closed form to 1e-3 over 100 frames; a ball bouncing to
`e²` of its height within 5%; a box held below its friction angle and
sliding above it; ten boxes standing 1,000 frames with the top one
inside a millimetre; a sphere and a box resting on the exact height to
the micron with no drift; Blender's own settings driving a crate to rest
on a mesh floor, out through `poses`; two runs bit-identical; a
frictionless ball keeping its energy over ten seconds.

**Not working: the four shapes the narrowphase answers with GJK/EPA.**
A capsule, cylinder, cone and convex hull collide and roughly settle,
but they creep — 15 to 18 mm over three seconds at ten substeps and ten
projections — and the cone and hull can sit a centimetre off their
height. `curved_shapes_rest_only_roughly` is `#[ignore]`d with the
measured numbers. The fault is the quality of the points and normals
EPA hands over for a curved or flat-faced convex, not the solver over
them: the sphere and box paths (closed form, SAT) are exact under the
same solver, and nothing is launched or loses energy any more. Snapping
the contact to a box's own face plane, the way `tri_pair` does for a
mesh triangle, was tried and changed nothing measurable, so the normal
being slightly off axis is *not* the cause; the next suspect is the
manifold's point set — `epa::manifold` picks the deepest witness plus
four tilted supports, which for a rim or a circular face are not the
points that hold it. Budget: 4 to 6 days, and it is the prerequisite
for M3 (the GPU path would inherit the same manifold).

**Also not wired yet:** an active body given a Mesh shape is simulated
as its convex hull, and `World::hull_fallback` lists those objects so an
open can say so - as §2.2 says it should, the way `skipped_modifiers`
reports a dropped modifier. The list is filled (`sim/blend.rs`) and
nothing reads it, so today the substitution is silent.

**Five bugs fixed in the solver while getting here**, each with the test
that catches it:

| Bug | What it cost |
|---|---|
| The friction cone compared a point's whole tangential ask against its *share* of the manifold's normal impulse | Friction almost never held: a box slid 3.3 m down a slope well inside its friction angle. No substep or iteration count helped, which is what marked it as a predicate bug rather than convergence |
| Friction was skipped whenever the normal pass had nothing left to do | The second and later projections of a substep held nothing, though the drift was still there |
| Penetration recovery became velocity | 3 cm at a 600 Hz substep *is* 18 m/s: the capsule was launched to z = 22 and the ball came back to four times its drop. Now split: the recovery moves the pose and is kept out of the derived velocity (Bullet's split impulse), rate-limited by `World::max_recover` |
| The restitution pass drove *every* normal velocity to its target, every substep | Once recovery spanned several substeps it cancelled the separation the bounce had just made. It now only ever applies an impulse that separates |
| Blender's `solver_iterations` was divided by the substep count | Bullet runs that many in *each* internal step, so the default file was left on one projection a substep, and the stack settled 2 mm instead of 0.07 mm |

The test for "at rest" now reads against `g h / 2`, the speed a body
carries when a frame ends on half a substep's kick of gravity: a sphere
and a box sit at exactly that, so an absolute budget near it was a
number only one substep count could meet.

## 6. Risks

1. **WGSL limits.** No float atomics, no `f64`, subgroups optional. The design
   uses sort-and-segment reductions and colouring instead; anything that
   assumes float atomics (Warp's normal averaging, many published GPU
   solvers) must be reworked. Positions are `f32`: large worlds need
   positions relative to a local origin.
2. **Jacobi convergence for stacking.** GPU contacts use Jacobi with averaging,
   which is weaker than Gauss-Seidel on tall stacks. Mitigation: substeps, the
   CPU twin for small scenes, per-frame contact colouring if needed.
3. **Blender parity.** Blender runs Bullet for rigid bodies and an implicit
   mass-spring solver for cloth. XPBD will not reproduce Blender's bakes; files
   tuned in Blender will move differently. Mitigation: parity fixtures with
   stated tolerances, a documented stiffness-to-compliance mapping, and
   reading Blender's baked point caches where present so baked shots play
   exactly.
4. **Determinism across machines.** Same device, same result; different
   vendors differ (FMA contraction, transcendental precision). Renders of
   simulated shots must play from the cache, never re-simulate.
5. **One device, one queue.** Physics dispatches share the queue with the
   render thread (`rendered/worker.rs`) and the 5 s readback guard
   (`gpu/mod.rs`). Steps must be many small dispatches, and the render thread's
   slicing (`sample_in`) must leave room for them.
6. **The CPU round trip today.** Every viewport frame is read back and handed
   to gpui on DirectX 11. Physics with CPU-side hand-off adds a second trip;
   R3 and R5 remove both, R5 is the sizeable one.
7. **Instancing format change.** R1 changes the face word, the hit record, the
   raster prepass target and every kernel that names the instance. It touches
   most of `trace/`; do it in one change with the existing tests as the guard.
8. **Mesh-mesh collision.** Active bodies with Mesh shapes become convex hulls;
   Blender users rely on that combination sometimes. Surface it in the open
   report.
9. **Scope.** Unreal-parity items (Lumen, Nanite, Chaos destruction) are each
   multi-person-year systems. They are listed so the gap is visible, not as
   plans.
10. **Concurrent change in phiber.** The UI is being reworked in parallel; the
    property panels and playback code named here may move. The integration
    points (settings in `SceneModel`, `reinstance`, `repose`, playhead) are the
    stable seams.

---

## 7. Drift between the docs and the code

| Where | Says | Code |
|---|---|---|
| `README.md` table | Solid is "phiber canvas (today)" | Solid is phiray's Workbench kernel (`solid.wgsl`, `workbench.wgsl`); phiber maps every shading mode to phiray (`gpu_mode`, `phiber/src/view/draw/canvas/view3d/draw.rs:57`); the CPU painter is only the no-adapter fallback |
| `README.md` table | Preview is "wgpu raster, mipmapped textures, tiled previews" | Preview is a deterministic compute ray cast with IBL (`preview.wgsl`); there are no tiled previews |
| `README.md` tree | five source areas | also `ibl`, `taa`, `smaa`, `outline`, `proc`, `instance`, `pose`, `denoise`, `image`, and `Mode::Wire` |
| `src/lib.rs`, `mode.rs:7` | "Three modes"; Solid "phiber canvas today" | Four modes; Solid is in phiray |
| `trace/mod.rs:1` | "primary ray, sun with shadow, short-range ambient occlusion, sky gradient" | A full path tracer and three more modes |
| `trace/kernel.rs:295` | MatCap "drawn as Studio"; Texture colour "as Material" | `solid.wgsl:69` samples the MatCap; `solid.wgsl:34` samples the texture |
| `jobs.md` | "4 sampled per pixel" | One file lamp per shading point picked by power, plus one emitter-triangle and one environment sample (`light.wgsl` `direct`) |
| `jobs.md` | "today sort/pack run on the CPU" | Sort is on the GPU (`bvh/radix.rs`, `bvh/sort.rs`); only packing is on the CPU |
| `jobs.md` | "Up to 16 file lights" | `LIGHTS = 32` (`trace/kernel.rs:485`) |
| `jobs.md` | Instancing (BVH per mesh plus top level) is a todo | A tree per object plus a top tree exists (`instance.rs`, `trees.rs`); geometry is still not shared |
| `reference/warp/ANALYSIS.md` section 7 | Solid "nothing on GPU yet"; Preview "wgpu raster" | Both on the GPU, both compute ray casts |
| This audit's brief | phiphx reads Blender's rigid body and cloth settings | It reads none; `report.rs` lists modifier names only |
| This audit's brief | Warp's viewer has shadows | It has none |
