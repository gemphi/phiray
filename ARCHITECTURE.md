# phiray — architecture

Solid boxes are in the crate today. Dashed boxes are to build, each named
with the reference it is read from (`reference/<repo>`) or the Apple header
it answers. Nothing here is measured; the speed list is
[docs/APPLE.md](docs/APPLE.md).

## The modules today and what they use

```mermaid
graph LR
    m_trace["trace<br/>Scene · Tracer · trees · pose · lamps · mat · proc · ibl<br/>Trace · Solid · Wire · Preview · Adapt ·<br/>Taa · Smaa · Fx · Glow · Show · Edge"]
    m_sim["sim<br/>broad · narrow: gjk, epa, sat ·<br/>contact · world"]
    m_kernels["kernels<br/>Denoise · Convolve · Bilateral ·<br/>MatMul · Skin"]
    m_mode["mode<br/>Wire · Solid · Preview · Rendered"]
    m_bvh["bvh<br/>sah: Bvh · Node · Aabb<br/>gpu: Morton · radix sort ·<br/>Karras · Refit · Spread"]
    m_gpu["gpu<br/>Gpu · GpuArray · Pass ·<br/>Visibility raster · readback"]
    m_kernel["kernel<br/>Kernel · Bind · Dispatch · BLOCK"]
    m_hot["hot<br/>live WGSL reload"]
    m_trace --> m_bvh
    m_trace --> m_gpu
    m_trace --> m_kernel
    m_trace --> m_kernels
    m_trace --> m_mode
    m_sim --> m_bvh
    m_kernels --> m_gpu
    m_kernels --> m_kernel
    m_bvh --> m_gpu
    m_bvh --> m_kernel
    m_gpu --> m_kernel
    m_kernel --> m_hot
```

## The modules to build and what each stands on

```mermaid
graph LR
    m_clock["clock<br/>GPU time a dispatch, counters a pixel<br/>MTLCounters"]
    m_prim["prim<br/>one intersect seam, many kinds: splat ·<br/>ellipsoid · brick · sparse voxel · curve"]
    m_medium["medium<br/>overlap stack, closed-form transmittance<br/>volumetric_primitives"]
    m_sample["sample<br/>the sample set of a frame: foveation layers,<br/>blue noise, adaptive in the viewport<br/>CLOD-3DGS · HVVR · DONERF"]
    m_beam["beam<br/>tile frustum to a BVH cut<br/>HVVR"]
    m_vis["vis<br/>the visibility record for traced modes too:<br/>face · instance · u · v, as gpu::Visibility<br/>holds for Solid and Wire today<br/>DRTK · pytorch3d · HVVR"]
    m_lod["lod<br/>detail as a prefix count, frame-time budget<br/>CLOD-3DGS"]
    m_recon["recon<br/>temporal reuse, layer compositor, cache combine<br/>HVVR · MTLFXTemporalScaler · MPSSVGF · HairMSNN"]
    m_surface["surface<br/>analytic vector texture · per-face texture ·<br/>hair BSDF<br/>AnalyticTextXR · Replica-Dataset · HairMSNN"]
    m_grad["grad<br/>edge gradients over vis, soft blend<br/>DRTK · pytorch3d"]
    m_hw["hw<br/>hardware ray query behind a feature<br/>MTLAccelerationStructure"]
    t_trace["trace"]
    t_bvh["bvh"]
    t_gpu["gpu"]
    t_kernels["kernels"]
    m_medium --> m_prim
    m_lod --> m_prim
    m_lod --> m_clock
    m_beam --> m_sample
    m_recon --> m_vis
    m_grad --> m_vis
    m_prim --> t_bvh
    m_beam --> t_bvh
    m_hw --> t_bvh
    m_clock --> t_gpu
    m_vis --> t_gpu
    m_recon --> t_kernels
    m_sample --> t_trace
    m_surface --> t_trace
    m_prim --> t_trace
    m_vis --> t_trace
    classDef plan stroke-dasharray: 6 4;
    class m_clock,m_prim,m_medium,m_sample,m_beam,m_vis,m_lod,m_recon,m_surface,m_grad,m_hw plan;
```

## The frame, as every reference renderer divides it

Read from nine Meta renderers: each keeps stage 6's output as a buffer and
runs 7 to 9 as passes over it. phiray does so for Solid and Wire (the
raster prepass); for Preview and Rendered its megakernel fuses 3 to 8.

```mermaid
graph LR
    s1["1 scene compile"] --- h1["Scene · Atlas · Mat<br/>trace/scene.rs"]
    s2["2 frame update"] --- h2["repose · reinstance · retree<br/>trace/pose.rs · kernels::Skin"]
    subgraph mega["Trace megakernel: one dispatch a sample"]
        s3["3 sample set"]
        s4["4 coarse visibility"]
        s5["5 ordering"]
        s6["6 fine visibility"]
        s7["7 shade"]
        s8["8 resolve"]
    end
    s3 --- h3["camera.wgsl · pixel filter · PCG hash"]
    s4 --- h4["Solid, Wire: raster prepass, gpu::Visibility<br/>Preview, Rendered: none, a ray starts at the root"]
    s5 --- h5["closest hit · any hit"]
    s6 --- h6["bvh.wgsl: trace · trace_blas · tri_t<br/>vis2.wgsl: primary reads the prepass"]
    s7 --- h7["light.wgsl · shade.wgsl ·<br/>texture.wgsl · procedural.wgsl"]
    s8 --- h8["running sums a pixel · Adapt"]
    s9["9 reconstruct"] --- h9["Denoise: four a-trous passes · Taa · Smaa"]
    s10["10 present"] --- h10["Fx · Glow · Show · Edge ·<br/>readback to phiber"]
    s11["11 backward"] --- h11["none"]
    h1 -.- p1["prim: kinds compiled to flat arrays<br/>lod: detail ordered as a prefix"]
    h2 -.- p2["top tree rebuilt on the GPU: Morton · Karras · Refit<br/>mvp bvh.cu, kernels already in bvh/"]
    h3 -.- p3["sample: foveation layers, blue noise, ray footprint"]
    h4 -.- p4["beam: tile frustum to a BVH cut"]
    h5 -.- p5["collect all overlaps · ordered intervals<br/>radix sort for splats, already in bvh/radix.rs"]
    h6 -.- p6["vis: the same four words kept for traced modes<br/>hw: ray query under the same seam"]
    h7 -.- p7["medium · surface kinds"]
    h8 -.- p8["adaptive in the viewport · prefix plus cache"]
    h9 -.- p9["recon: temporal reuse, layer compositor"]
    h10 -.- p10["GPU-resident present"]
    h11 -.- p11["grad: edge gradients over vis"]
    classDef plan stroke-dasharray: 6 4;
    class p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 plan;
```

## Pages, one subject each

- [frame](docs/frame.md): one sample and its tail today, by mode, and the split to make
- [primitives](docs/primitives.md): the kinds a ray can meet and the one seam they share
- [accel](docs/accel.md): trees, refit, node format, beam cut, ray query
- [sampling](docs/sampling.md): which rays a frame casts, and how many
- [reconstruct](docs/reconstruct.md): from sums to a shown image
- [gradients](docs/gradients.md): the differentiable lane

Folders and order of work: [PLAN.md](PLAN.md) ·
What each part must do: [SPECS.md](SPECS.md) ·
Apple's standards and the speed list: [docs/APPLE.md](docs/APPLE.md) ·
What was ported: [docs/PORTS.md](docs/PORTS.md) ·
The references: [reference/README.md](reference/README.md) ·
What is left to do: [docs/ROADMAP.md](docs/ROADMAP.md)
