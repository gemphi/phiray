# reference

Trees phiray reads and ports from. None is built.

- `warp/`: NVIDIA Warp; the GPU design follows `warp/ANALYSIS.md`.
- Meta (facebookresearch) rendering, 3D and vision repositories, as git submodules:
  - `ocean/`: C++ framework for computer vision and augmented reality
  - `DRTK/`: differentiable rendering toolkit
  - `pytorch3d/`: deep learning components for 3D data
  - `scenescript/`: scene layout as a structured language (SceneScript)
  - `efm3d/`: benchmark for egocentric 3D foundation models
  - `hot3d/`: hand and object tracking in 3D from egocentric video
  - `projectaria_tools/`: C++/Python toolkit for Project Aria data
- Meta renderers and ray casters (added 2026-11-10, shallow submodules):
  - `vkraygs/`: hardware-rasterised ray-based Gaussian splatting (C++, Vulkan)
  - `CLOD-3DGS/`: continuous level of detail for fast splat rendering (Vulkan)
  - `HVVR/`: hybrid CPU/GPU ray caster with hierarchical visibility
  - `volumetric_primitives/`: ray-traced volumetric primitives for scattering media (Mitsuba)
  - `HairMSNN/`: path-traced hair with learned high-order scattering and radiance caching
  - `lightplane/`: memory-efficient differentiable radiance field renderer
  - `mvp/`, `neuralvolumes/`: mixtures of volumetric primitives, ray-marched volumes
  - `DONERF/`: depth oracle that cuts the samples a ray needs
  - `DeepFovea/`: foveated rendering, the periphery rebuilt from few pixels
  - `AnalyticTextXR/`: reference renderer for real-time vector graphics
  - `NSVF/`: sparse voxel octree ray marching
  - `SteepGS/`: density control for compact Gaussian splats
  - `Replica-Dataset/`: ReplicaSDK, the renderer for the Replica scenes
- Meta scene representations and view synthesis:
  - `hyperreel/`, `neural-light-fields/`, `TMPI/` (tiled multiplane images)
  - `Online-3DGS-Monocular/`, `egocentric_splats/`, `4DGT/` (splats from video)
  - `localrf/`, `robust-dynrf/`, `nonrigid_nerf/`, `SceNeRFlow/`, `PlatoNeRF/` (radiance fields)
- Apple's rendering standards, by path (SDK files, they stay where they are):
  - Metal, MetalFX, MetalKit, QuartzCore, Model I/O headers: `gemphi/phi/apple-source/ruskit/Interfaces/MacOSX/`
  - Metal Performance Shaders: `gemphi/phi/apple-source/ruskit/MetalPerformanceShaders.framework`
  - the mapping onto phiray and the speed list: `../docs/APPLE.md`

Characters, rigs, animation and geometry sit with `phiphx` in
`crates/phiphx/reference`. Robotics repositories sit with `phiace` in
`crates/phiace/reference`.
