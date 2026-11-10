# reference

Source phiray reads and ports from. None is built. Thirty-two Meta
(facebookresearch) repositories are git submodules, shallow where they were
added shallow; Warp is a copied subset. One line each, in the order of the
work they feed.

- `warp/`: NVIDIA Warp, a read-only subset (Apache-2.0); the GPU design follows `warp/ANALYSIS.md`

Rendering, ray casting and visibility:

- `HVVR/`: hybrid CPU/GPU ray caster with hierarchical visibility, beams and wide BVH nodes
- `DRTK/`: differentiable rendering toolkit, a visibility buffer and its edge gradients
- `pytorch3d/`: deep learning components for 3D data, the coarse-to-fine rasteriser
- `vkraygs/`: hardware-rasterised ray-based Gaussian splatting (C++, Vulkan)
- `CLOD-3DGS/`: continuous level of detail for fast splat rendering (Vulkan)
- `volumetric_primitives/`: ray-traced volumetric primitives for scattering media (Mitsuba)
- `mvp/`: mixtures of volumetric primitives, a BVH over boxes ray-marched on the GPU
- `neuralvolumes/`: neural volumes, a voxel grid ray-marched from images
- `NSVF/`: neural sparse voxel fields, sparse voxel octree ray marching
- `lightplane/`: memory-efficient differentiable radiance field renderer
- `HairMSNN/`: path-traced hair with learned high-order scattering and radiance caching
- `DONERF/`: depth oracle that cuts the samples a ray needs
- `DeepFovea/`: foveated rendering, the periphery rebuilt from few pixels
- `AnalyticTextXR/`: reference renderer for real-time vector graphics
- `SteepGS/`: density control for compact Gaussian splats
- `Replica-Dataset/`: ReplicaSDK, the renderer for the Replica scenes

Scene representations and view synthesis:

- `hyperreel/`: 6-DoF video with ray-conditioned sampling
- `neural-light-fields/`: neural light fields with a ray-space embedding
- `TMPI/`: tiled multiplane images for view synthesis
- `Online-3DGS-Monocular/`: Gaussian splats built online from monocular video
- `egocentric_splats/`: Gaussian splats from egocentric (Project Aria) captures
- `4DGT/`: 4D Gaussian transformer, dynamic splats from video
- `localrf/`: progressively optimised local radiance fields for long camera paths
- `robust-dynrf/`: robust dynamic radiance fields from casual video
- `nonrigid_nerf/`: non-rigid radiance fields for deforming scenes
- `SceNeRFlow/`: time-consistent reconstruction of general dynamic scenes
- `PlatoNeRF/`: 3D reconstruction from single-view two-bounce lidar

Vision, capture and scene understanding:

- `ocean/`: C++ framework for computer vision and augmented reality
- `scenescript/`: scene layout as a structured language (SceneScript)
- `efm3d/`: benchmark for egocentric 3D foundation models
- `hot3d/`: hand and object tracking in 3D from egocentric video
- `projectaria_tools/`: C++/Python toolkit for Project Aria data

Apple's rendering standards, by path (SDK files, they stay where they are):

- Metal, MetalFX, MetalKit, QuartzCore, Model I/O headers: `gemphi/phi/apple-source/ruskit/Interfaces/MacOSX/`
- Metal Performance Shaders: `gemphi/phi/apple-source/ruskit/MetalPerformanceShaders.framework`
- the mapping onto phiray and the speed list: `../docs/APPLE.md`

Characters, rigs, animation and geometry sit with `phiphx` in
`crates/phiphx/reference`. Robotics repositories sit with `phiace` in
`crates/phiace/reference`.
