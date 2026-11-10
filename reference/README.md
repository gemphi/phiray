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

Robotics repositories sit with `phiace` in `crates/phiace/ext`.
