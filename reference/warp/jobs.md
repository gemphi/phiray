# Jobs — `phiray/reference/warp`

| State | Open | Done |
|---|---|---|
| Reference vendored + analysed | 9 | 2 |

## Todo

- [ ] `Kernel` trait (`const WGSL`, `type Params: Pod`) + hashed pipeline cache
- [ ] `Pass::dispatch(n)` with 256 workgroups and x→y spill (§2)
- [ ] `GpuArray<T>` (grow-by-doubling) + mapped staging ring; mmap file → ring (§3)
- [ ] Launch recording: build a step list once, replay per frame (§2)
- [ ] GPU skinning kernel writing vertex buffers in place (Solid + Preview)
- [ ] LBVH in compute: bounds → Morton (group in high bits) → radix sort → hierarchy → packed leaves (§4)
- [ ] Per-frame refit kernel after skinning (§4)
- [ ] WGSL traversal: 16-byte halves, 32-slot stack, slab + Möller (§4)
- [ ] Texture array + `textureSampleLevel` in the path tracer; tiled preview pass (§5, §6)

## Done

- [x] Warp subset vendored with licenses and pinned commit (`SOURCE.txt`)
- [x] `ANALYSIS.md`: kernel, memory, BVH, texture, render mechanisms mapped to wgpu

> Format: status table, `- [ ]` open, `- [x]` done. Move lines, don't delete them.
