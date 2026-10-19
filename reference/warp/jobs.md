# Jobs — `phiray/reference/warp`

| State | Open | Done |
|---|---|---|
| Reference vendored + analysed | 7 | 12 |

> **This list was read against the code on 2026-10-18.** Seven of the nine
> lines below had been sitting under *Todo* while the code was already
> there — the list was describing the plan, not the tree. Every line now
> names the file that proves it, so the next read is a check and not a
> guess. Where a line is half done it is split: the done half moved, the
> open half stayed and says what is still missing.

## Todo

- [ ] **Mapped staging ring.** `GpuArray::read_first` (`gpu/array.rs:140`)
      creates a staging buffer per read and drops it. Warp's pinned-memory
      analogue is a small pool of `MAP_WRITE`/`MAP_READ` buffers reused for
      the life of the device. Readback is on every viewport frame, so this
      is one allocation per frame we are still paying.
- [ ] **`mmap` the file into the ring.** Nothing maps `.blend`/`.phx`; the
      reader copies through `Vec`. §3 of `ANALYSIS.md`.
- [ ] **Skinning without the round trip.** `kernels/skin.rs` is the kernel
      and it is correct; `phiber/src/v2/scene/skin.rs:13` says the quiet
      part out loud — *"below this many vertices the CPU is faster than an
      upload + readback"*. The skinned vertices come back to the CPU and go
      up again as a vertex buffer. Warp's `RegisteredGLBuffer` trick (§3) is
      free for us because the compute buffer and the vertex buffer are on
      one device: the kernel should write the vertex buffer in place and
      `GPU_MIN_VERTS` should then be deletable.
- [ ] **Leaf packing in compute.** The sort has moved (below); `Bvh::assemble`
      has not, so the build still reads the sort order back to pack small
      and deep subtrees into leaves. That readback is the last hop in the
      LBVH build, and `Bvh::sort_codes` stays on the CPU path meanwhile.
- [ ] **Hardware texture sampling.** `Atlas` (`trace/kernel.rs:455`) is a
      storage buffer with a CPU-built mip chain, sampled by hand. §5 wants
      a `texture_2d_array` + `textureSampleLevel`, which moves filtering
      into the sampler and drops the manual bilinear from the inner loop.
      The mip chain itself is already there and keeps its meaning.
- [ ] **Tiled preview pass** (§6): many cameras, one dispatch, for
      thumbnails and the asset browser. The tile *seams* are solved
      (`Tracer::seam_rows`) — the multi-camera pass is not.
- [ ] **Pipeline cache on disk.** `Kernel::key` is FNV-1a over name +
      source and `Gpu::pipeline` caches in memory, so a kernel compiles
      once per *process*. Warp caches to the user cache dir and so skips
      the compile on the second run; `wgpu::PipelineCache` is the hook.

## Done

- [x] Warp subset vendored with licenses and pinned commit (`SOURCE.txt`)
- [x] `ANALYSIS.md`: kernel, memory, BVH, texture, render mechanisms mapped to wgpu
- [x] `Kernel` trait — `const WGSL`, `type Params: Pod`, `BINDS`, hashed
      pipeline cache per device (`kernel.rs`, `gpu/mod.rs:pipeline`)
- [x] 256-wide blocks and the lean x→y grid (`Dispatch::new`, `kernel.rs`),
      with a test that every item is covered once up to 20 M
- [x] `GpuArray<T>` grown by doubling, never per frame (`gpu/array.rs:50`)
- [x] Launch recording and replay — build the step list once, write only
      params per frame (`gpu/pass.rs`: `add`, `set`, `run`). This is the
      whole of Warp's graph capture and it is the reason the tracer is fast.
- [x] GPU skinning kernel, CSR influences, B-Bone segments (`kernels/skin.rs`)
- [x] LBVH on the GPU: scene bounds, 30-bit Morton, Karras links
      (`bvh/kernels.rs`) — all but the sort
- [x] Per-frame refit as a recorded Jacobi ping-pong (`bvh::Refit`,
      `GpuBvh::update`), so a posed frame refits instead of rebuilding
- [x] Node layout and traversal exactly as §4 asks: two 16-byte halves with
      the leaf bit in the index (`bvh/node.rs:9`), 32-slot function stack,
      near child first (`trace/wgsl/bvh.wgsl:55`)

- [x] **Radix sort in compute** (`bvh/radix.rs`, `bvh/sort.rs`,
      `bvh/spread.rs`) — `cub::DeviceRadixSort` translated to WGSL, so the
      build no longer comes back to the CPU to sort. Eight 8-bit digits
      over one 64-bit key `(code, index)`: packing the slot into the key
      makes every key distinct, which is what lets the scatter use
      `atomicAdd` and still be deterministic — a least-significant-digit
      sort needs stability, and an atomic cursor cannot give it. `Spread`
      then hands `Karras` the sorted codes where they lie. Proven two ways
      without a GPU: `sort.rs`'s test models the passes in Rust and
      asserts the order equals `Bvh::sort_codes`'s value for value
      (including a thousand codes over 97 values, ~10 collisions each),
      and `tests/wgsl.rs` type-checks every kernel through naga.
- [x] **A kernel's WGSL is checked without an adapter** (`tests/wgsl.rs`).
      Until now a typo in a kernel surfaced as a pipeline error on a
      machine with a GPU; naga — the front end wgpu itself hands WGSL to —
      validates all nine BVH kernels in a unit test.

> Format: status table, `- [ ]` open, `- [x]` done. Move lines, don't delete them.
