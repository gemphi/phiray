// The list of pixels Adaptive Sampling has NOT settled, for the next
// sample to trace — Cycles' active-path compaction
// (`integrator/path_state.h`'s queues), in its simplest form.
//
// Without it a settled pixel still costs its slot in a warp: the tracer
// returns at once for it (`adapt_settled`), but the 31 other threads of
// its warp are still walking the tree, so a frame half settled ran at
// nearly the price of one not settled at all. With it the next sample's
// threads take only pixels that still need samples, packed together.
//
// Two entry points, two dispatches, because a dispatch cannot order its
// own threads: `gather_clear` zeroes the count, `gather` appends. Each
// workgroup reserves one run of the list for its own pixels, so a run is
// 256 neighbouring pixels at most, in the order they lie — the next
// sample's warps keep their rays close together.
//
// Words (`Tracer::PX_WORDS`): the list at `32n + k`, its length at `34n`.
// The image is bound as atomics here, because the count is one; every
// other word is read and written through `atomicLoad` / `atomicStore`.
@group(0) @binding(0) var<uniform> p: Camera;
@group(0) @binding(1) var<storage, read_write> px: array<atomic<u32>>;

var<workgroup> held: atomic<u32>;
var<workgroup> run: u32;

@compute @workgroup_size(256)
fn gather_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (item(gid, p.groups_x) == 0u) { atomicStore(&px[34u * p.n], 0u); }
}

@compute @workgroup_size(256)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if (lane == 0u) { atomicStore(&held, 0u); }
    workgroupBarrier();
    let i = item(gid, p.groups_x);
    // Settled after the dilation (`adapt_seen`): not traced again.
    let open = i < p.n && p.adapt.w != 0.0 && atomicLoad(&px[30u * p.n + i]) == 0u;
    var at = 0u;
    if (open) { at = atomicAdd(&held, 1u); }
    workgroupBarrier();
    if (lane == 0u) { run = atomicAdd(&px[34u * p.n], atomicLoad(&held)); }
    workgroupBarrier();
    if (open) { atomicStore(&px[32u * p.n + run + at], i); }
}
