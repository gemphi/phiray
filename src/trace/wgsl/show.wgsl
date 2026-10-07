// The display step on its own: the tail of `util.wgsl`'s `emit`, over the
// linear sums already in the image buffer, with nothing traced.
//
// This exists so a composited image can be shown without a second
// implementation of the view transform. Blender's compositor runs on the
// render's **scene-linear** pixels and the view transform is applied after
// it (`RE_RenderFrame`, then the display pipeline), so a host that
// composites has to hand the linear result back and ask for it to be shown
// — `Tracer::show`. Compositing the displayed pixels instead would put the
// grade after AgX, which is not the grade Blender applied before it, and is
// wrong in a way that looks plausible.
//
// `emit` and this kernel must stay the same arithmetic; it is written here
// as its own copy only because WGSL has no way to share a function between
// two shader modules that do not include the same file. `util.wgsl` is not
// included here on purpose: it brings the whole tracer's prelude with it.

@group(0) @binding(0) var<uniform> p: Camera;
@group(0) @binding(1) var<storage, read_write> px: array<u32>;

fn three(a: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(px[a]), bitcast<f32>(px[a + 1u]), bitcast<f32>(px[a + 2u]));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    // `p.frame` is the sample the sums hold, so the mean is over `frame + 1`
    // — exactly as `emit` divides.
    let k = f32(p.frame + 1u);
    let l = three(p.n + 3u * i);
    let gs = three(20u * p.n + 3u * i);
    let cov = bitcast<f32>(px[23u * p.n + i]);
    let film = bitcast<f32>(px[24u * p.n + i]);
    let shown = view_transform_of(l / k, p.view) * (1.0 - cov / k) + gs / k;
    let b = 4u * p.n + 3u * i;
    px[b] = bitcast<u32>(shown.x);
    px[b + 1u] = bitcast<u32>(shown.y);
    px[b + 2u] = bitcast<u32>(shown.z);
    px[i] = pack(shown, film / k);
}
