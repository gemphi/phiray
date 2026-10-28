// The visibility prepass: Workbench's way of finding what each pixel sees
// (`workbench_prepass_vert.glsl` + the depth test), rasterized, not traced.
// Every face of the scene goes through the raster pipeline as three
// vertices pulled from the tracer's own buffers (`face`, `vert`, the
// instance table), projected exactly as `solid.wgsl` casts its rays — so
// the hit it writes per pixel is the one the ray would have found — and
// the depth test keeps the nearest. Each mesh is one instanced draw over
// the instances that show it (`Trees::draws`): `instance_index` names the
// draw slot, `draw_inst` the instance placing this copy. The fragment
// writes the face, the instance and its two barycentrics, each a whole
// word: the instance took the word the distance along the ray used to
// have, so `vis2.wgsl` works that out again from the face it names.
// Rounding the barycentrics into half a word instead cost Wireframe a
// least significant bit on a few dozen pixels of a tiled frame, which
// `tiles_laid_together_are_the_whole_frame` compares byte for byte. Depth is reversed (`p.vis.w`'s near plane over the distance,
// greater is nearer), for float precision far from the eye.

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) tri: u32,
    @location(1) bary: vec3<f32>,
    @location(2) @interpolate(flat) inst: u32,
}

// Corner `k` of face `f`, in the world, placed by instance `i`.
fn corner_at(f: u32, k: u32, i: u32) -> vec3<f32> {
    let v = face(f);
    let id = select(select(v.x, v.y, k == 1u), v.z, k == 2u);
    let ph = vert(id).p;
    let w = inst_at(i);
    if ((data[w + 27u] & 1u) != 0u) { return ph.xyz; }
    return vec3<f32>(dot(inst_row(w), ph), dot(inst_row(w + 4u), ph), dot(inst_row(w + 8u), ph));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let f = vi / 3u;
    let k = vi % 3u;
    let inst = draw_inst(ii);
    let P = corner_at(f, k, inst);
    // Camera space: `right`, `up`, `back` about the pivot.
    let c = P - p.pivot;
    let cx = dot(c, p.right);
    let cy = dot(c, p.up);
    let cz = dot(c, p.back);
    var clip: vec4<f32>;
    if (p.persp != 0u) {
        // The eye sits `dist / units` along `back`; `wv` is the depth along
        // the view axis. The ray through canvas point `(vx, vy)` meets this
        // point where `vx = cx · dist / wv` (`solid.wgsl`: `d = right·vx +
        // up·vy − back·dist`), and the canvas point is `2 (vx·scale +
        // pan.x)` across, `2 · aspect (vy·scale − pan.y)` up.
        let wv = p.dist / p.units - cz;
        clip = vec4<f32>(2.0 * (cx * p.dist * p.scale + p.pan.x * wv), 2.0 * p.aspect * (cy * p.dist * p.scale - p.pan.y * wv), bitcast<f32>(p.vis.w), wv);
    } else {
        // Orthographic: the ray origin is offset `(vx, vy) / units` from the
        // pivot, so `vx = cx · units`; depth spans the scene's reach either
        // side of the pivot (`p.vis.z`), nearer (toward `back`) greater.
        let reach = bitcast<f32>(p.vis.z);
        clip = vec4<f32>(2.0 * (cx * p.units * p.scale + p.pan.x), 2.0 * p.aspect * (cy * p.units * p.scale - p.pan.y), (cz + reach) / (2.0 * reach), 1.0);
    }
    // The sample's jitter (`taa_jitter` casts the rays the same way).
    clip.x += p.taa.x / f32(p.w) * clip.w;
    clip.y += p.taa.y / f32(p.h) * clip.w;
    var out: VOut;
    out.clip = clip;
    out.tri = f;
    out.bary = vec3<f32>(f32(k == 0u), f32(k == 1u), f32(k == 2u));
    out.inst = inst;
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<u32> {
    return vec4<u32>(in.tri, in.inst, bitcast<u32>(in.bary.y), bitcast<u32>(in.bary.z));
}
