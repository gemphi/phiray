// What the visibility prepass (`raster.wgsl`) saw, for Solid and
// Wireframe: binding 9 after `binds2.wgsl`'s image. Four words a pixel —
// face, instance, barycentric u and v (`raster.wgsl`) — in rows `p.vis.y` pixels apart
// (the copy out of the render target pads its rows); `p.vis.x` says there
// is one to read, else the primary ray is traced as before (X-ray, tiles).
@group(0) @binding(9) var<storage, read> vis: array<u32>;

fn primary(o: vec3<f32>, d: vec3<f32>, i: u32) -> Hit {
    if (p.vis.x == 0u) { return trace(o, d, INF, false); }
    let at = ((i / p.w) * p.vis.y + i % p.w) * 4u;
    var h = Hit(INF, vis[at], bitcast<f32>(vis[at + 2u]), bitcast<f32>(vis[at + 3u]), vis[at + 1u]);
    if (h.tri == 0xffffffffu) { return h; }
    // How far along the ray the face it named sits: the prepass spends its
    // fourth word on the barycentric, so the distance is worked out from
    // the face, placed by its instance. `o + d · t = hit` inverts to this
    // whatever the ray's length, which is the `t` the traversal returns.
    let f = tri_at(h);
    let hit = f.a * (1.0 - h.u - h.v) + f.b * h.u + f.c * h.v;
    h.t = dot(hit - o, d) / max(dot(d, d), 1e-20);
    return h;
}
