// What the visibility prepass (`raster.wgsl`) saw, for Solid and
// Wireframe: binding 9 after `binds2.wgsl`'s image. Four words a pixel —
// face, barycentric u and v, distance — in rows `p.vis.y` pixels apart
// (the copy out of the render target pads its rows); `p.vis.x` says there
// is one to read, else the primary ray is traced as before (X-ray, tiles).
@group(0) @binding(9) var<storage, read> vis: array<u32>;

fn primary(o: vec3<f32>, d: vec3<f32>, i: u32) -> Hit {
    if (p.vis.x == 0u) { return trace(o, d, INF, false); }
    let at = ((i / p.w) * p.vis.y + i % p.w) * 4u;
    return Hit(bitcast<f32>(vis[at + 3u]), vis[at], bitcast<f32>(vis[at + 1u]), bitcast<f32>(vis[at + 2u]));
}
