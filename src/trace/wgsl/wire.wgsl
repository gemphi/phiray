// Wireframe shading (`V3D_SHADING_WIREFRAME`): the mesh's edges, depth-tested.
//
// One primary ray per pixel, as in Solid. Where it hits, the distance from the
// hit point to each of the triangle's three edges decides whether this pixel
// is on a wire: for barycentrics (a, b, c) the perpendicular distance to the
// edge opposite each vertex is that vertex's weight times the triangle's
// height on that edge, `2 * area / |edge|`. Comparing that against the pixel's
// world footprint gives a line of constant screen width at any depth.
//
// With X-ray off it is a real ray cast, so the wire is hidden-surface correct: an edge
// behind a face does not draw. The CPU stroke path it replaces had no depth
// test at all, so every edge in the scene drew over every other.
//
// In this mode `p.env` carries the wire colour — there is no environment to
// reflect in a wireframe.
//
// X-ray (`show_xray_wireframe`, on in a new file with alpha 0) sees through:
// every surface along the ray gives its wire, and those behind the nearest
// are faded to `1 − xray_alpha` (`overlay_xray_fade`), as the floor grid is.

// The most surfaces X-ray sees through along one ray.
const WIRE_LAYERS: u32 = 32u;

// How much of this pixel the wire of the surface `h` covers, at `t` along
// the ray.
fn wire_at(h: Hit, t_at: f32, pix: f32) -> f32 {
    let t = tri_at(h);
    let bary = vec3<f32>(1.0 - h.u - h.v, h.u, h.v);

    // Twice the triangle's area, and the length of the edge opposite each
    // vertex: BC, CA, AB.
    let area2 = length(cross(t.b - t.a, t.c - t.a));
    let e = vec3<f32>(length(t.c - t.b), length(t.a - t.c), length(t.b - t.a));
    let heights = area2 / max(e, vec3<f32>(1e-12));
    let dist = min(min(bary.x * heights.x, bary.y * heights.y), bary.z * heights.z);

    // The pixel's footprint where the ray landed: constant under an
    // orthographic view, growing with distance under a perspective one.
    var foot = pix;
    if (p.persp != 0u) { foot = pix * t_at * p.units / p.dist; }
    // Half a line width, plus half a pixel of feather so the wire does not
    // crawl as the view turns.
    let half = foot * 0.75;
    return 1.0 - smoothstep(half, half + foot * 0.5, dist);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = pixel(gid);
    if (i >= p.n) { return; }
    let sp = frame_point(i) + taa_jitter();
    let su = sp.x;
    let sv = sp.y;
    let vx = (su - p.pan.x) / p.scale;
    let vy = -(sv / p.aspect - p.pan.y) / p.scale;
    var o: vec3<f32>;
    var d: vec3<f32>;
    if (p.persp != 0u) {
        o = p.pivot + p.back * (p.dist / p.units);
        d = normalize(p.right * vx + p.up * vy - p.back * p.dist);
    } else {
        o = p.pivot + (p.right * vx + p.up * vy + p.back * 1.0e4) / p.units;
        d = -p.back;
    }
    let pix = 1.0 / (p.scale * p.units * frame_size().x);
    // The theme's sky, with its sweep when it has one (`sky_at`).
    let sky = sky_at(i);

    if (p.see_through > 0.0) {
        // Every surface's wire, the nearest whole and the rest faded: what
        // the pixel keeps of the background under all of them.
        var keep = 1.0;
        var near = INF;
        var ahead = 0.0;
        var last = 0xffffffffu;
        var last_inst = 0u;
        for (var k = 0u; k < WIRE_LAYERS; k++) {
            let h = trace(o + d * ahead, d, INF, false);
            if (h.tri == 0xffffffffu) { break; }
            let t = ahead + h.t;
            ahead = t + max(t, 1.0) * 1e-5;
            if (h.tri == last && h.inst == last_inst) { continue; }
            last = h.tri;
            last_inst = h.inst;
            let fade = select(p.see_through, 1.0, near >= INF);
            near = min(near, t);
            keep *= 1.0 - wire_at(h, t, pix) * fade;
        }
        var g = grid_over(o, d, INF, pix);
        if (abs(d.z) > 1e-6 && -o.z / d.z > near) { g *= p.see_through; }
        outline_store(i, 0u);
        taa_sample(i, mix(sky * (1.0 - g.a) + g.rgb, p.env.rgb, 1.0 - keep));
        return;
    }
    let h = primary(o, d, i);
    if (h.tri == 0xffffffffu) {
        outline_store(i, 0u);
        taa_sample(i, floor_grid(o, d, INF, pix, sky));
        return;
    }
    outline_store(i, h.inst + 1u);
    let behind = floor_grid(o, d, h.t, pix, sky);
    taa_sample(i, mix(behind, p.env.rgb, wire_at(h, h.t, pix)));
}
