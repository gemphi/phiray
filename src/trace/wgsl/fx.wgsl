// Workbench's Solid effects over the opaque pass: the composite's cavity
// effect — screen-space ambient occlusion over a jittered disk of
// neighbours (`workbench_cavity.bsl.hh`, `cavity_compute`) — and its
// curvature effect, one pixel against its four neighbours
// (`workbench_curvature.bsl.hh`, `curvature_compute`), then the object
// outline drawn after them (`workbench_effect_outline_frag.glsl`), all
// read off the object-id, normal and depth buffers `fx_store` left
// (`solid.wgsl`). The colour is then put through the view transform and
// the floor grid laid over it — the sample `taa.wgsl` gathers. Nothing
// here for X-ray (Workbench's transparent pass has none) or the Wireframe.

@group(0) @binding(0) var<uniform> p: Camera;
@group(0) @binding(1) var<storage, read_write> px: array<u32>;
// EEVEE's blue-noise jitter, `[cos φ, sin φ, bn₁]` a texel (`blue_noise.rs`).
@group(0) @binding(2) var<storage, read> noise: array<u32>;

const TAU: f32 = 6.2831853;
// `CavityEffect::max_samples_`.
const MAX_SAMPLES: u32 = 512u;

// Pixel `(x, y)` clamped to the image, as a texture fetch at its edge is.
fn at(x: i32, y: i32) -> u32 {
    return u32(clamp(y, 0, i32(p.h) - 1)) * p.w + u32(clamp(x, 0, i32(p.w) - 1));
}
fn id_at(j: u32) -> u32 { return px[17u * p.n + j]; }
fn normal_at(j: u32) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(px[18u * p.n + j]), bitcast<f32>(px[19u * p.n + j]));
}
fn z_at(j: u32) -> f32 { return bitcast<f32>(px[20u * p.n + j]); }
fn three(a: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(px[a]), bitcast<f32>(px[a + 1u]), bitcast<f32>(px[a + 2u]));
}

// The whole frame's size and this tile's place in it (`util.wgsl`'s).
fn frame_size() -> vec2<f32> {
    if (p.tile.z != 0u) { return vec2<f32>(f32(p.tile.z), f32(p.tile.w)); }
    return vec2<f32>(f32(p.w), f32(p.h));
}

// `workbench_normal_decode` of what the opaque pass stored: the view
// normal's x and y, z toward the viewer.
fn normal3_at(j: u32) -> vec3<f32> {
    let n = normal_at(j);
    return vec3<f32>(n, sqrt(max(1.0 - dot(n, n), 0.0)));
}

// `drw_point_screen_to_view`: the view-space point (Blender units) at
// screen position `c` (pixels across this tile, the pixel centre at
// `+0.5`) and view depth `z` — the canvas point `solid.wgsl`'s `main`
// shoots through, scaled out to the depth under perspective.
fn view_pos(c: vec2<f32>, z: f32) -> vec3<f32> {
    let sp = (vec2<f32>(f32(p.tile.x), f32(p.tile.y)) + c) / frame_size() - 0.5;
    var v = vec2<f32>((sp.x - p.pan.x) / p.scale, -(sp.y / p.aspect - p.pan.y) / p.scale);
    if (p.persp != 0u) { v *= -z / p.dist; }
    return vec3<f32>(v, z);
}

// `BLI_hammersley_1d`: the radical inverse of `n` in base 2.
fn radical_inverse(n: u32) -> f32 {
    return f32(reverseBits(n)) * 2.3283064365386963e-10;
}

// `cavity_compute`: Alchemy screen-space ambient obscurance over
// `ssao_samples` points of the Hammersley disk (`load_samples_buf`) —
// this sample's run of them (`CavityEffect::init`) — each turned and slid
// by the pixel's blue noise (`load_jitter_tx`), a `cavity_distance` wide
// at the pixel's depth. Returns `(cavities, edges)`, the valley and ridge
// factors applied.
fn cavity(i: u32, x: i32, y: i32) -> vec2<f32> {
    let ssao_samples = p.ssao.x;
    let sample_count = min(p.ssao.y * ssao_samples, MAX_SAMPLES);
    let max_iter = max(sample_count / ssao_samples, 1u);
    let run = p.frame % max_iter;
    let start = ssao_samples * run;
    let end = ssao_samples * (run + 1u);
    let count_inv = 1.0 / f32(end - start);

    let z = z_at(i);
    let centre = vec2<f32>(f32(x), f32(y)) + 0.5;
    let position = view_pos(centre, z);
    let normal = normal3_at(i);

    // The jitter texel: the pixel's place in the frame, GL's rows up from
    // the bottom, repeated every 64.
    let size = frame_size();
    let jx = (p.tile.x + u32(x)) % 64u;
    let jy = (u32(size.y) - 1u - (p.tile.y + u32(y))) % 64u;
    let t = 3u * (64u * jy + jx);
    let rot_x = vec2<f32>(bitcast<f32>(noise[t]), bitcast<f32>(noise[t + 1u]));
    let rot_y = vec2<f32>(-rot_x.y, rot_x.x);
    let slide = clamp(bitcast<f32>(noise[t + 2u]) - 0.5, -0.499, 0.499) / f32(sample_count);

    // `cavity_distance` across the screen at this depth: the frame's
    // view-space span is `1 / scale` across and `1 / (scale · aspect)` up,
    // out to the depth under perspective (Blender's `winmat` over
    // `homcoord`), in pixels of the frame.
    var span = vec2<f32>(1.0 / p.scale, 1.0 / (p.scale * p.aspect));
    if (p.persp != 0u) { span *= -z / p.dist; }
    let offset = p.cavity.w / span * size;

    var cavities = 0.0;
    var edges = 0.0;
    for (var k = start; k < end && k < MAX_SAMPLES; k++) {
        // The disk sample: its direction and its distance from the centre
        // (more of them near it, so near the shadow).
        let it_add = f32(k / ssao_samples) * 0.499;
        let r = fract((f32(k) + 0.5 + it_add) / f32(ssao_samples));
        let phi = radical_inverse(k) * TAU + it_add;
        let dir = vec2<f32>(cos(phi), sin(phi));
        var jittered = vec2<f32>(dot(dir, rot_x), dot(dir, rot_y));
        jittered *= r + slide;
        // Screen up is the row above.
        let c = centre + vec2<f32>(jittered.x * offset.x, -jittered.y * offset.y);
        if (c.x < 0.0 || c.y < 0.0 || c.x >= f32(p.w) || c.y >= f32(p.h)) { continue; }
        let j = u32(c.y) * p.w + u32(c.x);
        let background = id_at(j) == 0u;
        // The background takes the centre's depth, a distance further.
        var s_pos = view_pos(c, select(z_at(j), z, background));
        if (background) { s_pos.z -= p.cavity.w; }

        let d = s_pos - position;
        let len = max(length(d), 1e-6);
        let f_cavities = dot(d, normal);
        let f_edge = -f_cavities;
        let f_bias = 0.05 * len + 0.0001;
        let attenuation = 1.0 / (len * (1.0 + len * len * p.cavity.z));
        if (f_cavities > -f_bias) { cavities += f_cavities * attenuation; }
        if (f_edge > f_bias) { edges += f_edge * attenuation; }
    }
    cavities *= count_inv;
    edges *= count_inv;
    // Don't let the cavity wash out the surface.
    return vec2<f32>(clamp(cavities * p.cavity.x, 0.0, 1.0), edges * p.cavity.y);
}

// `curvature_soft_clamp`.
fn soft_clamp(curvature: f32, control: f32) -> f32 {
    if (curvature < 0.5 / control) { return curvature * (1.0 - curvature * control); }
    return 0.25 / control;
}

// `curvature_compute`: nothing across an object's edge or over the
// background; else the normals' slope, ridges brightened and valleys
// darkened, each soft-clamped by its factor.
fn curvature(up: u32, down: u32, right: u32, left: u32) -> f32 {
    if (id_at(up) != id_at(down) || id_at(right) != id_at(left)) { return 0.0; }
    if (id_at(up) == id_at(right) && id_at(right) == 0u) { return 0.0; }
    let diff = (normal_at(up).y - normal_at(down).y) + (normal_at(right).x - normal_at(left).x);
    if (diff < 0.0) { return -2.0 * soft_clamp(-diff, p.fx.z); }
    return 2.0 * soft_clamp(diff, p.fx.y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let x = i32(i % p.w);
    let y = i32(i / p.w);
    // `ui_scale` pixels away, `U.pixelsize`; screen up is the row above.
    let o = i32(max(round(p.fx.w), 1.0));
    let up = at(x, y - o);
    let down = at(x, y + o);
    let right = at(x + o, y);
    let left = at(x - o, y);
    var lin = three(7u * p.n + 3u * i);
    let id = id_at(i);
    if (id != 0u && (p.ssao.x > 0u || p.fx.y > 0.0 || p.fx.z > 0.0)) {
        // The composite: `(1 − cavity) (1 + edges) (1 + curvature)`, clamped.
        var ce = vec2<f32>(0.0);
        if (p.ssao.x > 0u) { ce = cavity(i, x, y); }
        var curv = 0.0;
        if (p.fx.y > 0.0 || p.fx.z > 0.0) { curv = curvature(up, down, right, left); }
        lin *= clamp((1.0 - ce.x) * (1.0 + ce.y) * (1.0 + curv), 0.0, 4.0);
    }
    if (p.fx.x > 0.0) {
        // `outline_opacity`: a quarter per neighbour of another object,
        // `object_outline_color` blended over, premultiplied.
        let same = vec4<f32>(f32(id_at(up) == id), f32(id_at(down) == id), f32(id_at(right) == id), f32(id_at(left) == id));
        let opacity = 1.0 - dot(same, vec4<f32>(0.25));
        lin = lin * (1.0 - opacity) + p.outline.rgb * opacity;
    }
    let g = three(10u * p.n + 3u * i);
    let cover = bitcast<f32>(px[13u * p.n + i]);
    let shown = srgb(lin) * (1.0 - cover) + g;
    let a = 14u * p.n + 3u * i;
    px[a] = bitcast<u32>(shown.x);
    px[a + 1u] = bitcast<u32>(shown.y);
    px[a + 2u] = bitcast<u32>(shown.z);
}
