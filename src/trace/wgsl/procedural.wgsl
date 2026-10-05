// Procedural textures per pixel, Cycles' SVM (`noise.wgsl`,
// `voronoi.wgsl`, `pattern.wgsl`; `displace.h`'s Bump here): a material
// carries two procedural slots (`Mat::color_proc`, `bump_proc`; 40 words
// each, `PROC_WORDS`) — a texture node, its coordinates (Generated, Object
// or UV through a Mapping node's location and scale), an optional colour
// ramp baked to a 256 × 1 atlas image, and what it drives: the base colour
// or the mix weight for the first slot, the normal through a Bump node
// for the second.
//
// The slot's words (`Proc` in `mat.rs`): 0 kind, 1 coords, 2 ramp,
// 3 target, 4–6 location, 7–9 scale, 10 Scale, 11 Detail, 12 Roughness,
// 13 Lacunarity, 14 Distortion / Randomness, 15 Strength, 16 variant
// (noise type, Voronoi feature, gradient type, wave type), 17 direction
// (Voronoi metric, bands direction, brick offset frequency), 18 rings
// (rings direction, brick squash frequency, magic depth), 19 profile,
// 20 Offset (brick offset amount, wave Detail Scale), 21 Gain (brick
// squash amount, wave Detail Roughness), 22 Smoothness (wave Phase
// Offset, brick Mortar Size), 23 Exponent (brick Mortar Smooth), 24 Bias,
// 25 Brick Width, 26 Row Height, 28–30 Color1, 31–33 Color2, 34–36 Mortar.

const PROC_WORDS: u32 = 40u;
// `Proc::kind`'s low byte.
const PROC_NONE: u32 = 0u;
const PROC_NOISE_FAC: u32 = 1u;
const PROC_NOISE_COLOR: u32 = 2u;
const PROC_VORONOI_DISTANCE: u32 = 3u;
const PROC_VORONOI_COLOR: u32 = 4u;
const PROC_VORONOI_RADIUS: u32 = 5u;
const PROC_CHECKER_FAC: u32 = 6u;
const PROC_CHECKER_COLOR: u32 = 7u;
const PROC_BRICK_FAC: u32 = 8u;
const PROC_BRICK_COLOR: u32 = 9u;
const PROC_GRADIENT: u32 = 10u;
const PROC_MAGIC_FAC: u32 = 11u;
const PROC_MAGIC_COLOR: u32 = 12u;
const PROC_WAVE: u32 = 13u;
// `Proc::kind`'s flag: Normalize on (a Voronoi, or a Noise whose type
// has one).
const PROC_NORMALIZE: u32 = 256u;
// `Proc::target`.
const PROC_BASE: u32 = 1u;
const PROC_MIX: u32 = 2u;
// Blender's Bump node Filter Width default.
const BUMP_FILTER_WIDTH: f32 = 0.1;

// ---- The slot (`Mat::color_proc` at `m + 32`, `bump_proc` at `m + 72`) ----

fn proc_kind(w: u32) -> u32 { return data[w] & 0xffu; }

fn proc_color(w: u32) -> vec3<f32> { return vec3<f32>(tf(w), tf(w + 1u), tf(w + 2u)); }

// The texture's coordinate for a hit: the object-space point `pobj`
// (Generated normalizes it by the object's bounding box, `inst_at + 32`,
// as `ATTR_STD_GENERATED` does; Object takes it as is; UV the face's),
// through the Mapping node's location and scale.
fn proc_coord(w: u32, pobj: vec3<f32>, uv: vec2<f32>, inst: u32) -> vec3<f32> {
    var co = pobj;
    if (data[w + 1u] == 0u) {
        let lo = vec3<f32>(tf(inst + 32u), tf(inst + 33u), tf(inst + 34u));
        let size = vec3<f32>(tf(inst + 35u), tf(inst + 36u), tf(inst + 37u));
        co = (pobj - lo) / max(size, vec3<f32>(1e-6));
    } else if (data[w + 1u] == 2u) {
        co = vec3<f32>(uv, 0.0);
    }
    let location = vec3<f32>(tf(w + 4u), tf(w + 5u), tf(w + 6u));
    let scale = vec3<f32>(tf(w + 7u), tf(w + 8u), tf(w + 9u));
    return co * scale + location;
}

// The texture at `co`: `.w` its scalar output, `.rgb` its colour output.
fn proc_tex(w: u32, co0: vec3<f32>) -> vec4<f32> {
    let kind = proc_kind(w);
    let normalize = (data[w] & PROC_NORMALIZE) != 0u;
    let co = co0 * tf(w + 10u);
    let detail = tf(w + 11u);
    let roughness = tf(w + 12u);
    let lacunarity = tf(w + 13u);
    let distortion = tf(w + 14u);
    let variant = data[w + 16u];
    switch kind {
        case PROC_NOISE_FAC, PROC_NOISE_COLOR: {
            return noise_texture_3d(co, detail, roughness, lacunarity, tf(w + 20u), tf(w + 21u), distortion, variant, normalize);
        }
        case PROC_VORONOI_DISTANCE, PROC_VORONOI_COLOR, PROC_VORONOI_RADIUS: {
            return voronoi_texture_3d(co, variant, data[w + 17u], detail, roughness, lacunarity, tf(w + 22u), tf(w + 23u), distortion, normalize);
        }
        case PROC_CHECKER_FAC, PROC_CHECKER_COLOR: {
            return checker_texture(co, proc_color(w + 28u), proc_color(w + 31u));
        }
        case PROC_BRICK_FAC, PROC_BRICK_COLOR: {
            return brick_texture(co, proc_color(w + 28u), proc_color(w + 31u), proc_color(w + 34u), tf(w + 22u), tf(w + 23u), tf(w + 24u), tf(w + 25u), tf(w + 26u), tf(w + 20u), i32(data[w + 17u]), tf(w + 21u), i32(data[w + 18u]));
        }
        case PROC_GRADIENT: {
            // The Gradient node has no Scale.
            let f = gradient_fac(co0, variant);
            return vec4<f32>(vec3<f32>(f), f);
        }
        case PROC_MAGIC_FAC, PROC_MAGIC_COLOR: {
            return magic_texture(co0, tf(w + 10u), i32(data[w + 18u]), distortion);
        }
        case PROC_WAVE: {
            let f = wave_fac(variant, data[w + 17u], data[w + 18u], data[w + 19u], co, distortion, detail, tf(w + 20u), tf(w + 21u), tf(w + 22u));
            return vec4<f32>(vec3<f32>(f), f);
        }
        default: { return vec4<f32>(0.5, 0.5, 0.5, 0.5); }
    }
}

// The slot's value for a hit: the scalar or colour output, through the
// ramp when there is one (`.rgb` the colour, `.w` the scalar — a ramp's
// output as a float is its luminance, as Blender converts).
fn proc_value(w: u32, pobj: vec3<f32>, uv: vec2<f32>, inst: u32) -> vec4<f32> {
    let kind = proc_kind(w);
    let t = proc_tex(w, proc_coord(w, pobj, uv, inst));
    var scalar = t.w;
    var color = vec3<f32>(t.w);
    if (kind == PROC_NOISE_COLOR || kind == PROC_VORONOI_COLOR || kind == PROC_CHECKER_COLOR || kind == PROC_BRICK_COLOR || kind == PROC_MAGIC_COLOR) {
        color = t.rgb;
        scalar = dot(t.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    let ramp = data[w + 2u];
    if (ramp != NO_TEX) {
        let u = clamp(scalar, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0;
        let c = sample_raw(ramp, vec2<f32>(u, 0.5)).rgb;
        return vec4<f32>(c, dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)));
    }
    return vec4<f32>(color, scalar);
}

// `svm_node_set_bump`: the normal perturbed by the height's change across
// the pixel's footprint — `dP.dx`, `dP.dy` two tangents `foot` long —
// with the node's Strength and Distance, Blender's filter width.
fn bump_normal(w: u32, n: vec3<f32>, pobj: vec3<f32>, uv: vec2<f32>, inst: u32, foot: f32) -> vec3<f32> {
    let strength = max(tf(w + 15u), 0.0);
    let dist = tf(w + 3u);
    let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let dx = normalize(cross(n, axis)) * max(foot, 1e-5);
    let dy = cross(n, normalize(dx)) * max(foot, 1e-5);
    let h_c = proc_value(w, pobj, uv, inst).w;
    let h_x = proc_value(w, pobj + dx * BUMP_FILTER_WIDTH, uv, inst).w;
    let h_y = proc_value(w, pobj + dy * BUMP_FILTER_WIDTH, uv, inst).w;
    let rx = cross(dy, n);
    let ry = cross(n, dx);
    let det = dot(dx, rx);
    let surfgrad = (h_x - h_c) * rx + (h_y - h_c) * ry;
    let out = BUMP_FILTER_WIDTH * abs(det) * n - dist * sign(det) * surfgrad;
    if (dot(out, out) < 1e-24) { return n; }
    return normalize(strength * normalize(out) + (1.0 - strength) * n);
}

// A hit's object-space point: the instance's inverse over the world point.
fn object_point(inst: u32, hit: vec3<f32>) -> vec3<f32> {
    if ((data[inst + 27u] & 1u) != 0u) { return hit; }
    let h = vec4<f32>(hit, 1.0);
    return vec3<f32>(dot(inst_row(inst + 12u), h), dot(inst_row(inst + 16u), h), dot(inst_row(inst + 20u), h));
}

// The weight of a material's second lobe at a hit: the mix slot's value
// when the first slot drives the mix, else the constant.
fn mix_weight(m: u32, pobj: vec3<f32>, uv: vec2<f32>, inst: u32) -> f32 {
    let w = m + 32u;
    if (proc_kind(w) != PROC_NONE && (data[w + 3u] & PROC_MIX) != 0u) {
        return clamp(proc_value(w, pobj, uv, inst).w, 0.0, 1.0);
    }
    return mat_mix(m);
}

// The base colour a material's first slot gives a hit, or `base` as it is.
fn proc_base(m: u32, base: vec3<f32>, pobj: vec3<f32>, uv: vec2<f32>, inst: u32) -> vec3<f32> {
    let w = m + 32u;
    if (proc_kind(w) != PROC_NONE && (data[w + 3u] & PROC_BASE) != 0u) {
        return max(proc_value(w, pobj, uv, inst).rgb, vec3<f32>(0.0));
    }
    return base;
}

// The normal a material's bump slot gives a hit, or `n` as it is.
fn proc_normal(m: u32, n: vec3<f32>, pobj: vec3<f32>, uv: vec2<f32>, inst: u32, foot: f32) -> vec3<f32> {
    let w = m + 72u;
    if (proc_kind(w) == PROC_NONE) { return n; }
    return bump_normal(w, n, pobj, uv, inst, foot);
}
