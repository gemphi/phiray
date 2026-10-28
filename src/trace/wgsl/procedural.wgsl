// Procedural textures per pixel, Cycles' SVM (`noise.wgsl`,
// `voronoi.wgsl`, `pattern.wgsl`; `displace.h`'s Bump here): a material
// carries two procedural slots (`Mat::color_proc`, `bump_proc`; 40 words
// each, `PROC_WORDS`) — a texture node, its coordinates (Generated, Object
// or UV through a Mapping node, whole: location, rotation, scale and
// type), an optional colour
// ramp baked to a 256 × 1 atlas image, and what it drives: the base colour
// or the mix weight for the first slot, the normal through a Bump node
// for the second.
//
// The slot's words (`Proc` in `mat.rs`): 0 kind, 1 coords, 2 ramp,
// 3 target, 4–6 location, 7–9 scale, 10 Scale, 11 Detail, 12 Roughness,
// 13 Lacunarity, 14 Distortion / Randomness, 15 Strength, 16 variant
// (noise type, Voronoi feature, gradient type, wave type, White Noise
// dimensions, Gabor type), 17 direction
// (Voronoi metric, bands direction, brick offset frequency), 18 rings
// (rings direction, brick squash frequency, magic depth), 19 profile,
// 20 Offset (brick offset amount, wave Detail Scale), 21 Gain (brick
// squash amount, wave Detail Roughness), 22 Smoothness (wave Phase
// Offset, brick Mortar Size), 23 Exponent (brick Mortar Smooth), 24 Bias,
// 25 Brick Width, 26 Row Height, 27 Mapping type, 28–30 Color1,
// 31–33 Color2, 34–36 Mortar, 37–39 Mapping Rotation, 40 the Math
// node's operation (`MATH_NONE` for none), 41–42 its second and third
// Value sockets, 43 its Clamp.

const PROC_WORDS: u32 = 44u;
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
const PROC_WHITE_NOISE_FAC: u32 = 14u;
const PROC_WHITE_NOISE_COLOR: u32 = 15u;
const PROC_GABOR_VALUE: u32 = 16u;
const PROC_GABOR_PHASE: u32 = 17u;
const PROC_GABOR_INTENSITY: u32 = 18u;
// Fresnel ▸ Fac, Layer Weight ▸ Fresnel and ▸ Facing: the hit's angle to
// the ray (`svm_node_fresnel`, `svm_node_layer_weight`), from `proc_n`
// and `proc_wi`, which the kernel sets per hit before reading a slot.
const PROC_FRESNEL: u32 = 19u;
const PROC_LAYER_FRESNEL: u32 = 20u;
const PROC_LAYER_FACING: u32 = 21u;
// `Proc::kind`'s flag: Normalize on (a Voronoi, or a Noise whose type
// has one).
const PROC_NORMALIZE: u32 = 256u;
// `Proc::target`.
const PROC_BASE: u32 = 1u;
const PROC_MIX: u32 = 2u;
// The second colour slot (`Mat::color_proc2` at `m + 120`): the other
// colour of a Base Color Mix node. It keeps the node's blend mode where a
// first slot keeps its target, and the Factor where a bump slot keeps its
// Strength.
// Blender's Bump node Filter Width default.
const BUMP_FILTER_WIDTH: f32 = 0.1;

// The hit the view-dependent slots read: its shading normal and the
// direction back along the ray (Cycles' `sd->N`, `sd->wi`).
var<private> proc_n: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);
var<private> proc_wi: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

// `fresnel_dielectric_cos` (`closure/bsdf_util.h`): the reflectance at
// `cosi` for a relative IOR `eta`, 1 past total internal reflection.
fn fresnel_dielectric_cos(cosi: f32, eta: f32) -> f32 {
    let c = abs(cosi);
    var g = eta * eta - 1.0 + c * c;
    if (g <= 0.0) { return 1.0; }
    g = sqrt(g);
    let a = (g - c) / (g + c);
    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

// `svm_node_fresnel`: the IOR is the slot's `detail`; a backface sees it
// inverted (the kernel hands a normal already facing the ray, so the
// side is the geometric one's: `proc_n` points at the ray on both).
fn fresnel_slot(ior: f32) -> f32 {
    return fresnel_dielectric_cos(dot(proc_wi, proc_n), max(ior, 1e-5));
}

// `svm_node_layer_weight`: Fresnel at `1 / (1 − blend)`, or Facing as
// `1 − cosᵖ` with the Blend remapped to the power.
fn layer_weight_slot(blend0: f32, facing: bool) -> f32 {
    if (!facing) {
        let eta = 1.0 / max(1.0 - blend0, 1e-5);
        return fresnel_dielectric_cos(dot(proc_wi, proc_n), eta);
    }
    var f = abs(dot(proc_wi, proc_n));
    if (blend0 != 0.5) {
        let b = clamp(blend0, 0.0, 1.0 - 1e-5);
        let p = select(0.5 / (1.0 - b), 2.0 * b, b < 0.5);
        f = pow(f, p);
    }
    return 1.0 - f;
}

// ---- The slot (`Mat::color_proc` at `m + 32`, `bump_proc` at `m + 76`) ----

fn proc_kind(w: u32) -> u32 { return data[w] & 0xffu; }

fn proc_color(w: u32) -> vec3<f32> { return vec3<f32>(tf(w), tf(w + 1u), tf(w + 2u)); }

// Blender's Euler XYZ as Cycles builds it (`euler_to_transform` in
// `util/transform.h`), whose rows are that function's `t.x`, `t.y`, `t.z`
// — so these are its columns, since WGSL takes a matrix column by column.
// `rot * v` is then `transform_direction` and `transpose(rot) * v` is
// `transform_direction_transposed`, which is how `svm_mapping` spends it.
fn euler_xyz(r: vec3<f32>) -> mat3x3<f32> {
    let c = cos(r);
    let s = sin(r);
    return mat3x3<f32>(
        vec3<f32>(c.y * c.z, c.y * s.z, -s.y),
        vec3<f32>(s.y * s.x * c.z - c.x * s.z, s.y * s.x * s.z + c.x * c.z, c.y * s.x),
        vec3<f32>(s.y * c.x * c.z + s.x * s.z, s.y * c.x * s.z - s.x * c.z, c.y * c.x),
    );
}

// Cycles' `safe_divide`: zero where the divisor is.
fn safe_div(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ok = abs(b) > vec3<f32>(1e-12);
    return select(vec3<f32>(0.0), a / select(vec3<f32>(1.0), b, ok), ok);
}

// Cycles' `safe_normalize`: the vector itself when it has no length.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    return select(v, v / l, l > 1e-20);
}

// The texture's coordinate for a hit: the object-space point `pobj`
// (Generated normalizes it by the mesh's bounding box, `inst_at + 32`,
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
    let rot = euler_xyz(vec3<f32>(tf(w + 37u), tf(w + 38u), tf(w + 39u)));
    switch data[w + 27u] {
        // Texture: the inverse of Point, which is how a texture's own
        // transform reads — move, unrotate, then unscale.
        case 1u: { return safe_div(transpose(rot) * (co - location), scale); }
        // Vector: a direction, so no offset.
        case 2u: { return rot * (co * scale); }
        // Normal: scaled the other way and renormalized.
        case 3u: { return normalize_or_zero(rot * safe_div(co, scale)); }
        // Point, the default.
        default: { return rot * (co * scale) + location; }
    }
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
        case PROC_WHITE_NOISE_FAC, PROC_WHITE_NOISE_COLOR: {
            // The White Noise node has no Scale: the coordinate is hashed
            // as it arrives, W in the Offset word.
            return white_noise_texture(co0, tf(w + 20u), variant);
        }
        case PROC_FRESNEL: {
            let f = fresnel_slot(detail);
            return vec4<f32>(vec3<f32>(f), f);
        }
        case PROC_LAYER_FRESNEL, PROC_LAYER_FACING: {
            let f = layer_weight_slot(detail, kind == PROC_LAYER_FACING);
            return vec4<f32>(vec3<f32>(f), f);
        }
        case PROC_GABOR_VALUE, PROC_GABOR_PHASE, PROC_GABOR_INTENSITY: {
            // Frequency, Anisotropy and Orientation 2D ride in the Detail,
            // Roughness and Lacunarity words; Orientation 3D in Color1.
            let g = gabor_texture(co, variant, detail, roughness, lacunarity, proc_color(w + 28u));
            let f = select(select(g.x, g.y, kind == PROC_GABOR_PHASE), g.z, kind == PROC_GABOR_INTENSITY);
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
    if (kind == PROC_NOISE_COLOR || kind == PROC_VORONOI_COLOR || kind == PROC_CHECKER_COLOR || kind == PROC_BRICK_COLOR || kind == PROC_MAGIC_COLOR || kind == PROC_WHITE_NOISE_COLOR) {
        color = t.rgb;
        scalar = dot(t.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    let ramp = data[w + 2u];
    if (ramp != NO_TEX) {
        let u = clamp(scalar, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0;
        let c = sample_raw(ramp, vec2<f32>(u, 0.5)).rgb;
        color = c;
        scalar = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    // The Math node between the texture and the socket (`svm_math`). It is
    // a scalar operation, so a colour output becomes the grey it maps to —
    // which is what Blender does when a Math node feeds a colour socket.
    let math = data[w + 40u];
    if (math != MATH_NONE) {
        scalar = svm_math(math, scalar, tf(w + 41u), tf(w + 42u));
        if (data[w + 43u] != 0u) { scalar = clamp(scalar, 0.0, 1.0); }
        color = vec3<f32>(scalar);
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

// The base colour a material's slots give a hit, or `base` as it is: the
// first slot's texture, and over it — through the Mix node's own blend
// mode and factor (`mix.wgsl`'s `ramp_blend`) — the second slot's, when
// the Base Color is a Mix of two textures.
fn proc_base(m: u32, base: vec3<f32>, pobj: vec3<f32>, uv: vec2<f32>, inst: u32) -> vec3<f32> {
    let w = m + 32u;
    if (proc_kind(w) == PROC_NONE || (data[w + 3u] & PROC_BASE) == 0u) { return base; }
    let first = max(proc_value(w, pobj, uv, inst).rgb, vec3<f32>(0.0));
    let w2 = m + 120u;
    if (proc_kind(w2) == PROC_NONE) { return first; }
    let second = max(proc_value(w2, pobj, uv, inst).rgb, vec3<f32>(0.0));
    return max(ramp_blend(data[w2 + 3u], tf(w2 + 15u), first, second), vec3<f32>(0.0));
}

// The normal a material's bump slot gives a hit, or `n` as it is.
fn proc_normal(m: u32, n: vec3<f32>, pobj: vec3<f32>, uv: vec2<f32>, inst: u32, foot: f32) -> vec3<f32> {
    let w = m + 76u;
    if (proc_kind(w) == PROC_NONE) { return n; }
    return bump_normal(w, n, pobj, uv, inst, foot);
}
