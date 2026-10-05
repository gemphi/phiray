// The Voronoi Texture node, Cycles' SVM ported (`kernel/svm/voronoi.h`, the
// 3D case): the features F1, F2, Smooth F1, Distance to Edge and N-Sphere
// Radius over the Euclidean, Manhattan, Chebychev and Minkowski metrics,
// fractal over Detail / Roughness / Lacunarity, normalized when the node
// asks (`fractal_voronoi_x_fx`, `fractal_voronoi_distance_to_edge`). The
// cell hash is `hash_int3_to_float3` (`noise.wgsl`).

// `NodeTexVoronoi.feature` (`SHD_VORONOI_*`).
const VORONOI_F1: u32 = 0u;
const VORONOI_F2: u32 = 1u;
const VORONOI_SMOOTH_F1: u32 = 2u;
const VORONOI_DISTANCE_TO_EDGE: u32 = 3u;
const VORONOI_N_SPHERE_RADIUS: u32 = 4u;
// `NodeTexVoronoi.distance`.
const VORONOI_EUCLIDEAN: u32 = 0u;
const VORONOI_MANHATTAN: u32 = 1u;
const VORONOI_CHEBYCHEV: u32 = 2u;
const VORONOI_MINKOWSKI: u32 = 3u;

// `VoronoiParams`, the node's inputs after its clamps.
struct VParams {
    feature: u32,
    metric: u32,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    smoothness: f32,
    exponent: f32,
    randomness: f32,
    max_distance: f32,
    normalize: bool,
}

// `voronoi_distance` for the metric.
fn voronoi_distance(a: vec3<f32>, b: vec3<f32>, p: VParams) -> f32 {
    let d = abs(a - b);
    switch p.metric {
        case VORONOI_MANHATTAN: { return d.x + d.y + d.z; }
        case VORONOI_CHEBYCHEV: { return max(d.x, max(d.y, d.z)); }
        case VORONOI_MINKOWSKI: {
            let e = max(p.exponent, 1e-6);
            return pow(pow(d.x, e) + pow(d.y, e) + pow(d.z, e), 1.0 / e);
        }
        default: { return length(a - b); }
    }
}

// An octave's result: `.rgb` the cell's colour, `.w` the distance.
fn voronoi_f1(p: VParams, coord: vec3<f32>) -> vec4<f32> {
    let cell_f = floor(coord);
    let local = coord - cell_f;
    let cell = vec3<i32>(cell_f);
    var min_d = 3.0e38;
    var nearest = vec3<i32>(0);
    for (var k = -1; k <= 1; k++) {
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let off = vec3<i32>(i, j, k);
                let pos = vec3<f32>(off) + hash_int3_to_float3(cell + off) * p.randomness;
                let d = voronoi_distance(pos, local, p);
                if (d < min_d) { nearest = off; min_d = d; }
            }
        }
    }
    return vec4<f32>(hash_int3_to_float3(cell + nearest), min_d);
}

fn voronoi_f2(p: VParams, coord: vec3<f32>) -> vec4<f32> {
    let cell_f = floor(coord);
    let local = coord - cell_f;
    let cell = vec3<i32>(cell_f);
    var d1 = 3.0e38; var d2 = 3.0e38;
    var off1 = vec3<i32>(0); var off2 = vec3<i32>(0);
    for (var k = -1; k <= 1; k++) {
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let off = vec3<i32>(i, j, k);
                let pos = vec3<f32>(off) + hash_int3_to_float3(cell + off) * p.randomness;
                let d = voronoi_distance(pos, local, p);
                if (d < d1) { d2 = d1; d1 = d; off2 = off1; off1 = off; }
                else if (d < d2) { d2 = d; off2 = off; }
            }
        }
    }
    return vec4<f32>(hash_int3_to_float3(cell + off2), d2);
}

// `voronoi_smooth_f1`: the 5 × 5 × 5 neighbourhood blended by the
// smoothness, with Cycles' correction terms.
fn voronoi_smooth_f1(p: VParams, coord: vec3<f32>) -> vec4<f32> {
    let cell_f = floor(coord);
    let local = coord - cell_f;
    let cell = vec3<i32>(cell_f);
    var smooth_d = 0.0;
    var smooth_c = vec3<f32>(0.0);
    var h = -1.0;
    for (var k = -2; k <= 2; k++) {
        for (var j = -2; j <= 2; j++) {
            for (var i = -2; i <= 2; i++) {
                let off = vec3<i32>(i, j, k);
                let pos = vec3<f32>(off) + hash_int3_to_float3(cell + off) * p.randomness;
                let d = voronoi_distance(pos, local, p);
                h = select(smoothstep(0.0, 1.0, 0.5 + 0.5 * (smooth_d - d) / p.smoothness), 1.0, h == -1.0);
                var correction = p.smoothness * h * (1.0 - h);
                smooth_d = mix(smooth_d, d, h) - correction;
                correction /= 1.0 + 3.0 * p.smoothness;
                smooth_c = mix(smooth_c, hash_int3_to_float3(cell + off), h) - correction;
            }
        }
    }
    return vec4<f32>(smooth_c, smooth_d);
}

// `voronoi_distance_to_edge`: the closest point, then the nearest of the
// perpendicular bisectors toward the others.
fn voronoi_distance_to_edge(p: VParams, coord: vec3<f32>) -> f32 {
    let cell_f = floor(coord);
    let local = coord - cell_f;
    let cell = vec3<i32>(cell_f);
    var to_closest = vec3<f32>(0.0);
    var min_d = 3.0e38;
    for (var k = -1; k <= 1; k++) {
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let off = vec3<i32>(i, j, k);
                let v = vec3<f32>(off) + hash_int3_to_float3(cell + off) * p.randomness - local;
                let d = dot(v, v);
                if (d < min_d) { min_d = d; to_closest = v; }
            }
        }
    }
    min_d = 3.0e38;
    for (var k = -1; k <= 1; k++) {
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let off = vec3<i32>(i, j, k);
                let v = vec3<f32>(off) + hash_int3_to_float3(cell + off) * p.randomness - local;
                let perp = v - to_closest;
                if (dot(perp, perp) > 0.0001) {
                    min_d = min(min_d, dot((to_closest + v) * 0.5, normalize(perp)));
                }
            }
        }
    }
    return min_d;
}

// `voronoi_n_sphere_radius`: half the distance from the closest point to
// its own closest neighbour.
fn voronoi_n_sphere_radius(p: VParams, coord: vec3<f32>) -> f32 {
    let cell_f = floor(coord);
    let local = coord - cell_f;
    let cell = vec3<i32>(cell_f);
    var closest = vec3<f32>(0.0);
    var closest_off = vec3<i32>(0);
    var min_d = 3.0e38;
    for (var k = -1; k <= 1; k++) {
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let off = vec3<i32>(i, j, k);
                let pos = vec3<f32>(off) + hash_int3_to_float3(cell + off) * p.randomness;
                let d = dot(pos - local, pos - local);
                if (d < min_d) { min_d = d; closest = pos; closest_off = off; }
            }
        }
    }
    min_d = 3.0e38;
    var neighbour = vec3<f32>(0.0);
    for (var k = -1; k <= 1; k++) {
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                if (i == 0 && j == 0 && k == 0) { continue; }
                let off = vec3<i32>(i, j, k) + closest_off;
                let pos = vec3<f32>(off) + hash_int3_to_float3(cell + off) * p.randomness;
                let d = dot(closest - pos, closest - pos);
                if (d < min_d) { min_d = d; neighbour = pos; }
            }
        }
    }
    return distance(neighbour, closest) * 0.5;
}

// One octave of F1 / F2 / Smooth F1.
fn voronoi_octave(p: VParams, coord: vec3<f32>) -> vec4<f32> {
    if (p.feature == VORONOI_F2) { return voronoi_f2(p, coord); }
    if (p.feature == VORONOI_SMOOTH_F1 && p.smoothness != 0.0) { return voronoi_smooth_f1(p, coord); }
    return voronoi_f1(p, coord);
}

// `fractal_voronoi_x_fx`: the octaves summed by the roughness, normalized
// by the largest distance the metric allows when asked.
fn fractal_voronoi_x_fx(p: VParams, coord: vec3<f32>) -> vec4<f32> {
    var amplitude = 1.0; var max_amplitude = 0.0; var scale = 1.0;
    var out = vec4<f32>(0.0);
    let zero_input = p.detail == 0.0 || p.roughness == 0.0;
    let n = i32(ceil(p.detail));
    for (var i = 0; i <= n; i++) {
        let octave = voronoi_octave(p, coord * scale);
        if (zero_input) { max_amplitude = 1.0; out = octave; break; }
        if (f32(i) <= p.detail) {
            max_amplitude += amplitude;
            out += octave * amplitude;
            scale *= p.lacunarity;
            amplitude *= p.roughness;
        } else {
            let remainder = p.detail - floor(p.detail);
            if (remainder != 0.0) {
                max_amplitude = mix(max_amplitude, max_amplitude + amplitude, remainder);
                out = mix(out, out + octave * amplitude, remainder);
            }
        }
    }
    if (p.normalize) {
        return vec4<f32>(out.rgb / max_amplitude, out.w / (max_amplitude * p.max_distance));
    }
    return out;
}

// `fractal_voronoi_distance_to_edge`: the fractalization with lerps where
// fBm adds.
fn fractal_voronoi_distance_to_edge(p: VParams, coord: vec3<f32>) -> f32 {
    var amplitude = 1.0; var max_amplitude = p.max_distance; var scale = 1.0;
    var dist = 8.0;
    let zero_input = p.detail == 0.0 || p.roughness == 0.0;
    let n = i32(ceil(p.detail));
    for (var i = 0; i <= n; i++) {
        let octave = voronoi_distance_to_edge(p, coord * scale);
        if (zero_input) { dist = octave; break; }
        if (f32(i) <= p.detail) {
            max_amplitude = mix(max_amplitude, p.max_distance / scale, amplitude);
            dist = mix(dist, min(dist, octave / scale), amplitude);
            scale *= p.lacunarity;
            amplitude *= p.roughness;
        } else {
            let remainder = p.detail - floor(p.detail);
            if (remainder != 0.0) {
                let lerp_amplitude = mix(max_amplitude, p.max_distance / scale, amplitude);
                max_amplitude = mix(max_amplitude, lerp_amplitude, remainder);
                let lerp_dist = mix(dist, min(dist, octave / scale), amplitude);
                dist = mix(dist, min(dist, lerp_dist), remainder);
            }
        }
    }
    if (p.normalize) { return dist / max_amplitude; }
    return dist;
}

// `svm_node_tex_voronoi` for a 3D coordinate: `.w` the Distance (the
// Radius for N-Sphere Radius), `.rgb` the Color (black for the features
// without one). The node's clamps: Detail 0 ..= 15, Roughness and
// Randomness 0 ..= 1, Smoothness halved into 0 ..= 0.5.
fn voronoi_texture_3d(co: vec3<f32>, feature: u32, metric: u32, detail: f32, roughness: f32, lacunarity: f32, smoothness: f32, exponent: f32, randomness: f32, normalize: bool) -> vec4<f32> {
    var p: VParams;
    p.feature = feature;
    p.metric = metric;
    p.detail = clamp(detail, 0.0, 15.0);
    p.roughness = clamp(roughness, 0.0, 1.0);
    p.lacunarity = lacunarity;
    p.smoothness = clamp(smoothness * 0.5, 0.0, 0.5);
    p.exponent = exponent;
    p.randomness = clamp(randomness, 0.0, 1.0);
    p.normalize = normalize;
    p.max_distance = 0.0;
    if (feature == VORONOI_DISTANCE_TO_EDGE) {
        p.max_distance = 0.5 + 0.5 * p.randomness;
        return vec4<f32>(vec3<f32>(0.0), fractal_voronoi_distance_to_edge(p, co));
    }
    if (feature == VORONOI_N_SPHERE_RADIUS) {
        return vec4<f32>(vec3<f32>(0.0), voronoi_n_sphere_radius(p, co));
    }
    let half = vec3<f32>(0.5 + 0.5 * p.randomness);
    p.max_distance = voronoi_distance(vec3<f32>(0.0), half, p) * select(1.0, 2.0, feature == VORONOI_F2);
    return fractal_voronoi_x_fx(p, co);
}
