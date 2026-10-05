// Perlin noise and its fractals, Cycles' SVM ported (`kernel/svm/noise.h`,
// `fractal_noise.h`, `noisetex.h`, `util/hash.h`): the Noise Texture node's
// five types — fBm, Multifractal, Hybrid Multifractal, Ridged Multifractal,
// Hetero Terrain — over Detail, Roughness, Lacunarity, Offset and Gain, the
// Color output three offset evaluations, Distortion three more.

// `NodeTexNoise.type` (`SHD_NOISE_*`).
const NOISE_MULTIFRACTAL: u32 = 0u;
const NOISE_FBM: u32 = 1u;
const NOISE_HYBRID_MULTIFRACTAL: u32 = 2u;
const NOISE_RIDGED_MULTIFRACTAL: u32 = 3u;
const NOISE_HETERO_TERRAIN: u32 = 4u;

// ---- Jenkins' lookup3 (`util/hash.h`: `mix`, `final`, `hash_uint2/3`) ----

fn rot(x: u32, k: u32) -> u32 { return (x << k) | (x >> (32u - k)); }

fn jenkins_final(a0: u32, b0: u32, c0: u32) -> u32 {
    var a = a0; var b = b0; var c = c0;
    c ^= b; c -= rot(b, 14u);
    a ^= c; a -= rot(c, 11u);
    b ^= a; b -= rot(a, 25u);
    c ^= b; c -= rot(b, 16u);
    a ^= c; a -= rot(c, 4u);
    b ^= a; b -= rot(a, 14u);
    c ^= b; c -= rot(b, 24u);
    return c;
}

fn hash_uint2(kx: u32, ky: u32) -> u32 {
    let s = 0xdeadbeefu + (2u << 2u) + 13u;
    return jenkins_final(s + kx, s + ky, s);
}

fn hash_uint3(kx: u32, ky: u32, kz: u32) -> u32 {
    let s = 0xdeadbeefu + (3u << 2u) + 13u;
    return jenkins_final(s + kx, s + ky, s + kz);
}

// `uint_to_float_incl`.
fn hash_float2_to_float(k: vec2<f32>) -> f32 {
    return f32(hash_uint2(bitcast<u32>(k.x), bitcast<u32>(k.y))) * (1.0 / 4294967295.0);
}

// `hash_pcg3d_i` → `hash_int3_to_float3`: three numbers in 0 ..= 1 for a cell.
fn hash_int3_to_float3(k: vec3<i32>) -> vec3<f32> {
    var v = bitcast<vec3<u32>>(k) * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v = v & vec3<u32>(0x7FFFFFFFu);
    return vec3<f32>(v) * (1.0 / 2147483647.0);
}

// ---- Perlin (`noise.h`) ----

fn fade(t: f32) -> f32 { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

fn negate_if(v: f32, c: u32) -> f32 { return select(v, -v, c != 0u); }

fn grad3(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15u;
    let u = select(y, x, h < 8u);
    let vt = select(z, x, h == 12u || h == 14u);
    let v = select(vt, y, h < 4u);
    return negate_if(u, h & 1u) + negate_if(v, h & 2u);
}

fn tri_mix(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32, x: f32, y: f32, z: f32) -> f32 {
    let x1 = 1.0 - x; let y1 = 1.0 - y; let z1 = 1.0 - z;
    return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) + z * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x));
}

fn perlin_3d(x: f32, y: f32, z: f32) -> f32 {
    let fl = vec3<f32>(floor(x), floor(y), floor(z));
    let X = bitcast<u32>(i32(fl.x)); let Y = bitcast<u32>(i32(fl.y)); let Z = bitcast<u32>(i32(fl.z));
    let fx = x - fl.x; let fy = y - fl.y; let fz = z - fl.z;
    let u = fade(fx); let v = fade(fy); let w = fade(fz);
    return tri_mix(
        grad3(hash_uint3(X, Y, Z), fx, fy, fz),
        grad3(hash_uint3(X + 1u, Y, Z), fx - 1.0, fy, fz),
        grad3(hash_uint3(X, Y + 1u, Z), fx, fy - 1.0, fz),
        grad3(hash_uint3(X + 1u, Y + 1u, Z), fx - 1.0, fy - 1.0, fz),
        grad3(hash_uint3(X, Y, Z + 1u), fx, fy, fz - 1.0),
        grad3(hash_uint3(X + 1u, Y, Z + 1u), fx - 1.0, fy, fz - 1.0),
        grad3(hash_uint3(X, Y + 1u, Z + 1u), fx, fy - 1.0, fz - 1.0),
        grad3(hash_uint3(X + 1u, Y + 1u, Z + 1u), fx - 1.0, fy - 1.0, fz - 1.0),
        u, v, w);

}

// `snoise_3d`: in −1 ..= 1, the texture repeating every 100000 so the
// floating point holds.
fn snoise_3d(p0: vec3<f32>) -> f32 {
    let correction = 0.5 * vec3<f32>(select(vec3<f32>(0.0), vec3<f32>(1.0), abs(p0) >= vec3<f32>(1000000.0)));
    let p = p0 - 100000.0 * trunc(p0 / 100000.0) + correction;
    return 0.9820 * perlin_3d(p.x, p.y, p.z);
}

// `noise_fbm` (3D), `normalize` on: in 0 ..= 1.
fn noise_fbm(p: vec3<f32>, detail: f32, roughness: f32, lacunarity: f32, normalize: bool) -> f32 {
    var fscale = 1.0; var amp = 1.0; var maxamp = 0.0; var sum = 0.0;
    let n = i32(detail);
    for (var i = 0; i <= n; i++) {
        sum += snoise_3d(fscale * p) * amp;
        maxamp += amp;
        amp *= roughness;
        fscale *= lacunarity;
    }
    let rmd = detail - floor(detail);
    if (rmd != 0.0) {
        let sum2 = sum + snoise_3d(fscale * p) * amp;
        if (normalize) { return mix(0.5 * sum / maxamp + 0.5, 0.5 * sum2 / (maxamp + amp) + 0.5, rmd); }
        return mix(sum, sum2, rmd);
    }
    if (normalize) { return 0.5 * sum / maxamp + 0.5; }
    return sum;
}

// `noise_multi_fractal`.
fn noise_multi_fractal(p0: vec3<f32>, detail: f32, roughness: f32, lacunarity: f32) -> f32 {
    var p = p0; var value = 1.0; var pwr = 1.0;
    for (var i = 0; i <= i32(detail); i++) {
        value *= pwr * snoise_3d(p) + 1.0;
        pwr *= roughness;
        p *= lacunarity;
    }
    let rmd = detail - floor(detail);
    if (rmd != 0.0) { value *= rmd * pwr * snoise_3d(p) + 1.0; }
    return value;
}

// `noise_hetero_terrain`: the first octave unscaled, the rest scaled by
// the value so far.
fn noise_hetero_terrain(p0: vec3<f32>, detail: f32, roughness: f32, lacunarity: f32, offset: f32) -> f32 {
    var p = p0; var pwr = roughness;
    var value = offset + snoise_3d(p);
    p *= lacunarity;
    for (var i = 1; i <= i32(detail); i++) {
        value += (snoise_3d(p) + offset) * pwr * value;
        pwr *= roughness;
        p *= lacunarity;
    }
    let rmd = detail - floor(detail);
    if (rmd != 0.0) { value += rmd * (snoise_3d(p) + offset) * pwr * value; }
    return value;
}

// `noise_hybrid_multi_fractal`.
fn noise_hybrid_multi_fractal(p0: vec3<f32>, detail: f32, roughness: f32, lacunarity: f32, offset: f32, gain: f32) -> f32 {
    var p = p0; var pwr = 1.0; var value = 0.0; var weight = 1.0;
    for (var i = 0; weight > 0.001 && i <= i32(detail); i++) {
        weight = min(weight, 1.0);
        let signal = (snoise_3d(p) + offset) * pwr;
        pwr *= roughness;
        value += weight * signal;
        weight *= gain * signal;
        p *= lacunarity;
    }
    let rmd = detail - floor(detail);
    if (rmd != 0.0 && weight > 0.001) {
        weight = min(weight, 1.0);
        value += rmd * weight * (snoise_3d(p) + offset) * pwr;
    }
    return value;
}

// `noise_ridged_multi_fractal`.
fn noise_ridged_multi_fractal(p0: vec3<f32>, detail: f32, roughness: f32, lacunarity: f32, offset: f32, gain: f32) -> f32 {
    var p = p0; var pwr = roughness;
    var signal = offset - abs(snoise_3d(p));
    signal *= signal;
    var value = signal;
    for (var i = 1; i <= i32(detail); i++) {
        p *= lacunarity;
        let weight = clamp(signal * gain, 0.0, 1.0);
        signal = offset - abs(snoise_3d(p));
        signal *= signal * weight;
        value += signal * pwr;
        pwr *= roughness;
    }
    return value;
}

// `noise_select`: the node's type.
fn noise_select(p: vec3<f32>, detail: f32, roughness: f32, lacunarity: f32, offset: f32, gain: f32, kind: u32, normalize: bool) -> f32 {
    switch kind {
        case NOISE_MULTIFRACTAL: { return noise_multi_fractal(p, detail, roughness, lacunarity); }
        case NOISE_HYBRID_MULTIFRACTAL: { return noise_hybrid_multi_fractal(p, detail, roughness, lacunarity, offset, gain); }
        case NOISE_RIDGED_MULTIFRACTAL: { return noise_ridged_multi_fractal(p, detail, roughness, lacunarity, offset, gain); }
        case NOISE_HETERO_TERRAIN: { return noise_hetero_terrain(p, detail, roughness, lacunarity, offset); }
        default: { return noise_fbm(p, detail, roughness, lacunarity, normalize); }
    }
}

// `random_float3_offset`.
fn random_float3_offset(seed: f32) -> vec3<f32> {
    return vec3<f32>(100.0 + hash_float2_to_float(vec2<f32>(seed, 0.0)) * 100.0, 100.0 + hash_float2_to_float(vec2<f32>(seed, 1.0)) * 100.0, 100.0 + hash_float2_to_float(vec2<f32>(seed, 2.0)) * 100.0);
}

// `noise_texture_3d` (`svm_node_tex_noise`): `.w` the Fac, `.rgb` the
// Color; Detail clamped to 0 ..= 15 and Roughness to at least 0, as the
// node does.
fn noise_texture_3d(co: vec3<f32>, detail0: f32, roughness0: f32, lacunarity: f32, offset: f32, gain: f32, distortion: f32, kind: u32, normalize: bool) -> vec4<f32> {
    let detail = clamp(detail0, 0.0, 15.0);
    let roughness = max(roughness0, 0.0);
    var pp = co;
    if (distortion != 0.0) {
        pp += vec3<f32>(snoise_3d(pp + random_float3_offset(0.0)) * distortion, snoise_3d(pp + random_float3_offset(1.0)) * distortion, snoise_3d(pp + random_float3_offset(2.0)) * distortion);
    }
    let value = noise_select(pp, detail, roughness, lacunarity, offset, gain, kind, normalize);
    let g = noise_select(pp + random_float3_offset(3.0), detail, roughness, lacunarity, offset, gain, kind, normalize);
    let b = noise_select(pp + random_float3_offset(4.0), detail, roughness, lacunarity, offset, gain, kind, normalize);
    return vec4<f32>(value, g, b, value);
}
