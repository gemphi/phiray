// The rest of Jenkins' lookup3 family (`util/hash.h`), for the nodes that
// hash a coordinate rather than interpolate one: White Noise
// (`svm/white_noise.h`) and Gabor (`svm/gabor.h`). `noise.wgsl` holds
// `rot`, `jenkins_final`, `hash_uint2/3` and `hash_float2_to_float`, which
// Perlin needs; these are the four-argument mixer and the float-to-float
// shapes built on it.

// `mix(a, b, c)` — lookup3's non-final round, which only `hash_uint4` uses.
fn jenkins_mix(a0: u32, b0: u32, c0: u32) -> vec3<u32> {
    var a = a0; var b = b0; var c = c0;
    a -= c; a ^= rot(c, 4u); c += b;
    b -= a; b ^= rot(a, 6u); a += c;
    c -= b; c ^= rot(b, 8u); b += a;
    a -= c; a ^= rot(c, 16u); c += b;
    b -= a; b ^= rot(a, 19u); a += c;
    c -= b; c ^= rot(b, 4u); b += a;
    return vec3<u32>(a, b, c);
}

fn hash_uint(kx: u32) -> u32 {
    let s = 0xdeadbeefu + (1u << 2u) + 13u;
    return jenkins_final(s + kx, s, s);
}

fn hash_uint4(kx: u32, ky: u32, kz: u32, kw: u32) -> u32 {
    let s = 0xdeadbeefu + (4u << 2u) + 13u;
    let m = jenkins_mix(s + kx, s + ky, s + kz);
    return jenkins_final(m.x + kw, m.y, m.z);
}

// `uint_to_float_incl`: [0, uint_max] -> [0, 1].
const HASH_SCALE: f32 = 1.0 / 4294967295.0;

fn hash_float_to_float(k: f32) -> f32 {
    return f32(hash_uint(bitcast<u32>(k))) * HASH_SCALE;
}

fn hash_float3_to_float(k: vec3<f32>) -> f32 {
    return f32(hash_uint3(bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z))) * HASH_SCALE;
}

fn hash_float4_to_float(k: vec4<f32>) -> f32 {
    return f32(hash_uint4(bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z), bitcast<u32>(k.w))) * HASH_SCALE;
}

fn hash_float_to_float3(k: f32) -> vec3<f32> {
    return vec3<f32>(hash_float_to_float(k), hash_float2_to_float(vec2<f32>(k, 1.0)), hash_float2_to_float(vec2<f32>(k, 2.0)));
}

fn hash_float2_to_float2(k: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(hash_float2_to_float(k), hash_float3_to_float(vec3<f32>(k, 1.0)));
}

fn hash_float2_to_float3(k: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(hash_float2_to_float(k), hash_float3_to_float(vec3<f32>(k, 1.0)), hash_float3_to_float(vec3<f32>(k, 2.0)));
}

fn hash_float3_to_float2(k: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(hash_float3_to_float(k), hash_float4_to_float(vec4<f32>(k, 1.0)));
}

fn hash_float3_to_float3(k: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(hash_float3_to_float(k), hash_float4_to_float(vec4<f32>(k, 1.0)), hash_float4_to_float(vec4<f32>(k, 2.0)));
}

fn hash_float4_to_float2(k: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(hash_float4_to_float(k), hash_float4_to_float(vec4<f32>(k.w, k.x, k.y, k.z)));
}

fn hash_float4_to_float3(k: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(hash_float4_to_float(k), hash_float4_to_float(vec4<f32>(k.w, k.x, k.y, k.z)), hash_float4_to_float(vec4<f32>(k.z, k.w, k.x, k.y)));
}

// `svm_node_tex_white_noise`: the coordinate hashed, in as many dimensions
// as the node is set to (`bNode.custom1`: 1 takes W alone, 2 the XY, 3 the
// XYZ, 4 all four). `.w` is the Value output and `.rgb` the Color.
fn white_noise_texture(co: vec3<f32>, w: f32, dimensions: u32) -> vec4<f32> {
    switch dimensions {
        case 1u: { return vec4<f32>(hash_float_to_float3(w), hash_float_to_float(w)); }
        case 2u: { return vec4<f32>(hash_float2_to_float3(co.xy), hash_float2_to_float(co.xy)); }
        case 4u: {
            let k = vec4<f32>(co, w);
            return vec4<f32>(hash_float4_to_float3(k), hash_float4_to_float(k));
        }
        default: { return vec4<f32>(hash_float3_to_float3(co), hash_float3_to_float(co)); }
    }
}
