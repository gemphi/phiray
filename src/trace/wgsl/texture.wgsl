fn texel(base: u32, w: u32, x: i32, y: i32) -> vec4<f32> {
    let c = atlas(base + u32(y) * w + u32(x));
    return vec4<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), f32(c >> 24u)) / 255.0;
}

// Colour textures are sRGB; normal maps are read raw (`sample_raw`).
fn sample_tex(id: u32, uv: vec2<f32>) -> vec4<f32> {
    let s = sample_raw(id, uv);
    return vec4<f32>(pow(s.rgb, vec3<f32>(2.2)), s.a);
}

// Emitted radiance at barycentrics `bary` (emission texture when linked).
fn emit_at(t: Tri, bary: vec3<f32>) -> vec3<f32> {
    if (t.maps.z == NO_TEX) { return t.emit; }
    let uv = vec2<f32>(dot(t.us, bary), dot(t.vs, bary));
    return t.emit * sample_tex(t.maps.z, uv).rgb;
}

// Tangent-space normal from triangle `t`'s normal map, around shading normal `n`.
fn mapped_normal(t: Tri, n: vec3<f32>, bary: vec3<f32>) -> vec3<f32> {
    return mapped_normal_lod(t, n, bary, 0.0);
}

// The same, the map read at mip level `lod`.
fn mapped_normal_lod(t: Tri, n: vec3<f32>, bary: vec3<f32>, lod: f32) -> vec3<f32> {
    let uv = vec2<f32>(dot(t.us, bary), dot(t.vs, bary));
    let raw = sample_raw_lod(t.nmap, uv, lod).xyz * 2.0 - 1.0;
    let ts = normalize(mix(vec3<f32>(0.0, 0.0, 1.0), raw, clamp(t.bump, 0.0, 10.0)));
    let e1 = t.b - t.a;
    let e2 = t.c - t.a;
    let du1 = t.us.y - t.us.x;
    let dv1 = t.vs.y - t.vs.x;
    let du2 = t.us.z - t.us.x;
    let dv2 = t.vs.z - t.vs.x;
    let det = du1 * dv2 - du2 * dv1;
    if (abs(det) < 1e-12) { return n; }
    let tu = (e1 * dv2 - e2 * dv1) / det;
    let tv = (e2 * du1 - e1 * du2) / det;
    let tan = normalize(tu - n * dot(n, tu));
    let bit = cross(n, tan) * select(-1.0, 1.0, dot(cross(n, tan), tv) >= 0.0);
    return normalize(tan * ts.x + bit * ts.y + n * ts.z);
}

// Bilinear, repeat-wrapped, v up (Blender) → row 0 at the top.
// An atlas entry's head: its first texel, width, height and — an image
// added with its mip chain (`Atlas::add`) — how many levels follow it, in
// the width's top byte.
fn head_base(id: u32) -> u32 { return atlas(1u + 3u * id); }
fn head_w(id: u32) -> u32 { return atlas(2u + 3u * id) & 0xffffffu; }
fn head_h(id: u32) -> u32 { return atlas(3u + 3u * id); }
fn head_levels(id: u32) -> u32 { return max(atlas(2u + 3u * id) >> 24u, 1u); }

// Bilinear at mip level `lod` of image `id`: two levels blended when it
// falls between them. Level 0 is the image; past the last level, the last.
fn sample_raw_lod(id: u32, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    if (id >= atlas(0u)) { return vec4<f32>(1.0); }
    let top = f32(head_levels(id) - 1u);
    let l = clamp(lod, 0.0, top);
    let k = u32(floor(l));
    let a = sample_raw(id + k, uv);
    if (f32(k) >= top) { return a; }
    return mix(a, sample_raw(id + k + 1u, uv), l - floor(l));
}

fn sample_tex_lod(id: u32, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    let s = sample_raw_lod(id, uv, lod);
    return vec4<f32>(pow(s.rgb, vec3<f32>(2.2)), s.a);
}

fn sample_raw(id: u32, uv: vec2<f32>) -> vec4<f32> {
    if (id >= atlas(0u)) { return vec4<f32>(1.0); }
    let base = head_base(id);
    let w = head_w(id);
    let h = head_h(id);
    let fx = fract(uv.x) * f32(w) - 0.5;
    let fy = (1.0 - fract(uv.y)) * f32(h) - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let wi = i32(w);
    let hi = i32(h);
    let xa = (x0 % wi + wi) % wi;
    let xb = (xa + 1) % wi;
    let ya = (y0 % hi + hi) % hi;
    let yb = (ya + 1) % hi;
    let top = mix(texel(base, w, xa, ya), texel(base, w, xb, ya), tx);
    let bot = mix(texel(base, w, xa, yb), texel(base, w, xb, yb), tx);
    return mix(top, bot, ty);
}


// One RGBE texel decoded to linear radiance (`Atlas::add_hdr`).
fn env_texel(base: u32, w: u32, x: i32, y: i32) -> vec3<f32> {
    let c = atlas(base + u32(y) * w + u32(x));
    let e = c >> 24u;
    if (e == 0u) { return vec3<f32>(0.0); }
    let f = exp2(f32(e) - 136.0);
    return vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) * f;
}

// Bilinear sample of an RGBE environment map: `uv.x` wraps (the sphere joins
// behind the camera), `uv.y` clamps at the poles, row 0 is the north pole.
fn sample_env(id: u32, uv: vec2<f32>) -> vec3<f32> {
    if (id >= atlas(0u)) { return vec3<f32>(0.0); }
    let base = head_base(id);
    let w = head_w(id);
    let h = head_h(id);
    let wi = i32(w);
    let hi = i32(h);
    let fx = fract(uv.x) * f32(w) - 0.5;
    let fy = clamp(uv.y, 0.0, 1.0) * f32(h) - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let xa = (x0 % wi + wi) % wi;
    let xb = (xa + 1) % wi;
    let ya = clamp(y0, 0, hi - 1);
    let yb = clamp(y0 + 1, 0, hi - 1);
    let top = mix(env_texel(base, w, xa, ya), env_texel(base, w, xb, ya), tx);
    let bot = mix(env_texel(base, w, xa, yb), env_texel(base, w, xb, yb), tx);
    return mix(top, bot, ty);
}
