// Workbench's Depth of Field (`workbench_effect_dof.cc` and its five
// shaders): a gather over a half-resolution copy of the colour, its
// circle of confusion from the depth (`dof_calculate_coc`), after the
// composite and before the anti-aliasing gather, every sample — the
// blur's noise offset walks with the sample, so the frames converge.
// The stages run as steps of the effects kernel (`Fx`), `p.slice.z` the
// stage, each over its own count of pixels: 1 `dof_prepare` (half
// resolution: colour and CoC), 2 and 3 `dof_downsample` (quarter, eighth
// — the texture's mips), 4 `dof_blur1` (the 49-sample bokeh gather), 5
// `dof_blur2` (the 3 × 3 median), 6 `dof_resolve` (full resolution: the
// blur blended in where the CoC is wide), which then finishes the pixel
// as the composite does without the effect. Buffers live in the image's
// scratch words: the half-resolution colour at `21n` and its CoC at
// `24n`, the mips and the blur between `4n` and `7n` (`dof_region`).

// `MAX_COC_SIZE`.
const MAX_COC: f32 = 100.0;
// `kernel_radius_`: the gather's samples sit on rings 0 ..= 3 of a
// 7 × 7 square mapped onto the disk, 49 in all.
const DOF_KERNEL_RADIUS: i32 = 3;

// The widths of the half, quarter and eighth images (`DofPass::init`'s
// `resolution / 2`, floored, at least 1).
fn dof_size(mip: u32) -> vec2<u32> {
    var s = vec2<u32>(max(p.w / 2u, 1u), max(p.h / 2u, 1u));
    for (var m = 0u; m < mip; m++) { s = vec2<u32>(max(s.x / 2u, 1u), max(s.y / 2u, 1u)); }
    return s;
}

// Where mip `mip`'s colour (3 words a pixel) and CoC (1 word) start, and
// the blur's colour: the half image after the grid sums, the rest in the
// display sums' words, which Solid leaves empty.
fn dof_region(mip: u32) -> vec2<u32> {
    if (mip == 0u) { return vec2<u32>(21u * p.n, 24u * p.n); }
    let q = dof_size(1u);
    let qn = q.x * q.y;
    let m1 = vec2<u32>(4u * p.n, 4u * p.n + 3u * qn);
    if (mip == 1u) { return m1; }
    let e = dof_size(2u);
    let en = e.x * e.y;
    return vec2<u32>(m1.y + qn, m1.y + qn + 3u * en);
}
fn dof_blur_at() -> u32 {
    let m2 = dof_region(2u);
    let e = dof_size(2u);
    return m2.y + e.x * e.y;
}

// `dof_calculate_coc` of the view depth under a pixel (`z_at`; nothing
// drawn there reads as the far plane), in the frame's pixels.
fn dof_coc(z_in: f32) -> f32 {
    var z = z_in;
    if (z >= 0.0) { z = p.dof.w; }
    let invsensor = p.dof.z * frame_size().x;
    return p.dof.x * (p.dof.y / z - 1.0) * invsensor;
}
fn dof_encode(near: f32, far: f32) -> u32 { return pack2x16unorm(vec2<f32>(near, far) / MAX_COC); }
fn dof_decode(w: u32) -> vec2<f32> { return unpack2x16unorm(w); }

// A texel of mip `mip`, clamped to its edge.
fn dof_color_fetch(mip: u32, t: vec2<i32>) -> vec3<f32> {
    let s = dof_size(mip);
    let c = vec2<u32>(clamp(t, vec2<i32>(0), vec2<i32>(s) - 1));
    let a = dof_region(mip).x + 3u * (c.y * s.x + c.x);
    return three(a);
}
fn dof_coc_fetch(mip: u32, t: vec2<i32>) -> vec2<f32> {
    let s = dof_size(mip);
    let c = vec2<u32>(clamp(t, vec2<i32>(0), vec2<i32>(s) - 1));
    return dof_decode(px[dof_region(mip).y + c.y * s.x + c.x]);
}
fn dof_blur_fetch(t: vec2<i32>) -> vec3<f32> {
    let s = dof_size(0u);
    let c = vec2<u32>(clamp(t, vec2<i32>(0), vec2<i32>(s) - 1));
    return three(dof_blur_at() + 3u * (c.y * s.x + c.x));
}

// `texture(tx, uv)`: bilinear over the mip, the edge extended.
fn dof_color_sample(mip: u32, uv: vec2<f32>) -> vec3<f32> {
    let f = uv * vec2<f32>(dof_size(mip)) - 0.5;
    let t = vec2<i32>(floor(f));
    let w = fract(f);
    let a = mix(dof_color_fetch(mip, t), dof_color_fetch(mip, t + vec2<i32>(1, 0)), w.x);
    let b = mix(dof_color_fetch(mip, t + vec2<i32>(0, 1)), dof_color_fetch(mip, t + vec2<i32>(1, 1)), w.x);
    return mix(a, b, w.y);
}
fn dof_coc_sample(uv: vec2<f32>) -> vec2<f32> {
    let f = uv * vec2<f32>(dof_size(0u)) - 0.5;
    let t = vec2<i32>(floor(f));
    let w = fract(f);
    let a = mix(dof_coc_fetch(0u, t), dof_coc_fetch(0u, t + vec2<i32>(1, 0)), w.x);
    let b = mix(dof_coc_fetch(0u, t + vec2<i32>(0, 1)), dof_coc_fetch(0u, t + vec2<i32>(1, 1)), w.x);
    return mix(a, b, w.y);
}
fn dof_blur_sample(uv: vec2<f32>) -> vec3<f32> {
    let f = uv * vec2<f32>(dof_size(0u)) - 0.5;
    let t = vec2<i32>(floor(f));
    let w = fract(f);
    let a = mix(dof_blur_fetch(t), dof_blur_fetch(t + vec2<i32>(1, 0)), w.x);
    let b = mix(dof_blur_fetch(t + vec2<i32>(0, 1)), dof_blur_fetch(t + vec2<i32>(1, 1)), w.x);
    return mix(a, b, w.y);
}
// `textureLod`: between the two mips either side of `lod`, 0 ..= 2.
fn dof_color_lod(uv: vec2<f32>, lod_in: f32) -> vec3<f32> {
    let lod = clamp(lod_in, 0.0, 2.0);
    let lo = u32(floor(lod));
    let hi = min(lo + 1u, 2u);
    return mix(dof_color_sample(lo, uv), dof_color_sample(hi, uv), fract(lod));
}

// `weighted_sum`: `safe_rcp` of a zero weight is 0.
fn dof_weighted(c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>, c4: vec3<f32>, w: vec4<f32>) -> vec3<f32> {
    let sum = w.x + w.y + w.z + w.w;
    if (sum == 0.0) { return vec3<f32>(0.0); }
    return (c1 * w.x + c2 * w.y + c3 * w.z + c4 * w.w) / sum;
}
fn dof_max4(v: vec4<f32>) -> f32 { return max(max(v.x, v.y), max(v.z, v.w)); }

// `dof_prepare` and `dof_downsample`'s CoC-aware reduction of the four
// texels under one of the next mip's: the far field's pixels take
// priority, each weighted by how close its CoC is to the largest.
fn dof_reduce(c1: vec3<f32>, c2: vec3<f32>, c3: vec3<f32>, c4: vec3<f32>, cocs_near: vec4<f32>, cocs_far: vec4<f32>, coc_near: f32, coc_far: f32) -> vec3<f32> {
    let near_weights = step(vec4<f32>(0.0), cocs_near) * clamp(1.0 - abs(coc_near - cocs_near), vec4<f32>(0.0), vec4<f32>(1.0));
    let far_weights = step(vec4<f32>(0.0), cocs_far) * clamp(1.0 - abs(coc_far - cocs_far), vec4<f32>(0.0), vec4<f32>(1.0));
    var w = near_weights;
    if (any(far_weights != vec4<f32>(0.0))) { w = far_weights; }
    return dof_weighted(c1, c2, c3, c4, w);
}

// Stage 1: the half image from the composite's colour and depth.
fn dof_prepare(i: u32) {
    let s = dof_size(0u);
    if (i >= s.x * s.y) { return; }
    let h = vec2<i32>(i32(i % s.x), i32(i / s.x)) * 2;
    let t1 = at(h.x, h.y);
    let t2 = at(h.x + 1, h.y + 1);
    let t3 = at(h.x + 1, h.y);
    let t4 = at(h.x, h.y + 1);
    let cocs_near = vec4<f32>(dof_coc(z_at(t1)), dof_coc(z_at(t2)), dof_coc(z_at(t3)), dof_coc(z_at(t4)));
    let cocs_far = -cocs_near;
    let coc_near = max(dof_max4(cocs_near), 0.0);
    let coc_far = max(dof_max4(cocs_far), 0.0);
    var color = dof_reduce(three(7u * p.n + 3u * t1), three(7u * p.n + 3u * t2), three(7u * p.n + 3u * t3), three(7u * p.n + 3u * t4), cocs_near, cocs_far, coc_near, coc_far);
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(3.0));
    let r = dof_region(0u);
    px[r.x + 3u * i] = bitcast<u32>(color.x);
    px[r.x + 3u * i + 1u] = bitcast<u32>(color.y);
    px[r.x + 3u * i + 2u] = bitcast<u32>(color.z);
    px[r.y + i] = dof_encode(coc_near, coc_far);
}

// Stages 2 and 3: mip `mip` from the one above it.
fn dof_downsample(mip: u32, i: u32) {
    let s = dof_size(mip);
    if (i >= s.x * s.y) { return; }
    let src = mip - 1u;
    let h = vec2<i32>(i32(i % s.x), i32(i / s.x)) * 2;
    let k1 = dof_coc_fetch(src, h) * MAX_COC;
    let k2 = dof_coc_fetch(src, h + vec2<i32>(1, 1)) * MAX_COC;
    let k3 = dof_coc_fetch(src, h + vec2<i32>(1, 0)) * MAX_COC;
    let k4 = dof_coc_fetch(src, h + vec2<i32>(0, 1)) * MAX_COC;
    let cocs_near = vec4<f32>(k1.x, k2.x, k3.x, k4.x);
    let cocs_far = vec4<f32>(k1.y, k2.y, k3.y, k4.y);
    let color = dof_reduce(dof_color_fetch(src, h), dof_color_fetch(src, h + vec2<i32>(1, 1)), dof_color_fetch(src, h + vec2<i32>(1, 0)), dof_color_fetch(src, h + vec2<i32>(0, 1)), cocs_near, cocs_far, dof_max4(cocs_near), dof_max4(cocs_far));
    let r = dof_region(mip);
    px[r.x + 3u * i] = bitcast<u32>(color.x);
    px[r.x + 3u * i + 1u] = bitcast<u32>(color.y);
    px[r.x + 3u * i + 2u] = bitcast<u32>(color.z);
    px[r.y + i] = dof_encode(dof_max4(cocs_near), dof_max4(cocs_far));
}

// `square_to_circle`: the unit square onto the disk, `(r, θ)`.
fn dof_square_to_circle(x: f32, y: f32) -> vec2<f32> {
    let q = 0.78539816;
    if (x > -y) {
        if (x > y) { return vec2<f32>(x, q * (y / x)); }
        return vec2<f32>(y, q * (2.0 - x / y));
    }
    if (x < y) { return vec2<f32>(-x, q * (4.0 + y / x)); }
    if (y != 0.0) { return vec2<f32>(-y, q * (6.0 - x / y)); }
    return vec2<f32>(-y, 0.0);
}

// `DofPass::setup_samples`: ring sample `(j, k)` of the kernel as the
// bokeh's `[x, y, r]` — shaped by the aperture's blades, turned by its
// rotation and squeezed by its ratio (`Camera::blades`, `::blade_spin`,
// `::aperture_ratio`, the camera's own).
fn dof_kernel_sample(j: i32, k: i32) -> vec3<f32> {
    let coord = vec2<f32>(f32(j), f32(k)) / f32(DOF_KERNEL_RADIUS);
    let rt = dof_square_to_circle(coord.x, coord.y);
    var r = rt.x;
    var t = rt.y;
    let blades = f32(p.blades);
    if (blades > 1.0) {
        let denom = t - (TAU / blades) * floor((blades * t + 3.14159265) / TAU);
        r *= cos(3.14159265 / blades) / cos(denom);
    }
    t += p.blade_spin;
    var ratio = p.aperture_ratio;
    if (ratio <= 0.0) { ratio = 1.0; }
    return vec3<f32>(r * cos(t) / ratio, r * sin(t), rt.x);
}

// `get_random_vector`: interleaved gradient noise over the blue noise's
// last channel, turned by the sample's offset.
fn dof_random_vector(frag: vec2<f32>, offset: f32) -> vec2<f32> {
    let ign = fract(offset + 52.9829189 * fract(0.06711056 * frag.x + 0.00583715 * frag.y));
    let t = 3u * (64u * (u32(frag.y) % 64u) + (u32(frag.x) % 64u));
    let bn = bitcast<f32>(noise[t + 2u]);
    let ang = TAU * fract(bn + offset);
    return vec2<f32>(cos(ang), sin(ang)) * sqrt(ign);
}

// Stage 4: the bokeh gather over the half image.
fn dof_blur1(i: u32) {
    let s = dof_size(0u);
    if (i >= s.x * s.y) { return; }
    let frag = vec2<f32>(f32(i % s.x), f32(i / s.x)) + 0.5;
    let inv = 1.0 / vec2<f32>(f32(p.w), f32(p.h));
    let uv = frag * inv * 2.0;
    let texel = vec2<i32>(uv * vec2<f32>(s));
    let k = dof_coc_fetch(0u, texel);
    let max_radius = max(k.x, k.y) * MAX_COC;
    let offset = f32(p.frame) / max(p.taa.w, 1.0);
    let jitter = dof_random_vector(frag, offset) * 0.2 * clamp(max_radius * 0.2 - 4.0, 0.0, 1.0);
    var color = vec3<f32>(0.0);
    var tot = 0.0;
    for (var ring = 0; ring <= DOF_KERNEL_RADIUS; ring++) {
        for (var j = -DOF_KERNEL_RADIUS; j <= DOF_KERNEL_RADIUS; j++) {
            for (var kk = -DOF_KERNEL_RADIUS; kk <= DOF_KERNEL_RADIUS; kk++) {
                if (abs(j) > ring || abs(kk) > ring) { continue; }
                if (abs(j) < ring && abs(kk) < ring) { continue; }
                let smp = dof_kernel_sample(j, kk);
                let tc = uv + (jitter + smp.xy) * inv * max_radius;
                let c = dof_coc_sample(tc);
                // `dof_decode_signed_coc`: the larger field's, signed.
                var coc = c.y * MAX_COC;
                if (c.x > c.y) { coc = c.x * MAX_COC; }
                let lod = log2(clamp((coc + min(coc, max_radius)) * 0.5 - 21.0, 0.0, 16.0) * 0.25);
                let radius = smp.z * max_radius;
                let weight = coc * smoothstep(radius - 0.5, radius + 0.5, coc);
                color += dof_color_lod(tc, lod) * weight;
                tot += weight;
            }
        }
    }
    var out = dof_color_sample(0u, uv);
    if (tot > 0.0) { out = color / tot; }
    let b = dof_blur_at() + 3u * i;
    px[b] = bitcast<u32>(out.x);
    px[b + 1u] = bitcast<u32>(out.y);
    px[b + 2u] = bitcast<u32>(out.z);
}

// `s2(a, b)`: the window's `a` takes the lesser, `b` the greater, per channel.
fn dof_swap(v: ptr<function, array<vec3<f32>, 9>>, a: i32, b: i32) {
    let t = (*v)[a];
    (*v)[a] = min(t, (*v)[b]);
    (*v)[b] = max(t, (*v)[b]);
}

// Stage 5: McGuire and Whitson's 3 × 3 median over the blur, its window
// opened only where the CoC passes 9, back into the half image.
fn dof_blur2(i: u32) {
    let s = dof_size(0u);
    if (i >= s.x * s.y) { return; }
    let pixel = 1.0 / vec2<f32>(s);
    let uv = (vec2<f32>(f32(i % s.x), f32(i / s.x)) + 0.5) * pixel;
    let k = dof_coc_sample(uv);
    let rad = clamp(max(k.x, k.y) * MAX_COC - 9.0, 0.0, 1.0);
    var v: array<vec3<f32>, 9>;
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            v[(dx + 1) * 3 + (dy + 1)] = dof_blur_sample(uv + vec2<f32>(f32(dx), f32(dy)) * pixel * rad);
        }
    }
    // mnmx6, mnmx5, mnmx4, mnmx3: the median is what is left in the middle.
    dof_swap(&v, 0, 3); dof_swap(&v, 1, 4); dof_swap(&v, 2, 5);
    dof_swap(&v, 0, 1); dof_swap(&v, 0, 2); dof_swap(&v, 4, 5); dof_swap(&v, 3, 5);
    dof_swap(&v, 1, 2); dof_swap(&v, 3, 4); dof_swap(&v, 1, 3); dof_swap(&v, 1, 6); dof_swap(&v, 4, 6); dof_swap(&v, 2, 6);
    dof_swap(&v, 2, 3); dof_swap(&v, 4, 7); dof_swap(&v, 2, 4); dof_swap(&v, 3, 7);
    dof_swap(&v, 4, 8); dof_swap(&v, 3, 8); dof_swap(&v, 3, 4);
    let r = dof_region(0u);
    px[r.x + 3u * i] = bitcast<u32>(v[4].x);
    px[r.x + 3u * i + 1u] = bitcast<u32>(v[4].y);
    px[r.x + 3u * i + 2u] = bitcast<u32>(v[4].z);
}

// Stage 6: the blur over the composite where the CoC is wide
// (`DRW_STATE_BLEND_CUSTOM`: `colour · (1 − blend) + blur · blend`), then
// the pixel finished as the composite finishes it.
fn dof_resolve(i: u32) {
    if (i >= p.n) { return; }
    let s = dof_size(0u);
    let uv = (vec2<f32>(f32(i % p.w), f32(i / p.w)) + 0.5) * 0.5 / vec2<f32>(s);
    let coc = dof_coc(z_at(i));
    let blend = smoothstep(1.0, 3.0, abs(coc));
    let lin = three(7u * p.n + 3u * i) * (1.0 - blend) + dof_color_sample(0u, uv) * blend;
    fx_finish(i, lin);
}

fn dof_stage(stage: u32, i: u32) {
    switch stage {
        case 1u: { dof_prepare(i); }
        case 2u: { dof_downsample(1u, i); }
        case 3u: { dof_downsample(2u, i); }
        case 4u: { dof_blur1(i); }
        case 5u: { dof_blur2(i); }
        default: { dof_resolve(i); }
    }
}
