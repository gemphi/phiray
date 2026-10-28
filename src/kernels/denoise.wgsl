fn f_at(i: u32) -> f32 { return bitcast<f32>(px[i]); }

fn v3_at(base: u32, i: u32) -> vec3<f32> {
    let o = base + 3u * i;
    return vec3<f32>(f_at(o), f_at(o + 1u), f_at(o + 2u));
}

fn put3(base: u32, i: u32, v: vec3<f32>) {
    let o = base + 3u * i;
    px[o] = bitcast<u32>(v.x);
    px[o + 1u] = bitcast<u32>(v.y);
    px[o + 2u] = bitcast<u32>(v.z);
}

// Means over the samples THIS pixel took (`util.wgsl`'s `adapt_count`, the
// word `emit` keeps at `31n + i`). With Adaptive Sampling a settled pixel
// stops early, so the frame's count (`strength.w`, 1 / samples) is not its
// divisor: dividing a pixel that settled at 40 of 256 by 256 darkened it
// sixfold, and the filter spread those dark pixels into blotches over the
// whole frame — the broken 256-sample pose 23 of 2026-10-20.
fn inv_n(i: u32) -> f32 { return 1.0 / f32(max(px[31u * p.n + i], 1u)); }
//
// Everything here is display-space, because that is what the tracer averages
// (`util.wgsl`'s `emit`) and therefore what `pixels` shows: the mean of the
// transformed samples, not the transform of the mean. Filtering scene-linear
// values and transforming once at the end would be a different image
// altogether — brighter wherever one sample was very bright.
fn albedo_of(i: u32) -> vec3<f32> { return srgb(v3_at(7u * p.n, i) * inv_n(i)); }
fn normal_of(i: u32) -> vec3<f32> { return normalize(v3_at(10u * p.n, i) + vec3<f32>(1e-9)); }
fn depth_of(i: u32) -> f32 { return f_at(13u * p.n + i) * inv_n(i); }

// Shown radiance with the albedo divided out, so what is filtered is the
// lighting and not the texture printed on it.
fn demodulated(i: u32) -> vec3<f32> {
    return v3_at(1u * p.n, i) * inv_n(i) / max(albedo_of(i), vec3<f32>(0.05));
}

fn read_in(i: u32) -> vec3<f32> {
    if (p.round == 0u) { return demodulated(i); }
    return v3_at(select(17u, 14u, (p.round & 1u) == 1u) * p.n, i);
}

const KERNEL: array<f32, 3> = array<f32, 3>(0.375, 0.25, 0.0625);

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let x = i32(i % p.w);
    let y = i32(i / p.w);
    let centre = read_in(i);
    let cn = normal_of(i);
    let ca = albedo_of(i);
    let cd = depth_of(i);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let sx = x + dx * i32(p.step);
            let sy = y + dy * i32(p.step);
            if (sx < 0 || sy < 0 || sx >= i32(p.w) || sy >= i32(p.h)) { continue; }
            let j = u32(sy) * p.w + u32(sx);
            // A ray that met nothing and one that hit a surface never share a
            // neighbourhood, whatever their other channels say.
            if ((depth_of(j) < 0.0) != (cd < 0.0)) { continue; }
            let tap = read_in(j);
            // Edge stops: how far this tap's guides are from the centre's.
            let dn = max(0.0, 1.0 - dot(cn, normal_of(j)));
            let da = length(albedo_of(j) - ca);
            let dd = abs(depth_of(j) - cd) / max(abs(cd), 1e-3);
            let stop = exp(-dn * p.strength.x - da * p.strength.y - dd * p.strength.z);
            let k = KERNEL[abs(dx)] * KERNEL[abs(dy)];
            sum += tap * k * stop;
            weight += k * stop;
        }
    }
    let out = select(centre, sum / weight, weight > 0.0);
    if (p.last == 0u) {
        put3(select(14u, 17u, (p.round & 1u) == 1u) * p.n, i, out);
        return;
    }
    let k = inv_n(i);
    // The share of this pixel's samples that met something, as `emit`
    // counts it: `k - gone` samples, at least one's worth. A pixel none of
    // whose samples met anything has no light, whatever its taps lent it —
    // divided by a share of nothing, the surface light it borrowed came out
    // a million times over, the pastel halo round every figure on a
    // transparent film (LOOK-01, 2026-10-27).
    let met = 1.0 - f_at(33u * p.n + i) * k;
    // Put the texture back: this is the filtered scene-linear radiance.
    let lin = select(vec3<f32>(0.0), out * max(albedo_of(i), vec3<f32>(0.05)), met > 0.5 * k);
    // A finished render's it replaces the linear sums, so `Tracer::radiance`
    // — and an EXR — get the denoised image too; a refining viewport
    // (`p.keep`) keeps its sums, so the next sample lands on the truth.
    // Back to a sum over this pixel's own samples.
    if (p.keep == 0u) { put3(1u * p.n, i, lin / k); }
    // Then the view transform once, and the grid overlay the tracer
    // accumulated, exactly as `emit` composes an un-denoised pixel.
    let cov = f_at(23u * p.n + i) * k;
    let grid = v3_at(20u * p.n, i) * k;
    // `lin` is a mean over every sample; the light is over the ones that
    // met something (`emit`), so it is scaled up by the share that did.
    let shown = view_transform_of(lin / max(met, k), p.view) * (1.0 - cov) + grid;
    put3(4u * p.n, i, shown);
    px[i] = pack(shown, f_at(24u * p.n + i) * k);
}
