// Adaptive Sampling's convergence check — a direct port of Cycles'
// `film_adaptive_sampling_convergence_check`
// (`intern/cycles/kernel/film/adaptive_sampling.h`).
//
// `I` is the full image's running sum, `A` the half image's
// (`util.wgsl`'s `emit`). Both estimate the same mean, so scaled by the
// sample count their disagreement is the pixel's error. Normalising by
// intensity — by its square root below 1 — is Cycles' exposure rule: a
// bright pixel tolerates more absolute error than a dark one, because the
// eye and the view transform both compress it.
//
// Runs as a tail step every sample and returns at once unless this is a
// step boundary past the minimum (`AdaptiveSampling::need_filter`), so the
// pass is recorded once and never re-recorded for it.
//
// `p.adapt`'s stage (`Camera::slice.z`) picks the step:
//   0  the convergence check itself, writing the RAW flag
//   1  the dilation, reading the raw flags of the four neighbours and
//      writing the flag the tracer actually reads
//
// The dilation is Cycles' `film_adaptive_sampling_filter_x` / `_y`. Cycles
// runs them as sequential scans along each row and then each column; here
// they are one parallel pass over the four-neighbourhood, which widens the
// unconverged set by the same one pixel each way. It needs two words —
// raw and final — because a parallel pass that read and wrote the same
// word would race with itself.
//
// Without this a pixel that merely LOOKS settled between two samples
// freezes at a noisy value while its neighbours keep refining, and the
// frame comes out salt-and-peppered. That is exactly what happened when
// this was left out.

// `camera.wgsl` defines the struct; the uniform itself is declared in
// `layout.wgsl`, which this kernel does not include — so it declares it
// here, exactly as `taa.wgsl` does.
@group(0) @binding(0) var<uniform> p: Camera;
@group(0) @binding(1) var<storage, read_write> px: array<u32>;

fn adapt_on() -> bool { return p.adapt.w != 0.0; }
fn adapt_aux(i: u32) -> u32 { return 26u * p.n + 3u * i; }
fn adapt_flag(i: u32) -> u32 { return 29u * p.n + i; }
fn adapt_count(i: u32) -> u32 { return 31u * p.n + i; }
fn adapt_seen(i: u32) -> u32 { return 30u * p.n + i; }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    if (!adapt_on()) { return; }

    let sample = p.frame;
    let min_samples = u32(p.adapt.y);
    let step = max(u32(p.adapt.z), 1u);
    // `need_filter`: nothing before the minimum is past, then the last
    // sample of each step.
    if (sample <= min_samples || (sample & (step - 1u)) != step - 1u) { return; }

    let flag = adapt_flag(i);
    // Already settled: its state cannot change here.
    if (px[flag] != 0u) { return; }

    let a = p.n + 3u * i;
    let x = adapt_aux(i);
    let full = vec3<f32>(bitcast<f32>(px[a]), bitcast<f32>(px[a + 1u]), bitcast<f32>(px[a + 2u]));
    let half_img = vec3<f32>(bitcast<f32>(px[x]), bitcast<f32>(px[x + 1u]), bitcast<f32>(px[x + 2u]));

    // Cycles divides by the PIXEL's sample count (`pass_sample_count`),
    // not by the frame number — with adaptive sampling they part company
    // as soon as the first pixel settles.
    let taken = max(px[adapt_count(i)], 1u);
    let scale = 1.0 / f32(taken);
    let d = abs(full - half_img);
    let error_difference = (d.x + d.y + d.z) * scale;
    let intensity = (full.x + full.y + full.z) * scale;
    var error_normalize = intensity;
    if (intensity < 1.0) { error_normalize = sqrt(intensity); }
    let error = error_difference / (0.0001 + error_normalize);
    if (error < p.adapt.x) { px[flag] = 1u; }
}

// Stage 1: a pixel counts as settled only if it and all four of its
// neighbours are. One pass, read from `adapt_flag`, write to `adapt_seen`.
@compute @workgroup_size(256)
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    if (!adapt_on()) { return; }
    let x = i % p.w;
    let y = i / p.w;
    var all = px[adapt_flag(i)];
    if (x > 0u) { all = min(all, px[adapt_flag(i - 1u)]); }
    if (x + 1u < p.w) { all = min(all, px[adapt_flag(i + 1u)]); }
    if (y > 0u) { all = min(all, px[adapt_flag(i - p.w)]); }
    if (y + 1u < p.h) { all = min(all, px[adapt_flag(i + p.w)]); }
    px[adapt_seen(i)] = all;
}
