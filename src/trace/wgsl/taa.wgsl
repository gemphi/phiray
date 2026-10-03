// Workbench's anti-aliasing gather (`workbench_effect_taa_frag.glsl` and
// the resolve in `workbench_effect_smaa_frag.glsl`): the sample the Solid
// or Wireframe kernel just drew, kept per pixel in the image's scratch
// words (`taa_sample`, `util.wgsl`), is gathered over the 3 × 3
// neighbourhood with Blackman-Harris weights for this sample's jitter
// (`p.taa.xy`, `setup_taa_weights`) in `log2(colour + 1)` space — so a
// highlight does not alias — into the running sums (frame 0 starts them),
// and the shown pixel is the sums over the weight accumulated so far
// (`p.taa.z`), back out of log space. The first frame's SMAA pass is not
// here yet.

@group(0) @binding(0) var<uniform> p: Camera;
@group(0) @binding(1) var<storage, read_write> px: array<u32>;

const FILTER_WIDTH: f32 = 2.0;

fn blackman_harris(x: f32) -> f32 {
    if (x > FILTER_WIDTH * 0.5) { return 0.0; }
    let t = 6.2831853 * clamp(x / FILTER_WIDTH + 0.5, 0.0, 1.0);
    return 0.35875 - 0.48829 * cos(t) + 0.14128 * cos(2.0 * t) - 0.01168 * cos(3.0 * t);
}

fn sample_at(x: i32, y: i32) -> vec3<f32> {
    let cx = u32(clamp(x, 0, i32(p.w) - 1));
    let cy = u32(clamp(y, 0, i32(p.h) - 1));
    let a = 14u * p.n + 3u * (cy * p.w + cx);
    return vec3<f32>(bitcast<f32>(px[a]), bitcast<f32>(px[a + 1u]), bitcast<f32>(px[a + 2u]));
}

fn pack(c: vec3<f32>, a: f32) -> u32 {
    let q = vec3<u32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0 + 0.5);
    let qa = u32(clamp(a, 0.0, 1.0) * 255.0 + 0.5);
    return q.r | (q.g << 8u) | (q.b << 16u) | (qa << 24u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let x = i32(i % p.w);
    let y = i32(i / p.w);
    var gathered = vec3<f32>(0.0);
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            let co = vec2<f32>(f32(dx), f32(dy)) - p.taa.xy;
            let w = blackman_harris(length(co));
            // Clamp infinite inputs, then log2 space (#112211).
            let c = clamp(sample_at(x + dx, y + dy), vec3<f32>(0.0), vec3<f32>(1e10));
            gathered += log2(c + 1.0) * w;
        }
    }
    let s = p.n + 3u * i;
    let kept = select(vec3<f32>(0.0), vec3<f32>(bitcast<f32>(px[s]), bitcast<f32>(px[s + 1u]), bitcast<f32>(px[s + 2u])), p.frame > 0u);
    let sum = kept + gathered;
    px[s] = bitcast<u32>(sum.x);
    px[s + 1u] = bitcast<u32>(sum.y);
    px[s + 2u] = bitcast<u32>(sum.z);
    let shown = exp2(sum / max(p.taa.z, 1e-6)) - 1.0;
    px[i] = pack(shown, 1.0);
}
