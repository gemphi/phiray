// Bloom / Glare (Blender's Eevee "Bloom" and the compositor's Glare "Fog
// Glow"): bright parts of the scene-linear image bleed into their
// surroundings. It runs on the SETTLED frame, on the scene-linear radiance
// mean — never on the display pixels — because glare is a scene-linear
// effect applied BEFORE the view transform, exactly as Blender runs the
// compositor on the render's linear pixels and the view transform after
// (`show.wgsl` says the same for grading). Compositing after AgX would
// bloom the wrong, clamped values.
//
// Four stages over one `Camera::slice.x`, a separable Gaussian like
// `bloom.wgsl` in the appkit reference, re-expressed as compute kernels
// over phiray's flat pixel buffer (that shader is fragment + sampler; this
// one reads neighbours by index):
//   0 threshold  — bright pass of the radiance mean into `glow` buffer A
//   1 blur-x     — A → B, 9-tap Gaussian across
//   2 blur-y     — B → A, 9-tap Gaussian down
//   3 composite  — radiance mean + A·intensity, then the view transform and
//                  the pack, replacing the display pixel `emit`/`show` wrote.
//
// `px` is the tracer's image (scene-linear radiance mean at `n + 3i`,
// display at `4n + 3i`, packed at `i`). `glow` is a scratch buffer of
// `2·3n` floats: buffer A at `0`, buffer B at `3n`. `p.fx` carries
// `[threshold, intensity, radius_px, on]` (`Tracer::glow`).

@group(0) @binding(0) var<uniform> p: Camera;
@group(0) @binding(1) var<storage, read_write> px: array<u32>;
@group(0) @binding(2) var<storage, read_write> glow: array<u32>;

// 9-tap Gaussian, sigma ~ 2 (the weights `bloom.wgsl` uses).
const W0: f32 = 0.227027;
const W1: f32 = 0.1945946;
const W2: f32 = 0.1216216;
const W3: f32 = 0.054054;
const W4: f32 = 0.016216;

// Read / write a triple in the glare scratch buffer. WGSL cannot portably
// take a pointer-to-storage-array as a function parameter (naga rejects it),
// so these name the `glow` global directly rather than being handed a buffer.
fn glow_rgb(a: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(glow[a]), bitcast<f32>(glow[a + 1u]), bitcast<f32>(glow[a + 2u]));
}

fn glow_put(a: u32, v: vec3<f32>) {
    glow[a] = bitcast<u32>(v.x);
    glow[a + 1u] = bitcast<u32>(v.y);
    glow[a + 2u] = bitcast<u32>(v.z);
}

// The scene-linear radiance mean of pixel `i` — the same quantity
// `show.wgsl` reads, divided by the samples that pixel actually took.
fn radiance_mean(i: u32) -> vec3<f32> {
    let k = f32(max(px[31u * p.n + i], 1u));
    let gone = bitcast<f32>(px[33u * p.n + i]);
    let l = vec3<f32>(bitcast<f32>(px[p.n + 3u * i]), bitcast<f32>(px[p.n + 3u * i + 1u]), bitcast<f32>(px[p.n + 3u * i + 2u]));
    return l / max(k - gone, 1.0);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Clamp a pixel coordinate to the image (edge-extend, as a sampler would).
fn at(x: i32, y: i32) -> u32 {
    let cx = u32(clamp(x, 0, i32(p.w) - 1));
    let cy = u32(clamp(y, 0, i32(p.h) - 1));
    return cy * p.w + cx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    if (p.fx.w == 0.0) { return; }

    let stage = p.slice.x;
    let x = i32(i % p.w);
    let y = i32(i / p.w);
    // The blur step in texels: wider radius spreads the glare further.
    let step = max(1, i32(round(p.fx.z)));
    let a = 3u * i;            // buffer A slot for this pixel
    let bslot = 3u * p.n + a;  // buffer B slot

    if (stage == 0u) {
        // Bright pass with a soft knee: everything above the threshold,
        // scaled by how far over it the luma is (`bloom.wgsl`).
        let c = radiance_mean(i);
        let l = luma(c);
        let contrib = max(0.0, l - p.fx.x) / max(l, 1e-4);
        glow_put(a, c * contrib);
        return;
    }
    if (stage == 1u) {
        // Horizontal Gaussian: A → B.
        var sum = glow_rgb(a) * W0;
        let w = array<f32, 4>(W1, W2, W3, W4);
        for (var k = 1; k <= 4; k = k + 1) {
            let o = k * step;
            sum += glow_rgb(3u * at(x + o, y)) * w[k - 1];
            sum += glow_rgb(3u * at(x - o, y)) * w[k - 1];
        }
        glow_put(bslot, sum);
        return;
    }
    if (stage == 2u) {
        // Vertical Gaussian: B → A.
        var sum = glow_rgb(bslot) * W0;
        let w = array<f32, 4>(W1, W2, W3, W4);
        for (var k = 1; k <= 4; k = k + 1) {
            let o = k * step;
            sum += glow_rgb(3u * p.n + 3u * at(x, y + o)) * w[k - 1];
            sum += glow_rgb(3u * p.n + 3u * at(x, y - o)) * w[k - 1];
        }
        glow_put(a, sum);
        return;
    }
    // stage == 3: composite the blurred glare onto the radiance mean, in
    // scene-linear, then run the same display step `show.wgsl` runs so the
    // pixel shown and read back both carry the glare.
    let bloom = glow_rgb(a);
    let lin = radiance_mean(i) + bloom * p.fx.y;
    // The overlays and coverage `emit` composited, kept as they were.
    let k = f32(max(px[31u * p.n + i], 1u));
    let gs = vec3<f32>(bitcast<f32>(px[20u * p.n + 3u * i]), bitcast<f32>(px[20u * p.n + 3u * i + 1u]), bitcast<f32>(px[20u * p.n + 3u * i + 2u]));
    let cov = bitcast<f32>(px[23u * p.n + i]);
    let film = bitcast<f32>(px[24u * p.n + i]);
    let shown = view_transform_of(lin, p.view) * (1.0 - cov / k) + gs / k;
    let b = 4u * p.n + 3u * i;
    px[b] = bitcast<u32>(shown.x);
    px[b + 1u] = bitcast<u32>(shown.y);
    px[b + 2u] = bitcast<u32>(shown.z);
    px[i] = pack(shown, film / k);
}
