// The pattern textures, Cycles' SVM ported: Checker (`kernel/svm/
// checker.h`), Brick (`brick.h`), Gradient (`gradient.h`), Magic
// (`magic.h`) and Wave (`wave.h`, its distortion `noise_fbm`).

// `NodeTexGradient.gradient_type` (`SHD_BLEND_*`).
const BLEND_LINEAR: u32 = 0u;
const BLEND_QUADRATIC: u32 = 1u;
const BLEND_EASING: u32 = 2u;
const BLEND_DIAGONAL: u32 = 3u;
const BLEND_RADIAL: u32 = 4u;
const BLEND_QUADRATIC_SPHERE: u32 = 5u;
const BLEND_SPHERICAL: u32 = 6u;
// `NodeTexWave`: type, bands / rings direction, profile.
const WAVE_BANDS: u32 = 0u;
const WAVE_RINGS: u32 = 1u;
const WAVE_DIRECTION_X: u32 = 0u;
const WAVE_DIRECTION_Y: u32 = 1u;
const WAVE_DIRECTION_Z: u32 = 2u;
const WAVE_DIRECTION_DIAGONAL: u32 = 3u;
const WAVE_PROFILE_SIN: u32 = 0u;
const WAVE_PROFILE_SAW: u32 = 1u;
const WAVE_PROFILE_TRI: u32 = 2u;

const TWO_PI: f32 = 6.283185307179586;

// `svm_checker`: 1 on one set of cells, 0 on the other.
fn checker_fac(p0: vec3<f32>) -> f32 {
    // Avoid precision issues on unit coordinates.
    let p = (p0 + 0.000001) * 0.999999;
    let c = abs(vec3<i32>(floor(p)));
    let xy = (c.x % 2) == (c.y % 2);
    let z = (c.z % 2) == 1;
    return select(0.0, 1.0, xy == z);
}

// `svm_node_tex_checker`: `.rgb` Color1 on the ones, Color2 on the
// zeros; `.w` the Fac.
fn checker_texture(co: vec3<f32>, color1: vec3<f32>, color2: vec3<f32>) -> vec4<f32> {
    let f = checker_fac(co);
    return vec4<f32>(select(color2, color1, f == 1.0), f);
}

// `brick_noise`: Blender's fast integer noise, in 0 ..= 1.
fn brick_noise(n0: u32) -> f32 {
    var n = (n0 + 1013u) & 0x7fffffffu;
    n = (n >> 13u) ^ n;
    let nn = (n * (n * n * 60493u + 19990303u) + 1376312589u) & 0x7fffffffu;
    return 0.5 * (f32(nn) / 1073741824.0);
}

// `svm_brick`: `.x` the brick's tint, `.y` how much mortar the point is.
fn brick_cell(p: vec3<f32>, mortar_size: f32, mortar_smooth: f32, bias: f32, brick_width0: f32, row_height: f32, offset_amount: f32, offset_frequency: i32, squash_amount: f32, squash_frequency: i32) -> vec2<f32> {
    var brick_width = brick_width0;
    var offset = 0.0;
    let rownum = i32(floor(p.y / row_height));
    if (offset_frequency != 0 && squash_frequency != 0) {
        brick_width *= select(squash_amount, 1.0, (rownum % squash_frequency) != 0);
        offset = select(brick_width * offset_amount, 0.0, (rownum % offset_frequency) != 0);
    }
    let bricknum = i32(floor((p.x + offset) / brick_width));
    let x = (p.x + offset) - brick_width * f32(bricknum);
    let y = p.y - row_height * f32(rownum);
    let tint = clamp(brick_noise((u32(rownum) << 16u) + (u32(bricknum) & 0xFFFFu)) + bias, 0.0, 1.0);
    var min_dist = min(min(x, y), min(brick_width - x, row_height - y));
    var mortar = 0.0;
    if (min_dist >= mortar_size) {
        mortar = 0.0;
    } else if (mortar_smooth == 0.0) {
        mortar = 1.0;
    } else {
        min_dist = 1.0 - min_dist / mortar_size;
        mortar = smoothstep(0.0, 1.0, min_dist / mortar_smooth);
    }
    return vec2<f32>(tint, mortar);
}

// `svm_node_tex_brick`: `.rgb` the Color, `.w` the Fac (the mortar).
fn brick_texture(co: vec3<f32>, color1: vec3<f32>, color2: vec3<f32>, mortar: vec3<f32>, mortar_size: f32, mortar_smooth: f32, bias: f32, brick_width: f32, row_height: f32, offset_amount: f32, offset_frequency: i32, squash_amount: f32, squash_frequency: i32) -> vec4<f32> {
    let f2 = brick_cell(co, mortar_size, mortar_smooth, bias, brick_width, row_height, offset_amount, offset_frequency, squash_amount, squash_frequency);
    let tint = f2.x;
    let f = f2.y;
    var c1 = color1;
    if (f != 1.0) { c1 = (1.0 - tint) * color1 + tint * color2; }
    return vec4<f32>(c1 * (1.0 - f) + mortar * f, f);
}

// `svm_gradient`, saturated as the node does.
fn gradient_fac(p: vec3<f32>, kind: u32) -> f32 {
    var f = 0.0;
    switch kind {
        case BLEND_LINEAR: { f = p.x; }
        case BLEND_QUADRATIC: { let r = max(p.x, 0.0); f = r * r; }
        case BLEND_EASING: { let r = clamp(p.x, 0.0, 1.0); let t = r * r; f = 3.0 * t - 2.0 * t * r; }
        case BLEND_DIAGONAL: { f = (p.x + p.y) * 0.5; }
        case BLEND_RADIAL: { f = atan2(p.y, p.x) / TWO_PI + 0.5; }
        default: {
            // Biased a little so a unit vector gives exactly zero.
            let r = max(0.999999 - length(p), 0.0);
            f = select(r, r * r, kind == BLEND_QUADRATIC_SPHERE);
        }
    }
    return clamp(f, 0.0, 1.0);
}

// One of the Magic texture's distortion rounds (`svm_magic`'s `n > k`
// steps): `s` the three running values, `k` the round.
fn magic_round(s: vec3<f32>, k: i32, distortion: f32) -> vec3<f32> {
    var x = s.x; var y = s.y; var z = s.z;
    switch k {
        case 0: { y = -cos(x - y + z) * distortion; }
        case 1: { x = cos(x - y - z) * distortion; }
        case 2: { z = sin(-x - y - z) * distortion; }
        case 3: { x = -cos(-x + y - z) * distortion; }
        case 4: { y = -sin(-x + y + z) * distortion; }
        case 5: { y = -cos(-x + y + z) * distortion; }
        case 6: { x = cos(x + y + z) * distortion; }
        case 7: { z = sin(x + y - z) * distortion; }
        case 8: { x = -cos(-x - y + z) * distortion; }
        default: { y = -sin(x - y + z) * distortion; }
    }
    return vec3<f32>(x, y, z);
}

// `svm_magic`: `.rgb` the Color, `.w` the Fac (its mean).
fn magic_texture(p: vec3<f32>, scale: f32, depth: i32, distortion0: f32) -> vec4<f32> {
    // Periodic inputs keep the sums in float range (Cycles' note).
    let px = (p.x * scale) % TWO_PI;
    let py = (p.y * scale) % TWO_PI;
    let pz = (p.z * scale) % TWO_PI;
    var s = vec3<f32>(sin((px + py + pz) * 5.0), cos((-px + py - pz) * 5.0), -cos((-px - py + pz) * 5.0));
    var distortion = distortion0;
    if (depth > 0) {
        s *= distortion;
        for (var k = 0; k < min(depth, 10); k++) { s = magic_round(s, k, distortion); }
    }
    if (distortion != 0.0) {
        distortion *= 2.0;
        s /= distortion;
    }
    let c = vec3<f32>(0.5) - s;
    return vec4<f32>(c, (c.x + c.y + c.z) / 3.0);
}

// `svm_wave`: bands or rings along a direction, a sine, saw or triangle
// profile, distorted by an fBm.
fn wave_fac(kind: u32, bands_dir: u32, rings_dir: u32, profile: u32, p0: vec3<f32>, distortion: f32, detail: f32, dscale: f32, droughness: f32, phase: f32) -> f32 {
    let p = (p0 + 0.000001) * 0.999999;
    var n = 0.0;
    if (kind == WAVE_BANDS) {
        switch bands_dir {
            case WAVE_DIRECTION_X: { n = p.x * 20.0; }
            case WAVE_DIRECTION_Y: { n = p.y * 20.0; }
            case WAVE_DIRECTION_Z: { n = p.z * 20.0; }
            default: { n = (p.x + p.y + p.z) * 10.0; }
        }
    } else {
        var rp = p;
        switch rings_dir {
            case WAVE_DIRECTION_X: { rp *= vec3<f32>(0.0, 1.0, 1.0); }
            case WAVE_DIRECTION_Y: { rp *= vec3<f32>(1.0, 0.0, 1.0); }
            case WAVE_DIRECTION_Z: { rp *= vec3<f32>(1.0, 1.0, 0.0); }
            default: {}
        }
        n = length(rp) * 20.0;
    }
    n += phase;
    if (distortion != 0.0) {
        n += distortion * (noise_fbm(p * dscale, detail, droughness, 2.0, true) * 2.0 - 1.0);
    }
    if (profile == WAVE_PROFILE_SIN) { return 0.5 + 0.5 * sin(n - TWO_PI * 0.25); }
    n /= TWO_PI;
    if (profile == WAVE_PROFILE_SAW) { return n - floor(n); }
    return abs(n - floor(n + 0.5)) * 2.0;
}
