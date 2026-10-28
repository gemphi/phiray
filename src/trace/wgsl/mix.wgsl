// Blender's Mix node over two colours, per pixel: `ramp_blend`
// (`blenkernel/intern/material.cc`, what Cycles' `svm_mix` compiles to)
// with every `MA_RAMP_*` mode, so a material whose base colour is two
// procedural textures through a Mix is shaded as the mix and not as one
// of them (`procedural.wgsl`'s second colour slot).
//
// Only MIX touches alpha, which this does not carry: the slots are RGB.
// The HSV modes are no-ops where the colour they take their channel from
// is grey, as Blender guards them.

const RAMP_BLEND: u32 = 0u;
const RAMP_ADD: u32 = 1u;
const RAMP_MULT: u32 = 2u;
const RAMP_SUB: u32 = 3u;
const RAMP_SCREEN: u32 = 4u;
const RAMP_DIV: u32 = 5u;
const RAMP_DIFF: u32 = 6u;
const RAMP_DARK: u32 = 7u;
const RAMP_LIGHT: u32 = 8u;
const RAMP_OVERLAY: u32 = 9u;
const RAMP_DODGE: u32 = 10u;
const RAMP_BURN: u32 = 11u;
const RAMP_HUE: u32 = 12u;
const RAMP_SAT: u32 = 13u;
const RAMP_VAL: u32 = 14u;
const RAMP_COLOR: u32 = 15u;
const RAMP_SOFT: u32 = 16u;
const RAMP_LINEAR: u32 = 17u;
const RAMP_EXCLUSION: u32 = 18u;

// `rgb_to_hsv` (`math_color.c`): hue and saturation in 0 ..= 1, value the
// largest channel.
fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    var h = 0.0;
    if (d != 0.0) {
        if (mx == c.r) {
            h = fract(((c.g - c.b) / d) / 6.0);
        } else if (mx == c.g) {
            h = (((c.b - c.r) / d) + 2.0) / 6.0;
        } else {
            h = (((c.r - c.g) / d) + 4.0) / 6.0;
        }
    }
    return vec3<f32>(h, select(0.0, d / mx, mx != 0.0), mx);
}

// `hsv_to_rgb`.
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let n = vec3<f32>(5.0, 3.0, 1.0);
    let k = (n + hsv.x * 6.0) % 6.0;
    return hsv.z - hsv.z * hsv.y * clamp(min(k, 4.0 - k), vec3<f32>(0.0), vec3<f32>(1.0));
}

// `ramp_blend`: `col` blended over `r_col` by `fac`.
fn ramp_blend(mode: u32, fac: f32, r_col: vec3<f32>, col: vec3<f32>) -> vec3<f32> {
    let facm = 1.0 - fac;
    let one = vec3<f32>(1.0);
    switch mode {
        case RAMP_ADD: { return r_col + fac * col; }
        case RAMP_MULT: { return r_col * (facm + fac * col); }
        case RAMP_SUB: { return r_col - fac * col; }
        case RAMP_SCREEN: { return one - (facm + fac * (one - col)) * (one - r_col); }
        case RAMP_DIV: {
            // A zero divisor leaves the channel as it was, as Blender does.
            let nz = col != vec3<f32>(0.0);
            let d = facm * r_col + fac * r_col / select(one, col, nz);
            return select(r_col, d, nz);
        }
        case RAMP_DIFF: { return facm * r_col + fac * abs(r_col - col); }
        case RAMP_DARK: { return min(r_col, col) * fac + r_col * facm; }
        case RAMP_LIGHT: { return max(r_col, col) * fac + r_col * facm; }
        case RAMP_OVERLAY: {
            let lo = r_col * (facm + 2.0 * fac * col);
            let hi = one - (facm + 2.0 * fac * (one - col)) * (one - r_col);
            return select(hi, lo, r_col < vec3<f32>(0.5));
        }
        case RAMP_DODGE: {
            let tmp = one - fac * col;
            let d = select(min(r_col / max(tmp, vec3<f32>(1e-12)), one), one, tmp <= vec3<f32>(0.0));
            return select(r_col, d, r_col != vec3<f32>(0.0));
        }
        case RAMP_BURN: {
            let tmp = facm + fac * col;
            let b = clamp(one - (one - r_col) / max(tmp, vec3<f32>(1e-12)), vec3<f32>(0.0), one);
            return select(b, vec3<f32>(0.0), tmp <= vec3<f32>(0.0));
        }
        case RAMP_HUE: {
            let c = rgb_to_hsv(col);
            if (c.y == 0.0) { return r_col; }
            let r = rgb_to_hsv(r_col);
            return facm * r_col + fac * hsv_to_rgb(vec3<f32>(c.x, r.y, r.z));
        }
        case RAMP_SAT: {
            let r = rgb_to_hsv(r_col);
            if (r.y == 0.0) { return r_col; }
            let c = rgb_to_hsv(col);
            return hsv_to_rgb(vec3<f32>(r.x, facm * r.y + fac * c.y, r.z));
        }
        case RAMP_VAL: {
            let r = rgb_to_hsv(r_col);
            let c = rgb_to_hsv(col);
            return hsv_to_rgb(vec3<f32>(r.x, r.y, facm * r.z + fac * c.z));
        }
        case RAMP_COLOR: {
            let c = rgb_to_hsv(col);
            if (c.y == 0.0) { return r_col; }
            let r = rgb_to_hsv(r_col);
            return facm * r_col + fac * hsv_to_rgb(vec3<f32>(c.x, c.y, r.z));
        }
        case RAMP_SOFT: {
            let sc = one - (one - col) * (one - r_col);
            return facm * r_col + fac * ((one - r_col) * col * r_col + r_col * sc);
        }
        case RAMP_LINEAR: {
            // Blender splits on the *blend* colour: above a half it
            // lightens by 2(col − 0.5), below it darkens by 2·col − 1.
            return r_col + fac * select(2.0 * col - one, 2.0 * (col - vec3<f32>(0.5)), col > vec3<f32>(0.5));
        }
        case RAMP_EXCLUSION: { return max(facm * r_col + fac * (r_col + col - 2.0 * r_col * col), vec3<f32>(0.0)); }
        // RAMP_BLEND, and any mode a newer Blender adds.
        default: { return facm * r_col + fac * col; }
    }
}
