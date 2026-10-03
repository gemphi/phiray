// Blender's colour management, shared by the tracer and the denoiser. The
// caller passes the `Camera::view` words rather than reading a uniform, so
// this file needs no particular params struct.

// View transforms (Blender's colour management, `config.ocio`), sampled
// from Blender 5.0: display-encoded grey over log2(scene value) in
// [-14.947861, 10.052138] (64 steps). Filmic is exactly this curve per
// channel; AgX applies it between an inset and an outset matrix (fitted to
// Blender's output: mean error 1.6/255).
const VT_LO: f32 = -14.947861;
const VT_HI: f32 = 10.052138;
var<private> FILMIC: array<f32, 64> = array<f32, 64>(0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.001109, 0.002798, 0.004849, 0.007337, 0.010324, 0.013937, 0.018276, 0.023498, 0.029783, 0.037296, 0.046297, 0.057015, 0.069747, 0.084803, 0.102511, 0.123240, 0.147317, 0.175060, 0.206725, 0.242476, 0.282333, 0.326138, 0.373576, 0.424074, 0.476835, 0.530496, 0.584086, 0.636811, 0.687598, 0.735542, 0.779871, 0.820052, 0.855760, 0.886888, 0.913539, 0.935939, 0.954360, 0.969213, 0.980878, 0.989775, 0.996255, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000);
var<private> AGX: array<f32, 64> = array<f32, 64>(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000015, 0.000094, 0.000307, 0.000714, 0.001401, 0.002474, 0.004091, 0.006433, 0.009787, 0.014516, 0.021097, 0.030173, 0.042574, 0.057306, 0.073883, 0.092617, 0.114256, 0.138886, 0.167071, 0.199246, 0.235968, 0.277732, 0.324850, 0.377462, 0.435243, 0.495499, 0.553566, 0.607904, 0.657973, 0.703691, 0.745144, 0.782605, 0.816304, 0.846638, 0.873680, 0.897952, 0.919880, 0.939710, 0.957668, 0.973975, 0.988673, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000);
const AGX_IN: mat3x3<f32> = mat3x3<f32>(0.498585, 0.125929, 0.097476, 0.346742, 0.681639, 0.191866, 0.097839, 0.123183, 0.678532);
const AGX_OUT: mat3x3<f32> = mat3x3<f32>(2.055217, -0.287805, -0.221530, -0.912124, 1.396234, -0.286374, -0.141522, -0.097287, 1.507346);

fn vt_curve(x: f32, agx: bool) -> f32 {
    let t = clamp((log2(max(x, 1e-10)) - VT_LO) / (VT_HI - VT_LO), 0.0, 1.0) * 63.0;
    let i = u32(floor(t));
    let j = min(i + 1u, 63u);
    let f = t - floor(t);
    if (agx) { return mix(AGX[i], AGX[j], f); }
    return mix(FILMIC[i], FILMIC[j], f);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

// Scene-linear radiance → display, as Blender's colour management does it:
// exposure, the view transform, then gamma.
fn view_transform_of(lin: vec3<f32>, view: vec4<f32>) -> vec3<f32> {
    let c = max(lin, vec3<f32>(0.0)) * exp2(view.y);
    var d: vec3<f32>;
    let kind = u32(view.x + 0.5);
    if (kind == 1u) {
        d = vec3<f32>(vt_curve(c.r, false), vt_curve(c.g, false), vt_curve(c.b, false));
    } else if (kind == 2u) {
        let i = AGX_IN * c;
        let y = vec3<f32>(vt_curve(i.r, true), vt_curve(i.g, true), vt_curve(i.b, true));
        d = srgb(AGX_OUT * srgb_to_linear(y));
    } else {
        d = srgb(c);
    }
    var g = view.z;
    if (g <= 0.0) { g = 1.0; }
    return pow(clamp(d, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / g));
}

// Blender's "Standard" view transform: scene-linear to sRGB display.
fn srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

fn pack(c: vec3<f32>, a: f32) -> u32 {
    let q = vec3<u32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0 + 0.5);
    let qa = u32(clamp(a, 0.0, 1.0) * 255.0 + 0.5);
    return q.r | (q.g << 8u) | (q.b << 16u) | (qa << 24u);
}
