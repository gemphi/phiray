// The Math node, Cycles' `svm_math` ported whole (`svm/math_util.h`, with
// its helpers from `util/math_base.h`): all 41 of `NodeMathOperation`,
// over the node's three Value sockets and its Clamp checkbox.
//
// A texture slot carries one of these as the operation between the texture
// and the socket it drives (`procedural.wgsl`'s `MATH_NONE`), which is the
// commonest node to find in that position and the one that, unhandled,
// made the texture vanish from the material entirely.

// `NodeMathOperation` (`DNA_node_types.h`). `MATH_NONE` is ours: no node.
const MATH_NONE: u32 = 0xffffffffu;
const MATH_ADD: u32 = 0u;
const MATH_SUBTRACT: u32 = 1u;
const MATH_MULTIPLY: u32 = 2u;
const MATH_DIVIDE: u32 = 3u;
const MATH_SINE: u32 = 4u;
const MATH_COSINE: u32 = 5u;
const MATH_TANGENT: u32 = 6u;
const MATH_ARCSINE: u32 = 7u;
const MATH_ARCCOSINE: u32 = 8u;
const MATH_ARCTANGENT: u32 = 9u;
const MATH_POWER: u32 = 10u;
const MATH_LOGARITHM: u32 = 11u;
const MATH_MINIMUM: u32 = 12u;
const MATH_MAXIMUM: u32 = 13u;
const MATH_ROUND: u32 = 14u;
const MATH_LESS_THAN: u32 = 15u;
const MATH_GREATER_THAN: u32 = 16u;
const MATH_MODULO: u32 = 17u;
const MATH_ABSOLUTE: u32 = 18u;
const MATH_ARCTAN2: u32 = 19u;
const MATH_FLOOR: u32 = 20u;
const MATH_CEIL: u32 = 21u;
const MATH_FRACTION: u32 = 22u;
const MATH_SQRT: u32 = 23u;
const MATH_INV_SQRT: u32 = 24u;
const MATH_SIGN: u32 = 25u;
const MATH_EXPONENT: u32 = 26u;
const MATH_RADIANS: u32 = 27u;
const MATH_DEGREES: u32 = 28u;
const MATH_SINH: u32 = 29u;
const MATH_COSH: u32 = 30u;
const MATH_TANH: u32 = 31u;
const MATH_TRUNC: u32 = 32u;
const MATH_SNAP: u32 = 33u;
const MATH_WRAP: u32 = 34u;
const MATH_COMPARE: u32 = 35u;
const MATH_MULTIPLY_ADD: u32 = 36u;
const MATH_PINGPONG: u32 = 37u;
const MATH_SMOOTH_MIN: u32 = 38u;
const MATH_SMOOTH_MAX: u32 = 39u;
const MATH_FLOORED_MODULO: u32 = 40u;

// `util/math_base.h`'s guards. Each returns zero where the operation has
// no answer, which is what Cycles does rather than carry a NaN.
fn safe_divide(a: f32, b: f32) -> f32 { return select(0.0, a / b, b != 0.0); }
fn safe_sqrt(a: f32) -> f32 { return sqrt(max(a, 0.0)); }
fn safe_inv_sqrt(a: f32) -> f32 { return select(0.0, 1.0 / sqrt(a), a > 0.0); }
fn safe_log(a: f32, b: f32) -> f32 { return select(safe_divide(log(a), log(b)), 0.0, a <= 0.0 || b <= 0.0); }
fn safe_modulo(a: f32, b: f32) -> f32 { return select(0.0, a - trunc(a / b) * b, b != 0.0); }
fn safe_floored_modulo(a: f32, b: f32) -> f32 { return select(0.0, a - floor(a / b) * b, b != 0.0); }
fn compatible_sign(f: f32) -> f32 { return select(select(-1.0, 1.0, f > 0.0), 0.0, f == 0.0); }
fn compatible_atan2(y: f32, x: f32) -> f32 { return select(atan2(y, x), 0.0, x == 0.0 && y == 0.0); }

// `safe_powf` over `compatible_powf`: a negative base with a fractional
// exponent has no real answer, `x^0` is 1 including `0^0`, and a negative
// base with a whole exponent keeps its sign.
fn safe_pow(a: f32, b: f32) -> f32 {
    if (b == 0.0) { return 1.0; }
    if (a == 0.0) { return 0.0; }
    if (a > 0.0) { return pow(a, b); }
    if (b != trunc(b)) { return 0.0; }
    let m = pow(-a, b);
    return select(m, -m, safe_modulo(-b, 2.0) != 0.0);
}

fn math_wrap(value: f32, hi: f32, lo: f32) -> f32 {
    let range = hi - lo;
    return select(lo, value - range * floor((value - lo) / range), range != 0.0);
}

fn math_pingpong(a: f32, b: f32) -> f32 {
    return select(0.0, abs(fract((a - b) / (b * 2.0)) * b * 2.0 - b), b != 0.0);
}

fn math_smooth_min(a: f32, b: f32, k: f32) -> f32 {
    if (k == 0.0) { return min(a, b); }
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * h * k * (1.0 / 6.0);
}

// `svm_math`: `a` is the texture's value, `b` and `c` the node's other two
// sockets.
fn svm_math(kind: u32, a: f32, b: f32, c: f32) -> f32 {
    switch kind {
        case MATH_ADD: { return a + b; }
        case MATH_SUBTRACT: { return a - b; }
        case MATH_MULTIPLY: { return a * b; }
        case MATH_DIVIDE: { return safe_divide(a, b); }
        case MATH_POWER: { return safe_pow(a, b); }
        case MATH_LOGARITHM: { return safe_log(a, b); }
        case MATH_SQRT: { return safe_sqrt(a); }
        case MATH_INV_SQRT: { return safe_inv_sqrt(a); }
        case MATH_ABSOLUTE: { return abs(a); }
        case MATH_RADIANS: { return a * (PI / 180.0); }
        case MATH_DEGREES: { return a * (180.0 / PI); }
        case MATH_MINIMUM: { return min(a, b); }
        case MATH_MAXIMUM: { return max(a, b); }
        case MATH_LESS_THAN: { return select(0.0, 1.0, a < b); }
        case MATH_GREATER_THAN: { return select(0.0, 1.0, a > b); }
        case MATH_ROUND: { return floor(a + 0.5); }
        case MATH_FLOOR: { return floor(a); }
        case MATH_CEIL: { return ceil(a); }
        case MATH_FRACTION: { return a - floor(a); }
        case MATH_MODULO: { return safe_modulo(a, b); }
        case MATH_FLOORED_MODULO: { return safe_floored_modulo(a, b); }
        case MATH_TRUNC: { return select(ceil(a), floor(a), a >= 0.0); }
        case MATH_SNAP: { return floor(safe_divide(a, b)) * b; }
        case MATH_WRAP: { return math_wrap(a, b, c); }
        case MATH_PINGPONG: { return math_pingpong(a, b); }
        case MATH_SINE: { return sin(a); }
        case MATH_COSINE: { return cos(a); }
        case MATH_TANGENT: { return tan(a); }
        case MATH_SINH: { return sinh(a); }
        case MATH_COSH: { return cosh(a); }
        case MATH_TANH: { return tanh(a); }
        case MATH_ARCSINE: { return asin(clamp(a, -1.0, 1.0)); }
        case MATH_ARCCOSINE: { return acos(clamp(a, -1.0, 1.0)); }
        case MATH_ARCTANGENT: { return atan(a); }
        case MATH_ARCTAN2: { return compatible_atan2(a, b); }
        case MATH_SIGN: { return compatible_sign(a); }
        case MATH_EXPONENT: { return exp(a); }
        // `FLT_EPSILON`, so an exact comparison still answers 1.
        case MATH_COMPARE: { return select(0.0, 1.0, a == b || abs(a - b) <= max(c, 1.1920929e-7)); }
        case MATH_MULTIPLY_ADD: { return a * b + c; }
        case MATH_SMOOTH_MIN: { return math_smooth_min(a, b, c); }
        case MATH_SMOOTH_MAX: { return -math_smooth_min(-a, -b, c); }
        default: { return 0.0; }
    }
}
