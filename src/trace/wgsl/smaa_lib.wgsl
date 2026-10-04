// SMAA 1x (`gpu_shader_smaa_lib.glsl`, SMAA_PRESET_HIGH: threshold 0.1,
// 16 search steps, 8 diagonal, corner rounding 25) as Workbench runs it
// over the first anti-aliasing samples (`workbench_effect_smaa_frag.glsl`).
// The library's textures are this kernel's buffers: the colour it detects
// edges in is the gather's running sums (`taa.wgsl`, words n..4n — Blender's
// `taa_accumulation_tx_`, in log2 space, not yet over the weight), the
// edges live in words 17n..19n and the blending weights in 19n..23n — the
// opaque pass's ids and normals and the Depth of Field's half image, which
// the frame is done with by now — and the area and search tables are the
// `tables` binding (`smaa.rs`). Texture fetches are bilinear with the edge
// clamped, as the library asks; texcoords are `(x + ½) / w` with row 0 at
// the top, a mirror of Blender's, which SMAA does not mind.

@group(0) @binding(0) var<uniform> p: Camera;
@group(0) @binding(1) var<storage, read_write> px: array<u32>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;

const SMAA_THRESHOLD: f32 = 0.1;
const SMAA_MAX_SEARCH_STEPS: f32 = 16.0;
const SMAA_MAX_SEARCH_STEPS_DIAG: f32 = 8.0;
const SMAA_CORNER_ROUNDING_NORM: f32 = 0.25;
const SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR: f32 = 2.0;
const SMAA_AREATEX_MAX_DISTANCE: f32 = 16.0;
const SMAA_AREATEX_MAX_DISTANCE_DIAG: f32 = 20.0;
const SMAA_AREATEX_PIXEL_SIZE: vec2<f32> = vec2<f32>(1.0 / 160.0, 1.0 / 560.0);
const SMAA_AREATEX_SUBTEX_SIZE: f32 = 1.0 / 7.0;
const SMAA_SEARCHTEX_SIZE: vec2<f32> = vec2<f32>(66.0, 33.0);
const SMAA_SEARCHTEX_PACKED_SIZE: vec2<f32> = vec2<f32>(64.0, 16.0);
const AREA_BYTES: u32 = 160u * 560u * 2u;

// `SMAA_RT_METRICS`: a texel's size and the image's.
fn rt() -> vec4<f32> {
    return vec4<f32>(1.0 / f32(p.w), 1.0 / f32(p.h), f32(p.w), f32(p.h));
}

// Workbench's `smaa_mix_factor_`: 1 − sample / 4, so SMAA fades out as
// the gather's samples land; 0 from the fourth, where it is not run — and
// with anti-aliasing off (`p.taa.w`, the count, 0: `draw_aa` false).
fn smaa_mix() -> f32 {
    if (p.taa.w < 1.0) { return 0.0; }
    return 1.0 - clamp(f32(p.frame) / 4.0, 0.0, 1.0);
}

fn texel(x: i32, y: i32) -> u32 {
    return u32(clamp(y, 0, i32(p.h) - 1)) * p.w + u32(clamp(x, 0, i32(p.w) - 1));
}

// The colour (the gather's log-space sums) at a texel.
fn sums_texel(x: i32, y: i32) -> vec3<f32> {
    let a = p.n + 3u * texel(x, y);
    return vec3<f32>(bitcast<f32>(px[a]), bitcast<f32>(px[a + 1u]), bitcast<f32>(px[a + 2u]));
}

// Bilinear in texcoords (`SMAASampleLevelZero` of `color_tx`).
fn sums_at(uv: vec2<f32>) -> vec3<f32> {
    let t = uv * rt().zw - 0.5;
    let i = vec2<i32>(floor(t));
    let f = t - floor(t);
    let top = mix(sums_texel(i.x, i.y), sums_texel(i.x + 1, i.y), f.x);
    let bottom = mix(sums_texel(i.x, i.y + 1), sums_texel(i.x + 1, i.y + 1), f.x);
    return mix(top, bottom, f.y);
}

fn edges_texel(x: i32, y: i32) -> vec2<f32> {
    let j = texel(x, y);
    return vec2<f32>(bitcast<f32>(px[17u * p.n + j]), bitcast<f32>(px[18u * p.n + j]));
}

fn edges_store(i: u32, e: vec2<f32>) {
    px[17u * p.n + i] = bitcast<u32>(e.x);
    px[18u * p.n + i] = bitcast<u32>(e.y);
}

// `SMAASampleLevelZero(edgesTex, uv)`: bilinear, which the diagonal and
// orthogonal searches read two texels at once through.
fn edges_at(uv: vec2<f32>) -> vec2<f32> {
    let t = uv * rt().zw - 0.5;
    let i = vec2<i32>(floor(t));
    let f = t - floor(t);
    let top = mix(edges_texel(i.x, i.y), edges_texel(i.x + 1, i.y), f.x);
    let bottom = mix(edges_texel(i.x, i.y + 1), edges_texel(i.x + 1, i.y + 1), f.x);
    return mix(top, bottom, f.y);
}

// `SMAASampleLevelZeroOffset(edgesTex, uv, offset)`.
fn edges_off(uv: vec2<f32>, off: vec2<i32>) -> vec2<f32> {
    return edges_at(uv + vec2<f32>(off) * rt().xy);
}

fn weights_texel(x: i32, y: i32) -> vec4<f32> {
    let j = texel(x, y);
    return vec4<f32>(bitcast<f32>(px[19u * p.n + j]), bitcast<f32>(px[20u * p.n + j]), bitcast<f32>(px[21u * p.n + j]), bitcast<f32>(px[22u * p.n + j]));
}

fn weights_store(i: u32, w: vec4<f32>) {
    px[19u * p.n + i] = bitcast<u32>(w.x);
    px[20u * p.n + i] = bitcast<u32>(w.y);
    px[21u * p.n + i] = bitcast<u32>(w.z);
    px[22u * p.n + i] = bitcast<u32>(w.w);
}

fn table_byte(b: u32) -> f32 {
    return f32((tables[b >> 2u] >> ((b & 3u) * 8u)) & 255u) / 255.0;
}

fn area_texel(x: i32, y: i32) -> vec2<f32> {
    let b = (u32(clamp(y, 0, 559)) * 160u + u32(clamp(x, 0, 159))) * 2u;
    return vec2<f32>(table_byte(b), table_byte(b + 1u));
}

// The area table, filtered (`GPU_texture_filter_mode(smaa_area_tx_, true)`).
fn area_at(uv: vec2<f32>) -> vec2<f32> {
    let t = uv * vec2<f32>(160.0, 560.0) - 0.5;
    let i = vec2<i32>(floor(t));
    let f = t - floor(t);
    let top = mix(area_texel(i.x, i.y), area_texel(i.x + 1, i.y), f.x);
    let bottom = mix(area_texel(i.x, i.y + 1), area_texel(i.x + 1, i.y + 1), f.x);
    return mix(top, bottom, f.y);
}

fn search_texel(x: i32, y: i32) -> f32 {
    return table_byte(AREA_BYTES + u32(clamp(y, 0, 15)) * 64u + u32(clamp(x, 0, 63)));
}

// The search table, filtered.
fn search_at(uv: vec2<f32>) -> f32 {
    let t = uv * vec2<f32>(64.0, 16.0) - 0.5;
    let i = vec2<i32>(floor(t));
    let f = t - floor(t);
    let top = mix(search_texel(i.x, i.y), search_texel(i.x + 1, i.y), f.x);
    let bottom = mix(search_texel(i.x, i.y + 1), search_texel(i.x + 1, i.y + 1), f.x);
    return mix(top, bottom, f.y);
}

// `SMAADecodeDiagBilinearAccess`: a bilinear fetch of two edge texels
// back to which were set.
fn decode_diag2(e: vec2<f32>) -> vec2<f32> {
    return round(vec2<f32>(e.x * abs(5.0 * e.x - 5.0 * 0.75), e.y));
}

fn decode_diag4(e: vec4<f32>) -> vec4<f32> {
    return round(vec4<f32>(e.x * abs(5.0 * e.x - 5.0 * 0.75), e.y, e.z * abs(5.0 * e.z - 5.0 * 0.75), e.w));
}

// `SMAASearchDiag1`: how far a diagonal line of edges runs from `uv`
// along `dir`, in texels, and the edges where it stopped.
fn search_diag1(uv: vec2<f32>, dir: vec2<f32>, e: ptr<function, vec2<f32>>) -> vec2<f32> {
    var coord = vec4<f32>(uv, -1.0, 1.0);
    let t = vec3<f32>(rt().xy, 1.0);
    while (coord.z < SMAA_MAX_SEARCH_STEPS_DIAG - 1.0 && coord.w > 0.9) {
        let step = coord.xyz + t * vec3<f32>(dir, 1.0);
        *e = edges_at(step.xy);
        coord = vec4<f32>(step, dot(*e, vec2<f32>(0.5, 0.5)));
    }
    return coord.zw;
}

// `SMAASearchDiag2`, a quarter texel over, the fetch decoded.
fn search_diag2(uv: vec2<f32>, dir: vec2<f32>, e: ptr<function, vec2<f32>>) -> vec2<f32> {
    var coord = vec4<f32>(uv.x + 0.25 * rt().x, uv.y, -1.0, 1.0);
    let t = vec3<f32>(rt().xy, 1.0);
    while (coord.z < SMAA_MAX_SEARCH_STEPS_DIAG - 1.0 && coord.w > 0.9) {
        let step = coord.xyz + t * vec3<f32>(dir, 1.0);
        *e = decode_diag2(edges_at(step.xy));
        coord = vec4<f32>(step, dot(*e, vec2<f32>(0.5, 0.5)));
    }
    return coord.zw;
}

// `SMAAAreaDiag`: the diagonal half of the area table.
fn area_diag(dist: vec2<f32>, e: vec2<f32>, offset: f32) -> vec2<f32> {
    var tc = vec2<f32>(SMAA_AREATEX_MAX_DISTANCE_DIAG) * e + dist;
    tc = SMAA_AREATEX_PIXEL_SIZE * tc + 0.5 * SMAA_AREATEX_PIXEL_SIZE;
    tc.x += 0.5;
    tc.y += SMAA_AREATEX_SUBTEX_SIZE * offset;
    return area_at(tc);
}

// `SMAASearchLength`: what the last fetched pair of edges adds to a search.
fn search_length(e: vec2<f32>, offset: f32) -> f32 {
    var scale = SMAA_SEARCHTEX_SIZE * vec2<f32>(0.5, -1.0);
    var bias = SMAA_SEARCHTEX_SIZE * vec2<f32>(offset, 1.0);
    scale += vec2<f32>(-1.0, 1.0);
    bias += vec2<f32>(0.5, -0.5);
    scale *= 1.0 / SMAA_SEARCHTEX_PACKED_SIZE;
    bias *= 1.0 / SMAA_SEARCHTEX_PACKED_SIZE;
    return search_at(scale * e + bias);
}

fn search_x_left(uv: vec2<f32>, end: f32) -> f32 {
    var e = vec2<f32>(0.0, 1.0);
    var tc = uv;
    while (tc.x > end && e.y > 0.8281 && e.x == 0.0) {
        e = edges_at(tc);
        tc -= vec2<f32>(2.0, 0.0) * rt().xy;
    }
    let offset = -(255.0 / 127.0) * search_length(e, 0.0) + 3.25;
    return rt().x * offset + tc.x;
}

fn search_x_right(uv: vec2<f32>, end: f32) -> f32 {
    var e = vec2<f32>(0.0, 1.0);
    var tc = uv;
    while (tc.x < end && e.y > 0.8281 && e.x == 0.0) {
        e = edges_at(tc);
        tc += vec2<f32>(2.0, 0.0) * rt().xy;
    }
    let offset = -(255.0 / 127.0) * search_length(e, 0.5) + 3.25;
    return -rt().x * offset + tc.x;
}

fn search_y_up(uv: vec2<f32>, end: f32) -> f32 {
    var e = vec2<f32>(1.0, 0.0);
    var tc = uv;
    while (tc.y > end && e.x > 0.8281 && e.y == 0.0) {
        e = edges_at(tc);
        tc -= vec2<f32>(0.0, 2.0) * rt().xy;
    }
    let offset = -(255.0 / 127.0) * search_length(e.yx, 0.0) + 3.25;
    return rt().y * offset + tc.y;
}

fn search_y_down(uv: vec2<f32>, end: f32) -> f32 {
    var e = vec2<f32>(1.0, 0.0);
    var tc = uv;
    while (tc.y < end && e.x > 0.8281 && e.y == 0.0) {
        e = edges_at(tc);
        tc += vec2<f32>(0.0, 2.0) * rt().xy;
    }
    let offset = -(255.0 / 127.0) * search_length(e.yx, 0.5) + 3.25;
    return -rt().y * offset + tc.y;
}

// `SMAAArea`: the orthogonal half of the area table.
fn area(dist: vec2<f32>, e1: f32, e2: f32, offset: f32) -> vec2<f32> {
    var tc = vec2<f32>(SMAA_AREATEX_MAX_DISTANCE) * round(4.0 * vec2<f32>(e1, e2)) + dist;
    tc = SMAA_AREATEX_PIXEL_SIZE * tc + 0.5 * SMAA_AREATEX_PIXEL_SIZE;
    tc.y = SMAA_AREATEX_SUBTEX_SIZE * offset + tc.y;
    return area_at(tc);
}

// `SMAADetectHorizontalCornerPattern`: less blending at a line's corners.
fn horizontal_corner(w: vec2<f32>, tc: vec4<f32>, d: vec2<f32>) -> vec2<f32> {
    let left_right = step(d.xy, d.yx);
    var rounding = (1.0 - SMAA_CORNER_ROUNDING_NORM) * left_right;
    rounding /= left_right.x + left_right.y;
    var factor = vec2<f32>(1.0, 1.0);
    factor.x -= rounding.x * edges_off(tc.xy, vec2<i32>(0, 1)).x;
    factor.x -= rounding.y * edges_off(tc.zw, vec2<i32>(1, 1)).x;
    factor.y -= rounding.x * edges_off(tc.xy, vec2<i32>(0, -2)).x;
    factor.y -= rounding.y * edges_off(tc.zw, vec2<i32>(1, -2)).x;
    return w * clamp(factor, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn vertical_corner(w: vec2<f32>, tc: vec4<f32>, d: vec2<f32>) -> vec2<f32> {
    let left_right = step(d.xy, d.yx);
    var rounding = (1.0 - SMAA_CORNER_ROUNDING_NORM) * left_right;
    rounding /= left_right.x + left_right.y;
    var factor = vec2<f32>(1.0, 1.0);
    factor.x -= rounding.x * edges_off(tc.xy, vec2<i32>(1, 0)).y;
    factor.x -= rounding.y * edges_off(tc.zw, vec2<i32>(1, 1)).y;
    factor.y -= rounding.x * edges_off(tc.xy, vec2<i32>(-2, 0)).y;
    factor.y -= rounding.y * edges_off(tc.zw, vec2<i32>(-2, 1)).y;
    return w * clamp(factor, vec2<f32>(0.0), vec2<f32>(1.0));
}
