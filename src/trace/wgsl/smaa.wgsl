// SMAA's three passes (`workbench_effect_smaa_frag.glsl`, `SMAA_STAGE`
// 0–2), one dispatch each after the gather (`taa.wgsl`), picked by
// `p.slice.z` (1 edges, 2 blending weights, 3 the blend and resolve), on
// the first four samples alone (`smaa_mix()`): the aliased first frames
// are smoothed while the jittered ones land, and the fourth onward is the
// gather's own resolve, untouched.

// `SMAALumaEdgeDetectionPS` with `SMAA_LUMA_WEIGHT` (1, 1, 1, 1): the
// luma is the sum of the channels, the alpha a constant that drops out.
fn luma(x: i32, y: i32) -> f32 {
    let c = sums_texel(x, y);
    return c.x + c.y + c.z;
}

fn edge_detect(i: u32, x: i32, y: i32) {
    let l = luma(x, y);
    let l_left = luma(x - 1, y);
    let l_top = luma(x, y - 1);
    var delta = vec4<f32>(abs(l - vec2<f32>(l_left, l_top)), 0.0, 0.0);
    var edges = step(vec2<f32>(SMAA_THRESHOLD), delta.xy);
    if (edges.x + edges.y == 0.0) {
        edges_store(i, vec2<f32>(0.0));
        return;
    }
    let l_right = luma(x + 1, y);
    let l_bottom = luma(x, y + 1);
    delta = vec4<f32>(delta.xy, abs(l - vec2<f32>(l_right, l_bottom)));
    var max_delta = max(delta.xy, delta.zw);
    let l_leftleft = luma(x - 2, y);
    let l_toptop = luma(x, y - 2);
    delta = vec4<f32>(delta.xy, abs(vec2<f32>(l_left, l_top) - vec2<f32>(l_leftleft, l_toptop)));
    max_delta = max(max_delta, delta.zw);
    let final_delta = max(max_delta.x, max_delta.y);
    // Local contrast adaptation.
    edges *= step(vec2<f32>(final_delta), SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta.xy);
    edges_store(i, edges);
}

// `SMAACalculateDiagWeights`: the two diagonal patterns through `uv`.
fn diag_weights(uv: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
    let m = rt();
    var weights = vec2<f32>(0.0);
    var end = vec2<f32>(0.0);
    var d = vec4<f32>(0.0);
    if (e.x > 0.0) {
        let xz = search_diag1(uv, vec2<f32>(-1.0, 1.0), &end);
        d.x = xz.x + f32(end.y > 0.9);
        d.z = xz.y;
    }
    let yw = search_diag1(uv, vec2<f32>(1.0, -1.0), &end);
    d.y = yw.x;
    d.w = yw.y;
    if (d.x + d.y > 2.0) {
        let coords = vec4<f32>(-d.x + 0.25, d.x, d.y, -d.y - 0.25) * m.xyxy + uv.xyxy;
        var c = vec4<f32>(edges_off(coords.xy, vec2<i32>(-1, 0)), edges_off(coords.zw, vec2<i32>(1, 0)));
        c = decode_diag4(c).yxwz;
        var cc = vec2<f32>(2.0) * c.xz + c.yw;
        cc = select(cc, vec2<f32>(0.0), d.zw >= vec2<f32>(0.9));
        weights += area_diag(d.xy, cc, 0.0);
    }
    let xz = search_diag2(uv, vec2<f32>(-1.0, -1.0), &end);
    d.x = xz.x;
    d.z = xz.y;
    if (edges_off(uv, vec2<i32>(1, 0)).x > 0.0) {
        let yw2 = search_diag2(uv, vec2<f32>(1.0, 1.0), &end);
        d.y = yw2.x + f32(end.y > 0.9);
        d.w = yw2.y;
    } else {
        d.y = 0.0;
        d.w = 0.0;
    }
    if (d.x + d.y > 2.0) {
        let coords = vec4<f32>(-d.x, -d.x, d.y, d.y) * m.xyxy + uv.xyxy;
        let c = vec4<f32>(edges_off(coords.xy, vec2<i32>(-1, 0)).y, edges_off(coords.xy, vec2<i32>(0, -1)).x, edges_off(coords.zw, vec2<i32>(1, 0)).yx);
        var cc = vec2<f32>(2.0) * c.xz + c.yw;
        cc = select(cc, vec2<f32>(0.0), d.zw >= vec2<f32>(0.9));
        weights += area_diag(d.xy, cc, 0.0).yx;
    }
    return weights;
}

// `SMAABlendingWeightCalculationPS` with its vertex stage's offsets.
fn blending_weights(i: u32, x: i32, y: i32) {
    let m = rt();
    let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) * m.xy;
    let pixcoord = uv * m.zw;
    let offset0 = m.xyxy * vec4<f32>(-0.25, -0.125, 1.25, -0.125) + uv.xyxy;
    let offset1 = m.xyxy * vec4<f32>(-0.125, -0.25, -0.125, 1.25) + uv.xyxy;
    let offset2 = m.xxyy * vec4<f32>(-2.0, 2.0, -2.0, 2.0) * SMAA_MAX_SEARCH_STEPS + vec4<f32>(offset0.xz, offset1.yw);
    var weights = vec4<f32>(0.0);
    var e = edges_texel(x, y);
    if (e.y > 0.0) {
        // An edge at the top: a horizontal line, diagonal first.
        let dw = diag_weights(uv, e);
        weights.x = dw.x;
        weights.y = dw.y;
        if (weights.x == -weights.y) {
            var coords = vec3<f32>(search_x_left(offset0.xy, offset2.x), offset1.y, 0.0);
            let e1 = edges_at(coords.xy).x;
            coords.z = search_x_right(offset0.zw, offset2.y);
            let d = abs(round(m.zz * vec2<f32>(coords.x, coords.z) - pixcoord.xx));
            let e2 = edges_off(coords.zy, vec2<i32>(1, 0)).x;
            let w = horizontal_corner(area(sqrt(d), e1, e2, 0.0), vec4<f32>(coords.x, uv.y, coords.z, uv.y), d);
            weights.x = w.x;
            weights.y = w.y;
        } else {
            e.x = 0.0;
        }
    }
    if (e.x > 0.0) {
        // An edge at the left: a vertical line.
        var coords = vec3<f32>(offset0.x, search_y_up(offset1.xy, offset2.z), 0.0);
        let e1 = edges_at(coords.xy).y;
        coords.z = search_y_down(offset1.zw, offset2.w);
        let d = abs(round(m.ww * vec2<f32>(coords.y, coords.z) - pixcoord.yy));
        let e2 = edges_off(coords.xz, vec2<i32>(0, 1)).y;
        let w = vertical_corner(area(sqrt(d), e1, e2, 0.0), vec4<f32>(uv.x, coords.y, uv.x, coords.z), d);
        weights.z = w.x;
        weights.w = w.y;
    }
    weights_store(i, weights);
}

fn pack(c: vec3<f32>, a: f32) -> u32 {
    let q = vec3<u32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0 + 0.5);
    let qa = u32(clamp(a, 0.0, 1.0) * 255.0 + 0.5);
    return q.r | (q.g << 8u) | (q.b << 16u) | (qa << 24u);
}

// `SMAANeighborhoodBlendingPS`, then the stage-2 resolve: the blend mixed
// with the plain sums by `mix_factor`, over the weight accumulated, out of
// log space — the pixel the gather resolved, smoothed.
fn blend(i: u32, x: i32, y: i32) {
    let m = rt();
    let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) * m.xy;
    let here = weights_texel(x, y);
    // Right, top, bottom, left.
    let a = vec4<f32>(weights_texel(x + 1, y).w, weights_texel(x, y + 1).y, here.z, here.x);
    let plain = sums_texel(x, y);
    var color = plain;
    if (dot(a, vec4<f32>(1.0)) >= 1e-5) {
        let h = max(a.x, a.z) > max(a.y, a.w);
        let blending_offset = select(vec4<f32>(0.0, a.y, 0.0, a.w), vec4<f32>(a.x, 0.0, a.z, 0.0), h);
        var blending_weight = select(a.yw, a.xz, h);
        blending_weight /= dot(blending_weight, vec2<f32>(1.0));
        let coord = blending_offset * vec4<f32>(m.xy, -m.xy) + uv.xyxy;
        color = blending_weight.x * sums_at(coord.xy) + blending_weight.y * sums_at(coord.zw);
    }
    let mixf = smaa_mix();
    var out = color * mixf + plain * (1.0 - mixf);
    out /= max(p.taa.z, 1e-6);
    px[i] = pack(exp2(out) - 1.0, 1.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n || smaa_mix() <= 0.0) { return; }
    let x = i32(i % p.w);
    let y = i32(i / p.w);
    switch (p.slice.z) {
        case 1u: { edge_detect(i, x, y); }
        case 2u: { blending_weights(i, x, y); }
        default: { blend(i, x, y); }
    }
}
