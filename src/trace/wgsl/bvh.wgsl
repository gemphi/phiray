// The slab test over a wide node's four children at once: each lane is one
// child. An empty slot is masked out by its child word — `0` is never a
// child, the root being node 0 — rather than by its box: an inverted box
// of infinities gives tmin = -INF and tmax = +INF under the arithmetic
// below, which reads as a hit at distance zero on node 0, and the walk
// never ends. That was a two-second GPU watchdog reset on the first frame.
// Returns each child's entry distance, or INF.
fn box4(n: Wide, o: vec3<f32>, inv: vec3<f32>, best: f32) -> vec4<f32> {
    let t1x = (n.lox - o.x) * inv.x;
    let t2x = (n.hix - o.x) * inv.x;
    let t1y = (n.loy - o.y) * inv.y;
    let t2y = (n.hiy - o.y) * inv.y;
    let t1z = (n.loz - o.z) * inv.z;
    let t2z = (n.hiz - o.z) * inv.z;
    let tmin = max(max(min(t1x, t2x), min(t1y, t2y)), min(t1z, t2z));
    let tmax = min(min(max(t1x, t2x), max(t1y, t2y)), max(t1z, t2z));
    let entry = max(tmin, vec4<f32>(0.0));
    let hit = (tmax >= entry) & (tmin < vec4<f32>(best)) & (n.c != vec4<u32>(0u));
    return select(vec4<f32>(INF), entry, hit);
}

// Push the hit children of a wide node far-to-near so the nearest is walked
// first: a 4-lane sort by distance done as a fixed compare network, then
// the far ones onto the stack and the nearest returned (0xffffffff if none).
fn push_near_first(t: vec4<f32>, c: vec4<u32>, stack: ptr<function, array<u32, 32>>, sp: ptr<function, u32>) -> u32 {
    var tt = t;
    var cc = c;
    // Sort network for 4 (5 compare-swaps): ascending by t.
    if (tt[0] > tt[1]) { let a = tt[0]; tt[0] = tt[1]; tt[1] = a; let b = cc[0]; cc[0] = cc[1]; cc[1] = b; }
    if (tt[2] > tt[3]) { let a = tt[2]; tt[2] = tt[3]; tt[3] = a; let b = cc[2]; cc[2] = cc[3]; cc[3] = b; }
    if (tt[0] > tt[2]) { let a = tt[0]; tt[0] = tt[2]; tt[2] = a; let b = cc[0]; cc[0] = cc[2]; cc[2] = b; }
    if (tt[1] > tt[3]) { let a = tt[1]; tt[1] = tt[3]; tt[3] = a; let b = cc[1]; cc[1] = cc[3]; cc[3] = b; }
    if (tt[1] > tt[2]) { let a = tt[1]; tt[1] = tt[2]; tt[2] = a; let b = cc[1]; cc[1] = cc[2]; cc[2] = b; }
    // Farthest first onto the stack, so the nearest comes off first.
    if (tt[3] < INF) { (*stack)[*sp] = cc[3]; *sp = *sp + 1u; }
    if (tt[2] < INF) { (*stack)[*sp] = cc[2]; *sp = *sp + 1u; }
    if (tt[1] < INF) { (*stack)[*sp] = cc[1]; *sp = *sp + 1u; }
    if (tt[0] < INF) { return cc[0]; }
    return 0xffffffffu;
}

// Möller–Trumbore; returns (t, u, v) or t = INF.
fn tri_t(ta: vec3<f32>, tb: vec3<f32>, tc: vec3<f32>, o: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let e1 = tb - ta;
    let e2 = tc - ta;
    let pv = cross(d, e2);
    let det = dot(e1, pv);
    if (abs(det) < 1e-12) { return vec3<f32>(INF, 0.0, 0.0); }
    let inv = 1.0 / det;
    let tv = o - ta;
    let u = dot(tv, pv) * inv;
    if (u < 0.0 || u > 1.0) { return vec3<f32>(INF, 0.0, 0.0); }
    let q = cross(tv, e1);
    let v = dot(d, q) * inv;
    if (v < 0.0 || u + v > 1.0) { return vec3<f32>(INF, 0.0, 0.0); }
    let tt = dot(e2, q) * inv;
    if (tt <= 0.0) { return vec3<f32>(INF, 0.0, 0.0); }
    return vec3<f32>(tt, u, v);
}

fn safe_inv(d: vec3<f32>) -> vec3<f32> {
    let s = select(vec3<f32>(1.0), vec3<f32>(-1.0), d < vec3<f32>(0.0));
    return 1.0 / select(d, s * 1e-9, abs(d) < vec3<f32>(1e-9));
}

// Instance `i`'s tree, walked in the object's own space: the ray is taken
// there by the instance's inverse (its direction left unnormalised, so `t`
// means the same along both rays), and a hit's `t` is the world's.
//
// The walk is over the four-wide tree (`Wide`): a node's four children are
// slab-tested at once, the hits pushed far-to-near, and the nearest taken
// next. A child word with LEAF set is a face range [start, end) rather than
// a node — the leaf's faces are tested straight from the parent, so leaves
// cost no node fetch at all.
fn trace_blas(i: u32, ow: vec3<f32>, dw: vec3<f32>, first: bool, best: ptr<function, Hit>) {
    let w = inst_at(i);
    let root = data[w + 24u];
    if (root == 0xffffffffu) { return; }
    // A shadow ray (any hit) passes an instance that casts none (word 27, bit 1).
    if (first && (data[w + 27u] & 2u) != 0u) { return; }
    var o = ow;
    var d = dw;
    if ((data[w + 27u] & 1u) == 0u) {
        let i0 = inst_row(w + 12u);
        let i1 = inst_row(w + 16u);
        let i2 = inst_row(w + 20u);
        let oh = vec4<f32>(ow, 1.0);
        o = vec3<f32>(dot(i0, oh), dot(i1, oh), dot(i2, oh));
        d = vec3<f32>(dot(i0.xyz, dw), dot(i1.xyz, dw), dot(i2.xyz, dw));
    }
    let inv = safe_inv(d);
    var stack: array<u32, 32>;
    var sp = 0u;
    var cur = root;
    loop {
        let nd = node(cur);
        let t = box4(nd, o, inv, (*best).t);
        // Leaves first, straight from this node: their faces are tested
        // here and they never go on the stack.
        var tn = t;
        for (var k = 0u; k < 4u; k++) {
            if (t[k] < INF && (nd.c[k] & LEAF) != 0u) {
                tn[k] = INF;
                for (var id = nd.c[k] & ~LEAF; id < nd.e[k]; id++) {
                    // Shadow rays pass glass and cut-out (alpha) surfaces, and
                    // the test goes BEFORE the intersection on purpose: reading
                    // the material is one cached fetch, while `tri_t` costs
                    // three vertex loads. Moving it after the hit test to "save
                    // a read" measured 29% SLOWER on pose 26 (254s -> 328s),
                    // because it threw away the early-out that skips those
                    // vertex loads entirely.
                    if (first) {
                        let m = mat_at(id);
                        if (tf(m + 6u) > 0.5 || tf(m + 3u) < 0.5) { continue; }
                    }
                    let v = face(id);
                    let h = tri_t(vert(v.x).p.xyz, vert(v.y).p.xyz, vert(v.z).p.xyz, o, d);
                    if (h.x < (*best).t) {
                        *best = Hit(h.x, id, h.y, h.z);
                        if (first) { return; }
                    }
                }
            }
        }
        // Then the inner children, nearest next, the rest stacked.
        let next = push_near_first(tn, nd.c, &stack, &sp);
        if (next != 0xffffffffu) { cur = next; continue; }
        if (sp == 0u) { break; }
        sp--;
        cur = stack[sp];
    }
}

// The top tree, over the instances' world bounds: each leaf names the
// instances (`inst_order`) whose own trees the ray then walks.
fn trace(o: vec3<f32>, d: vec3<f32>, tmax: f32, first: bool) -> Hit {
    var best = Hit(tmax, 0xffffffffu, 0.0, 0.0);
    let inv = safe_inv(d);
    var stack: array<u32, 32>;
    var sp = 0u;
    var cur = 0u;
    loop {
        let nd = node(cur);
        let t = box4(nd, o, inv, best.t);
        var tn = t;
        for (var k = 0u; k < 4u; k++) {
            if (t[k] < INF && (nd.c[k] & LEAF) != 0u) {
                tn[k] = INF;
                for (var s = nd.c[k] & ~(LEAF | INST); s < nd.e[k]; s++) {
                    trace_blas(inst_order(s), o, d, first, &best);
                    if (first && best.tri != 0xffffffffu) { return best; }
                }
            }
        }
        let next = push_near_first(tn, nd.c, &stack, &sp);
        if (next != 0xffffffffu) { cur = next; continue; }
        if (sp == 0u) { break; }
        sp--;
        cur = stack[sp];
    }
    return best;
}
