fn box_t(n: Node, o: vec3<f32>, inv: vec3<f32>, best: f32) -> f32 {
    let t1 = (n.lo - o) * inv;
    let t2 = (n.hi - o) * inv;
    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tmax >= max(tmin, 0.0) && tmin < best) { return max(tmin, 0.0); }
    return INF;
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
    if (box_t(node(root), o, inv, (*best).t) >= INF) { return; }
    var stack: array<u32, 32>;
    var sp = 0u;
    var cur = root;
    loop {
        let nd = node(cur);
        if ((nd.li & LEAF) != 0u) {
            for (var id = nd.li & ~LEAF; id < nd.ri; id++) {
                // Shadow rays pass glass and cut-out (alpha) surfaces.
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
        } else {
            let tl = box_t(node(nd.li), o, inv, (*best).t);
            let tr = box_t(node(nd.ri), o, inv, (*best).t);
            if (tl < INF && tr < INF) {
                let near = select(nd.ri, nd.li, tl <= tr);
                let far = select(nd.li, nd.ri, tl <= tr);
                stack[sp] = far;
                sp++;
                cur = near;
                continue;
            }
            if (tl < INF) { cur = nd.li; continue; }
            if (tr < INF) { cur = nd.ri; continue; }
        }
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
    if (box_t(node(0u), o, inv, best.t) >= INF) { return best; }
    var stack: array<u32, 32>;
    var sp = 0u;
    var cur = 0u;
    loop {
        let nd = node(cur);
        if ((nd.li & LEAF) != 0u) {
            for (var s = nd.li & ~(LEAF | INST); s < nd.ri; s++) {
                trace_blas(inst_order(s), o, d, first, &best);
                if (first && best.tri != 0xffffffffu) { return best; }
            }
        } else {
            let tl = box_t(node(nd.li), o, inv, best.t);
            let tr = box_t(node(nd.ri), o, inv, best.t);
            if (tl < INF && tr < INF) {
                let near = select(nd.ri, nd.li, tl <= tr);
                let far = select(nd.li, nd.ri, tl <= tr);
                stack[sp] = far;
                sp++;
                cur = near;
                continue;
            }
            if (tl < INF) { cur = nd.li; continue; }
            if (tr < INF) { cur = nd.ri; continue; }
        }
        if (sp == 0u) { break; }
        sp--;
        cur = stack[sp];
    }
    return best;
}
