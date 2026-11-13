fn box_t(n: Node, o: vec3<f32>, inv: vec3<f32>, best: f32) -> f32 {
    let t1 = (n.lo - o) * inv;
    let t2 = (n.hi - o) * inv;
    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tmax >= max(tmin, 0.0) && tmin < best) { return max(tmin, 0.0); }
    return INF;
}

// The ray's own frame for the watertight triangle test: Woop, Benthin,
// Benthin and Wald, "Watertight Ray/Triangle Intersection", JCGT 2(1)
// 2013, §3 — the axis the direction is largest along becomes the depth
// axis and the other two are sheared by it (`s.x`, `s.y`), `s.z` the
// reciprocal of the depth component. Worked out once a ray, so every
// triangle it meets is tested in exactly the same frame — that sameness
// is the whole point (`tri_t`). Cycles carries the same test in
// `kernel/geom/triangle_intersect.h`.
struct RaySpace { kz: u32, flip: u32, s: vec3<f32> }

// A point with its axes in that order: the three rotated so the depth
// axis is `z`, then the other two swapped when the ray runs backwards
// down it, which keeps the triangles' winding and the sign of the
// determinant. Written out as swizzles rather than indexed by a runtime
// axis number (`v[kz]`, as the paper's listing has it): naga lowers a
// dynamically indexed vector through a function-scope variable, which on
// this Vulkan backend is local memory, and that cost a 51 M triangle
// frame about 40% a sample.
fn ray_axes(v: vec3<f32>, kz: u32, flip: bool) -> vec3<f32> {
    let p = select(select(vec3<f32>(v.z, v.x, v.y), vec3<f32>(v.y, v.z, v.x), kz == 0u), v, kz == 2u);
    return select(p, vec3<f32>(p.y, p.x, p.z), flip);
}

fn ray_space(d: vec3<f32>) -> RaySpace {
    let a = abs(d);
    let kz = select(select(2u, 1u, a.y >= a.z), 0u, a.x >= a.y && a.x >= a.z);
    let flip = ray_axes(d, kz, false).z < 0.0;
    let dr = ray_axes(d, kz, flip);
    let sz = 1.0 / dr.z;
    return RaySpace(kz, u32(flip), vec3<f32>(dr.x * sz, dr.y * sz, sz));
}

// Watertight ray/triangle intersection (Woop et al. 2013, listing 1).
// Returns (t, u, v) — `u` and `v` the weights of `tb` and `tc`, as
// Möller–Trumbore's were — or t = INF.
//
// Each of the three edge values is a 2 × 2 determinant of the two vertices
// on that edge alone, taken in the ray's frame. Two triangles sharing an
// edge name its two vertices in opposite order, so their edge values come
// out `a·b − c·d` and `c·d − a·b`: each product rounds to the same float
// and round-to-nearest is symmetric, so the two are exact negations and
// the pair can never disagree about which side of the edge the ray fell.
// An edge value of exactly zero counts as inside for both (the test below
// rejects only when the three disagree in sign), so a ray down a shared
// edge hits both rather than neither — a hole is impossible — and the
// strict `h.x < (*best).t` in `trace_blas` gives the edge to whichever
// the leaf order reaches first. Möller–Trumbore in its place tested `u`
// and `v` after dividing by each triangle's own determinant, which left
// six pixels of a two-triangle card showing the sky.
//
// The paper's `double` fallback for an edge value lost to cancellation is
// not here: WGSL has no f64, and watertightness does not need it. The
// fallback only resolves *which* of the pair owns a borderline edge; the
// exact negation above is what rules out the hole, and a float zero is
// already inside for both. Refining in single precision (an `fma`
// determinant) would be worse than nothing, since it is not provably
// correctly rounded and so could break the negation that matters.
fn tri_t(ta: vec3<f32>, tb: vec3<f32>, tc: vec3<f32>, o: vec3<f32>, r: RaySpace) -> vec3<f32> {
    let miss = vec3<f32>(INF, 0.0, 0.0);
    let flip = r.flip != 0u;
    let a = ray_axes(ta - o, r.kz, flip);
    let b = ray_axes(tb - o, r.kz, flip);
    let c = ray_axes(tc - o, r.kz, flip);
    let ax = a.x - r.s.x * a.z;
    let ay = a.y - r.s.y * a.z;
    let bx = b.x - r.s.x * b.z;
    let by = b.y - r.s.y * b.z;
    let cx = c.x - r.s.x * c.z;
    let cy = c.y - r.s.y * c.z;
    // The edge value opposite each vertex, so `ea / det` weights `ta`.
    let ea = cx * by - cy * bx;
    let eb = ax * cy - ay * cx;
    let ec = bx * ay - by * ax;
    // No back-face culling: inside is "none negative, or none positive".
    if ((ea < 0.0 || eb < 0.0 || ec < 0.0) && (ea > 0.0 || eb > 0.0 || ec > 0.0)) { return miss; }
    let det = ea + eb + ec;
    if (det == 0.0) { return miss; }
    // The depth of each vertex along the ray, then the hit's over the
    // three weights — the paper's `Az`, `Bz`, `Cz` and `T`.
    let tt = ea * (r.s.z * a.z) + eb * (r.s.z * b.z) + ec * (r.s.z * c.z);
    // `tt` and `det` share the determinant's sign, so in front of the
    // origin is `tt / det > 0` without the divide.
    let sg = select(1.0, -1.0, det < 0.0);
    if (tt * sg <= 0.0) { return miss; }
    let inv = 1.0 / det;
    return vec3<f32>(tt * inv, eb * inv, ec * inv);
}

fn safe_inv(d: vec3<f32>) -> vec3<f32> {
    let s = select(vec3<f32>(1.0), vec3<f32>(-1.0), d < vec3<f32>(0.0));
    return 1.0 / select(d, s * 1e-9, abs(d) < vec3<f32>(1e-9));
}

// Instance `i`'s tree, walked in the object's own space: the ray is taken
// there by the instance's inverse (its direction left unnormalised, so `t`
// means the same along both rays), and a hit's `t` is the world's. The
// hit remembers the instance: its mesh may be shown by many.
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
    // One frame for the whole walk: the triangles share it, so they agree.
    let rs = ray_space(d);
    var stack: array<u32, 32>;
    var sp = 0u;
    var cur = root;
    loop {
        let nd = node(cur);
        if ((nd.li & LEAF) != 0u) {
            for (var id = nd.li & ~LEAF; id < nd.ri; id++) {
                let v = face(id);
                let h = tri_t(vert(v.x).p.xyz, vert(v.y).p.xyz, vert(v.z).p.xyz, o, rs);
                if (h.x < (*best).t) {
                    // Shadow rays pass glass and cut-out (alpha) surfaces.
                    // The triangle is tested first: most candidates miss,
                    // and only a hit needs its material read.
                    if (first) {
                        let m = mat_word(v.w);
                        if (tf(m + 6u) > 0.5 || tf(m + 3u) < 0.5) { continue; }
                    }
                    *best = Hit(h.x, id, h.y, h.z, i);
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
    var best = Hit(tmax, 0xffffffffu, 0.0, 0.0, 0u);
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
