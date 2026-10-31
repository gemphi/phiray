//! The watertight ray/triangle test, CPU twin of `bvh.wgsl`'s `ray_space`
//! and `tri_t` — Woop, Benthin, Benthin and Wald, "Watertight
//! Ray/Triangle Intersection", JCGT 2(1), 2013. Cycles carries the same
//! test in `kernel/geom/triangle_intersect.h`.
//!
//! Möller–Trumbore, which stood here before, divides by each triangle's
//! own determinant and only then asks whether `u` and `v` are in range.
//! Two triangles that share an edge reach that question by different
//! arithmetic, so a ray falling exactly on the edge can be rejected by
//! both: a pinhole. Every quad in every mesh has a shared diagonal, so
//! this was a speckle of background wherever a ray grazed one — read as
//! noise at high sample counts, plain in Material Preview, where one
//! sample is the frame.
//!
//! The watertight test shears the triangle into the ray's own frame —
//! picked from the ray alone, so both triangles get the same one — and
//! tests three edge values, each a 2 × 2 determinant of the two vertices
//! on that edge and nothing else. The pair name a shared edge's vertices
//! in opposite order, so their edge values come out `a·b − c·d` and
//! `c·d − a·b`: the products round to the same float and round-to-nearest
//! is symmetric, so the two are exact negations and can never disagree
//! which side of the edge the ray fell. A zero is inside for both, so the
//! ray hits two triangles rather than none, and the traversal's strict
//! `h.x < best.t` hands the edge to the first in leaf order.
//!
//! The paper's `double` fallback for an edge value lost to cancellation
//! is left out: WGSL has no f64, and watertightness does not need it —
//! the fallback only decides *which* of the pair owns a borderline edge,
//! while the exact negation is what rules out the hole. This twin leaves
//! it out too, so it is the kernel's arithmetic and not a better one.

/// A point or a direction, as the kernels keep them.
pub type V3 = [f32; 3];

/// The ray's own frame (Woop et al. §3): `kz` the axis the direction is
/// largest along, `flip` whether it runs backwards down it, and `s` the
/// two shear constants with the reciprocal of the depth component.
#[derive(Clone, Copy, Debug)]
pub struct RaySpace {
    pub kz: usize,
    pub flip: bool,
    pub s: [f32; 3],
}

/// `bvh.wgsl`'s `ray_axes`: a point with the depth axis last, then the
/// other two swapped when the ray runs backwards down it — which keeps
/// the triangles' winding, and with it the sign of the determinant.
/// Swizzles, not a vector indexed by a runtime axis number: naga lowers a
/// dynamic vector index through a function-scope variable, which on
/// Vulkan is local memory, and cost a 51 M triangle frame ~40% a sample.
pub fn ray_axes(v: V3, kz: usize, flip: bool) -> V3 {
    let p = match kz {
        0 => [v[1], v[2], v[0]],
        1 => [v[2], v[0], v[1]],
        _ => v,
    };
    match flip {
        true => [p[1], p[0], p[2]],
        false => p,
    }
}

/// `bvh.wgsl`'s `ray_space`: worked out once a ray, so every triangle the
/// ray meets is sheared the same way.
pub fn ray_space(d: V3) -> RaySpace {
    let a = [d[0].abs(), d[1].abs(), d[2].abs()];
    let kz = match a[0] >= a[1] && a[0] >= a[2] {
        true => 0,
        false => match a[1] >= a[2] {
            true => 1,
            false => 2,
        },
    };
    let flip = ray_axes(d, kz, false)[2] < 0.0;
    let dr = ray_axes(d, kz, flip);
    let sz = 1.0 / dr[2];
    RaySpace { kz, flip, s: [dr[0] * sz, dr[1] * sz, sz] }
}

/// `bvh.wgsl`'s `tri_t`: the hit as `(t, u, v)`, `u` and `v` the weights
/// of `tb` and `tc` as Möller–Trumbore's were, or `None` for a miss. No
/// back-face culling, and a hit behind the origin is a miss.
pub fn tri_t(ta: V3, tb: V3, tc: V3, o: V3, r: &RaySpace) -> Option<(f32, f32, f32)> {
    let rel = |v: V3| ray_axes([v[0] - o[0], v[1] - o[1], v[2] - o[2]], r.kz, r.flip);
    let (a, b, c) = (rel(ta), rel(tb), rel(tc));
    let shear = |v: V3| (v[0] - r.s[0] * v[2], v[1] - r.s[1] * v[2]);
    let ((ax, ay), (bx, by), (cx, cy)) = (shear(a), shear(b), shear(c));
    // The edge value opposite each vertex, so `ea / det` weights `ta`.
    let ea = cx * by - cy * bx;
    let eb = ax * cy - ay * cx;
    let ec = bx * ay - by * ax;
    // No back-face culling: inside is "none negative, or none positive".
    if (ea < 0.0 || eb < 0.0 || ec < 0.0) && (ea > 0.0 || eb > 0.0 || ec > 0.0) {
        return None;
    }
    let det = ea + eb + ec;
    if det == 0.0 {
        return None;
    }
    // The depth of each vertex along the ray, then the hit's over the
    // three weights — the paper's `Az`, `Bz`, `Cz` and `T`.
    let tt = ea * (r.s[2] * a[2]) + eb * (r.s[2] * b[2]) + ec * (r.s[2] * c[2]);
    // `tt` and `det` share the determinant's sign, so in front of the
    // origin is `tt / det > 0` without the divide.
    let sg = match det < 0.0 {
        true => -1.0,
        false => 1.0,
    };
    if tt * sg <= 0.0 {
        return None;
    }
    let inv = 1.0 / det;
    Some((tt * inv, eb * inv, ec * inv))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Möller–Trumbore test this file replaced, kept as the witness
    /// for the holes: `_temp/phiray-moller-trumbore-tri_t/tri_t.wgsl` is
    /// the WGSL it was, line for line.
    fn moller_trumbore(ta: V3, tb: V3, tc: V3, o: V3, d: V3) -> Option<(f32, f32, f32)> {
        let sub = |p: V3, q: V3| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
        let dot = |p: V3, q: V3| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
        let cross = |p: V3, q: V3| [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
        let (e1, e2) = (sub(tb, ta), sub(tc, ta));
        let pv = cross(d, e2);
        let det = dot(e1, pv);
        let inv = 1.0 / det;
        let tv = sub(o, ta);
        let u = dot(tv, pv) * inv;
        let q = cross(tv, e1);
        let v = dot(d, q) * inv;
        let t = dot(e2, q) * inv;
        if det.abs() < 1e-12 || u < 0.0 || u > 1.0 || v < 0.0 || u + v > 1.0 {
            return None;
        }
        match t > 0.0 { true => Some((t, u, v)), false => None }
    }

    const W: u32 = 48;

    /// `tests/preview.rs`'s own scene: `card(3.0, 2.0, None)` — a quad at
    /// y = 2 as two triangles over the diagonal from `q[0]` to `q[2]`.
    fn card() -> [[V3; 3]; 2] {
        let (half, y) = (3.0f32, 2.0f32);
        let q = [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]];
        [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    }

    /// `preview.wgsl`'s primary ray for pixel `(x, y)` under
    /// `tests/preview.rs`'s camera: perspective, `right` +X, `up` +Z,
    /// `back` −Y, the pivot at the origin and `dist` 4, so the eye is at
    /// (0, −4, 0) and `frame_point` is the pixel centre, −0.5 ..= 0.5.
    fn preview_ray(x: u32, y: u32) -> (V3, V3) {
        let sx = (x as f32 + 0.5) / W as f32 - 0.5;
        let sy = (y as f32 + 0.5) / W as f32 - 0.5;
        let d = [sx, 4.0, -sy];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        ([0.0, -4.0, 0.0], [d[0] / l, d[1] / l, d[2] / l])
    }

    /// How many of the card's two triangles the pixel's ray hits.
    fn covered(x: u32, y: u32, watertight: bool) -> usize {
        let (o, d) = preview_ray(x, y);
        let r = ray_space(d);
        card()
            .into_iter()
            .filter(|t| match watertight {
                true => tri_t(t[0], t[1], t[2], o, &r).is_some(),
                false => moller_trumbore(t[0], t[1], t[2], o, d).is_some(),
            })
            .count()
    }

    /// Every pixel of the quad's shared diagonal hits the quad. The
    /// diagonal runs from `q[0]` to `q[2]`, the line x = z in the plane
    /// y = 2, and a primary ray reaches it exactly where `x + y = W − 1`
    /// — the 48 pixels the luma dump of the failing Preview frame found
    /// six sky-coloured holes among. Where the ray falls on the edge both
    /// triangles take it, at the same `t`; `trace_blas` keeps a hit only
    /// when it is strictly nearer, so the first in leaf order wins.
    /// Which pixels Möller–Trumbore dropped is not asserted here: that
    /// turns on the last bit of the direction, and the kernel's
    /// `normalize` is the GPU's, not this twin's — `tests/preview.rs`
    /// holds the frame itself to the same claim.
    #[test]
    fn no_ray_falls_through_a_quads_shared_edge() {
        let diagonal = || (0..W).map(|x| (x, W - 1 - x));
        let holes = diagonal().filter(|&(x, y)| covered(x, y, true) == 0).collect::<Vec<_>>();
        assert!(holes.is_empty(), "the shared diagonal leaks at {holes:?}");
        let both = diagonal().filter(|&(x, y)| covered(x, y, true) == 2).count();
        assert!(both > 0, "a ray on the edge is inside both triangles, not just inside one: {both}");
    }

    /// Not only the camera's rays: a ray aimed at any point along a
    /// shared edge, from either side and well off the axis, hits one of
    /// the two triangles or both — never neither. Möller–Trumbore drops
    /// some of the same rays, which is the hole this file closed.
    #[test]
    fn a_ray_aimed_at_a_shared_edge_hits_one_side_or_both() {
        let [a, b] = card();
        let (p0, p2) = (a[0], a[2]);
        let mut x = 0x2545_F491u32;
        let mut rand = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 1_000_003) as f32 / 1_000_003.0
        };
        let mut dropped = 0;
        for _ in 0..200_000 {
            let f = 0.02 + 0.96 * rand();
            let aim = [p0[0] + f * (p2[0] - p0[0]), p0[1], p0[2] + f * (p2[2] - p0[2])];
            let side = match rand() < 0.5 { true => -1.0, false => 1.0 };
            let o = [8.0 * (rand() - 0.5), p0[1] + side * (0.5 + 9.0 * rand()), 8.0 * (rand() - 0.5)];
            let d = [aim[0] - o[0], aim[1] - o[1], aim[2] - o[2]];
            let r = ray_space(d);
            let one = tri_t(a[0], a[1], a[2], o, &r).is_some();
            let two = tri_t(b[0], b[1], b[2], o, &r).is_some();
            assert!(one || two, "a hole at {aim:?} from {o:?}");
            let old = moller_trumbore(a[0], a[1], a[2], o, d).is_some() || moller_trumbore(b[0], b[1], b[2], o, d).is_some();
            dropped += usize::from(!old);
        }
        assert!(dropped > 0, "Möller–Trumbore let some of these through: it dropped {dropped}");
    }

    /// The hit itself is unchanged: `t` and the weights are
    /// Möller–Trumbore's, in the same two slots, so every reader of a hit
    /// stays right — and the weights rebuild the point the ray reached.
    #[test]
    fn the_hit_is_the_one_moller_trumbore_found() {
        let [a, _] = card();
        let mut agreed = 0;
        for x in 0..W {
            for y in 0..W {
                let (o, d) = preview_ray(x, y);
                let (Some(n), Some(m)) = (tri_t(a[0], a[1], a[2], o, &ray_space(d)), moller_trumbore(a[0], a[1], a[2], o, d)) else {
                    continue;
                };
                agreed += 1;
                assert!((n.0 - m.0).abs() < 1e-3 * m.0, "t at ({x}, {y}): {n:?} vs {m:?}");
                assert!((n.1 - m.1).abs() < 1e-3 && (n.2 - m.2).abs() < 1e-3, "u, v at ({x}, {y}): {n:?} vs {m:?}");
                let w = 1.0 - n.1 - n.2;
                let px = w * a[0][0] + n.1 * a[1][0] + n.2 * a[2][0];
                assert!((px - (o[0] + n.0 * d[0])).abs() < 1e-3, "the weights place the hit at ({x}, {y})");
            }
        }
        assert!(agreed > 400, "the triangle covers a good part of the frame: {agreed}");
    }
}
