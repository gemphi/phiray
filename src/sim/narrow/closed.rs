//! Closed-form pairs (ROADMAP.md §2.2 "Collision", narrowphase):
//! sphere-sphere, sphere-capsule and capsule-capsule by the closest points
//! of their segments; sphere-box and sphere-triangle by the closest point
//! on the box or triangle. Every function names its first shape `a`, and
//! the manifold's normal points from `a` to `b`.

use super::super::math::*;
use super::manifold::Manifold;

/// Two spheres; a contact when their gap is under `slop`.
pub fn spheres(ca: V3, ra: f32, cb: V3, rb: f32, slop: f32) -> Manifold {
    let d = sub(cb, ca);
    let dist = len(d);
    if dist - ra - rb > slop {
        return Manifold::none();
    }
    let n = if dist > 0.0 { scale(d, 1.0 / dist) } else { [0.0, 0.0, 1.0] };
    Manifold::one(n, add(ca, scale(n, ra)), sub(cb, scale(n, rb)))
}

/// The point of segment `ab` nearest `p`.
pub fn closest_on_segment(p: V3, a: V3, b: V3) -> V3 {
    let ab = sub(b, a);
    let l2 = len2(ab);
    let t = if l2 > 0.0 { (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    add(a, scale(ab, t))
}

/// The nearest points of segments `p1q1` and `p2q2` (Ericson 5.1.9).
pub fn closest_segments(p1: V3, q1: V3, p2: V3, q2: V3) -> (V3, V3) {
    let (d1, d2, r) = (sub(q1, p1), sub(q2, p2), sub(p1, p2));
    let (a, e, f) = (len2(d1), len2(d2), dot(d2, r));
    let eps = 1e-12;
    let (mut s, mut t);
    if a <= eps && e <= eps {
        return (p1, p2);
    }
    if a <= eps {
        (s, t) = (0.0, (f / e).clamp(0.0, 1.0));
    } else {
        let c = dot(d1, r);
        if e <= eps {
            (s, t) = ((-c / a).clamp(0.0, 1.0), 0.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            s = if denom != 0.0 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            t = (b * s + f) / e;
            if t < 0.0 {
                (s, t) = ((-c / a).clamp(0.0, 1.0), 0.0);
            }
            if t > 1.0 {
                (s, t) = (((b - c) / a).clamp(0.0, 1.0), 1.0);
            }
        }
    }
    (add(p1, scale(d1, s)), add(p2, scale(d2, t)))
}

/// The point of triangle `abc` nearest `p` (Ericson 5.1.5).
pub fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add(a, scale(ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add(a, scale(ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    add(add(a, scale(ab, vb * denom)), scale(ac, vc * denom))
}

/// Sphere `a` against the capsule whose straight part is `seg`.
pub fn sphere_capsule(c: V3, r: f32, seg: (V3, V3), rc: f32, slop: f32) -> Manifold {
    spheres(c, r, closest_on_segment(c, seg.0, seg.1), rc, slop)
}

pub fn capsules(sa: (V3, V3), ra: f32, sb: (V3, V3), rb: f32, slop: f32) -> Manifold {
    let (pa, pb) = closest_segments(sa.0, sa.1, sb.0, sb.1);
    spheres(pa, ra, pb, rb, slop)
}

/// Sphere `a` against a box at `x`, `q` with `half` extents.
pub fn sphere_box(c: V3, r: f32, x: V3, q: Q, half: V3, slop: f32) -> Manifold {
    let cl = unrotate(q, sub(c, x));
    let p = map3(|k| cl[k].clamp(-half[k], half[k]));
    let d = sub(cl, p);
    let dist = len(d);
    // Centre inside: push out through the nearest face.
    if dist == 0.0 {
        let gap = map3(|k| half[k] - cl[k].abs());
        let k = (0..3).min_by(|&i, &j| gap[i].partial_cmp(&gap[j]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(2);
        let mut out = [0.0; 3];
        out[k] = 1f32.copysign(cl[k]);
        let mut face = cl;
        face[k] = half[k].copysign(cl[k]);
        let n = neg(rotate(q, out));
        return Manifold::one(n, add(c, scale(n, r)), add(x, rotate(q, face)));
    }
    if dist - r > slop {
        return Manifold::none();
    }
    let n = neg(rotate(q, scale(d, 1.0 / dist)));
    Manifold::one(n, add(c, scale(n, r)), add(x, rotate(q, p)))
}

/// Sphere `a` against triangle `t`.
pub fn sphere_triangle(c: V3, r: f32, t: [V3; 3], slop: f32) -> Manifold {
    let p = closest_on_triangle(c, t[0], t[1], t[2]);
    let d = sub(p, c);
    let dist = len(d);
    if dist - r > slop {
        return Manifold::none();
    }
    let n = if dist > 0.0 { scale(d, 1.0 / dist) } else { norm(cross(sub(t[1], t[0]), sub(t[2], t[0]))) };
    Manifold::one(n, add(c, scale(n, r)), p)
}
