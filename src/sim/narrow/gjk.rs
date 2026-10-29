//! GJK (ROADMAP.md §2.2 "Collision", narrowphase: anything with a cylinder
//! or cone, hulls, and capsules against boxes): the distance between two
//! convex shapes through their support functions, with the witness
//! points, or, when they overlap, the simplex around the origin that EPA
//! grows. The simplex is reduced by the closest point on it (Ericson 5.1.5
//! for the triangle), not by Johnson's sub-algorithm.

use super::super::math::*;
use super::super::shape::support_world;
use super::super::state::Shape;
use std::cmp::Ordering;

/// One convex thing in world: a placed shape or a mesh triangle.
#[derive(Clone, Copy)]
pub enum Convex<'a> {
    Shape { s: &'a Shape, hull: &'a [[f32; 4]], x: V3, q: Q },
    Tri([V3; 3]),
}

impl Convex<'_> {
    pub fn support(&self, d: V3) -> V3 {
        match *self {
            Convex::Shape { s, hull, x, q } => support_world(s, hull, x, q, d),
            Convex::Tri(t) => t.into_iter().max_by(|a, b| dot(*a, d).partial_cmp(&dot(*b, d)).unwrap_or(Ordering::Equal)).unwrap_or(t[0]),
        }
    }

    pub fn centre(&self) -> V3 {
        match *self {
            Convex::Shape { x, .. } => x,
            Convex::Tri(t) => scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0),
        }
    }
}

/// A vertex of the Minkowski difference `a - b` with the points it came from.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vert {
    pub p: V3,
    pub a: V3,
    pub b: V3,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Simplex {
    pub v: [Vert; 4],
    pub n: usize,
}

pub enum Gjk {
    /// Nearest points and their distance.
    Apart { a: V3, b: V3, dist: f32 },
    /// The shapes overlap; the simplex holds the origin or touches it.
    Overlap(Simplex),
}

pub fn minkowski(a: &Convex, b: &Convex, d: V3) -> Vert {
    let (pa, pb) = (a.support(d), b.support(neg(d)));
    Vert { p: sub(pa, pb), a: pa, b: pb }
}

/// Weights of the origin's nearest point on segment `p0 p1`.
fn on_segment(p0: V3, p1: V3) -> [f32; 2] {
    let d = sub(p1, p0);
    let l2 = len2(d);
    let t = if l2 > 0.0 { (-dot(p0, d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    [1.0 - t, t]
}

/// Weights of the origin's nearest point on a triangle (Ericson 5.1.5 with the point at the origin).
fn on_triangle(p: [V3; 3]) -> [f32; 3] {
    let (ab, ac) = (sub(p[1], p[0]), sub(p[2], p[0]));
    let (d1, d2) = (-dot(ab, p[0]), -dot(ac, p[0]));
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let (d3, d4) = (-dot(ab, p[1]), -dot(ac, p[1]));
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let (d5, d6) = (-dot(ab, p[2]), -dot(ac, p[2]));
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let denom = 1.0 / (va + vb + vc);
    [va * denom, vb * denom, vc * denom]
}

/// Keeps the vertices carrying weight; returns the point they weigh to.
fn reduce(s: &mut Simplex, w: [f32; 4]) -> (V3, [f32; 4]) {
    let (mut kept, mut n, mut c) = ([Vert::default(); 4], 0, [0.0; 3]);
    let mut weights = [0.0; 4];
    for i in 0..s.n {
        if w[i] > 0.0 {
            kept[n] = s.v[i];
            weights[n] = w[i];
            c = add(c, scale(s.v[i].p, w[i]));
            n += 1;
        }
    }
    *s = Simplex { v: kept, n };
    (c, weights)
}

/// The simplex's nearest point to the origin and its weights, the simplex
/// reduced to what carries them; `None` once a tetrahedron holds the origin.
fn closest(s: &mut Simplex) -> Option<(V3, [f32; 4])> {
    let p = |i: usize| s.v[i].p;
    match s.n {
        1 => Some((p(0), [1.0, 0.0, 0.0, 0.0])),
        2 => {
            let w = on_segment(p(0), p(1));
            Some(reduce(s, [w[0], w[1], 0.0, 0.0]))
        }
        3 => {
            let w = on_triangle([p(0), p(1), p(2)]);
            Some(reduce(s, [w[0], w[1], w[2], 0.0]))
        }
        _ => {
            let faces = [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2], [1, 3, 2, 0]];
            let vol = dot(cross(sub(p(1), p(0)), sub(p(2), p(0))), sub(p(3), p(0)));
            let mut best: Option<(f32, [f32; 4])> = None;
            for [i, j, k, l] in faces {
                let n = cross(sub(p(j), p(i)), sub(p(k), p(i)));
                let outward = if dot(n, sub(p(l), p(i))) > 0.0 { neg(n) } else { n };
                let outside = dot(outward, p(i)) < 0.0 || vol.abs() < 1e-12;
                if !outside {
                    continue;
                }
                let w = on_triangle([p(i), p(j), p(k)]);
                let c = add(add(scale(p(i), w[0]), scale(p(j), w[1])), scale(p(k), w[2]));
                let mut full = [0.0; 4];
                (full[i], full[j], full[k]) = (w[0], w[1], w[2]);
                if best.is_none_or(|(d, _)| len2(c) < d) {
                    best = Some((len2(c), full));
                }
            }
            best.map(|(_, w)| reduce(s, w))
        }
    }
}

/// The distance between `a` and `b`, or the simplex EPA starts from.
pub fn gjk(a: &Convex, b: &Convex) -> Gjk {
    let mut d = sub(a.centre(), b.centre());
    if len2(d) == 0.0 {
        d = [1.0, 0.0, 0.0];
    }
    let mut s = Simplex { v: [minkowski(a, b, d); 4], n: 1 };
    let mut last = (s.v[0].a, s.v[0].b, len(s.v[0].p));
    for _ in 0..64 {
        let Some((c, w)) = closest(&mut s) else { return Gjk::Overlap(s) };
        let c2 = len2(c);
        if c2 < 1e-12 {
            return Gjk::Overlap(s);
        }
        let (mut wa, mut wb) = ([0.0; 3], [0.0; 3]);
        for i in 0..s.n {
            wa = add(wa, scale(s.v[i].a, w[i]));
            wb = add(wb, scale(s.v[i].b, w[i]));
        }
        last = (wa, wb, c2.sqrt());
        let v = minkowski(a, b, neg(c));
        let known = s.v[..s.n].iter().any(|k| len2(sub(k.p, v.p)) < 1e-14);
        if c2 - dot(v.p, c) <= 1e-6 * c2 || known {
            break;
        }
        s.v[s.n] = v;
        s.n += 1;
    }
    Gjk::Apart { a: last.0, b: last.1, dist: last.2 }
}
