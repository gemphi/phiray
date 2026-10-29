//! EPA (ROADMAP.md §2.2 "Collision", narrowphase): grows GJK's simplex
//! around the origin of the Minkowski difference until a face of the
//! polytope reaches its surface; that face's normal is the penetration
//! direction, its distance the depth, and its barycentric witness points
//! the deepest pair. The other points of a resting manifold come from
//! supports tilted around the normal, the roadmap's rotated support
//! queries (four here, one per tangent direction).

use super::super::math::*;
use super::gjk::{Convex, Simplex, Vert, minkowski};
use super::manifold::{Manifold, Point};
use std::cmp::Ordering;

/// The deepest point `a` of the first shape, `depth` inside the second
/// along `normal` (from `a` to `b`); the point on the second is `a - normal * depth`.
#[derive(Clone, Copy, Debug)]
pub struct Epa {
    pub normal: V3,
    pub depth: f32,
    pub a: V3,
}

const MAX_VERTS: usize = 64;
const MAX_FACES: usize = 192;

#[derive(Clone, Copy)]
struct Face {
    v: [usize; 3],
    n: V3,
    d: f32,
}

/// A face with its normal pointing away from `centre`, and its distance.
fn face(verts: &[Vert], i: usize, j: usize, k: usize, centre: V3) -> Face {
    let (pi, pj, pk) = (verts[i].p, verts[j].p, verts[k].p);
    let n = norm(cross(sub(pj, pi), sub(pk, pi)));
    if dot(n, sub(pi, centre)) < 0.0 {
        return Face { v: [i, k, j], n: neg(n), d: -dot(n, pi) };
    }
    Face { v: [i, j, k], n, d: dot(n, pi) }
}

/// Whether `p` lies off the simplex's affine hull.
fn independent(s: &Simplex, p: V3) -> bool {
    let d = |i: usize| sub(s.v[i].p, s.v[0].p);
    let eps = 1e-10;
    match s.n {
        1 => len2(sub(p, s.v[0].p)) > eps,
        2 => len2(cross(d(1), sub(p, s.v[0].p))) > eps,
        _ => dot(cross(d(1), d(2)), sub(p, s.v[0].p)).abs() > eps,
    }
}

/// Grows a point, segment or triangle into a tetrahedron with supports
/// along directions the simplex does not span.
fn blow_up(a: &Convex, b: &Convex, s: &mut Simplex) -> bool {
    let axes = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    let mut tries = 0;
    while s.n < 4 && tries < 12 {
        let d = match s.n {
            1 => axes[tries % 6],
            2 => {
                let e = norm(sub(s.v[1].p, s.v[0].p));
                let (t1, t2) = basis(e);
                [t1, t2, neg(t1), neg(t2)][tries % 4]
            }
            _ => {
                let n = norm(cross(sub(s.v[1].p, s.v[0].p), sub(s.v[2].p, s.v[0].p)));
                if tries % 2 == 0 { n } else { neg(n) }
            }
        };
        let v = minkowski(a, b, d);
        if independent(s, v.p) {
            s.v[s.n] = v;
            s.n += 1;
        }
        tries += 1;
    }
    s.n == 4
}

/// The point of the first shape under the point `f.n * f.d` of face `f`.
fn witness(verts: &[Vert], f: &Face) -> V3 {
    let (p0, p1, p2) = (verts[f.v[0]].p, verts[f.v[1]].p, verts[f.v[2]].p);
    let (v0, v1, v2) = (sub(p1, p0), sub(p2, p0), sub(scale(f.n, f.d), p0));
    let (d00, d01, d11, d20, d21) = (dot(v0, v0), dot(v0, v1), dot(v1, v1), dot(v2, v0), dot(v2, v1));
    let denom = d00 * d11 - d01 * d01;
    let (w1, w2) = if denom.abs() > 1e-20 { ((d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom) } else { (0.0, 0.0) };
    let w = [1.0 - w1 - w2, w1, w2];
    (0..3).fold([0.0; 3], |acc, k| add(acc, scale(verts[f.v[k]].a, w[k])))
}

/// Penetration of overlapping `a` and `b` from GJK's simplex; `None` when
/// the simplex cannot be grown around the origin.
pub fn epa(a: &Convex, b: &Convex, mut s: Simplex) -> Option<Epa> {
    if !blow_up(a, b, &mut s) {
        return None;
    }
    let mut verts: Vec<Vert> = s.v.to_vec();
    let centre = scale(verts.iter().fold([0.0; 3], |c, v| add(c, v.p)), 0.25);
    let mut faces: Vec<Face> = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]].iter().map(|f| face(&verts, f[0], f[1], f[2], centre)).collect();
    let nearest = |faces: &[Face]| (0..faces.len()).min_by(|&i, &j| faces[i].d.partial_cmp(&faces[j].d).unwrap_or(Ordering::Equal)).unwrap_or(0);
    for _ in 0..64 {
        let f = faces[nearest(&faces)];
        let v = minkowski(a, b, f.n);
        let reach = dot(v.p, f.n);
        if reach - f.d < 1e-5 * f.d.max(1.0) || verts.len() >= MAX_VERTS || faces.len() >= MAX_FACES {
            return Some(Epa { normal: f.n, depth: f.d, a: witness(&verts, &f) });
        }
        let vi = verts.len();
        verts.push(v);
        let mut horizon: Vec<[usize; 2]> = Vec::new();
        faces.retain(|f| {
            let visible = dot(f.n, v.p) > f.d;
            if visible {
                for e in 0..3 {
                    let edge = [f.v[e], f.v[(e + 1) % 3]];
                    match horizon.iter().position(|h| h[0] == edge[1] && h[1] == edge[0]) {
                        Some(at) => {
                            horizon.remove(at);
                        }
                        None => horizon.push(edge),
                    }
                }
            }
            !visible
        });
        for e in horizon {
            faces.push(face(&verts, e[0], e[1], vi, centre));
        }
    }
    let f = faces[nearest(&faces)];
    Some(Epa { normal: f.n, depth: f.d, a: witness(&verts, &f) })
}

/// The deepest pair plus the points supports tilted about the normal
/// reach, so a flat face or a rim rests on up to four points.
pub fn manifold(a: &Convex, b: &Convex, e: &Epa, slop: f32) -> Manifold {
    let n = e.normal;
    let (t1, t2) = basis(n);
    let mut cands = vec![Point { a: e.a, b: sub(e.a, scale(n, e.depth)) }];
    let (sa0, sb0) = (a.support(n), b.support(neg(n)));
    for t in [t1, neg(t1), t2, neg(t2)] {
        let d = norm(add(n, scale(t, 0.15)));
        let pa = a.support(d);
        let depth = dot(sub(pa, sb0), n);
        if depth > -slop {
            cands.push(Point { a: pa, b: sub(pa, scale(n, depth)) });
        }
        let pb = b.support(neg(d));
        let depth = dot(sub(sa0, pb), n);
        if depth > -slop {
            cands.push(Point { a: add(pb, scale(n, depth)), b: pb });
        }
    }
    Manifold::gather(n, &cands)
}
