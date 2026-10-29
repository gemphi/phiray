//! SAT with reference-face clipping (ROADMAP.md §2.2 "Collision",
//! narrowphase: box-box and convex-triangle): the axis of least overlap
//! among the face normals and the edge crosses, then Sutherland-Hodgman
//! clipping of the incident face against the reference face's side planes
//! for up to four points in one frame. Boxes and triangles are both a
//! `Poly`; a hull with faces would be one too.

use super::super::math::*;
use super::closed::closest_segments;
use super::manifold::{Manifold, Point};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Default)]
pub struct Face {
    pub n: V3,
    pub v: [u8; 4],
    pub nv: usize,
}

/// A small polytope in world space; `edges[..axes]` have distinct directions.
pub struct Poly {
    pub v: [V3; 8],
    pub nv: usize,
    pub faces: [Face; 6],
    pub nf: usize,
    pub edges: [[u8; 2]; 12],
    pub ne: usize,
    pub axes: usize,
}

impl Poly {
    /// Corner `i` has bit `k` set when it lies at `+half[k]`.
    pub fn boxed(x: V3, q: Q, half: V3) -> Self {
        let mut v = [[0.0; 3]; 8];
        for (i, p) in v.iter_mut().enumerate() {
            let s = |bit: usize| if (i >> bit) & 1 == 1 { 1.0 } else { -1.0 };
            *p = add(x, rotate(q, [half[0] * s(0), half[1] * s(1), half[2] * s(2)]));
        }
        let axis = |k: usize, sign: f32| {
            let mut e = [0.0; 3];
            e[k] = sign;
            rotate(q, e)
        };
        let faces = [
            Face { n: axis(0, 1.0), v: [1, 3, 7, 5], nv: 4 },
            Face { n: axis(0, -1.0), v: [0, 4, 6, 2], nv: 4 },
            Face { n: axis(1, 1.0), v: [2, 6, 7, 3], nv: 4 },
            Face { n: axis(1, -1.0), v: [0, 1, 5, 4], nv: 4 },
            Face { n: axis(2, 1.0), v: [4, 5, 7, 6], nv: 4 },
            Face { n: axis(2, -1.0), v: [0, 2, 3, 1], nv: 4 },
        ];
        let edges = [[0, 1], [0, 2], [0, 4], [2, 3], [4, 5], [6, 7], [1, 3], [4, 6], [5, 7], [1, 5], [2, 6], [3, 7]];
        Self { v, nv: 8, faces, nf: 6, edges, ne: 12, axes: 3 }
    }

    /// A triangle as a two-sided face.
    pub fn triangle(t: [V3; 3]) -> Self {
        let n = norm(cross(sub(t[1], t[0]), sub(t[2], t[0])));
        let mut v = [[0.0; 3]; 8];
        v[..3].copy_from_slice(&t);
        let mut faces = [Face::default(); 6];
        faces[0] = Face { n, v: [0, 1, 2, 0], nv: 3 };
        faces[1] = Face { n: neg(n), v: [0, 2, 1, 0], nv: 3 };
        let mut edges = [[0u8; 2]; 12];
        edges[..3].copy_from_slice(&[[0, 1], [1, 2], [2, 0]]);
        Self { v, nv: 3, faces, nf: 2, edges, ne: 3, axes: 3 }
    }

    fn extent(&self, axis: V3) -> (f32, f32) {
        self.v[..self.nv].iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            let t = dot(*p, axis);
            (lo.min(t), hi.max(t))
        })
    }

    /// The face whose normal leans most along `d`.
    fn facing(&self, d: V3) -> &Face {
        let i = (0..self.nf).max_by(|&i, &j| dot(self.faces[i].n, d).partial_cmp(&dot(self.faces[j].n, d)).unwrap_or(Ordering::Equal)).unwrap_or(0);
        &self.faces[i]
    }

    fn edge(&self, e: usize) -> (V3, V3) {
        (self.v[self.edges[e][0] as usize], self.v[self.edges[e][1] as usize])
    }

    fn edge_dir(&self, e: usize) -> V3 {
        let (p, q) = self.edge(e);
        sub(q, p)
    }

    fn face_centre(&self, f: &Face) -> V3 {
        scale(f.v[..f.nv].iter().fold([0.0; 3], |a, &k| add(a, self.v[k as usize])), 1.0 / f.nv as f32)
    }
}

#[derive(Clone, Copy)]
enum Source {
    FaceA,
    FaceB,
    Edge(usize, usize),
}

struct Best {
    overlap: f32,
    axis: V3,
    source: Source,
}

/// The overlap along unit `u` and the a-to-b normal it implies.
fn overlap(a: &Poly, b: &Poly, u: V3) -> (f32, V3) {
    let (alo, ahi) = a.extent(u);
    let (blo, bhi) = b.extent(u);
    let (o1, o2) = (ahi - blo, bhi - alo);
    if o1 <= o2 { (o1, u) } else { (o2, neg(u)) }
}

/// Contacts between `a` and `b`, or none past `slop` apart. `prefer` forces
/// a mesh triangle's normal as the contact normal once no axis separates.
pub fn sat(a: &Poly, b: &Poly, slop: f32, prefer: Option<V3>) -> Manifold {
    let mut best = Best { overlap: f32::MAX, axis: [0.0; 3], source: Source::FaceA };
    let consider = |u: V3, source: Source, bias: f32, best: &mut Best| {
        let (o, n) = overlap(a, b, u);
        if o + bias < best.overlap {
            *best = Best { overlap: o, axis: n, source };
        }
        o >= -slop
    };
    for f in &a.faces[..a.nf] {
        if !consider(f.n, Source::FaceA, 0.0, &mut best) {
            return Manifold::none();
        }
    }
    for f in &b.faces[..b.nf] {
        if !consider(f.n, Source::FaceB, 0.0, &mut best) {
            return Manifold::none();
        }
    }
    for ea in 0..a.axes {
        for eb in 0..b.axes {
            let u = cross(a.edge_dir(ea), b.edge_dir(eb));
            let l = len(u);
            if l < 1e-6 {
                continue;
            }
            if !consider(scale(u, 1.0 / l), Source::Edge(ea, eb), 1e-3, &mut best) {
                return Manifold::none();
            }
        }
    }
    if let Some(n) = prefer {
        let (o, axis) = overlap(a, b, n);
        best = Best { overlap: o, axis, source: Source::FaceB };
    }
    match best.source {
        Source::FaceA => clip(a, b, best.axis, slop, false),
        Source::FaceB => clip(a, b, best.axis, slop, true),
        Source::Edge(ea, eb) => edge_contact(a, b, ea, eb, best.axis),
    }
}

/// Keeps the part of `poly` on the inner side of the plane through `p0` with normal `s`.
fn clip_plane(poly: &[V3], p0: V3, s: V3) -> Vec<V3> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let (cur, next) = (poly[i], poly[(i + 1) % poly.len()]);
        let (dc, dn) = (dot(sub(cur, p0), s), dot(sub(next, p0), s));
        if dc >= -1e-5 {
            out.push(cur);
        }
        if (dc >= -1e-5) != (dn >= -1e-5) {
            out.push(add(cur, scale(sub(next, cur), dc / (dc - dn))));
        }
    }
    out
}

/// Reference face on `b` when `on_b`, else on `a`; `n` is the a-to-b normal.
fn clip(a: &Poly, b: &Poly, n: V3, slop: f32, on_b: bool) -> Manifold {
    let (r, i, m) = if on_b { (b, a, neg(n)) } else { (a, b, n) };
    let rf = r.facing(m);
    let inc = i.facing(neg(m));
    let mut poly: Vec<V3> = inc.v[..inc.nv].iter().map(|&k| i.v[k as usize]).collect();
    let rc = r.face_centre(rf);
    for e in 0..rf.nv {
        let (p0, p1) = (r.v[rf.v[e] as usize], r.v[rf.v[(e + 1) % rf.nv] as usize]);
        let mut s = norm(cross(rf.n, sub(p1, p0)));
        if dot(s, sub(rc, p0)) < 0.0 {
            s = neg(s);
        }
        poly = clip_plane(&poly, p0, s);
    }
    let on = r.v[rf.v[0] as usize];
    let cands: Vec<Point> = poly
        .iter()
        .map(|&p| (p, dot(sub(on, p), rf.n)))
        .filter(|&(_, depth)| depth > -slop)
        .map(|(p, depth)| if on_b { Point { a: p, b: sub(p, scale(n, depth)) } } else { Point { a: add(p, scale(n, depth)), b: p } })
        .collect();
    Manifold::gather(n, &cands)
}

/// The nearest points of the two edges that reach farthest along the axis.
fn edge_contact(a: &Poly, b: &Poly, ea: usize, eb: usize, n: V3) -> Manifold {
    let pick = |p: &Poly, dir: V3, sign: f32| {
        let along = |e: usize| {
            let (p0, p1) = p.edge(e);
            sign * dot(add(p0, p1), n)
        };
        let parallel = |e: usize| len2(cross(p.edge_dir(e), dir)) < 1e-8 * len2(p.edge_dir(e)) * len2(dir);
        let e = (0..p.ne).filter(|&e| parallel(e)).max_by(|&i, &j| along(i).partial_cmp(&along(j)).unwrap_or(Ordering::Equal)).unwrap_or(0);
        p.edge(e)
    };
    let (a0, a1) = pick(a, a.edge_dir(ea), 1.0);
    let (b0, b1) = pick(b, b.edge_dir(eb), -1.0);
    let (pa, pb) = closest_segments(a0, a1, b0, b1);
    Manifold::one(n, pa, pb)
}
