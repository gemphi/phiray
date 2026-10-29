//! The narrowphase's answer for one pair (ROADMAP.md §2.2 "Collision"):
//! up to four point pairs under one normal, found in one frame with no
//! frame-to-frame matching. `gather` keeps the four that hold a resting
//! body best: the deepest, the farthest from it, then the widest spread.

use super::super::math::*;

/// One contact point pair: `a` on the first body's surface, `b` on the second's.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub a: V3,
    pub b: V3,
}

/// Up to four point pairs under one normal (from `a` to `b`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Manifold {
    pub normal: V3,
    pub points: [Point; 4],
    pub n: usize,
}

impl Manifold {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn one(normal: V3, a: V3, b: V3) -> Self {
        let mut m = Self { normal, ..Self::default() };
        m.push(a, b);
        m
    }

    /// Adds a pair; a fifth is dropped.
    pub fn push(&mut self, a: V3, b: V3) {
        if self.n < 4 {
            self.points[self.n] = Point { a, b };
            self.n += 1;
        }
    }

    /// The same contact seen from the other body.
    pub fn flip(mut self) -> Self {
        self.normal = neg(self.normal);
        for p in &mut self.points[..self.n] {
            std::mem::swap(&mut p.a, &mut p.b);
        }
        self
    }

    /// Penetration of pair `i` along the normal (negative when apart).
    pub fn depth(&self, i: usize) -> f32 {
        dot(sub(self.points[i].a, self.points[i].b), self.normal)
    }

    /// Re-derives every pair against the plane through `on` with unit
    /// normal `n` (a mesh triangle's): `a` stays, `b` is its foot on the
    /// plane. Ghost normals from a triangle's edges go away with this.
    pub fn along(mut self, n: V3, on: V3) -> Self {
        self.normal = n;
        for p in &mut self.points[..self.n] {
            let depth = dot(sub(p.a, on), n);
            p.b = sub(p.a, scale(n, depth));
        }
        self
    }

    /// The manifold of up to four of `cands` (pairs under `normal`), near
    /// duplicates dropped: the deepest, the farthest from it, the one
    /// spanning the most area with them, then the one farthest from all.
    pub fn gather(normal: V3, cands: &[Point]) -> Self {
        let mut pts: Vec<Point> = Vec::with_capacity(cands.len());
        for c in cands {
            if !pts.iter().any(|p| len2(sub(p.a, c.a)) < 1e-8) {
                pts.push(*c);
            }
        }
        let mut m = Self { normal, ..Self::default() };
        if pts.len() <= 4 {
            for p in &pts {
                m.push(p.a, p.b);
            }
            return m;
        }
        let depth = |p: &Point| dot(sub(p.a, p.b), normal);
        let pick = |pts: &[Point], score: &dyn Fn(&Point) -> f32| {
            (0..pts.len()).max_by(|&i, &j| score(&pts[i]).partial_cmp(&score(&pts[j])).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0)
        };
        let i0 = pick(&pts, &depth);
        let p0 = pts[i0];
        let i1 = pick(&pts, &|p| len2(sub(p.a, p0.a)));
        let p1 = pts[i1];
        let i2 = pick(&pts, &|p| len2(cross(sub(p.a, p0.a), sub(p.a, p1.a))));
        let p2 = pts[i2];
        let i3 = pick(&pts, &|p| [p0, p1, p2].iter().map(|c| len2(sub(p.a, c.a))).fold(f32::MAX, f32::min));
        let mut chosen = vec![i0, i1, i2, i3];
        chosen.dedup();
        for i in chosen {
            m.push(pts[i].a, pts[i].b);
        }
        m
    }
}
