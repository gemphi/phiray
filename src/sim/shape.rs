//! Shapes through one support function (ROADMAP.md §2.2 "State layout",
//! the `support` `match`): a convex shape is a function from a direction
//! to its farthest point, so GJK/EPA, the swept bounds and the manifold
//! queries take every kind through one path, and the WGSL twin is a
//! `switch` on the same values. Mesh shapes are passive triangle soups
//! and never take it.

use super::math::*;
use super::state::{Shape, ShapeKind};
use crate::bvh::Aabb;

/// The farthest point of `s` along `d`, in the shape's own space.
pub fn support(s: &Shape, hull: &[[f32; 4]], d: V3) -> V3 {
    match s.kind() {
        ShapeKind::Sphere => scale(norm(d), s.radius),
        ShapeKind::Box => map3(|k| s.half[k].copysign(d[k])),
        ShapeKind::Capsule => add(scale(norm(d), s.radius), [0.0, 0.0, s.half[2].copysign(d[2])]),
        ShapeKind::Cylinder => cylinder_support(s, d),
        ShapeKind::Cone => cone_support(s, d),
        ShapeKind::Hull => hull_support(&hull[s.data as usize..][..s.count as usize], d),
        ShapeKind::Mesh => unreachable!("mesh shapes never take the convex path"),
    }
}

/// `support` for a shape placed at `x` with orientation `q`, `d` in world.
pub fn support_world(s: &Shape, hull: &[[f32; 4]], x: V3, q: Q, d: V3) -> V3 {
    add(x, rotate(q, support(s, hull, unrotate(q, d))))
}

fn cylinder_support(s: &Shape, d: V3) -> V3 {
    let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let rim = if r > 0.0 { [d[0] / r * s.radius, d[1] / r * s.radius] } else { [0.0, 0.0] };
    [rim[0], rim[1], s.half[2].copysign(d[2])]
}

/// Apex at `+half` or the base rim at `-half`, whichever lies farther.
fn cone_support(s: &Shape, d: V3) -> V3 {
    let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let apex = [0.0, 0.0, s.half[2]];
    let base = if r > 0.0 { [d[0] / r * s.radius, d[1] / r * s.radius, -s.half[2]] } else { [0.0, 0.0, -s.half[2]] };
    if dot(d, apex) >= dot(d, base) { apex } else { base }
}

fn hull_support(points: &[[f32; 4]], d: V3) -> V3 {
    let mut best = ([0.0; 3], f32::MIN);
    for p in points {
        let v = [p[0], p[1], p[2]];
        let t = dot(v, d);
        if t > best.1 {
            best = (v, t);
        }
    }
    best.0
}

/// World bounds of a convex shape at a pose: its support along each axis.
pub fn bounds(s: &Shape, hull: &[[f32; 4]], x: V3, q: Q) -> Aabb {
    let axis = |k: usize, sign: f32| {
        let mut e = [0.0; 3];
        e[k] = sign;
        support_world(s, hull, x, q, e)[k]
    };
    Aabb::new(map3(|k| axis(k, -1.0)), map3(|k| axis(k, 1.0)))
}

/// Principal inertia of a solid of `mass` in the body frame; hulls and
/// meshes take the box of their bounds (Bullet approximates the same way).
pub fn inertia(s: &Shape, mass: f32) -> V3 {
    let boxed = |h: V3| {
        let (x, y, z) = (4.0 * h[0] * h[0], 4.0 * h[1] * h[1], 4.0 * h[2] * h[2]);
        scale([y + z, x + z, x + y], mass / 12.0)
    };
    let r2 = s.radius * s.radius;
    match s.kind() {
        ShapeKind::Sphere => [0.4 * mass * r2; 3],
        ShapeKind::Box => boxed(s.half),
        ShapeKind::Capsule => boxed([s.radius, s.radius, s.half[2] + s.radius]),
        ShapeKind::Cylinder => {
            let h2 = 4.0 * s.half[2] * s.half[2];
            [mass * (3.0 * r2 + h2) / 12.0, mass * (3.0 * r2 + h2) / 12.0, 0.5 * mass * r2]
        }
        ShapeKind::Cone => {
            let h2 = 4.0 * s.half[2] * s.half[2];
            let side = mass * (3.0 * r2 / 20.0 + 3.0 * h2 / 80.0);
            [side, side, 0.3 * mass * r2]
        }
        ShapeKind::Hull => boxed(s.half),
        ShapeKind::Mesh => boxed(s.half),
    }
}

/// The capsule's straight part in world: its two end centres.
pub fn segment(s: &Shape, x: V3, q: Q) -> (V3, V3) {
    let h = rotate(q, [0.0, 0.0, s.half[2]]);
    (sub(x, h), add(x, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_are_the_farthest_points() {
        let hull = [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [-1.0, -1.0, 3.0, 0.0]];
        let h = Shape::hull(0, 3, [1.0, 1.5, 1.5]);
        assert_eq!(support(&h, &hull, [0.0, 0.0, 1.0]), [-1.0, -1.0, 3.0]);
        assert_eq!(support(&Shape::boxed([1.0, 2.0, 3.0]), &[], [-1.0, 1.0, -0.5]), [-1.0, 2.0, -3.0]);
        let cyl = support(&Shape::cylinder(2.0, 1.0), &[], [3.0, 4.0, -1.0]);
        assert!(len(sub(cyl, [1.2, 1.6, -1.0])) < 1e-6);
        let cone = Shape::cone(1.0, 1.0);
        assert_eq!(support(&cone, &[], [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]);
        assert_eq!(support(&cone, &[], [1.0, 0.0, 0.0]), [1.0, 0.0, -1.0]);
        let q = q_axis([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2);
        let b = bounds(&Shape::boxed([2.0, 1.0, 0.5]), &[], [10.0, 0.0, 0.0], q);
        assert!(len(sub(b.lo, [9.0, -2.0, -0.5])) < 1e-5 && len(sub(b.hi, [11.0, 2.0, 0.5])) < 1e-5);
    }
}
