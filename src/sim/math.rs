//! Vectors, quaternions and matrices for the solver (ROADMAP.md §2.2 "State
//! layout"): plain `[f32; N]` arrays and free functions, so the CPU twin and
//! the WGSL kernels keep one layout and one arithmetic. Quaternions are
//! `xyzw`; matrices are Blender's `m[col][row]`.

pub type V3 = [f32; 3];
pub type Q = [f32; 4];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn mul3(a: V3, b: V3) -> V3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

pub fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn len2(a: V3) -> f32 {
    dot(a, a)
}

pub fn len(a: V3) -> f32 {
    len2(a).sqrt()
}

/// `a / |a|`, or zero for zero.
pub fn norm(a: V3) -> V3 {
    let l = len(a);
    if l > 0.0 { scale(a, 1.0 / l) } else { [0.0; 3] }
}

pub fn map3(f: impl Fn(usize) -> f32) -> V3 {
    [f(0), f(1), f(2)]
}

pub fn neg(a: V3) -> V3 {
    [-a[0], -a[1], -a[2]]
}

/// A unit vector at right angles to unit `n` (the axis `n` leans least on).
pub fn any_perp(n: V3) -> V3 {
    let k = (0..3).min_by(|&i, &j| n[i].abs().partial_cmp(&n[j].abs()).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
    let mut e = [0.0; 3];
    e[k] = 1.0;
    norm(cross(n, e))
}

/// Two unit tangents completing unit `n` to a right-handed frame.
pub fn basis(n: V3) -> (V3, V3) {
    let t = any_perp(n);
    (t, cross(n, t))
}

pub fn q_mul(a: Q, b: Q) -> Q {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

pub fn q_conj(q: Q) -> Q {
    [-q[0], -q[1], -q[2], q[3]]
}

pub fn q_norm(q: Q) -> Q {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l > 0.0 { [q[0] / l, q[1] / l, q[2] / l, q[3] / l] } else { [0.0, 0.0, 0.0, 1.0] }
}

/// `q v q*`: body to world.
pub fn rotate(q: Q, v: V3) -> V3 {
    let u = [q[0], q[1], q[2]];
    let t = scale(cross(u, v), 2.0);
    add(add(v, scale(t, q[3])), cross(u, t))
}

/// `q* v q`: world to body.
pub fn unrotate(q: Q, v: V3) -> V3 {
    rotate(q_conj(q), v)
}

/// The quaternion about unit `axis` by `angle`.
pub fn q_axis(axis: V3, angle: f32) -> Q {
    let (s, c) = (0.5 * angle).sin_cos();
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

/// `q + h/2 [w, 0] q`, normalised: one explicit step of the orientation.
pub fn q_integrate(q: Q, w: V3, h: f32) -> Q {
    let d = q_mul([w[0], w[1], w[2], 0.0], q);
    q_norm([q[0] + 0.5 * h * d[0], q[1] + 0.5 * h * d[1], q[2] + 0.5 * h * d[2], q[3] + 0.5 * h * d[3]])
}

/// The angular velocity that took `prev` to `q` in `h` (Muller 2020, eq. 11).
pub fn q_delta_w(q: Q, prev: Q, h: f32) -> V3 {
    let d = q_mul(q, q_conj(prev));
    let w = scale([d[0], d[1], d[2]], 2.0 / h);
    if d[3] < 0.0 { neg(w) } else { w }
}

/// The columns of the rotation matrix of `q`.
pub fn mat_of(q: Q) -> [V3; 3] {
    [rotate(q, [1.0, 0.0, 0.0]), rotate(q, [0.0, 1.0, 0.0]), rotate(q, [0.0, 0.0, 1.0])]
}

/// The unit quaternion of a rotation given as columns (Shepperd's method,
/// the branch chosen by the largest diagonal term).
pub fn q_from_cols(c: [V3; 3]) -> Q {
    let m = |r: usize, k: usize| c[k][r];
    let tr = m(0, 0) + m(1, 1) + m(2, 2);
    let diag = [m(0, 0), m(1, 1), m(2, 2)];
    let big = (0..3).max_by(|&i, &j| diag[i].partial_cmp(&diag[j]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
    let pick = if tr > 0.0 { 3 } else { big };
    let q = match pick {
        3 => {
            let s = (tr + 1.0).sqrt() * 2.0;
            [(m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s, (m(1, 0) - m(0, 1)) / s, 0.25 * s]
        }
        0 => {
            let s = (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt() * 2.0;
            [0.25 * s, (m(0, 1) + m(1, 0)) / s, (m(0, 2) + m(2, 0)) / s, (m(2, 1) - m(1, 2)) / s]
        }
        1 => {
            let s = (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt() * 2.0;
            [(m(0, 1) + m(1, 0)) / s, 0.25 * s, (m(1, 2) + m(2, 1)) / s, (m(0, 2) - m(2, 0)) / s]
        }
        _ => {
            let s = (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt() * 2.0;
            [(m(0, 2) + m(2, 0)) / s, (m(1, 2) + m(2, 1)) / s, 0.25 * s, (m(1, 0) - m(0, 1)) / s]
        }
    };
    q_norm(q)
}

/// Blender's `m[col][row]` of `translate(x) rotate(q) translate(-com) scale(s)`:
/// the object's matrix for a body whose centre of mass sits at `com` in the
/// scaled object space.
pub fn pose_matrix(x: V3, q: Q, com: V3, s: V3) -> [[f32; 4]; 4] {
    let col = |k: usize| {
        let mut e = [0.0; 3];
        e[k] = s[k];
        let r = rotate(q, e);
        [r[0], r[1], r[2], 0.0]
    };
    let t = sub(x, rotate(q, com));
    [col(0), col(1), col(2), [t[0], t[1], t[2], 1.0]]
}

/// Translation, rotation and scale of a Blender `m[col][row]` matrix. A
/// mirrored matrix keeps its handedness on the scale's z.
pub fn decompose(m: [[f32; 4]; 4]) -> (V3, Q, V3) {
    let col = |k: usize| [m[k][0], m[k][1], m[k][2]];
    let (c0, c1, c2) = (col(0), col(1), col(2));
    let flip = if dot(cross(c0, c1), c2) < 0.0 { -1.0 } else { 1.0 };
    let s = [len(c0), len(c1), len(c2) * flip];
    let unit = |c: V3, l: f32| if l != 0.0 { scale(c, 1.0 / l) } else { c };
    let q = q_from_cols([unit(c0, s[0]), unit(c1, s[1]), unit(c2, s[2])]);
    (col(3), q, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_round_trip() {
        let q = q_norm([0.2, -0.3, 0.4, 0.8]);
        let v = [1.0, 2.0, 3.0];
        let back = unrotate(q, rotate(q, v));
        assert!(len(sub(back, v)) < 1e-5);
        let q2 = q_from_cols(mat_of(q));
        let same = (0..4).all(|k| (q2[k] - q[k]).abs() < 1e-5) || (0..4).all(|k| (q2[k] + q[k]).abs() < 1e-5);
        assert!(same, "{q:?} {q2:?}");
        let (x, r, s) = decompose(pose_matrix([1.0, 2.0, 3.0], q, [0.0; 3], [2.0, 3.0, 4.0]));
        assert!(len(sub(x, [1.0, 2.0, 3.0])) < 1e-5 && len(sub(s, [2.0, 3.0, 4.0])) < 1e-4);
        assert!(len(sub(rotate(r, v), rotate(q, v))) < 1e-4);
    }

    #[test]
    fn angular_velocity_matches_integration() {
        let w = [0.3, -0.2, 0.5];
        let q = q_norm([0.1, 0.2, 0.3, 0.9]);
        let next = q_integrate(q, w, 1e-3);
        assert!(len(sub(q_delta_w(next, q, 1e-3), w)) < 1e-3);
    }
}
