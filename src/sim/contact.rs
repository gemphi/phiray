//! Contacts (ROADMAP.md §2.2 "Contacts and friction" and "One step" 4.2,
//! 4.4): every substep re-evaluates each contact from its body-local
//! anchors, projects the normal with static friction at position level
//! (Muller 2020 §3.5), then, with the velocities derived, applies dynamic
//! friction and restitution (§3.6). Pairs go Gauss-Seidel in sorted order;
//! the points of one pair's manifold go Jacobi: each asks for its own
//! impulse from the same state, the sum is applied once, scaled down so
//! that no point is over-corrected. Four corners on a floor then lift a
//! box straight up, where one corner after another would leave it turned
//! and sliding. That per-point work is what the GPU `Contacts` kernel
//! will compute and `Apply` sum. The normal points from `a` to `b`, so a
//! positive normal velocity is the pair closing.

use super::math::*;
use super::state::{Body, BodyConst, BodyPrev, Contact};

/// Inverse mass of `b` felt along `n` at arm `r` (Muller 2020 eq. 2, 3).
fn inv_mass_along(b: &Body, c: &BodyConst, r: V3, n: V3) -> f32 {
    if b.inv_mass == 0.0 {
        return 0.0;
    }
    let rn = unrotate(b.q, cross(r, n));
    b.inv_mass + dot(mul3(rn, c.inv_inertia), rn)
}

/// The two bodies of a pair, borrowed apart.
fn two(bodies: &mut [Body], i: usize, j: usize) -> (&mut Body, &mut Body) {
    let (lo, hi) = bodies.split_at_mut(i.max(j));
    if i < j { (&mut lo[i], &mut hi[0]) } else { (&mut hi[0], &mut lo[j]) }
}

fn arms(a: &Body, b: &Body, c: &Contact) -> (V3, V3) {
    (rotate(a.q, c.anchor_a), rotate(b.q, c.anchor_b))
}

/// Velocity of the contact point on `a` relative to the one on `b`.
fn relative_velocity(a: &Body, b: &Body, ra: V3, rb: V3) -> V3 {
    sub(add(a.v, cross(a.w, ra)), add(b.v, cross(b.w, rb)))
}

/// One point's ask: impulse `p` on `a` (`-p` on `b`) at its arms, for a
/// change of `need` along `dir` in the point's relative motion.
#[derive(Clone, Copy, Debug, Default)]
struct Share {
    p: V3,
    ra: V3,
    rb: V3,
    dir: V3,
    need: f32,
}

/// What the summed shares move each body by (positions or velocities).
struct Motion {
    da: V3,
    wa: V3,
    db: V3,
    wb: V3,
}

fn motion(a: &Body, ka: &BodyConst, b: &Body, kb: &BodyConst, shares: &[Share]) -> Motion {
    let turn = |b: &Body, k: &BodyConst, r: V3, p: V3| {
        if b.inv_mass == 0.0 { [0.0; 3] } else { rotate(b.q, mul3(unrotate(b.q, cross(r, p)), k.inv_inertia)) }
    };
    shares.iter().fold(Motion { da: [0.0; 3], wa: [0.0; 3], db: [0.0; 3], wb: [0.0; 3] }, |m, s| Motion {
        da: add(m.da, scale(s.p, a.inv_mass)),
        wa: add(m.wa, turn(a, ka, s.ra, s.p)),
        db: sub(m.db, scale(s.p, b.inv_mass)),
        wb: sub(m.wb, turn(b, kb, s.rb, s.p)),
    })
}

/// The largest scale, at most 1, under which no point gets more than it asked.
fn fit(m: &Motion, shares: &[Share]) -> f32 {
    shares.iter().fold(1.0, |alpha, s| {
        let got = dot(sub(add(m.da, cross(m.wa, s.ra)), add(m.db, cross(m.wb, s.rb))), s.dir);
        if got * s.need > 0.0 { alpha.min(s.need / got) } else { alpha }
    })
}

/// Moves the pair's poses by the shares; returns the scale applied.
fn move_poses(a: &mut Body, ka: &BodyConst, b: &mut Body, kb: &BodyConst, shares: &[Share]) -> f32 {
    let m = motion(a, ka, b, kb, shares);
    let alpha = fit(&m, shares);
    a.x = add(a.x, scale(m.da, alpha));
    a.q = q_integrate(a.q, scale(m.wa, alpha), 1.0);
    b.x = add(b.x, scale(m.db, alpha));
    b.q = q_integrate(b.q, scale(m.wb, alpha), 1.0);
    alpha
}

/// Changes the pair's velocities by the shares.
fn move_velocities(a: &mut Body, ka: &BodyConst, b: &mut Body, kb: &BodyConst, shares: &[Share]) {
    let m = motion(a, ka, b, kb, shares);
    let alpha = fit(&m, shares);
    a.v = add(a.v, scale(m.da, alpha));
    a.w = add(a.w, scale(m.wa, alpha));
    b.v = add(b.v, scale(m.db, alpha));
    b.w = add(b.w, scale(m.wb, alpha));
}

/// The runs of consecutive contacts of one pair.
fn groups(contacts: &mut [Contact], mut each: impl FnMut(&mut [Contact])) {
    let mut at = 0;
    while at < contacts.len() {
        let (a, b) = (contacts[at].a, contacts[at].b);
        let n = contacts[at..].iter().take_while(|c| c.a == a && c.b == b).count();
        each(&mut contacts[at..at + n]);
        at += n;
    }
}

/// Each contact's closing speed before the substep, for restitution.
pub fn presolve(bodies: &[Body], contacts: &mut [Contact], vn0: &mut [f32]) {
    for (c, v) in contacts.iter_mut().zip(vn0) {
        let (a, b) = (&bodies[c.a as usize], &bodies[c.b as usize]);
        let (ra, rb) = arms(a, b, c);
        *v = dot(relative_velocity(a, b, ra, rb), c.normal);
        (c.lambda_n, c.lambda_t) = (0.0, 0.0);
    }
}

/// Positional projection of every contact: the normals of a manifold,
/// then its static friction as far as `mu * lambda_n` allows.
///
/// `cap` is the deepest penetration one projection may push out, in
/// metres — the substep's whole recovery budget divided between its
/// projections (`World::max_recover`, `world.rs`).
///
/// A substep derives velocity from its pose change over `h`, so a
/// penetration resolved inside one substep *becomes* speed: 3 cm at a
/// 600 Hz substep is 18 m/s, and that is what launched the capsule of
/// `every_shape_comes_to_rest_on_the_ground` to z = 22 and bounced the
/// ball of `a_ball_bounces_to_e_squared_of_its_height` to four times its
/// drop. Contacts are found once a frame, so a body that arrives fast is
/// already centimetres in when the first one exists; this budget lets it
/// out over as many substeps as it takes instead. A resting contact
/// penetrates by a thousandth of the budget in a substep and so is still
/// corrected whole. Bullet reduces its error the same way, and Blender's
/// Rigid Body World calls the related control Split Impulse.
pub fn project(bodies: &mut [Body], consts: &[BodyConst], prev: &[BodyPrev], contacts: &mut [Contact], cap: f32, push: &mut [V3], spin: &mut [V3]) {
    groups(contacts, |group| {
        let (i, j) = (group[0].a as usize, group[0].b as usize);
        let (ka, kb) = (&consts[i], &consts[j]);
        let (a, b) = two(bodies, i, j);
        let mut shares = Vec::with_capacity(group.len());
        let mut dls = vec![0.0; group.len()];
        let mut owners: Vec<usize> = Vec::with_capacity(group.len());
        // The pure-recovery part of each point's correction, as a
        // fraction of it.
        let mut recovery = vec![0.0f32; group.len()];
        for (k, c) in group.iter().enumerate() {
            let (ra, rb) = arms(a, b, c);
            let n = c.normal;
            let d = dot(sub(add(a.x, ra), add(b.x, rb)), n);
            let w = inv_mass_along(a, ka, ra, n) + inv_mass_along(b, kb, rb, n);
            if d <= 0.0 || w == 0.0 {
                continue;
            }
            // How much of `d` this substep's own drift just made: that
            // much is a collision and belongs in the velocity the substep
            // derives. The rest was already there when the substep
            // began — the penetration a body that arrived fast is in
            // before a contact exists — and pushing it out is recovery,
            // which must move the pose without becoming speed.
            let (qa, qb) = (add(prev[i].x, rotate(prev[i].q, c.anchor_a)), add(prev[j].x, rotate(prev[j].q, c.anchor_b)));
            let closed = dot(sub(sub(add(a.x, ra), qa), sub(add(b.x, rb), qb)), n).max(0.0);
            let hit = d.min(closed);
            let fix = hit + (d - hit).min(cap);
            dls[k] = -fix / w;
            recovery[k] = match fix > 0.0 {
                true => (fix - hit) / fix,
                false => 0.0,
            };
            owners.push(k);
            shares.push(Share { p: scale(n, dls[k]), ra, rb, dir: neg(n), need: fix });
        }
        if !shares.is_empty() {
            let alpha = move_poses(a, ka, b, kb, &shares);
            for (c, dl) in group.iter_mut().zip(&dls) {
                c.lambda_n += dl * alpha;
            }
            // What of that motion was recovery, kept out of the velocity
            // `World::velocities` derives (Bullet's split impulse).
            let only: Vec<Share> = shares.iter().zip(&owners).map(|(s, &k)| Share { p: scale(s.p, recovery[k]), ..*s }).collect();
            if only.iter().any(|s| len2(s.p) > 0.0) {
                let m = motion(a, ka, b, kb, &only);
                push[i] = add(push[i], scale(m.da, alpha));
                spin[i] = add(spin[i], scale(m.wa, alpha));
                push[j] = add(push[j], scale(m.db, alpha));
                spin[j] = add(spin[j], scale(m.wb, alpha));
            }
        }

        // Static friction: cancel each point's tangential drift as far as
        // its cone allows. The ask is scaled by the manifold's own Jacobi
        // share *before* the cone sees it, because `lambda_n` is already
        // a share of the normal pass — a box's four corners each asked
        // for the whole drift and were tested against a quarter of the
        // normal impulse, so the cone held almost nowhere and a box slid
        // 3.3 m down a slope well inside its friction angle. What is over
        // the cone is clamped to it rather than dropped, so a contact
        // that must slide still takes all the friction it is owed.
        //
        // This runs whether or not the normal pass moved anything: on the
        // second projection of a substep the penetration is already gone,
        // and the drift to hold is not.
        shares.clear();
        owners.clear();
        let mut tangent = vec![[0.0f32; 3]; group.len()];
        let mu = ka.friction * kb.friction;
        for (k, c) in group.iter().enumerate() {
            dls[k] = 0.0;
            if c.lambda_n == 0.0 {
                continue;
            }
            let (ra, rb) = arms(a, b, c);
            let (pa, pb) = (add(a.x, ra), add(b.x, rb));
            let (qa, qb) = (add(prev[i].x, rotate(prev[i].q, c.anchor_a)), add(prev[j].x, rotate(prev[j].q, c.anchor_b)));
            let dp = sub(sub(pa, qa), sub(pb, qb));
            let dpt = sub(dp, scale(c.normal, dot(dp, c.normal)));
            let lt = len(dpt);
            if lt < 1e-9 {
                continue;
            }
            let t = scale(dpt, 1.0 / lt);
            let wt = inv_mass_along(a, ka, ra, t) + inv_mass_along(b, kb, rb, t);
            if wt == 0.0 {
                continue;
            }
            tangent[k] = t;
            dls[k] = -lt / wt;
            owners.push(k);
            shares.push(Share { p: scale(t, dls[k]), ra, rb, dir: neg(t), need: lt });
        }
        if shares.is_empty() {
            return;
        }
        let share = fit(&motion(a, ka, b, kb, &shares), &shares);
        for (s, &k) in shares.iter_mut().zip(&owners) {
            let want = dls[k] * share;
            let room = (mu * group[k].lambda_n.abs() - group[k].lambda_t.abs()).max(0.0);
            dls[k] = want.signum() * want.abs().min(room);
            s.p = scale(tangent[k], dls[k]);
        }
        let alpha = move_poses(a, ka, b, kb, &shares);
        for (c, dl) in group.iter_mut().zip(&dls) {
            c.lambda_t += dl * alpha;
        }
    });
}

/// Dynamic friction and restitution on the velocities (Muller 2020 §3.6)
/// for the contacts the projection touched. `h` is the substep and `g`
/// the gravity's size: a closing speed under `2 g h` bounces not at all.
/// Friction's impulse is capped at `mu |lambda_n| / h`; the normal impulse
/// then sets the closing speed to `-e` times what it was.
pub fn velocity_pass(bodies: &mut [Body], consts: &[BodyConst], contacts: &mut [Contact], vn0: &[f32], h: f32, g: f32) {
    let mut at = 0;
    groups(contacts, |group| {
        let before = &vn0[at..at + group.len()];
        at += group.len();
        let (i, j) = (group[0].a as usize, group[0].b as usize);
        let (ka, kb) = (&consts[i], &consts[j]);
        let (a, b) = two(bodies, i, j);
        let mu = ka.friction * kb.friction;
        let mut shares = Vec::with_capacity(group.len());
        for c in group.iter().filter(|c| c.lambda_n != 0.0) {
            let (ra, rb) = arms(a, b, c);
            let vrel = relative_velocity(a, b, ra, rb);
            let vt = sub(vrel, scale(c.normal, dot(vrel, c.normal)));
            let lt = len(vt);
            if lt < 1e-9 {
                continue;
            }
            let t = scale(vt, 1.0 / lt);
            let wt = inv_mass_along(a, ka, ra, t) + inv_mass_along(b, kb, rb, t);
            let j = (lt / wt).min(mu * c.lambda_n.abs() / h);
            shares.push(Share { p: scale(t, -j), ra, rb, dir: neg(t), need: j * wt });
        }
        if !shares.is_empty() {
            move_velocities(a, ka, b, kb, &shares);
        }
        shares.clear();
        for (c, &vn0) in group.iter().zip(before).filter(|(c, _)| c.lambda_n != 0.0) {
            let (ra, rb) = arms(a, b, c);
            let n = c.normal;
            let vn = dot(relative_velocity(a, b, ra, rb), n);
            let e = if vn0.abs() <= 2.0 * g * h { 0.0 } else { ka.restitution * kb.restitution };
            let wn = inv_mass_along(a, ka, ra, n) + inv_mass_along(b, kb, rb, n);
            let need = -e * vn0.max(0.0) - vn;
            // Only ever an impulse that separates the pair. This pass
            // runs every substep, and a penetration let out over several
            // of them keeps `lambda_n` alive for substeps after the
            // bounce — where `-e vn0 - vn` turns positive and would pull
            // the pair back together, cancelling the separation the
            // bounce had just made. Testing the sign of `vn` instead
            // would skip the impact substep itself: the projection has
            // already taken the closing speed out by then, so the bounce
            // is owed on `vn0`, not on what is left.
            if wn > 0.0 && need < -1e-12 {
                shares.push(Share { p: scale(n, need / wn), ra, rb, dir: n, need });
            }
        }
        if !shares.is_empty() {
            move_velocities(a, ka, b, kb, &shares);
        }
    });
}
