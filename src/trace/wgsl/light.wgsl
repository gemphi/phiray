fn onb(n: vec3<f32>, a: f32, b: f32) -> vec3<f32> {
    let tx = normalize(select(cross(n, vec3<f32>(0.0, 0.0, 1.0)), cross(n, vec3<f32>(1.0, 0.0, 0.0)), abs(n.z) > 0.9));
    let ty = cross(n, tx);
    return tx * a + ty * b;
}

// How much of a spot reaches a surface whose direction to the light is `l`
// (EEVEE's `light_spot_attenuation`, `eevee_light_lib.bsl.hh`). The cone's
// cross-section is an ellipse when the object's scale is not uniform:
// `spot_scale` is `scale.z / scale.xy` (`eevee_light.cc`'s `spot_size_inv`),
// so a spot stretched in Y lights further in Y. At `[1, 1]` the term is
// `cos` of the angle off the axis and this is batch 147's smoothstep
// exactly.
fn spot_mask(li: Light, l: vec3<f32>) -> f32 {
    // EEVEE works in the light's own frame, where +Z is the way the lamp
    // looks from (`-dir`) and X and Y are the object's. `l` is normalized
    // at both call sites, so `z` is the cosine of the angle off the axis.
    let z = -dot(l, li.dir);
    if (z <= 0.0) { return 0.0; }
    // A record with no axes or no scale is the round case, and so is the
    // common one: `z` is what the ellipse comes to at `[1, 1]`. EEVEE
    // keeps the same guard on the CPU, setting `spot_size_inv` to 1 for a
    // light with no cone so the field is never garbage.
    var ellipse = z;
    if (dot(li.right, li.right) > 1e-12 && dot(li.spot_scale, li.spot_scale) > 1e-12) {
        let d = -l;
        let xy = vec2<f32>(dot(d, li.right), dot(d, cross(li.dir, li.right))) * li.spot_scale;
        ellipse = inverseSqrt(1.0 + dot(xy, xy) / (z * z));
    }
    return smoothstep(li.cone, li.cone + li.blend, ellipse);
}

// Direction inside a cone of half-angle `ang` around `axis`.
fn cone_dir(axis: vec3<f32>, ang: f32, seed: ptr<function, u32>) -> vec3<f32> {
    if (ang <= 0.0) { return axis; }
    let r = ang * sqrt(rnd(seed));
    let ph = 2.0 * PI * rnd(seed);
    return normalize(axis + onb(axis, r * cos(ph), r * sin(ph)));
}

// Radiance arriving from direction `d`: the world's environment image when it
// has one, otherwise the world's own colour — uniform in every direction, as
// a Background node with no image is. There used to be a gradient here
// (`mix(0.35, 1.15, …)`, brighter above the horizon), which looked like a
// sky but is not what Blender lights with: it put about 2 % more on an
// upward-facing surface than the flat colour should, and more than that on
// a downward-facing one (`rendering.md` §54).
fn env_at(d: vec3<f32>) -> vec3<f32> {
    if (p.envmap == 0u) {
        return p.env.rgb;
    }
    // `direction_to_equirectangular` with the world Mapping node's Z spin.
    let c = cos(p.env_rot);
    let s = sin(p.env_rot);
    let dir = vec3<f32>(d.x * c - d.y * s, d.x * s + d.y * c, d.z);
    let u = -atan2(dir.y, dir.x) / (2.0 * PI) + 0.5;
    let v = 0.5 - atan2(dir.z, length(dir.xy)) / PI;
    return sample_env(p.envmap - 1u, vec2<f32>(u, v)) * p.env.rgb;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn visible(org: vec3<f32>, l: vec3<f32>, dist: f32) -> f32 {
    return select(1.0, 0.0, trace(org, l, dist, true).tri != 0xffffffffu);
}

// A light's share at a shading point before its shadow ray, and that ray:
// `c` is what arrives if nothing is in the way, `dir` and `dist` the ray
// that says whether something is. Every light is reduced to one of these
// so that `direct` can trace them all from a single call site.
struct Shadowed { c: vec3<f32>, dir: vec3<f32>, dist: f32 }

fn unshadowed() -> Shadowed { return Shadowed(vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 0.0); }

// A world is a light, image or no image. With one and no lamps it is the
// only light — Blender adds nothing on top of a world, and Material
// Preview's studio HDRI is exactly this case. The three-point rig below
// is a fallback for a scene with no light of *any* kind.
//
// This used to test the environment image alone, so a scene lit by a
// flat world colour got the rig on top of it: a dome of radiance L came
// back as `0.83 L + 0.068` where Blender gives exactly `albedo × L`
// (`rendering.md` §54). An image was already excluded for the same
// reason — it made every lamp-less HDRI scene about a fifth too bright.
fn uses_rig() -> bool {
    let world_lit = p.envmap != 0u || dot(p.env.rgb, p.env.rgb) > 0.0;
    return p.count == 0u && !world_lit;
}

// The rig, tied to the view: a shadowed key (`k` 0), a shadowed rim (1),
// and an unshadowed fill (`rig_fill`).
fn rig_query(k: u32, s: Surf, seed: ptr<function, u32>) -> Shadowed {
    if (k == 0u) {
        let key = cone_dir(p.sun, 0.06, seed);
        return Shadowed(brdf(s, key, vec3<f32>(2.0)), key, INF);
    }
    let rim = normalize(p.up * 0.6 - p.back * 0.8 + p.right * 0.2);
    return Shadowed(brdf(s, rim, vec3<f32>(0.9)), rim, INF);
}

fn rig_fill(s: Surf) -> vec3<f32> {
    let fill = normalize(-p.right * 0.8 + p.up * 0.2 + p.back * 0.6);
    return brdf(s, fill, vec3<f32>(0.45));
}

// One lamp per shading point, its light divided by the chance of the pick —
// Cycles' light distribution (`light_distribution_sample`). This took up to
// four lamps a point, each with its own shadow ray, and a path makes a
// shading point at every bounce: thirty and more rays a sample on pose 26,
// where Cycles spends about ten. The expected image is the same.
//
// The pick follows the light that arrives here (`lamp_reach`), not the
// power alone: in a city of lamps, the one across the street was drawn as
// often as the one beside the figure, and most shadow rays went to lamps
// too far off to matter. Cycles' Light Tree weighs by distance the same way.
fn lamp_query(s: Surf, seed: ptr<function, u32>) -> Shadowed {
    // In one pass, as a reservoir of one: each lamp replaces the one held
    // with the chance of its share of the light seen so far, and the single
    // random number is stretched back over what is left of it after each
    // choice, so it stays uniform without drawing another.
    var total = 0.0;
    var idx = 0u;
    var held = 0.0;
    var u = rnd(seed);
    for (var k = 0u; k < p.count; k++) {
        let w = lamp_reach(p.lights[k], s.pos);
        if (w <= 0.0) { continue; }
        total += w;
        let share = w / total;
        if (u < share) {
            idx = k;
            held = w;
            u = u / share;
        } else {
            u = (u - share) / (1.0 - share);
        }
    }
    if (total <= 0.0 || held <= 0.0) { return unshadowed(); }
    let li = p.lights[idx];
    let weight = total / held;
    if (li.kind == 0u) {
        let l = cone_dir(-li.dir, li.size * 0.5, seed);
        return Shadowed(brdf(s, l, li.color) * weight, l, INF);
    }
    let r1 = rnd(seed);
    let r2 = rnd(seed);
    if (li.kind == 3u) {
        return area_query(s, li, weight, r1, r2, rnd(seed));
    }
    let aim = li.pos + onb(normalize(s.pos - li.pos), (r1 - 0.5) * li.size, (r2 - 0.5) * li.size);
    let to = aim - s.pos;
    let d2 = max(dot(to, to), 1e-6);
    let l = to / sqrt(d2);
    // Irradiance at the surface, which is what the BSDF below expects:
    // a lamp's Power over the sphere it fills, `P / (4π d²)`. The BSDF
    // divides the diffuse albedo by π itself (`brdf_w`), so dividing by
    // π here as well — which this did, as `4π²d²` — made every lamp π
    // times too dim (`rendering.md` §54).
    var e = li.color / (4.0 * PI * d2);
    if (li.kind == 2u) {
        e *= spot_mask(li, l);
    }
    if (max(e.r, max(e.g, e.b)) < 1e-5) { return unshadowed(); }
    return Shadowed(brdf(s, l, e) * weight, l, sqrt(d2) * 0.999);
}

// How far off a surface its shading point sits, for a ray that came `t` to it:
// enough to clear the rounding in `o + d * t`. An orthographic camera's ray
// starts 1e4 behind the pivot, so that distance is no measure of the scene. At
// 1e-5 of it a floor shaded from 10 cm up, every lamp 2 m above it read 10 %
// too bright, and only its rounding, one part in four million, is real.
fn hit_lift(t: f32) -> f32 {
    let start = 1.0e4 / p.units;
    if (p.persp == 0u && t > start * 0.5) {
        return p.eps + abs(t - start) * 1e-5 + start * 2.4e-7;
    }
    return p.eps + t * 1e-5;
}

// How far past a surface a ray that goes on through it starts — glass, or a
// texel's alpha. `hit_lift` will not do: its `p.eps` is a share of the whole
// scene's extent, and a floor 741 m across (CW-POSES pose 24) made that a
// centimetre, the thickness of the car's glass roof. The ray then started
// beyond the pane's far face, crossed one face instead of two, was tinted by
// the square root of the base colour once instead of twice, and a roof that
// is nearly black glass glowed red.
//
// What has to be cleared is the rounding of the hit point, and that follows
// the size of its own coordinates, not the scene's: Cycles' `ray_offset`
// (`kernel/bvh/util.h`, after Ray Tracing Gems ch. 6) steps each coordinate
// 256 units in the last place, and 1/65536 near the origin. The share for
// the ray's length stays, since the point was made as `o + d * t`.
fn through_lift(pos: vec3<f32>, t: f32) -> f32 {
    let reach = max(max(abs(pos.x), abs(pos.y)), max(abs(pos.z), 0.5));
    return hit_lift(t) - p.eps + reach * (256.0 / 8388608.0);
}

// An area light seen from `s`: a point of its own shape, its radiance its flux
// over its area, `L = P / (π A)`, shining from one side only and judged from
// that point. Aimed from its centre, a surface over the middle of a wide light
// saw it edge on and got nothing (Lava City's 141 m disks, under the mannequin).
//
// The point comes one of two ways, as a chance `near` that grows with how much
// of the view the light fills. Evenly over the shape, as Cycles draws an
// ellipse (`area_light_sample`), is right for a light far off. Close to a wide
// one nearly all its light comes from the few points just below the surface,
// so that alone is rare spikes; Cycles has the BSDF's rays hitting the light to
// fill them in, and this has no such rays. So the other way draws a direction
// cosine-weighted about the light's normal and takes where it meets the plane,
// which puts points where the light is. A direction that misses the shape
// carries nothing. The pdf is the mix of both, so the image is the same.
fn area_query(s: Surf, li: Light, weight: f32, r1: f32, r2: f32, pick: f32) -> Shadowed {
    let height = dot(s.pos - li.pos, li.dir);
    if (height <= 0.0) { return unshadowed(); }
    let round = li.shape == AREA_DISK || li.shape == AREA_ELLIPSE;
    let half = vec2<f32>(li.size, li.size_y) * 0.5;
    let area = max(li.size * li.size_y * select(1.0, 0.25 * PI, round), 1e-8);
    let up = cross(li.dir, li.right);
    let off = s.pos - li.pos;
    let near = area / (area + PI * dot(off, off));
    var on = vec2<f32>(0.0);
    if (pick < near) {
        let spread = sqrt(r1 / max(1.0 - r1, 1e-6)) * height;
        on = vec2<f32>(cos(2.0 * PI * r2), sin(2.0 * PI * r2)) * spread + vec2<f32>(dot(off, li.right), dot(off, up));
        let unit = on / max(half, vec2<f32>(1e-8));
        let inside = select(max(abs(unit.x), abs(unit.y)) <= 1.0, dot(unit, unit) <= 1.0, round);
        if (!inside) { return unshadowed(); }
    } else {
        let disk = vec2<f32>(cos(2.0 * PI * r2), sin(2.0 * PI * r2)) * sqrt(r1);
        on = select(vec2<f32>(r1, r2) * 2.0 - 1.0, disk, round) * half;
    }
    let to = li.pos + li.right * on.x + up * on.y - s.pos;
    let d2 = max(dot(to, to), 1e-6);
    let l = to / sqrt(d2);
    let facing = max(dot(-l, li.dir), 0.0);
    // In area measure: `1 / A` evenly, `cos² / (π d²)` by direction.
    let pdf = (1.0 - near) / area + near * facing * facing / (PI * d2);
    let e = li.color / (PI * area) * facing / d2 / max(pdf, 1e-12);
    if (max(e.r, max(e.g, e.b)) < 1e-5) { return unshadowed(); }
    return Shadowed(brdf(s, l, e) * weight, l, sqrt(d2) * 0.999);
}

// How much of a lamp reaches `at`, which is what `lamp_query` picks by: a
// sun's irradiance as it is, any other lamp's power over the sphere at its
// distance. The distance is never taken below the lamp's own size or a small
// share of the scene, so a lamp close by is not picked so often that the
// others' rare picks weigh enough to sparkle. Every lamp keeps a chance, so
// nothing it lights goes missing.
fn lamp_reach(li: Light, at: vec3<f32>) -> f32 {
    let lum = dot(max(li.color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
    if (li.kind == 0u) { return lum; }
    let to = li.pos - at;
    let near = max(li.size * 0.5, p.ao * 0.05);
    return lum / (4.0 * PI * (dot(to, to) + near * near));
}

// Probability that `glow_query` picks face `f` (0 when it does not glow).
// The share of the light table a hit holds: its (face, instance) pair,
// found among the emitters sorted by face then instance (`lamp.rs`), 0
// when it is not one. An emissive mesh shown by many instances is one
// emitter an instance, so the instance tells them apart.
fn lamp_chance(h: Hit) -> f32 {
    let count = data[0];
    if (count == 0u) { return 0.0; }
    var lo = 0u;
    var hi = count - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let f = data[1u + mid];
        let i = data[1u + count + mid];
        if (f < h.tri || (f == h.tri && i < h.inst)) { lo = mid + 1u; } else { hi = mid; }
    }
    if (data[1u + lo] != h.tri || data[1u + count + lo] != h.inst) { return 0.0; }
    let prev = select(0.0, tf(2u * count + lo), lo > 0u);
    return tf(1u + 2u * count + lo) - prev;
}

// Next-event estimation: one emissive triangle (by power), one point on it.
fn glow_query(s: Surf, seed: ptr<function, u32>) -> Shadowed {
    let count = data[0];
    if (count == 0u) { return unshadowed(); }
    // Pick by power: binary search in the cdf stored after the ids.
    let u = rnd(seed);
    var lo = 0u;
    var hi = count - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (tf(1u + 2u * count + mid) < u) { lo = mid + 1u; } else { hi = mid; }
    }
    let prev = select(0.0, tf(2u * count + lo), lo > 0u);
    let chance = max(tf(1u + 2u * count + lo) - prev, 1e-8);
    let f = data[1u + lo];
    let t = tri_of(f, data[1u + count + lo], mat_at(f));
    var a = rnd(seed);
    var b = rnd(seed);
    if (a + b > 1.0) { a = 1.0 - a; b = 1.0 - b; }
    let q = t.a + (t.b - t.a) * a + (t.c - t.a) * b;
    let cr = cross(t.b - t.a, t.c - t.a);
    let area = 0.5 * length(cr);
    let to = q - s.pos;
    let d2 = max(dot(to, to), 1e-6);
    let l = to / sqrt(d2);
    let cos_l = abs(dot(normalize(cr), l));
    let e = emit_at(t, vec3<f32>(1.0 - a - b, a, b)) * cos_l * area / (chance * d2);
    // MIS weight against the cosine-sampled diffuse ray (power heuristic).
    let pl = chance * d2 / max(area * cos_l, 1e-8);
    let pb = max(dot(s.n, l), 0.0) / PI;
    let w = pl * pl / max(pl * pl + pb * pb, 1e-12);
    return Shadowed(brdf(s, l, e) * w, l, sqrt(d2) * 0.999);
}

// --- Environment sampling -------------------------------------------------
//
// `Atlas::add_env_cdf` stores, for an equirectangular world image, one
// conditional CDF over columns per row and a final row holding the marginal
// CDF over rows, weighted by luminance × sin θ. Drawing from it points rays at
// the bright parts of the sky instead of leaving them to find it by chance,
// which is what lets a cavity see daylight through its opening.

struct EnvHit { dir: vec3<f32>, radiance: vec3<f32>, pdf: f32 }

fn env_cdf_base() -> u32 { return head_base(p.env_cdf - 1u); }
fn env_cdf_w() -> u32 { return head_w(p.env_cdf - 1u); }
// The table's last row is the marginal, so the grid is one row shorter.
fn env_cdf_h() -> u32 { return head_h(p.env_cdf - 1u) - 1u; }

fn env_cdf(row: u32, i: u32) -> f32 {
    return bitcast<f32>(atlas(env_cdf_base() + row * env_cdf_w() + i));
}

// First index in `row` whose CDF value is at or above `u`.
fn env_find(row: u32, count: u32, u: f32) -> u32 {
    var lo = 0u;
    var hi = count - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (env_cdf(row, mid) < u) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn env_cell_pdf(row: u32, col: u32) -> f32 {
    let cw = env_cdf_w();
    let ch = env_cdf_h();
    let prev_row = select(0.0, env_cdf(ch, row - 1u), row > 0u);
    let prev_col = select(0.0, env_cdf(row, col - 1u), col > 0u);
    let pr = env_cdf(ch, row) - prev_row;
    let pc = env_cdf(row, col) - prev_col;
    // Density per unit of the [0,1]² image, not per cell.
    return pr * pc * f32(cw) * f32(ch);
}

// `(u, v)` of a world direction, matching `env_at`'s mapping.
fn env_uv(d: vec3<f32>) -> vec2<f32> {
    let c = cos(p.env_rot);
    let s = sin(p.env_rot);
    let dir = vec3<f32>(d.x * c - d.y * s, d.x * s + d.y * c, d.z);
    return vec2<f32>(
        fract(-atan2(dir.y, dir.x) / (2.0 * PI) + 0.5),
        clamp(0.5 - atan2(dir.z, length(dir.xy)) / PI, 0.0, 1.0),
    );
}

// Solid-angle density the table gives this direction (0 with no table).
fn env_pdf(d: vec3<f32>) -> f32 {
    if (p.env_cdf == 0u) { return 0.0; }
    let uv = env_uv(d);
    let cw = env_cdf_w();
    let ch = env_cdf_h();
    let col = min(u32(uv.x * f32(cw)), cw - 1u);
    let row = min(u32(uv.y * f32(ch)), ch - 1u);
    let sin_t = sin(PI * uv.y);
    if (sin_t < 1e-6) { return 0.0; }
    return env_cell_pdf(row, col) / (2.0 * PI * PI * sin_t);
}

// One direction drawn from the table, with its radiance and density.
fn env_sample(seed: ptr<function, u32>) -> EnvHit {
    let cw = env_cdf_w();
    let ch = env_cdf_h();
    let row = env_find(ch, ch, rnd(seed));
    let col = env_find(row, cw, rnd(seed));
    // Uniform inside the cell: the table is piecewise constant, so this is
    // exactly the density `env_cell_pdf` reports.
    let u = (f32(col) + rnd(seed)) / f32(cw);
    let v = (f32(row) + rnd(seed)) / f32(ch);
    let theta = PI * v;
    let phi = (0.5 - u) * 2.0 * PI;
    let sin_t = sin(theta);
    let local = vec3<f32>(sin_t * cos(phi), sin_t * sin(phi), cos(theta));
    // `env_uv` rotates the world into the image; going back undoes it.
    let c = cos(p.env_rot);
    let s = sin(p.env_rot);
    let dir = vec3<f32>(local.x * c + local.y * s, -local.x * s + local.y * c, local.z);
    if (sin_t < 1e-6) { return EnvHit(dir, vec3<f32>(0.0), 0.0); }
    let pdf = env_cell_pdf(row, col) / (2.0 * PI * PI * sin_t);
    return EnvHit(dir, sample_env(p.envmap - 1u, vec2<f32>(u, v)) * p.env.rgb, pdf);
}
