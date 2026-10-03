fn onb(n: vec3<f32>, a: f32, b: f32) -> vec3<f32> {
    let tx = normalize(select(cross(n, vec3<f32>(0.0, 0.0, 1.0)), cross(n, vec3<f32>(1.0, 0.0, 0.0)), abs(n.z) > 0.9));
    let ty = cross(n, tx);
    return tx * a + ty * b;
}

// Direction inside a cone of half-angle `ang` around `axis`.
fn cone_dir(axis: vec3<f32>, ang: f32, seed: ptr<function, u32>) -> vec3<f32> {
    if (ang <= 0.0) { return axis; }
    let r = ang * sqrt(rnd(seed));
    let ph = 2.0 * PI * rnd(seed);
    return normalize(axis + onb(axis, r * cos(ph), r * sin(ph)));
}

// GGX (Trowbridge–Reitz) microfacets, ported from Cycles
// (`kernel/closure/bsdf_microfacet.h`). `alpha` is roughness², as the
// Principled node sets it.

// Smith's Λ for GGX — Heitz 2014, equation 72 (`bsdf_lambda`).
fn ggx_lambda(alpha2: f32, cos_n: f32) -> f32 {
    let t = alpha2 * max(1.0 / max(cos_n * cos_n, 1e-8) - 1.0, 0.0);
    return 0.5 * (sqrt(1.0 + t) - 1.0);
}

// A visible microfacet normal for view direction `wi` in the local frame
// (n = +z) — Heitz 2018, "Sampling the GGX Distribution of Visible Normals"
// (`microfacet_ggx_sample_vndf`). Sampling *visible* normals rather than the
// distribution itself is what keeps the weight below to two Λ terms and the
// variance low at grazing angles.
fn ggx_sample_vndf(wi: vec3<f32>, alpha: f32, r1: f32, r2: f32) -> vec3<f32> {
    let wi_ = normalize(vec3<f32>(alpha * wi.x, alpha * wi.y, wi.z));
    let lensq = wi_.x * wi_.x + wi_.y * wi_.y;
    var t1: vec3<f32>;
    var t2: vec3<f32>;
    if (lensq > 1e-7) {
        t1 = vec3<f32>(-wi_.y, wi_.x, 0.0) / sqrt(lensq);
        t2 = cross(wi_, t1);
    } else {
        t1 = vec3<f32>(1.0, 0.0, 0.0);
        t2 = vec3<f32>(0.0, 1.0, 0.0);
    }
    var t = unit_disk(r1, r2);
    t.y = mix(sqrt(max(1.0 - t.x * t.x, 0.0)), t.y, 0.5 * (1.0 + wi_.z));
    let h_ = t1 * t.x + t2 * t.y + wi_ * sqrt(max(0.0, 1.0 - t.x * t.x - t.y * t.y));
    return normalize(vec3<f32>(alpha * h_.x, alpha * h_.y, max(0.0, h_.z)));
}

// The throughput a VNDF-sampled reflection carries: `eval / pdf` from
// `bsdf_microfacet_sample`, which is `F · (1 + Λi) / (1 + Λi + Λo)`. No
// D, no cosines — they cancel between eval and pdf.
fn ggx_weight(alpha2: f32, cos_ni: f32, cos_no: f32) -> f32 {
    let li = ggx_lambda(alpha2, cos_ni);
    let lo = ggx_lambda(alpha2, cos_no);
    return (1.0 + li) / (1.0 + li + lo);
}

// Radiance arriving from direction `d`: the world's environment image when it
// has one, otherwise a soft dome (brighter above the horizon).
fn env_at(d: vec3<f32>) -> vec3<f32> {
    if (p.envmap == 0u) {
        return p.env.rgb * mix(0.35, 1.15, clamp(d.z * 0.5 + 0.5, 0.0, 1.0));
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

struct Surf {
    pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, f0: vec3<f32>, rough: f32,
    // Subsurface: how far light wraps past the terminator (0 = none) and the
    // colour it takes on inside the medium (from the scattering radius).
    bleed: f32, sss_tint: vec3<f32>,
    // Fabric sheen and clearcoat.
    sheen: f32, sheen_rough: f32, sheen_tint: vec3<f32>,
    coat: f32, coat_rough: f32,
}

// A surface with only the base lobes (secondary hits, Workbench-style shading).
fn plain_surf(pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, f0: vec3<f32>, rough: f32) -> Surf {
    return Surf(pos, n, v, albedo, f0, rough, 0.0, vec3<f32>(1.0), 0.0, 0.5, vec3<f32>(1.0), 0.0, 0.03);
}

// Scattering radius (world units) as the tint light picks up under the skin:
// the longest channel sets the scale, the others darken towards it. Blender's
// default (1, 0.2, 0.1) × 0.05 m is what gives skin its red bleed.
fn scatter_tint(radius: vec3<f32>) -> vec3<f32> {
    let m = max(max(radius.x, radius.y), max(radius.z, 1e-6));
    return mix(vec3<f32>(1.0), radius / m, 0.85);
}

// How far light wraps around the terminator for a scattering radius: a wide
// radius on a small feature (an ear, a nostril) bleeds most.
fn scatter_bleed(weight: f32, radius: vec3<f32>) -> f32 {
    let m = max(max(radius.x, radius.y), max(radius.z, 0.0));
    return clamp(weight, 0.0, 1.0) * clamp(m * 24.0, 0.15, 1.0);
}

// Charlie sheen distribution (`bsdf_sheen.h`): a narrow rim lobe that is
// brightest at grazing angles, which is what reads as cotton or velvet fuzz.
fn charlie(nh: f32, rough: f32) -> f32 {
    let inv = 1.0 / max(rough * rough, 0.005);
    let sin2 = max(1.0 - nh * nh, 0.0);
    return (2.0 + inv) * pow(sin2, inv * 0.5) / (2.0 * PI);
}

// Density of the GGX sampler (`ggx_sample_vndf`, then the view direction
// mirrored in the sampled normal) for the outgoing direction `l`:
// `G1(v) · D(h) / (4 · n·v)` — Cycles' `bsdf_microfacet_eval` pdf.
fn ggx_pdf(s: Surf, l: vec3<f32>) -> f32 {
    let h = normalize(l + s.v);
    let nh = max(dot(s.n, h), 0.0);
    let a2 = max(s.rough * s.rough * s.rough * s.rough, 1e-4);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let dist = a2 / (PI * dd * dd);
    let nv = max(dot(s.n, s.v), 1e-4);
    let g1 = 1.0 / (1.0 + ggx_lambda(a2, nv));
    return g1 * dist / (4.0 * nv);
}

// Diffuse (optionally scattering) + GGX specular + sheen + coat, for light
// from `l` with irradiance `e`.
fn brdf(s: Surf, l: vec3<f32>, e: vec3<f32>) -> vec3<f32> {
    return brdf_w(s, l, e, 1.0, 1.0);
}

// The same, with the diffuse lobe weighted by `wd` and the specular lobe by
// `ws`: the multiple-importance weights of the estimator this is part of.
// Sheen and coat are gathered by light sampling alone, so they keep weight 1.
fn brdf_w(s: Surf, l: vec3<f32>, e: vec3<f32>, wd: f32, ws: f32) -> vec3<f32> {
    // Wrapped diffuse: subsurface light keeps arriving past the terminator.
    let raw = dot(s.n, l);
    let w = s.bleed * 0.75;
    let nl = max((raw + w) / (1.0 + w), 0.0) / (1.0 + w * 0.5);
    // Light that travelled through a thin part and left on the viewer's side.
    let thru = s.bleed * s.albedo * s.sss_tint * pow(max(dot(s.v, -l), 0.0), 3.0) * 0.35;
    if (nl <= 0.0) { return thru * e * wd; }
    let hv = normalize(l + s.v);
    let nh = max(dot(s.n, hv), 0.0);
    let hvv = max(dot(hv, s.v), 0.0);
    let a2 = max(s.rough * s.rough * s.rough * s.rough, 1e-4);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let dist = a2 / (PI * dd * dd);
    let fres = s.f0 + (1.0 - s.f0) * pow(1.0 - hvv, 5.0);
    // Cycles' `bsdf_microfacet_eval`: D · F · G2 / (4 · n·i · n·o), with the
    // n·o supplied by the `nl` this is multiplied by on the way out.
    let nv = max(dot(s.n, s.v), 1e-4);
    let g2 = 1.0 / (1.0 + ggx_lambda(a2, nv) + ggx_lambda(a2, max(raw, 1e-4)));
    let spec = fres * dist * g2 * 0.25 / nv;
    let diffuse = mix(s.albedo, s.albedo * s.sss_tint, s.bleed) / PI;
    let sheen = s.sheen * s.sheen_tint * charlie(nh, s.sheen_rough);
    // Clearcoat: a second, smoother dielectric lobe; what it reflects it also
    // takes away from the lobes beneath it.
    let ca2 = max(s.coat_rough * s.coat_rough * s.coat_rough * s.coat_rough, 1e-4);
    let cd = nh * nh * (ca2 - 1.0) + 1.0;
    let coat_f = 0.04 + 0.96 * pow(1.0 - hvv, 5.0);
    let coat = s.coat * coat_f * (ca2 / (PI * cd * cd)) * 0.25;
    let under = 1.0 - s.coat * coat_f;
    return ((diffuse * wd + spec * ws + sheen) * under + coat) * e * nl + thru * e * wd;
}

fn visible(org: vec3<f32>, l: vec3<f32>, dist: f32) -> f32 {
    return select(1.0, 0.0, trace(org, l, dist, true).tri != 0xffffffffu);
}

fn lights(s: Surf, seed: ptr<function, u32>) -> vec3<f32> {
    var c = vec3<f32>(0.0);
    // An environment image is a light. With one present and no lamps, it is
    // the only light — Blender adds nothing on top of a world, and Material
    // Preview's studio HDRI is exactly this case. The three-point rig below
    // is a fallback for a scene with no light of any kind, and adding it on
    // top of a world made every lamp-less HDRI scene about a fifth too bright.
    if (p.count == 0u && p.envmap != 0u) { return c; }
    if (p.count == 0u) {
        // Three-point rig tied to the view: key (shadowed), fill, rim.
        let key = cone_dir(p.sun, 0.06, seed);
        c += brdf(s, key, vec3<f32>(2.0)) * visible(s.pos, key, INF);
        let fill = normalize(-p.right * 0.8 + p.up * 0.2 + p.back * 0.6);
        c += brdf(s, fill, vec3<f32>(0.45));
        let rim = normalize(p.up * 0.6 - p.back * 0.8 + p.right * 0.2);
        c += brdf(s, rim, vec3<f32>(0.9)) * visible(s.pos, rim, INF);
        return c;
    }
    // Up to 4 lights per sample; with more, pick 4 at random and reweight.
    let many = p.count > 4u;
    let weight = select(1.0, f32(p.count) / 4.0, many);
    for (var k = 0u; k < min(p.count, 4u); k++) {
        var idx = k;
        if (many) { idx = min(u32(rnd(seed) * f32(p.count)), p.count - 1u); }
        let li = p.lights[idx];
        if (li.kind == 0u) {
            let l = cone_dir(-li.dir, li.size * 0.5, seed);
            c += brdf(s, l, li.color) * visible(s.pos, l, INF) * weight;
            continue;
        }
        let r1 = rnd(seed);
        let r2 = rnd(seed);
        let aim = li.pos + onb(normalize(s.pos - li.pos), (r1 - 0.5) * li.size, (r2 - 0.5) * li.size);
        let to = aim - s.pos;
        let d2 = max(dot(to, to), 1e-6);
        let l = to / sqrt(d2);
        var e = li.color / (4.0 * PI * PI * d2);
        if (li.kind == 2u) {
            e *= smoothstep(li.cone, mix(li.cone, 1.0, 0.15), dot(-l, li.dir));
        } else if (li.kind == 3u) {
            e *= max(dot(-l, li.dir), 0.0) * 2.0;
        }
        if (max(e.r, max(e.g, e.b)) < 1e-5) { continue; }
        c += brdf(s, l, e) * visible(s.pos, l, sqrt(d2) * 0.999) * weight;
    }
    return c;
}


// Probability that `glow_light` picks face `f` (0 when it does not glow).
fn lamp_chance(f: u32) -> f32 {
    let count = data[0];
    if (count == 0u) { return 0.0; }
    var lo = 0u;
    var hi = count - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (data[1u + mid] < f) { lo = mid + 1u; } else { hi = mid; }
    }
    if (data[1u + lo] != f) { return 0.0; }
    let prev = select(0.0, tf(count + lo), lo > 0u);
    return tf(1u + count + lo) - prev;
}

// Next-event estimation: one emissive triangle (by power), one point on it.
fn glow_light(s: Surf, seed: ptr<function, u32>) -> vec3<f32> {
    let count = data[0];
    if (count == 0u) { return vec3<f32>(0.0); }
    // Pick by power: binary search in the cdf stored after the ids.
    let u = rnd(seed);
    var lo = 0u;
    var hi = count - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (tf(1u + count + mid) < u) { lo = mid + 1u; } else { hi = mid; }
    }
    let prev = select(0.0, tf(count + lo), lo > 0u);
    let chance = max(tf(1u + count + lo) - prev, 1e-8);
    let t = tri_at(data[1u + lo]);
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
    return brdf(s, l, e) * w * visible(s.pos, l, sqrt(d2) * 0.999);
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

// Cosine-weighted direction over the hemisphere around `n`.
fn cosine_dir(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let r1 = rnd(seed);
    let rr = sqrt(r1);
    let ph = 2.0 * PI * rnd(seed);
    return normalize(onb(n, rr * cos(ph), rr * sin(ph)) + n * sqrt(max(0.0, 1.0 - r1)));
}

// Emission from a face reached by a cosine-sampled ray, weighted against
// `glow_light`, which samples the same emitters directly (power heuristic).
fn glow_mis(t: Tri, h: Hit, from_n: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let e = emit_at(t, vec3<f32>(1.0 - h.u - h.v, h.u, h.v));
    let chance = lamp_chance(h.tri);
    if (chance <= 0.0) { return e; }
    let cr = cross(t.b - t.a, t.c - t.a);
    let pl = chance * h.t * h.t / max(0.5 * length(cr) * abs(dot(normalize(cr), dir)), 1e-8);
    let pb = max(dot(from_n, dir), 0.0) / PI;
    return e * pb * pb / max(pb * pb + pl * pl, 1e-12);
}

// Next-event estimation against the world image: one direction drawn by the
// environment's own brightness, shadow-rayed, and weighted against the
// cosine-sampled ray that `indirect` sends (power heuristic, against the
// diffuse lobe — Cycles weights each lobe separately).
// One environment sample, weighted against the other estimators that see
// the same light: the cosine bounce for the diffuse lobe and — when the
// caller also takes a GGX sample (`mis_spec`) — that sample for the specular
// lobe. Without the second weight the specular energy of a bright sky was
// counted twice, once here and once by the reflection ray: +5 % on a metal
// under a flat sky, and a metal chest reflecting the studio sky at 103
// against Cycles' 73.
fn env_light(s: Surf, seed: ptr<function, u32>, mis_spec: bool) -> vec3<f32> {
    if (p.env_cdf == 0u || p.envmap == 0u) { return vec3<f32>(0.0); }
    let e = env_sample(seed);
    if (e.pdf <= 0.0) { return vec3<f32>(0.0); }
    let nl = dot(s.n, e.dir);
    if (nl <= 0.0) { return vec3<f32>(0.0); }
    let pb = nl / PI;
    let pe2 = e.pdf * e.pdf;
    let wd = pe2 / max(pe2 + pb * pb, 1e-12);
    var ws = 1.0;
    if (mis_spec) {
        let pg = ggx_pdf(s, e.dir);
        ws = pe2 / max(pe2 + pg * pg, 1e-12);
    }
    return brdf_w(s, e.dir, e.radiance / e.pdf, wd, ws) * visible(s.pos, e.dir, INF);
}

// Diffuse bounces: radiance arriving at a surface with normal `n` at `pos`.
//
// Each step sends one cosine-weighted ray, gathers what it lands on (direct
// light there plus any emitter, then the world), and carries on with the
// throughput scaled by that surface's albedo — Blender's Light Paths → Max
// Bounces → Diffuse. A cosine-sampled ray needs no `cos/pdf` factor, so the
// albedo *is* the throughput.
//
// The default depth is **1**, not Blender's 4, and that is deliberate: the
// tail below closes a truncated path with the world unoccluded, so at depth 1
// it is the calibrated one-bounce ambient this used to be and matches Cycles
// on Female-01 with no pixel more than 120/765 darker. Raising the depth
// replaces that over-bright tail with real, occluded bounces and a cavity goes
// *darker* — 82 such pixels at depth 2, 1417 at depth 6 — because a
// cosine-sampled ray has to find a cavity's opening by chance. Environment
// next-event estimation is what makes a deeper path pay off; until it lands,
// depth stays at 1 and `Camera::bounces` raises it for anyone who wants to
// measure.
//
// Without a world image and without glowing meshes there is nothing to gather,
// so one occlusion ray within `p.ao` stands in — Workbench's ambient occlusion.
fn indirect(pos: vec3<f32>, n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let lit = p.envmap != 0u;
    if (p.glow == 0u && !lit) {
        let dir = cosine_dir(n, seed);
        return env_at(dir) * mix(0.25, 1.0, visible(pos, dir, p.ao));
    }
    let sampled = p.env_cdf != 0u;
    let depth = select(select(1u, 4u, sampled), p.bounces, p.bounces != 0u);
    var acc = vec3<f32>(0.0);
    var thr = vec3<f32>(1.0);
    var org = pos;
    var nrm = n;
    // Set when a ray escapes: the path is then complete and needs no tail.
    var escaped = false;
    for (var b = 0u; b < depth; b++) {
        let dir = cosine_dir(nrm, seed);
        let h = trace(org, dir, INF, false);
        if (h.tri == 0xffffffffu) {
            // Weighted against `env_light`, which samples the same world.
            let pb = max(dot(nrm, dir), 0.0) / PI;
            let pe = env_pdf(dir);
            acc += thr * env_at(dir) * select(1.0, pb * pb / max(pb * pb + pe * pe, 1e-12), pe > 0.0);
            escaped = true;
            break;
        }
        let t = tri_at(h.tri);
        let ng0 = normalize(cross(t.b - t.a, t.c - t.a));
        let ng = select(-ng0, ng0, dot(ng0, dir) < 0.0);
        let hit = org + dir * h.t + ng * (p.eps + h.t * 1e-5);
        let base = max(t.color.rgb, vec3<f32>(0.0));
        let albedo = base * (1.0 - t.metal);
        acc += thr * glow_mis(t, h, nrm, dir);
        // A ray that never left the AO radius is treated as the flat world
        // would treat it, so a crevice does not go black without a world image.
        let near = select(1.0, 0.25, !lit && h.t <= p.ao);
        let s = plain_surf(hit, ng, -dir, albedo, mix(vec3<f32>(0.04), base, t.metal), max(t.rough, 0.3));
        acc += thr * (lights(s, seed) + glow_light(s, seed) + env_light(s, seed, false)) * near;
        thr *= albedo;
        if (max(thr.r, max(thr.g, thr.b)) < 0.02) { break; }
        org = hit;
        nrm = ng;
    }
    // A path cut off inside the scene would otherwise lose every bounce it
    // never got to take, which turns a crevice black. Close it with the world
    // seen from the last surface — the one-bounce ambient this used to be.
    // Without a sampling table the tail stands in for the light a cosine ray
    // rarely finds; with one, that light is sampled directly and the tail would
    // double it.
    if (!escaped && !sampled) { acc += thr * env_at(nrm) * 0.8; }
    return acc;
}

// Radiance leaving a secondary hit (flat normal, no texture, direct + ambient) —
// what a reflection sees, where a full path would not pay for itself.
fn bounce_light(h: Hit, org: vec3<f32>, d: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let t = tri_at(h.tri);
    let ng0 = normalize(cross(t.b - t.a, t.c - t.a));
    let ng = select(-ng0, ng0, dot(ng0, d) < 0.0);
    let pos = org + d * h.t + ng * (p.eps + h.t * 1e-5);
    let base = max(t.color.rgb, vec3<f32>(0.0));
    let s = plain_surf(pos, ng, -d, base * (1.0 - t.metal), mix(vec3<f32>(0.04), base, t.metal), max(t.rough, 0.3));
    return lights(s, seed) + glow_light(s, seed) + s.albedo * env_at(ng) * 0.8 + emit_at(t, vec3<f32>(1.0 - h.u - h.v, h.u, h.v));
}
