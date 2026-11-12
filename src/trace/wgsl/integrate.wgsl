// The estimators: direct light from every source under MIS, the world's
// own query, and the indirect bounce. The lights are in `light.wgsl`, the
// surface in `brdf.wgsl`.

// Cosine-weighted direction over the hemisphere around `n`.
fn cosine_dir(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let r1 = rnd(seed);
    let rr = sqrt(r1);
    let ph = 2.0 * PI * rnd(seed);
    return normalize(onb(n, rr * cos(ph), rr * sin(ph)) + n * sqrt(max(0.0, 1.0 - r1)));
}

// Emission from a face reached by a cosine-sampled ray, weighted against
// `glow_query`, which samples the same emitters directly (power heuristic).
fn glow_mis(t: Tri, h: Hit, from_n: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let e = emit_at(t, vec3<f32>(1.0 - h.u - h.v, h.u, h.v));
    let chance = lamp_chance(h);
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
fn env_query(s: Surf, seed: ptr<function, u32>, mis_spec: bool) -> Shadowed {
    if (p.env_cdf == 0u || p.envmap == 0u) { return unshadowed(); }
    let e = env_sample(seed);
    if (e.pdf <= 0.0) { return unshadowed(); }
    let nl = dot(s.n, e.dir);
    if (nl <= 0.0) { return unshadowed(); }
    let pb = nl / PI;
    let pe2 = e.pdf * e.pdf;
    let wd = pe2 / max(pe2 + pb * pb, 1e-12);
    var ws = 1.0;
    if (mis_spec) {
        let pg = ggx_pdf(s, e.dir);
        ws = pe2 / max(pe2 + pg * pg, 1e-12);
    }
    return Shadowed(brdf_w(s, e.dir, e.radiance / e.pdf, wd, ws), e.dir, INF);
}

// Every light at a shading point — the picked lamp, an emissive face, the
// world image, or the studio rig when the scene has no light at all —
// each with its shadow ray, traced from this ONE call site.
//
// It was three functions with seven `visible` calls between them, and a
// path calls them at the camera hit, at every bounce and at a reflection's
// hit: each call site is a whole BVH walk inlined into the kernel, which
// came to about twenty walks. That is what took NVIDIA's compiler three to
// five minutes on a new kernel and llvmpipe past 7 GB. The lights and the
// random numbers they draw are the same as before, in the same order.
// `with_env` is false for a reflection's hit, which takes the world as a
// flat term instead (`bounce_light`); `mis_spec` is `env_query`'s weight
// against a GGX sample the caller also takes.
fn direct(s: Surf, seed: ptr<function, u32>, with_env: bool, mis_spec: bool) -> vec3<f32> {
    let rig = uses_rig();
    var sum = select(vec3<f32>(0.0), rig_fill(s), rig);
    for (var k = 0u; k < 4u; k++) {
        var q = unshadowed();
        if (k == 0u) {
            if (rig) { q = rig_query(0u, s, seed); } else { q = lamp_query(s, seed); }
        } else if (k == 1u) {
            if (rig) { q = rig_query(1u, s, seed); }
        } else if (k == 2u) {
            q = glow_query(s, seed);
        } else if (with_env) {
            q = env_query(s, seed, mis_spec);
        }
        if (max(q.c.r, max(q.c.g, q.c.b)) > 0.0) {
            sum += q.c * visible(s.pos, q.dir, q.dist);
        }
    }
    return sum;
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
// `bounce` false is Max Bounces' Total 0 (or Diffuse 0): the world is still
// a light the surface sees — Cycles counts the background as direct — but a
// ray that meets geometry brings nothing back.
fn indirect(pos: vec3<f32>, n: vec3<f32>, seed: ptr<function, u32>, bounce: bool) -> vec3<f32> {
    // A world is something to gather, image or no image — the same
    // predicate the direct pass uses (§54). This tested the image alone, so
    // a scene whose world is a flat colour never took the path below at
    // all: it got one occlusion ray with a 0.25 floor instead, and
    // `Camera::bounces` did nothing for it (`rendering.md` §56).
    let lit = p.envmap != 0u || dot(p.env.rgb, p.env.rgb) > 0.0;
    if (p.glow == 0u && !lit) {
        let dir = cosine_dir(n, seed);
        return env_at(dir) * mix(0.25, 1.0, visible(pos, dir, p.ao));
    }
    let sampled = p.env_cdf != 0u;
    let depth = select(1u, select(select(1u, 4u, sampled), p.bounces, p.bounces != 0u), bounce);
    var acc = vec3<f32>(0.0);
    var thr = vec3<f32>(1.0);
    var org = pos;
    var nrm = n;
    // Set when a ray escapes: the path is then complete and needs no tail.
    var escaped = false;
    // The smallest density the path was sampled with (`min_ray_pdf`), for
    // Filter Glossy; and the ray carried on through glass, if it was.
    var min_pdf = 3.0e38;
    var through = false;
    var dir = vec3<f32>(0.0, 0.0, 1.0);
    var pb = 1.0;
    for (var b = 0u; b < depth; b++) {
        if (!through) {
            dir = cosine_dir(nrm, seed);
            pb = max(dot(nrm, dir), 0.0) / PI;
        }
        through = false;
        min_pdf = min(min_pdf, pb);
        let h = trace(org, dir, INF, false);
        if (h.tri == 0xffffffffu) {
            // Weighted against `env_query`, which samples the same world.
            let pe = env_pdf(dir);
            acc += thr * env_at(dir) * select(1.0, pb * pb / max(pb * pb + pe * pe, 1e-12), pe > 0.0);
            escaped = true;
            break;
        }
        if (!bounce) { break; }
        let t = tri_at(h);
        let ng0 = normalize(cross(t.b - t.a, t.c - t.a));
        let ng = select(-ng0, ng0, dot(ng0, dir) < 0.0);
        let hit = org + dir * h.t + ng * (p.eps + h.t * 1e-5);
        let base = max(t.color.rgb, vec3<f32>(0.0));
        let rough = max(t.rough, glossy_floor(min_pdf));
        // This hit came by a diffuse bounce, so Caustics decides whether it
        // reflects at all: without Reflective the metal and the dielectric
        // reflection are both gone, and the diffuse under the latter keeps
        // what it would have given up.
        let glossy = caustic_reflective();
        let fd = select(vec3<f32>(0.0), dielectric_f0(t.ior), glossy);
        let metal = select(0.0, t.metal, glossy);
        // Glass is not a diffuse surface: what it transmits is not
        // scattered here (`closure.h` takes the transmission's weight out of
        // everything under it).
        let tw = t.trans * (1.0 - t.metal);
        var albedo = base * (1.0 - t.metal) * (1.0 - tw);
        if (max(albedo.r, max(albedo.g, albedo.b)) > 0.0) {
            albedo *= under_layer(fd, rough, max(dot(ng, -dir), 1e-4));
        }
        acc += thr * glow_mis(t, h, nrm, dir);
        // A ray that never left the AO radius is treated as the flat world
        // would treat it, so a crevice does not go black without a world image.
        let near = select(1.0, 0.25, !lit && h.t <= p.ao);
        let s = plain_surf(hit, ng, -dir, albedo, mix(fd, base, metal), rough);
        acc += thr * direct(s, seed, true, false) * near;
        // With Refractive on, the share glass lets through carries on in
        // the direction it came — a thin pane's refraction, which is what
        // lights a car's inside through its windows — tinted once per face
        // as the camera's path tints it. Without it, Cycles drops the glass
        // closure after a diffuse bounce and that light never arrives.
        if (tw > 0.0 && caustic_refractive() && rnd(seed) < tw) {
            thr *= sqrt(base);
            let at = org + dir * h.t;
            org = at - ng * through_lift(at, h.t);
            through = true;
            continue;
        }
        thr *= albedo;
        // Russian roulette, as Cycles terminates a path
        // (`path_state_continuation_probability`).
        //
        // This was a hard cutoff — stop below a throughput of 0.02 — which
        // is cheap and *biased*: every path it drops loses the light it
        // would still have carried, and that light is exactly the light in
        // crevices, so the cutoff darkened them. It also barely fired. A
        // surface of albedo 0.8 is still at 0.41 after four bounces, so on
        // a bright scene almost every path ran the full depth.
        //
        // Roulette keeps a path with a probability equal to its own
        // throughput and scales the survivors up by the same factor, so the
        // expected radiance is unchanged — the image is the same, the ray
        // count is not. Dim paths die early, bright ones carry on, and
        // nothing is silently discarded.
        //
        // Capped at 0.95 so even a white surface sheds some paths, and not
        // applied to the last bounce, which has nothing left to terminate.
        if (b + 1u < depth) {
            let keep = clamp(max(thr.r, max(thr.g, thr.b)), 0.0, 0.95);
            if (keep < 1e-4 || rnd(seed) > keep) { break; }
            thr /= keep;
        }
        org = hit;
        nrm = ng;
    }
    // A path cut off inside the scene would otherwise lose every bounce it
    // never got to take, which turns a crevice black. Close it with the world
    // seen from the last surface — the one-bounce ambient this used to be.
    // Without a sampling table the tail stands in for the light a cosine ray
    // rarely finds; with one, that light is sampled directly and the tail would
    // double it.
    // A *uniform* world needs none of that: a cosine ray finds it with
    // exactly the probability the surface is open to it, so the path above
    // is already unbiased and the tail would be pure over-count. It is for
    // an environment image with no sampling table.
    if (bounce && !escaped && !sampled && p.envmap != 0u) { acc += thr * env_at(nrm) * 0.8; }
    return acc;
}

// Radiance leaving a secondary hit (flat normal, no texture, direct + ambient) —
// what a reflection sees, where a full path would not pay for itself.
// `min_pdf` is the density the reflection was sampled with, for Filter
// Glossy: a sharp mirror's is high and blurs nothing, a rough one's low.
fn bounce_light(h: Hit, org: vec3<f32>, d: vec3<f32>, min_pdf: f32, seed: ptr<function, u32>) -> vec3<f32> {
    let t = tri_at(h);
    let ng0 = normalize(cross(t.b - t.a, t.c - t.a));
    let ng = select(-ng0, ng0, dot(ng0, d) < 0.0);
    let pos = org + d * h.t + ng * (p.eps + h.t * 1e-5);
    let base = max(t.color.rgb, vec3<f32>(0.0));
    let rough = max(t.rough, glossy_floor(min_pdf));
    let fd = dielectric_f0(t.ior);
    var albedo = base * (1.0 - t.metal);
    if (max(albedo.r, max(albedo.g, albedo.b)) > 0.0) {
        albedo *= under_layer(fd, rough, max(dot(ng, -d), 1e-4));
    }
    let s = plain_surf(pos, ng, -d, albedo, mix(fd, base, t.metal), rough);
    return direct(s, seed, false, false) + s.albedo * env_at(ng) * 0.8 + emit_at(t, vec3<f32>(1.0 - h.u - h.v, h.u, h.v));
}
