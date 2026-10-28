// A point in a regular polygon of `corners` sides — the bokeh a bladed iris
// makes. Picks a corner, samples the triangle it spans, then rotates.
fn unit_polygon(corners: f32, spin: f32, ru: f32, rv: f32) -> vec2<f32> {
    var u = ru;
    var v = rv;
    let corner = floor(u * corners);
    u = u * corners - corner;
    u = sqrt(u);
    v = v * u;
    u = 1.0 - u;
    let angle = PI / corners;
    let q = vec2<f32>((u + v) * cos(angle), (u - v) * sin(angle));
    let rot = spin + corner * 2.0 * angle;
    return vec2<f32>(cos(rot) * q.x - sin(rot) * q.y, sin(rot) * q.x + cos(rot) * q.y);
}

// The aperture offset for one sample, in world units along `right` / `up`.
fn lens_offset(ru: f32, rv: f32) -> vec2<f32> {
    var bokeh: vec2<f32>;
    if (p.blades == 0u) {
        bokeh = unit_disk(ru, rv);
    } else {
        bokeh = unit_polygon(f32(p.blades), p.blade_spin, ru, rv);
    }
    // Anamorphic squeeze, as Cycles applies it: across only.
    var ratio = p.aperture_ratio;
    if (ratio <= 0.0) { ratio = 1.0; }
    bokeh.x /= ratio;
    return bokeh * p.aperture;
}

// Cycles' sample clamp: a contribution brighter than `limit` in its
// brightest channel is scaled down to it, hue kept; 0 is no clamp.
fn clamp_light(l: vec3<f32>, limit: f32) -> vec3<f32> {
    let m = max(l.r, max(l.g, l.b));
    if (limit <= 0.0 || m <= limit) { return l; }
    return l * (limit / m);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i = pixel(gid);
    if (i >= p.n) { return; }
    // Past the first sample, Adaptive Sampling hands this thread the k-th
    // pixel still open (`gather.wgsl`) rather than the k-th pixel, so a
    // warp is never mostly settled pixels idling beside one that is not.
    if (adapt_on() && p.frame > 0u) {
        if (i >= px[34u * p.n]) { return; }
        i = px[32u * p.n + i];
    }
    // A pixel Adaptive Sampling has settled is not sampled again
    // (`film_need_sample_pixel`). This is where the time is saved: on a
    // frame whose floor and background converge in tens of samples, most
    // of the image stops costing anything long before the target.
    //
    // The flags live in the image buffer, which a render at the same size
    // keeps (`Tracer::sample_in`), so a new image clears them here: without
    // this a pixel the LAST render settled was never sampled again and kept
    // showing that render.
    if (p.frame == 0u) {
        px[adapt_flag(i)] = 0u;
        px[adapt_seen(i)] = 0u;
    }
    if (adapt_settled(i)) { return; }
    let sp = frame_point(i);
    let su = sp.x;
    let sv = sp.y;
    var seed = hash(i * 9781u + p.frame * 6271u + 1u);
    // Where in the pixel, by Film ▸ Pixel Filter (`filter_offset2`).
    let jo = filter_offset2(rnd(&seed), rnd(&seed));
    let jx = jo.x / frame_size().x;
    let jy = jo.y / frame_size().y;
    let vx = (su + jx - p.pan.x) / p.scale;
    let vy = -((sv + jy) / p.aspect - p.pan.y) / p.scale;
    var o: vec3<f32>;
    var d: vec3<f32>;
    if (p.persp != 0u) {
        o = p.pivot + p.back * (p.dist / p.units);
        d = normalize(p.right * vx + p.up * vy - p.back * p.dist);
        if (p.aperture > 0.0) {
            // The plane of focus is perpendicular to the view axis, so the
            // distance along *this* ray is `focus / cos`, not `focus`
            // (Cycles: `ft = focaldistance / D.z`). Using `focus` directly
            // would curve the focus surface into a sphere and blur the
            // corners of every frame.
            let forward = max(dot(d, -p.back), 1.0e-4);
            let aim = o + d * (p.focus / forward);
            let lens = lens_offset(rnd(&seed), rnd(&seed));
            o = o + p.right * lens.x + p.up * lens.y;
            d = normalize(aim - o);
        }
    } else {
        o = p.pivot + (p.right * vx + p.up * vy + p.back * 1.0e4) / p.units;
        d = -p.back;
    }
    let pix = 1.0 / (p.scale * p.units * frame_size().x);
    // The theme's sky, with its sweep when it has one (`sky_at`).
    let sky = sky_at(i);
    // The selection outline's prepass: nothing until the first surface
    // says otherwise, so every return below leaves it right.
    outline_store(i, 0u);
    let o0 = o;
    let d0 = d;
    var first_t = INF;
    var col = vec3<f32>(0.0);
    var thr = vec3<f32>(1.0);
    // Primary path: alpha cut-outs and glass continue the ray, each kind
    // up to its Max Bounces row (`p.depth`, Cycles' `transparent_bounce`
    // and `transmission_bounce` counters); a path past either ends there.
    // Unset rows keep the old eight events of each.
    let depth_set = p.depth.x != 0u;
    let total = select(8u, p.depth.x - 1u, depth_set);
    let see_through = select(8u, p.depth.y - 1u, depth_set);
    let through = select(8u, min(p.depth.z - 1u, total), depth_set);
    // Total 0 is direct light only: no bounce of any kind.
    let glossy_on = !depth_set || (p.depth.w > 1u && total > 0u);
    let diffuse_on = !depth_set || (p.bounces > 0u && total > 0u);
    var n_see = 0u;
    var n_through = 0u;
    // Every pass either skips a cut-out (`n_see`) or shades a surface that
    // ends the path or refracts through it (`n_through`), so this many
    // passes is enough. The bound is a value, not a constant: a constant
    // invites the shader compiler to unroll this whole body that many times.
    let passes = see_through + through + 1u;
    for (var bounce = 0u; bounce < passes; bounce++) {
        let h = trace(o, d, INF, false);
        if (h.tri == 0xffffffffu) {
            if (bounce == 0u) {
                // With an environment image the background *is* the world —
                // unless it is hidden, as Material Preview's studio light is,
                // in which case it lights the scene and the sky is shown.
                // A transparent film (`p.film.x`): nothing met, nothing
                // shown — black at alpha 0, as Blender's render result.
                if (p.film.x != 0u) {
                    guide(i, vec3<f32>(0.0), -d, -1.0);
                    emit(i, vec3<f32>(0.0), vec4<f32>(0.0), 0.0, 1.0);
                    return;
                }
                if ((p.envmap != 0u && p.env_hidden == 0u) || p.film.y != 0u) {
                    let bg = env_at(d);
                    guide(i, bg, -d, -1.0);
                    emit(i, bg, vec4<f32>(0.0), 1.0, 0.0);
                    return;
                }
                // The sky is display-referred already — it is the theme's own
                // canvas colour, not radiance — so it rides the overlay term,
                // past the view transform, exactly as the grid beside it does.
                //
                // It used to go in as light: `srgb_to_linear(sky)` out through
                // `view_transform`. That is a round trip only under
                // `Standard`. With Filmic or AgX — Blender 4.x's default, so
                // most files — the background came back darker and
                // desaturated than the colour the theme asked for, and the
                // viewport dimmed the moment a file named a view transform.
                // Solid and Wireframe never transformed it (`solid.wgsl`,
                // `wire.wgsl`), so one theme gave three different backgrounds
                // across the four shading modes.
                let g = grid_over(o, d, INF, pix);
                guide(i, srgb_to_linear(sky), -d, -1.0);
                emit(i, vec3<f32>(0.0), vec4<f32>(sky * (1.0 - g.a) + g.rgb, 1.0), 1.0, 1.0);
                return;
            }
            col += thr * env_at(d);
            break;
        }
        if (bounce == 0u) {
            first_t = h.t;
            outline_store(i, h.inst + 1u);
        }
        // A Mix Shader's two closures: one of them per hit, by the weight
        // (`svm_node_mix_closure` weights the closures and the sampler picks
        // one by weight, which is unbiased); the other hits of the pixel's
        // samples take the other.
        // The face's own material first: its geometry, the point in the
        // object's space and the UV, which a procedural mix weight reads.
        let own = mat_at(h.tri);
        let geo = tri_with(h, own);
        let w = 1.0 - h.u - h.v;
        let bary = vec3<f32>(w, h.u, h.v);
        let uv = vec2<f32>(dot(geo.us, bary), dot(geo.vs, bary));
        let hit = o + d * h.t;
        let inst = inst_at(h.inst);
        let pobj = object_point(inst, hit);
        // A Fresnel or Layer Weight on the mix reads the smooth normal
        // facing the ray (`procedural.wgsl`'s `proc_n`, `proc_wi`).
        let n_own = geo.na * w + geo.nb * h.u + geo.nc * h.v;
        proc_wi = -d;
        proc_n = select(-d, normalize(n_own) * select(-1.0, 1.0, dot(n_own, d) < 0.0), dot(n_own, n_own) > 1e-12);
        var m = own;
        let other = mat_second(m);
        if (other != NO_MAT && rnd(&seed) < mix_weight(m, pobj, uv, inst)) { m = mat_word(other); }
        let t = tri_with(h, m);
        let ng0 = world_ng(h, t);
        let front = dot(ng0, d) < 0.0;
        let ng = select(-ng0, ng0, front);
        var n = t.na * w + t.nb * h.u + t.nc * h.v;
        n = select(ng, normalize(n) * select(-1.0, 1.0, front), dot(n, n) > 1e-12);
        if (t.nmap != NO_TEX) { n = mapped_normal(t, n, bary); }
        n = proc_normal(m, n, pobj, uv, inst, pix * h.t);
        if (dot(n, -d) < 0.0) { n = ng; }
        proc_n = n;
        let off = ng * (p.eps + h.t * 1e-5);
        let pos = hit + off;
        var base = proc_base(m, max(t.color.rgb, vec3<f32>(0.0)), pobj, uv, inst);
        var alpha = t.color.a;
        if (t.tex != NO_TEX) {
            let tc = sample_tex(t.tex, uv);
            base *= tc.rgb;
            alpha *= tc.a;
        }
        // Roughness / metallic are non-colour data (raw); emission is colour.
        var rough = t.rough;
        var metal = t.metal;
        if (t.maps.x != NO_TEX) { rough *= sample_raw(t.maps.x, uv).r; }
        if (t.maps.y != NO_TEX) { metal *= sample_raw(t.maps.y, uv).r; }
        let emit = emit_at(t, bary);
        // The first surface a pixel sees is what its guides describe.
        if (bounce == 0u) { guide(i, base, n, h.t); }
        if (rnd(&seed) >= alpha) {
            n_see++;
            if (n_see > see_through) { break; }
            o = hit - off;
            continue;
        }
        col += thr * emit;
        let bleed = scatter_bleed(t.sss, t.sss_radius);
        let fd = dielectric_f0(t.ior);
        let under = under_layer(fd, max(rough, 0.04), max(dot(n, -d), 1e-4));
        let s = Surf(
            pos, n, -d, base * (1.0 - metal) * under, mix(fd, base, metal), max(rough, 0.04),
            bleed, scatter_tint(t.sss_radius),
            t.sheen, t.sheen_rough, t.sheen_tint,
            t.coat, t.coat_rough,
        );
        let opaque = 1.0 - t.trans * (1.0 - metal);
        // Light Paths ▸ Clamping (`p.limit`): the camera hit's direct light
        // capped at Direct, everything that bounced at Indirect, as Cycles
        // caps a sample's contribution (`kernel/film/write.h`,
        // `film_clamp_light`). Indirect defaults to 10 there, and that is
        // what keeps a strong lamp seen through a glossy bounce from
        // leaving fireflies that Adaptive Sampling then never settles.
        var c = clamp_light(direct(s, &seed, true, true) * opaque, p.limit.x);
        var bounced = vec3<f32>(0.0);

        // Ambient takes the same scattering tint, so skin stays warm in shadow.
        bounced += mix(s.albedo, s.albedo * s.sss_tint, bleed) * indirect(pos, n, &seed, diffuse_on) * opaque;

        // Glossy reflection: one GGX sample the way Cycles takes it
        // (`bsdf_microfacet_sample`) — a visible microfacet normal, the view
        // direction mirrored in it, and the throughput `F · (1+Λi)/(1+Λi+Λo)`.
        // This replaced a cone around the mirror direction damped by an
        // ad-hoc `(1 − rough · 0.7)`: that lost the sky's highlights on every
        // metal (p95 100 against Cycles' 140 on THYMUS) because a cone of the
        // right width still averages a bright sun over its whole disc.
        let v = -d;
        let tx = normalize(onb(n, 1.0, 0.0));
        let ty = cross(n, tx);
        let ggx_a = s.rough * s.rough;
        let wi_local = vec3<f32>(dot(v, tx), dot(v, ty), dot(v, n));
        let h_local = ggx_sample_vndf(wi_local, ggx_a, rnd(&seed), rnd(&seed));
        let hw = normalize(tx * h_local.x + ty * h_local.y + n * h_local.z);
        let refl = reflect(d, hw);
        let cos_hi = max(dot(hw, v), 0.0);
        let cos_no = dot(n, refl);
        let f0 = s.f0;
        // Fresnel at the microfacet, not the macro normal. The glass path
        // below reuses it for the energy the reflection took.
        let fres = f0 + (1.0 - f0) * pow(1.0 - cos_hi, 5.0);
        if (glossy_on && dot(refl, ng) > 0.0 && cos_hi > 0.0 && cos_no > 0.0) {
            let rh = trace(pos, refl, INF, false);
            var rc = env_at(refl);
            // A reflection that reaches the world is weighted against the
            // environment sample in `env_query`, which sees the same sky;
            // one that lands on geometry has no other estimator.
            var wg = 1.0;
            if (rh.tri != 0xffffffffu) {
                rc = bounce_light(rh, pos, refl, ggx_pdf(s, refl), &seed);
            } else if (p.env_cdf != 0u) {
                let pg = ggx_pdf(s, refl);
                let pe = env_pdf(refl);
                wg = pg * pg / max(pg * pg + pe * pe, 1e-12);
            }
            bounced += fres * rc * ggx_weight(ggx_a * ggx_a, max(dot(n, v), 1e-4), cos_no) * wg;
        }
        c += clamp_light(bounced, p.limit.y);
        col += thr * c;

        // Glass: refract and keep going, tinted by the base colour.
        let tw = t.trans * (1.0 - metal);
        if (tw <= 0.0) { break; }
        n_through++;
        if (n_through > through) { break; }
        let eta = select(t.ior, 1.0 / t.ior, front);
        var nd = refract(d, n, eta);
        if (dot(nd, nd) < 1e-6) {
            // Total internal reflection: all of it comes back, untinted
            // (Cycles' reflection tint is the Specular Tint, white). This
            // took `1 − fres` as if some had left, which darkened every
            // glass edge seen from inside.
            nd = reflect(d, n);
            o = pos;
            thr *= tw;
        } else {
            o = hit - off;
            // Each crossing tints by the square root of the base colour
            // (`generalized_schlick_setup`'s `sqrt(clamped_base_color)`),
            // so in and out again comes to the base colour once. Tinting
            // by the whole base at each face made glass its colour squared
            // — deep purple where Blender draws lilac.
            thr *= tw * sqrt(base) * (vec3<f32>(1.0) - fres);
        }
        d = cone_dir(normalize(nd), s.rough * s.rough, &seed);
        if (max(thr.r, max(thr.g, thr.b)) < 0.01) { break; }
    }

    emit(i, col, grid_over(o0, d0, first_t, pix), 1.0, 0.0);
}
