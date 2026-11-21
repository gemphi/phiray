// The surface a light lands on and what it sends back: GGX, the Principled
// layers (sheen, clearcoat, subsurface tint) and the BRDF weights the
// integrator (`integrate.wgsl`) and every light query (`light.wgsl`) use.
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

struct Surf {
    pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, f0: vec3<f32>, rough: f32,
    // Subsurface: how far light wraps past the terminator (0 = none) and the
    // colour it takes on inside the medium (from the scattering radius).
    bleed: f32, sss_tint: vec3<f32>,
    // Fabric sheen and clearcoat.
    sheen: f32, sheen_rough: f32, sheen_tint: vec3<f32>,
    coat: f32, coat_rough: f32,
    // The dielectric layer's IOR and its share of the reflection (one minus
    // the metallic), for the true Fresnel curve (`fresnel_of`). An IOR of 0
    // leaves the surface on Schlick's curve, as the plain surfaces are.
    eta: f32, dielectric: f32,
}

// A surface with only the base lobes (secondary hits, Workbench-style shading).
fn plain_surf(pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, f0: vec3<f32>, rough: f32) -> Surf {
    return Surf(pos, n, v, albedo, f0, rough, 0.0, vec3<f32>(1.0), 0.0, 0.5, vec3<f32>(1.0), 0.0, 0.03, 0.0, 0.0);
}

// The reflectance of a surface at `cos`, the angle between the view and the
// microfacet. Schlick's fifth power from `f0`, and for the dielectric share
// the difference to the true curve of its IOR: Cycles' Principled gives its
// specular layer `exponent = -eta`, which `microfacet_fresnel`
// (`closure/bsdf_microfacet.h`) reads as "the real Fresnel, between F0 and
// F90". Schlick alone overshoots at glancing angles, the more the lower the
// IOR: at 80 degrees a floor of Specular IOR Level 0.115 (CW-POSES) reflects
// 0.28 and Schlick says 0.39, and the far floor washed out to the sky.
fn fresnel_of(s: Surf, cos: f32) -> vec3<f32> {
    let schlick = pow(1.0 - cos, 5.0);
    var f = s.f0 + (1.0 - s.f0) * schlick;
    if (s.eta > 1.0 && s.dielectric > 0.0) {
        let r = (s.eta - 1.0) / (s.eta + 1.0);
        let face_on = r * r;
        f += vec3<f32>(s.dielectric * (fresnel_dielectric_cos(cos, s.eta) - face_on - (1.0 - face_on) * schlick));
    }
    return max(f, vec3<f32>(0.0));
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

// The split-sum's second term — how much of a prefiltered reflection a
// surface of `f0` and `rough` shows at `nv` — in Karis's analytic fit.
fn env_brdf(f0: vec3<f32>, rough: f32, nv: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + vec3<f32>(ab.y);
}

// The Principled's dielectric reflection (`svm/closure.h`): F0 from the
// IOR the material carries. phiber hands over the IOR Specular IOR Level
// has already adjusted (`BlendBsdf::specular_eta`), so a level of 0.1 —
// skin, the pose files' floors — reflects a fifth of what 0.5 does. This
// was a constant 0.04 for every surface, which is IOR 1.5 at level 0.5.
fn dielectric_f0(ior: f32) -> vec3<f32> {
    let r = (ior - 1.0) / (ior + 1.0);
    return vec3<f32>(r * r);
}

// What reaches the diffuse beneath a specular layer of reflectance `f0`
// seen at `nv`: Cycles weights the lower closures by one minus the layer's
// albedo (`closure_layering_weight` over `bsdf_albedo`), so light the
// coat of gloss reflects is not also scattered by the colour under it.
// Without it every dielectric surface returned its reflection on top of
// its full diffuse — a few percent at normal incidence, far more at the
// grazing angles a floor is seen at.
fn under_layer(f0: vec3<f32>, rough: f32, nv: f32) -> f32 {
    let e = env_brdf(f0, rough, nv);
    return clamp(1.0 - max(e.r, max(e.g, e.b)), 0.0, 1.0);
}

// The same under a dielectric of IOR `eta` whose reflectance climbs by the
// true Fresnel curve (`fresnel_of`). `env_brdf` is fitted to Schlick's
// climb, so its climbing part is scaled by how the true curve compares at
// this angle: where Schlick overshoots, less is taken from the diffuse.
// Cycles reads this from a table (`bsdf_albedo`, the `ggx_gen_schlick_ior_s`
// lookup); this stands in for it.
fn under_layer_real(eta: f32, rough: f32, nv: f32) -> f32 {
    let r = (eta - 1.0) / (eta + 1.0);
    let face_on = r * r;
    let climb = env_brdf(vec3<f32>(0.0), rough, nv).r;
    let flat = env_brdf(vec3<f32>(1.0), rough, nv).r - climb;
    let schlick = (1.0 - face_on) * pow(1.0 - nv, 5.0);
    let real = fresnel_dielectric_cos(nv, eta) - face_on;
    let scale = clamp(real / max(schlick, 1e-5), 0.0, 1.0);
    return clamp(1.0 - (face_on * flat + climb * scale), 0.0, 1.0);
}

// Film ▸ Pixel Filter: where in its pixel a camera ray goes, in pixels from
// the centre, for two uniform numbers — Cycles' importance-sampled filter
// (`scene/film.cpp`'s `filter_table`, read by `camera.h`'s
// `filter_table_sample`). Each axis is drawn on its own, as Cycles reads
// its table once per axis. Blender's default is Blackman-Harris 1.5 px,
// which Cycles spreads over twice that width; a Gaussian over three times
// it. phiray's own (`p.integrator.x` 0) is a one-pixel box: sharper than any
// Blender render, which is why its edges never matched.
fn filter_offset2(u: f32, v: f32) -> vec2<f32> {
    let kind = u32(p.integrator.x + 0.5);
    let w = p.integrator.y;
    if (kind == 1u) { return (vec2<f32>(u, v) - 0.5) * w; }
    if (kind == 2u) {
        // `filter_func_gaussian` over 3w: exp(−2 (6x / 3w)²), a normal of
        // σ = w / 4 — Box–Muller, cut at the table's edge (±1.5 w).
        let r = sqrt(-2.0 * log(max(u, 1e-7))) * (w * 0.25);
        let a = 6.2831853 * v;
        return clamp(vec2<f32>(r * cos(a), r * sin(a)), vec2<f32>(-1.5 * w), vec2<f32>(1.5 * w));
    }
    if (kind == 3u) { return vec2<f32>(blackman_harris_at(u), blackman_harris_at(v)) * (2.0 * w); }
    return vec2<f32>(u, v) - 0.5;
}

// The inverse of the Blackman-Harris window's distribution, as a fraction of
// its width from the centre (−0.5..0.5): `filter_func_blackman_harris`
// integrates in closed form, and bisection on that is exact to the 20th bit
// for a cost a path does not notice.
fn blackman_harris_at(u: f32) -> f32 {
    let target_ = u * 6.2831853 * 0.35875;
    var lo = 0.0;
    var hi = 6.2831853;
    for (var k = 0u; k < 20u; k++) {
        let v = 0.5 * (lo + hi);
        let g = 0.35875 * v - 0.48829 * sin(v) + 0.07064 * sin(2.0 * v) - 0.0038933 * sin(3.0 * v);
        if (g < target_) { lo = v; } else { hi = v; }
    }
    return 0.5 * (lo + hi) / 6.2831853 - 0.5;
}

// Light Paths ▸ Caustics ▸ Filter Glossy: the least roughness a glossy lobe
// takes at a hit the path reached with `min_pdf` the smallest density it
// was sampled with so far — Cycles' blur (`surface_shader.h`: `blur_pdf =
// min_ray_pdf / filter_glossy`, and below 1 the lobe's alpha is raised to
// √(1 − blur_pdf) / 2). Filter Glossy 0 blurs nothing. phiray's own
// (`p.integrator.x` 0) is the 0.3 floor it always had.
fn glossy_floor(min_pdf: f32) -> f32 {
    if (p.integrator.x < 0.5) { return 0.3; }
    if (p.integrator.z <= 0.0) { return 0.0; }
    let blur_pdf = min_pdf / p.integrator.z;
    if (blur_pdf >= 1.0) { return 0.0; }
    return sqrt(sqrt(1.0 - blur_pdf) * 0.5);
}

// Light Paths ▸ Caustics ▸ Reflective and Refractive: whether a path that
// has bounced diffusely still meets glossy reflection, and glass, at its
// next hit (`svm/closure.h`'s `reflective_caustics`, `refractive_caustics`).
fn caustic_reflective() -> bool { return p.integrator.x < 0.5 || (u32(p.integrator.w + 0.5) & 1u) != 0u; }
fn caustic_refractive() -> bool { return p.integrator.x < 0.5 || (u32(p.integrator.w + 0.5) & 2u) != 0u; }

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
    let fres = fresnel_of(s, hvv);
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
