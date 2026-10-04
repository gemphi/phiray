// Material Preview's whole frame: one primary ray per pixel, every surface
// it meets shaded in full from what was precomputed — the environment's
// irradiance and its prefiltered copies (`ibl.rs`), the lamps with one
// shadow ray each, textures at the mip level the pixel's footprint asks for.
// No random number anywhere: one dispatch is the frame, the same every
// time, the way EEVEE draws Blender's Material Preview — never the sampled,
// refining image the path tracer makes of the same scene.

// Surfaces one ray looks through: alpha blending and glass.
const PREVIEW_LAYERS: u32 = 6u;

// The world footprint of one pixel at distance `t` along a camera ray.
fn footprint(t: f32) -> f32 {
    let per = 1.0 / (p.scale * frame_size().x);
    if (p.persp != 0u) { return t * per / max(p.dist, 1e-4); }
    return per / p.units;
}

// The mip level texture `tex` should be read at for a footprint `foot` on
// triangle `t`, seen at `cos` to its normal: how many texels the footprint
// spans, as a power of two, widened at a grazing angle.
fn mip_level(t: Tri, tex: u32, foot: f32, cos: f32) -> f32 {
    if (tex >= atlas(0u)) { return 0.0; }
    let uv_area = abs((t.us.y - t.us.x) * (t.vs.z - t.vs.x) - (t.us.z - t.us.x) * (t.vs.y - t.vs.x));
    let world_area = length(cross(t.b - t.a, t.c - t.a));
    if (uv_area <= 0.0 || world_area <= 0.0) { return 0.0; }
    let density = sqrt(uv_area / world_area) * f32(head_w(tex));
    return log2(max(foot * density / sqrt(max(cos, 0.1)), 1e-6));
}

// The environment's direction for `d`, the world Mapping node's spin in.
fn env_turned(d: vec3<f32>) -> vec3<f32> {
    let c = cos(p.env_rot);
    let s = sin(p.env_rot);
    return vec3<f32>(d.x * c - d.y * s, d.x * s + d.y * c, d.z);
}

// The diffuse light a surface facing `n` receives from the environment,
// over π: the nine irradiance coefficients evaluated at `n`.
fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    if (p.preview.x == 0u) { return env_at(n) * 0.5; }
    let base = head_base(p.preview.x - 1u);
    let d = env_turned(n);
    var y = array<f32, 9>(
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    );
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i++) {
        let c = vec3<f32>(bitcast<f32>(atlas(base + 3u * i)), bitcast<f32>(atlas(base + 3u * i + 1u)), bitcast<f32>(atlas(base + 3u * i + 2u)));
        e += c * y[i];
    }
    return max(e, vec3<f32>(0.0)) * p.env.rgb;
}

// The environment reflected along `d` by a surface of `rough`: the image
// itself for a mirror, the prefiltered copy of that roughness otherwise,
// blended between the two steps it falls between.
fn prefiltered(d: vec3<f32>, rough: f32) -> vec3<f32> {
    if (p.preview.y == 0u || p.preview.z < 2u) { return env_at(d); }
    let f = clamp(rough, 0.0, 1.0) * f32(p.preview.z - 1u);
    let k = u32(floor(f));
    let t = f - floor(f);
    let uv = env_uv(d);
    var a = env_at(d);
    if (k > 0u) { a = sample_env(p.preview.y - 1u + (k - 1u), uv) * p.env.rgb; }
    if (k + 1u >= p.preview.z) { return a; }
    let b = sample_env(p.preview.y - 1u + k, uv) * p.env.rgb;
    return mix(a, b, t);
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

// Direct light from the lamps, one shadow ray each — every lamp, no pick
// among them (`lights` draws four at random; nothing here is drawn).
fn preview_lamps(s: Surf) -> vec3<f32> {
    var c = vec3<f32>(0.0);
    if (p.count == 0u) {
        // The environment is the light when there is one; the rig stands in
        // for a scene with no light of any kind.
        if (p.envmap != 0u) { return c; }
        c += brdf(s, p.sun, vec3<f32>(2.0)) * visible(s.pos, p.sun, INF);
        let fill = normalize(-p.right * 0.8 + p.up * 0.2 + p.back * 0.6);
        c += brdf(s, fill, vec3<f32>(0.45));
        let rim = normalize(p.up * 0.6 - p.back * 0.8 + p.right * 0.2);
        c += brdf(s, rim, vec3<f32>(0.9)) * visible(s.pos, rim, INF);
        return c;
    }
    for (var k = 0u; k < min(p.count, 16u); k++) {
        let li = p.lights[k];
        if (li.kind == 0u) {
            let l = normalize(-li.dir);
            c += brdf(s, l, li.color) * visible(s.pos, l, INF);
            continue;
        }
        let to = li.pos - s.pos;
        let d2 = max(dot(to, to), 1e-6);
        let l = to / sqrt(d2);
        var e = li.color / (4.0 * PI * PI * d2);
        if (li.kind == 2u) {
            e *= smoothstep(li.cone, mix(li.cone, 1.0, 0.15), dot(-l, li.dir));
        } else if (li.kind == 3u) {
            e *= max(dot(-l, li.dir), 0.0) * 2.0;
        }
        if (max(e.r, max(e.g, e.b)) < 1e-5) { continue; }
        c += brdf(s, l, e) * visible(s.pos, l, sqrt(d2) * 0.999);
    }
    return c;
}

// What one surface adds: `rgb` its radiance, `a` how much of the ray it
// takes (its alpha), and what continues past it.
struct Layer { rgb: vec3<f32>, alpha: f32, thr: vec3<f32>, o: vec3<f32>, d: vec3<f32>, through: bool }

// Shade the surface `h` met along `d` from `o`, `t` into the ray: its
// material's one lobe, or a Mix Shader's two lerped by the weight — what
// EEVEE's evaluation of the mix comes to, deterministic as this kernel is.
// The continued ray is the heavier lobe's.
fn preview_surface(h: Hit, o: vec3<f32>, d: vec3<f32>, foot: f32, i: u32, first: bool) -> Layer {
    let m = mat_at(h.tri);
    let other = mat_second(m);
    let a = preview_lobe(h, m, o, d, foot, i, first);
    if (other == NO_MAT) { return a; }
    let w = mat_mix(m);
    let b = preview_lobe(h, mat_word(other), o, d, foot, i, false);
    let heavy = w > 0.5;
    return Layer(mix(a.rgb, b.rgb, w), mix(a.alpha, b.alpha, w), mix(a.thr, b.thr, w), select(a.o, b.o, heavy), select(a.d, b.d, heavy), a.through || b.through);
}

// One lobe: the surface `h` with the material whose first word is `m`.
fn preview_lobe(h: Hit, m: u32, o: vec3<f32>, d: vec3<f32>, foot: f32, i: u32, first: bool) -> Layer {
    let t = tri_with(h.tri, m);
    let w = 1.0 - h.u - h.v;
    let bary = vec3<f32>(w, h.u, h.v);
    let ng0 = normalize(cross(t.b - t.a, t.c - t.a));
    let front = dot(ng0, d) < 0.0;
    let ng = select(-ng0, ng0, front);
    var n = t.na * w + t.nb * h.u + t.nc * h.v;
    n = select(ng, normalize(n) * select(-1.0, 1.0, front), dot(n, n) > 1e-12);
    let cos = max(dot(ng, -d), 0.0);
    if (t.nmap != NO_TEX) { n = mapped_normal_lod(t, n, bary, mip_level(t, t.nmap, foot, cos)); }
    if (dot(n, -d) < 0.0) { n = ng; }
    let hit = o + d * h.t;
    let off = ng * (p.eps + h.t * 1e-5);
    let pos = hit + off;
    let uv = vec2<f32>(dot(t.us, bary), dot(t.vs, bary));
    var base = max(t.color.rgb, vec3<f32>(0.0));
    var alpha = t.color.a;
    if (t.tex != NO_TEX) {
        let tc = sample_tex_lod(t.tex, uv, mip_level(t, t.tex, foot, cos));
        base *= tc.rgb;
        alpha *= tc.a;
    }
    var rough = t.rough;
    var metal = t.metal;
    if (t.maps.x != NO_TEX) { rough *= sample_raw_lod(t.maps.x, uv, mip_level(t, t.maps.x, foot, cos)).r; }
    if (t.maps.y != NO_TEX) { metal *= sample_raw_lod(t.maps.y, uv, mip_level(t, t.maps.y, foot, cos)).r; }
    rough = clamp(rough, 0.04, 1.0);
    if (first) { guide(i, base, n, h.t); }

    let v = -d;
    let nv = max(dot(n, v), 1e-4);
    let tw = t.trans * (1.0 - metal);
    var f0 = mix(vec3<f32>(0.04), base, metal);
    if (tw > 0.0) { f0 = vec3<f32>(pow((t.ior - 1.0) / (t.ior + 1.0), 2.0)); }
    // Fresnel at the surface, dulled by roughness (Lagarde's form).
    let fres = f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - nv, 5.0);
    let albedo = base * (1.0 - metal) * (1.0 - tw);
    let s = plain_surf(pos, n, v, albedo, f0, rough);
    // The environment: diffuse from the irradiance, specular from the
    // prefiltered copy; then the lamps, shadowed.
    let r = reflect(d, n);
    var c = albedo * sh_irradiance(n) * (1.0 - fres)
        + prefiltered(r, rough) * env_brdf(f0, rough, nv)
        + preview_lamps(s)
        + emit_at(t, bary);
    // Glass: the ray continues, bent, carrying what the surface let through.
    if (tw > 0.0) {
        let eta = select(t.ior, 1.0 / t.ior, front);
        var nd = refract(d, n, eta);
        var next_o = hit - off;
        if (dot(nd, nd) < 1e-6) { nd = reflect(d, n); next_o = pos; }
        return Layer(c, alpha, alpha * tw * base * (vec3<f32>(1.0) - fres), next_o, normalize(nd), true);
    }
    // An alpha-blended surface: what is behind it shows through `1 − alpha`.
    return Layer(c, alpha, vec3<f32>(1.0 - alpha), hit - off, d, alpha < 0.999);
}

// What a ray that meets nothing past the first surface sees: the world's
// image where there is one (glass shows the studio light, as EEVEE's world
// probe does), else the sky.
fn preview_background(d: vec3<f32>, sky: vec3<f32>) -> vec3<f32> {
    if (p.envmap != 0u) { return env_at(d); }
    return srgb_to_linear(sky);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = pixel(gid);
    if (i >= p.n) { return; }
    let sp = frame_point(i);
    let vx = (sp.x - p.pan.x) / p.scale;
    let vy = -(sp.y / p.aspect - p.pan.y) / p.scale;
    var o: vec3<f32>;
    var d: vec3<f32>;
    if (p.persp != 0u) {
        o = p.pivot + p.back * (p.dist / p.units);
        d = normalize(p.right * vx + p.up * vy - p.back * p.dist);
    } else {
        o = p.pivot + (p.right * vx + p.up * vy + p.back * 1.0e4) / p.units;
        d = -p.back;
    }
    let pix = 1.0 / (p.scale * p.units * frame_size().x);
    let sky = sky_at(i);
    let o0 = o;
    let d0 = d;
    var first_t = INF;
    var col = vec3<f32>(0.0);
    var thr = vec3<f32>(1.0);
    var cover = 1.0;
    for (var layer = 0u; layer < PREVIEW_LAYERS; layer++) {
        let h = trace(o, d, INF, false);
        if (h.tri == 0xffffffffu) {
            if (layer == 0u) {
                if (p.film.x != 0u) {
                    guide(i, vec3<f32>(0.0), -d, -1.0);
                    emit(i, vec3<f32>(0.0), vec4<f32>(0.0), 0.0);
                    return;
                }
                if (p.envmap != 0u && p.env_hidden == 0u) {
                    let bg = env_at(d);
                    guide(i, bg, -d, -1.0);
                    emit(i, bg, vec4<f32>(0.0), 1.0);
                    return;
                }
                let g = grid_over(o, d, INF, pix);
                guide(i, srgb_to_linear(sky), -d, -1.0);
                emit(i, vec3<f32>(0.0), vec4<f32>(sky * (1.0 - g.a) + g.rgb, 1.0), 1.0);
                return;
            }
            // Seen past a surface: under a transparent film the share let
            // through is not covered; else the background shows there.
            if (p.film.x != 0u) {
                cover -= max(thr.r, max(thr.g, thr.b));
            } else {
                col += thr * preview_background(d, sky);
            }
            break;
        }
        if (layer == 0u) { first_t = h.t; }
        let lay = preview_surface(h, o, d, footprint(first_t), i, layer == 0u);
        col += thr * lay.alpha * lay.rgb;
        thr *= lay.thr;
        if (!lay.through || max(thr.r, max(thr.g, thr.b)) < 0.01) { break; }
        o = lay.o;
        d = lay.d;
    }
    emit(i, col, grid_over(o0, d0, first_t, pix), clamp(cover, 0.0, 1.0));
}
