// Solid mode's whole frame: one primary hit per pixel — rasterized by the
// visibility prepass (`raster.wgsl`, as Workbench finds its surfaces) or,
// where there is none, traced — shaded by the studio rig, shadowed by the
// one directional light when Shadow is on, jittered for the anti-aliasing
// gather (`taa.wgsl`). Workbench is deterministic: the samples differ by
// their jitter alone. The object outline and the cavity and curvature
// effects follow in `fx.wgsl`, over what `fx_store` leaves.

// The most surfaces X-ray sees through along one ray.
const XRAY_LAYERS: u32 = 32u;

// Workbench's colour for the surface `h` meets along `d`, scene-linear. A
// back face turns toward the viewer when `facing` (the opaque pass,
// `normal_encode`) and keeps its own normal when not (X-ray's transparent
// pass).
// What the opaque prepass leaves per pixel: the colour, and the normal in
// view space (`normal_encode`'s) for the curvature effect.
struct Shaded { col: vec3<f32>, nv: vec3<f32> }

fn solid_at(h: Hit, d: vec3<f32>, facing: bool) -> Shaded {
    let t = tri_at(h.tri);
    let w = 1.0 - h.u - h.v;
    let bary = vec3<f32>(w, h.u, h.v);
    let ng0 = normalize(cross(t.b - t.a, t.c - t.a));
    let turn = select(1.0, -1.0, facing && dot(ng0, d) >= 0.0);
    var n = t.na * w + t.nb * h.u + t.nc * h.v;
    n = select(ng0, normalize(n), dot(n, n) > 1e-12) * turn;
    if (t.nmap != NO_TEX) { n = mapped_normal(t, n, bary); }

    var base = max(t.color.rgb, vec3<f32>(0.0));
    var rough = t.rough;
    var metal = t.metal;
    let uv = vec2<f32>(dot(t.us, bary), dot(t.vs, bary));
    // Color ▸ Texture (`workbench_image_lib.glsl`): the image is the colour;
    // Solid's build uploads one in that mode alone.
    if (t.tex != NO_TEX) { base = sample_tex(t.tex, uv).rgb; }
    if (t.maps.x != NO_TEX) { rough *= sample_raw(t.maps.x, uv).r; }
    if (t.maps.y != NO_TEX) { metal *= sample_raw(t.maps.y, uv).r; }
    // Color ▸ Object / Random / Single / Attribute (`workbench_engine.cc`'s
    // `get_material`): the object's colour, a hue hashed from its name, the
    // one colour, or the vertex colour — each with `pack_data(0.0, 0.4, …)`:
    // no metal, roughness 0.4. Attribute over a mesh without one is
    // Material, as `SceneState::has_color` falls back.
    let painted = p.shade.y == 5u && p.insts.z != 0u;
    if (p.shade.y == 4u || p.shade.y == 1u || p.shade.y == 2u || painted) {
        let iw = inst_at(face(h.tri).w >> 16u);
        if (p.shade.y == 4u) { base = vec3<f32>(tf(iw + 28u), tf(iw + 29u), tf(iw + 30u)); }
        else if (p.shade.y == 1u) { base = random_color(data[iw + 31u]); }
        else if (p.shade.y == 2u) { base = p.single.rgb; }
        else {
            // The corners' sRGB bytes, linear, over the barycentrics.
            let v = face(h.tri);
            base = vertex_color(v.x) * w + vertex_color(v.y) * h.u + vertex_color(v.z) * h.v;
        }
        rough = 0.4;
        metal = 0.0;
    }

    // The rig is fixed to the camera, so the normal goes into view space:
    // +x right, +y up, +z toward the viewer.
    let nv = vec3<f32>(dot(n, p.right), dot(n, p.up), dot(n, p.back));
    let iv = vec3<f32>(dot(-d, p.right), dot(-d, p.up), dot(-d, p.back));
    // `Material::pack_data`: Workbench shades with the square root of the
    // roughness ("Disney" roughness), both values kept to a byte.
    let wb_rough = round(sqrt(clamp(rough, 0.0, 1.0)) * 255.0) / 255.0;
    let wb_metal = round(clamp(metal, 0.0, 1.0) * 255.0) / 255.0;
    // Lighting ▸ Flat (`WORKBENCH_LIGHTING_FLAT`): the colour itself.
    if (p.shade.x == 2u) { return Shaded(base, nv); }
    // Lighting ▸ MatCap (`get_matcap_lighting`): the matcap's diffuse image
    // by the view normal times the colour, its specular image added when
    // Specular Lighting is on.
    if (p.shade.x == 1u && p.matcap.x != 0u) {
        let uv = matcap_uv(iv, nv, p.matcap.z != 0u);
        var lit = sample_env(p.matcap.x - 1u, uv) * base;
        if (p.matcap.y != 0u && p.shade.w == 0u) { lit += sample_env(p.matcap.y - 1u, uv); }
        return Shaded(lit, nv);
    }
    return Shaded(workbench(base, wb_rough, wb_metal, nv, iv) + emit_at(t, bary), nv);
}

// `matcap_uv_compute` (`workbench_matcap_lib.glsl`): the view normal in a
// basis made from the incident vector, to the image's `0.004 ..= 0.996`;
// the image's rows run top-down here where GL's run up, so y is turned.
fn matcap_uv(i: vec3<f32>, n: vec3<f32>, flipped: bool) -> vec2<f32> {
    let a = 1.0 / (1.0 + i.z);
    let b = -i.x * i.y * a;
    let b1 = vec3<f32>(1.0 - i.x * i.x * a, b, -i.x);
    let b2 = vec3<f32>(b, 1.0 - i.y * i.y * a, -i.y);
    var uv = vec2<f32>(dot(b1, n), dot(b2, n));
    if (flipped) { uv.x = -uv.x; }
    uv = uv * 0.496 + 0.5;
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

// Workbench's Shadow (`get_shadow`, `workbench_world_light_lib.glsl`) for
// the opaque hit `h` along `d` with view normal `nv`: the factor the colour
// takes. The terminator is a `smoothstep` from `shadow_shift` to `shadow_
// focus` over how squarely the surface faces the light; where the stencil
// shadow volumes would cover the point — here, a ray against the light's
// travel meets a surface — it is fully in shadow; `shadow_intensity` is
// how dark either gets (`shadow_mul` / `shadow_add`).
fn shadow_factor(o: vec3<f32>, d: vec3<f32>, h: Hit, nv: vec3<f32>) -> f32 {
    let dir = p.shadow.xyz;
    // `shadow_direction_vs`: the travel direction in view space.
    let dir_vs = vec3<f32>(dot(dir, p.right), dot(dir, p.up), dot(dir, p.back));
    let light_factor = -dot(nv, dir_vs);
    var mix = smoothstep(p.shadow_fade.x, p.shadow_fade.y, light_factor);
    if (mix > 0.0) {
        // Toward the light from just off the surface, past the face itself.
        let org = o + d * h.t - dir * (p.eps + h.t * 1e-5);
        if (trace(org, -dir, INF, true).tri != 0xffffffffu) { mix = 0.0; }
    }
    return mix * p.shadow.w + (1.0 - p.shadow.w);
}

// Vertex `v`'s colour attribute (`Scene::paint`'s sRGB bytes), scene-linear.
fn vertex_color(v: u32) -> vec3<f32> {
    return srgb_to_linear(unpack4x8unorm(data[p.insts.z + v]).rgb);
}

// `BLI_hash_int_01(hash)` as a hue at saturation 0.5 and value 0.8
// (`hsv_to_rgb_v`): Color ▸ Random's colour for an object whose id name
// hashed to `hash` (`hash.rs`). Blender's scene-linear is Rec.709, so
// `rec709_to_scene_linear` changes nothing.
fn random_color(hash: u32) -> vec3<f32> {
    let seed = 0xdeadbeefu + 8u + 13u;
    var a = seed + hash;
    var b = seed;
    var c = seed;
    c ^= b; c -= (b << 14u) | (b >> 18u);
    a ^= c; a -= (c << 11u) | (c >> 21u);
    b ^= a; b -= (a << 25u) | (a >> 7u);
    c ^= b; c -= (b << 16u) | (b >> 16u);
    a ^= c; a -= (c << 4u) | (c >> 28u);
    b ^= a; b -= (a << 14u) | (a >> 18u);
    c ^= b; c -= (b << 24u) | (b >> 8u);
    let h = f32(c) * (1.0 / 4294967295.0);
    let s = 0.5;
    let v = 0.8;
    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let pp = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    switch (u32(i) % 6u) {
        case 0u: { return vec3<f32>(v, t, pp); }
        case 1u: { return vec3<f32>(q, v, pp); }
        case 2u: { return vec3<f32>(pp, v, t); }
        case 3u: { return vec3<f32>(pp, q, v); }
        case 4u: { return vec3<f32>(t, pp, v); }
        default: { return vec3<f32>(v, pp, q); }
    }
}

// The opaque pass's buffers for the effects kernel (`fx.wgsl`): the pixel's
// scene-linear colour (words 7n..10n), the grid over it (10n..13n and
// 13n), its object id — the instance plus one, 0 for the background, as
// Workbench's `object_id` buffer counts (17n) — its view normal (18n,
// 19n) and its view depth (20n: Blender's view z, negative ahead, in
// Blender units; 0 for the background — Workbench's depth buffer for the
// SSAO), in the scratch the denoiser never touches in Solid.
fn fx_store(i: u32, lin: vec3<f32>, g: vec4<f32>, id: u32, nv: vec3<f32>, z: f32) {
    let a = 7u * p.n + 3u * i;
    px[a] = bitcast<u32>(lin.x);
    px[a + 1u] = bitcast<u32>(lin.y);
    px[a + 2u] = bitcast<u32>(lin.z);
    let b = 10u * p.n + 3u * i;
    px[b] = bitcast<u32>(g.x);
    px[b + 1u] = bitcast<u32>(g.y);
    px[b + 2u] = bitcast<u32>(g.z);
    px[13u * p.n + i] = bitcast<u32>(g.w);
    px[17u * p.n + i] = id;
    px[18u * p.n + i] = bitcast<u32>(nv.x);
    px[19u * p.n + i] = bitcast<u32>(nv.y);
    px[20u * p.n + i] = bitcast<u32>(z);
}

// `calculate_transparent_weight` (`workbench_transparent_accum_frag.glsl`,
// McGuire & Bavoil's equation 7): nearer surfaces weigh more. `t` is along
// the ray. Blender's orthographic viewport puts its near plane half the clip
// end (500) behind the view, which weighs every surface at the clamp's floor.
fn xray_weight(t: f32, d: vec3<f32>) -> f32 {
    if (p.persp == 0u) { return 1e-2; }
    let z = t * p.units * dot(d, -p.back);
    let a = z / 5.0;
    var b = z / 200.0;
    b *= b;
    return clamp(10.0 / ((1e-5 + a * a) + b * (b * b)), 1e-2, 3e2);
}

// What X-ray sees along one ray: the colour, scene-linear, and the nearest
// surface (`INF` when none).
struct Xray { col: vec3<f32>, near: f32 }

// X-ray (`V3D_SHADING_XRAY`): every surface along the ray at the X-ray
// alpha, front and back faces alike, blended as Workbench blends them —
// weighted blended order-independent transparency, accumulated
// (`workbench_transparent_accum_frag.glsl`), then resolved over the
// background (`workbench_transparent_resolve_frag.glsl`).
fn xray(o: vec3<f32>, d: vec3<f32>, background: vec3<f32>) -> Xray {
    let alpha = 1.0 - p.see_through;
    var accum = vec3<f32>(0.0);
    var weight = 0.0;
    var reveal = 1.0;
    var near = INF;
    var ahead = 0.0;
    var last = 0xffffffffu;
    for (var k = 0u; k < XRAY_LAYERS; k++) {
        let h = trace(o + d * ahead, d, INF, false);
        if (h.tri == 0xffffffffu) { break; }
        let t = ahead + h.t;
        ahead = t + max(t, 1.0) * 1e-5;
        // The same face met again at the step's edge is not a new layer.
        if (h.tri == last) { continue; }
        last = h.tri;
        near = min(near, t);
        let a = alpha * clamp(tri_at(h.tri).color.a, 0.0, 1.0);
        let w = xray_weight(t, d) * a;
        accum += solid_at(Hit(t, h.tri, h.u, h.v), d, false).col * w;
        weight += w;
        reveal *= 1.0 - a;
    }
    let col = accum / clamp(weight, 1e-4, 5e4);
    return Xray(col * (1.0 - reveal) + background * reveal, near);
}

// The floor grid over X-ray: drawn through the surfaces, and faded where it
// is behind the nearest (`overlay_xray_fade`: overlays behind scene geometry
// take `1 − xray_alpha`).
fn xray_grid(o: vec3<f32>, d: vec3<f32>, near: f32, pix: f32, base: vec3<f32>) -> vec3<f32> {
    var g = grid_over(o, d, INF, pix);
    if (abs(d.z) > 1e-6 && -o.z / d.z > near) { g *= p.see_through; }
    return base * (1.0 - g.a) + g.rgb;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = pixel(gid);
    if (i >= p.n) { return; }
    let sp = frame_point(i) + taa_jitter();
    let su = sp.x;
    let sv = sp.y;
    let vx = (su - p.pan.x) / p.scale;
    let vy = -(sv / p.aspect - p.pan.y) / p.scale;
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
    // The theme's sky, with its sweep when it has one (`sky_at`).
    let sky = sky_at(i);

    if (p.see_through > 0.0) {
        let seen = xray(o, d, srgb_to_linear(sky));
        taa_sample(i, xray_grid(o, d, seen.near, pix, srgb(seen.col)));
        return;
    }
    var h = primary(o, d, i);
    // Backface Culling (`DRW_STATE_CULL_BACK`): a face turned away is not
    // drawn, so the ray goes on past it to what is behind.
    for (var k = 0u; k < XRAY_LAYERS && p.shade.z != 0u && h.tri != 0xffffffffu; k++) {
        let t = tri_at(h.tri);
        if (dot(cross(t.b - t.a, t.c - t.a), d) < 0.0) { break; }
        let ahead = h.t + max(h.t, 1.0) * 1e-5;
        let next = trace(o + d * ahead, d, INF, false);
        h = Hit(ahead + next.t, next.tri, next.u, next.v);
    }
    if (h.tri == 0xffffffffu) {
        let g = grid_over(o, d, INF, pix);
        fx_store(i, srgb_to_linear(sky), g, 0u, vec3<f32>(0.0), 0.0);
        taa_sample(i, sky * (1.0 - g.a) + g.rgb);
        return;
    }
    var s = solid_at(h, d, true);
    // Shadow (`V3D_SHADING_SHADOW`), after the lighting as the composite has
    // it; X-ray's transparent pass casts and takes none.
    if (p.shadow.w > 0.0) { s.col *= shadow_factor(o, d, h, s.nv); }
    let g = grid_over(o, d, h.t, pix);
    // The hit's view depth: along the view axis from the eye (perspective)
    // or the far origin (orthographic), in Blender units.
    fx_store(i, s.col, g, (face(h.tri).w >> 16u) + 1u, s.nv, -h.t * p.units * dot(d, -p.back));
    // Standard view transform: scene-linear → sRGB display.
    taa_sample(i, srgb(s.col) * (1.0 - g.a) + g.rgb);
}
