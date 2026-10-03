// Blender's Workbench (Solid mode) shading, on the GPU.
//
// Ports `BKE_studiolight_default` (blenkernel/intern/studiolight.cc:791) and
// `workbench_world_light_lib.glsl::get_world_lighting` verbatim — the same rig
// the CPU painter used, so Solid mode looks identical and costs one dispatch.
//
// Four directional lights fixed to the camera, each with its own colour and a
// `wrap` that softens its terminator; Blinn specular split by roughness and
// metallic. There is no sampling here: Solid is deterministic, so one pass is
// the whole frame.

struct Solid { vec: vec3<f32>, wrap: f32, col: vec3<f32>, _c: f32, spec: vec3<f32>, _s: f32 }

fn rig(i: u32) -> Solid {
    switch i {
        case 0u: { return Solid(vec3<f32>(-0.352546, 0.170931, -0.920051), 0.526620,
                                vec3<f32>(0.033103, 0.033103, 0.033103), 0.0,
                                vec3<f32>(0.266761, 0.266761, 0.266761), 0.0); }
        case 1u: { return Solid(vec3<f32>(-0.408163, 0.346939, 0.844415), 0.0,
                                vec3<f32>(0.521083, 0.538226, 0.538226), 0.0,
                                vec3<f32>(0.599030, 0.599030, 0.599030), 0.0); }
        case 2u: { return Solid(vec3<f32>(0.521739, 0.826087, 0.212999), 0.478261,
                                vec3<f32>(0.038403, 0.034357, 0.049530), 0.0,
                                vec3<f32>(0.106102, 0.125981, 0.158523), 0.0); }
        default: { return Solid(vec3<f32>(0.624519, -0.562067, -0.542269), 0.200000,
                                vec3<f32>(0.090838, 0.082080, 0.072255), 0.0,
                                vec3<f32>(0.106535, 0.084771, 0.066080), 0.0); }
    }
}

// `fast_rcp` ([Drobot2014a], `workbench_world_light_lib.glsl`): Blender's
// reciprocal is this bit-trick estimate, a few percent off `1 / v`, and its
// shading carries that error — so this one does too.
fn fast_rcp(v: f32) -> f32 {
    return bitcast<f32>(0x7eef370b - bitcast<i32>(v));
}

// `wrapped_lighting(NL, w)` — NL deliberately unclamped, which is what lets a
// light wrap past the terminator.
fn wrapped(nl: f32, w: f32) -> f32 {
    return clamp((nl + w) * fast_rcp((w + 1.0) * (w + 1.0)), 0.0, 1.0);
}

// `brdf_approx` — fresnel brightening at grazing angles, cut by roughness.
fn brdf_approx(spec: vec3<f32>, rough: f32, nv: f32) -> vec3<f32> {
    let fresnel = exp2(-8.35 * nv) * (1.0 - rough);
    return spec + (1.0 - spec) * fresnel;
}

// `blinn_specular` — normalized Blinn; pi is already in the light's power.
fn blinn(shininess: f32, spec_angle: f32, nl: f32) -> f32 {
    return pow(spec_angle, shininess) * nl * (shininess * 0.125 + 1.0);
}

// `get_world_lighting`. `n` and `i` are in VIEW space (+z toward the
// viewer), which is what fixes the rig to the camera; `i` is the incident
// vector toward the eye (`drw_view_incident_vector`: +z in an orthographic
// view, toward the eye from the point in a perspective one). `n` is used as
// given — the caller turns a back face's toward the viewer where Blender
// does (the opaque pass's `normal_encode`) and not where it does not (X-ray).
fn workbench(base: vec3<f32>, rough: f32, metal: f32, n_in: vec3<f32>, i: vec3<f32>) -> vec3<f32> {
    let n = normalize(n_in);

    // Metals have no diffuse and tint their highlight; dielectrics keep their
    // colour and reflect a dim neutral one.
    let diffuse_color = base * (1.0 - metal);
    var specular_color = vec3<f32>(0.05) + (base - vec3<f32>(0.05)) * metal;

    let r = 2.0 * dot(n, i) * n - i;
    var diffuse_light = vec3<f32>(0.0);
    var specular_light = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let l = rig(k);
        let w = l.wrap;
        diffuse_light += wrapped(dot(n, l.vec), w) * l.col;

        let half = normalize(l.vec + i);
        let spec_angle = clamp(dot(half, n), 0.0, 1.0);
        let spec_nl = clamp(dot(n, l.vec), 0.0, 1.0);
        // Gloss is reduced by the wrap, which simulates a bigger light.
        let gloss = (1.0 - rough) * (1.0 - w);
        let shininess = exp2(10.0 * gloss + 1.0);
        let spec = blinn(shininess, spec_angle, spec_nl);
        let spec_env = wrapped(dot(r, l.vec), w + (1.0 - w) * rough);
        specular_light += (spec + (spec_env - spec) * (w * w)) * l.spec;
    }

    specular_color = brdf_approx(specular_color, rough, clamp(dot(n, i), 0.0, 1.0));
    specular_light *= specular_color;
    // "Energy conservation with coloured specular looks strange. Limit this
    // strangeness by using mono-chromatic specular intensity."
    let spec_energy = dot(specular_color, vec3<f32>(0.33333));
    // Scene-linear. The caller encodes it for display with `srgb()` — the
    // Standard view transform Blender renders Workbench through. A 1.7 gain
    // used to stand in for that and left every midtone dark and hard.
    return max(diffuse_light * diffuse_color * (1.0 - spec_energy) + specular_light, vec3<f32>(0.0));
}
