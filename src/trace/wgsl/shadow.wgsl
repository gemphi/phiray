// Material Preview's soft shadows, EEVEE's model ported
// (`eevee_shadow_tracing.bsl.hh`'s `shadow_ray_generate_*`,
// `eevee_sampling_lib.bsl.hh`, `eevee_light_lib.bsl.hh`): a shadow ray is
// aimed at a point on the light's shape, not at its centre, so a light with
// a Radius (a Sun with an Angle) casts a penumbra.
//
// EEVEE takes one such ray per frame and lets its sixteen viewport samples
// average them. This kernel is one dispatch and the same frame every time
// (§22), so it cannot accumulate: it takes `SHADOW_TAPS` rays at once, over
// the Hammersley set EEVEE itself samples its lookup tables with
// (`hammersley_2d`), and averages their visibility. A light whose shape is
// a point — Blender's Shadow Soft Size at 0, which EEVEE also draws hard —
// takes the one ray at its centre.

// Rays per lamp over its shape. EEVEE's own budget is one ray and six
// steps through the shadow map (`shadow_ray_count`, `shadow_step_count`);
// eight is what a penumbra needs to read as a gradient rather than as
// eight hard edges, at eight times the shadow cost of batch 108's kernel.
const SHADOW_TAPS: u32 = 8u;
// Under this, a light's shape is a point and its shadow is hard.
const SHADOW_HARD: f32 = 1e-5;

// `van_der_corput_radical_inverse`.
fn radical_inverse(n: u32) -> f32 {
    return f32(reverseBits(n)) * 2.3283064365386963e-10;
}

// `hammersley_2d`.
fn hammersley_2d(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse(i));
}

// `sample_circle`: a point on the unit circle.
fn sample_circle(rand: f32) -> vec2<f32> {
    let phi = (rand - 0.5) * 2.0 * PI;
    let cos_phi = cos(phi);
    return vec2<f32>(cos_phi, sqrt(max(1.0 - cos_phi * cos_phi, 0.0)) * sign(phi));
}

// `sample_disk`: a point on the unit disk.
fn sample_disk(rand: vec2<f32>) -> vec2<f32> {
    return sample_circle(rand.y) * sqrt(rand.x);
}

// `sample_uniform_cone` about +Z, `cos_angle` the cone's half-angle.
fn sample_uniform_cone(rand: vec2<f32>, cos_angle: f32) -> vec3<f32> {
    let cos_theta = mix(cos_angle, 1.0, rand.x);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let circle = sample_circle(rand.y);
    return vec3<f32>(sin_theta * circle, cos_theta);
}

// `light_sphere_disk_radius`: the disk whose half-angle matches the
// sphere's, seen from `dist` away.
fn light_sphere_disk_radius(radius: f32, dist: f32) -> f32 {
    let ratio = radius / max(dist, 1e-6);
    return radius * inverseSqrt(max(1e-8, 1.0 - ratio * ratio));
}

// How much of lamp `li` reaches `pos`, in 0 ..= 1: the mean of
// `SHADOW_TAPS` rays aimed over its shape. `l` is the direction to its
// centre and `dist` how far that is (`INF` for a sun).
fn lamp_visible(pos: vec3<f32>, li: Light, l: vec3<f32>, dist: f32) -> f32 {
    // A Sun's shape is an angle: the ray goes inside that cone
    // (`shadow_ray_generate_directional`).
    if (li.kind == 0u) {
        let half_angle = li.size * 0.5;
        if (half_angle < SHADOW_HARD) { return visible(pos, l, INF); }
        let cos_angle = cos(min(half_angle, PI * 0.5));
        var sum = 0.0;
        for (var k = 0u; k < SHADOW_TAPS; k++) {
            let cone = sample_uniform_cone(hammersley_2d(k, SHADOW_TAPS), cos_angle);
            sum += visible(pos, normalize(l * cone.z + onb(l, cone.x, cone.y)), INF);
        }
        return sum / f32(SHADOW_TAPS);
    }
    if (li.size < SHADOW_HARD) { return visible(pos, l, dist); }
    // An Area light's taps spread over its own shape in its own plane
    // (`shadow_ray_generate_punctual` for `LIGHT_RECT`): a rectangle takes
    // the taps straight, a disk or an ellipse takes them through
    // `sample_disk`, and its axes are the object's — `right` across
    // `size`, the other way across `size_y`, which Blender leaves equal to
    // `size` for a Square and a Disk.
    if (li.kind == 3u) {
        let half = vec2<f32>(li.size, li.size_y) * 0.5;
        let round = li.shape == AREA_DISK || li.shape == AREA_ELLIPSE;
        let up = cross(li.dir, li.right);
        var sum = 0.0;
        for (var k = 0u; k < SHADOW_TAPS; k++) {
            let rand = hammersley_2d(k, SHADOW_TAPS);
            let on_shape = select(rand * 2.0 - 1.0, sample_disk(rand), round) * half;
            let to = li.pos + li.right * on_shape.x + up * on_shape.y - pos;
            sum += visible(pos, normalize(to), length(to) * 0.999);
        }
        return sum / f32(SHADOW_TAPS);
    }
    // A Point or Spot aims at a disk of its Radius turned to face the
    // surface (`shadow_ray_generate_punctual`): its shape is a sphere, so
    // the disk takes the radius that spans the same angle.
    let radius = light_sphere_disk_radius(li.size, dist);
    var sum = 0.0;
    for (var k = 0u; k < SHADOW_TAPS; k++) {
        let disk = sample_disk(hammersley_2d(k, SHADOW_TAPS)) * radius;
        let to = li.pos + onb(l, disk.x, disk.y) - pos;
        sum += visible(pos, normalize(to), length(to) * 0.999);
    }
    return sum / f32(SHADOW_TAPS);
}

// The three-point rig's key and rim, which have no light record of their
// own: a cone of `half_angle` about `l`, the same taps.
fn rig_visible(pos: vec3<f32>, l: vec3<f32>, half_angle: f32) -> f32 {
    let cos_angle = cos(half_angle);
    var sum = 0.0;
    for (var k = 0u; k < SHADOW_TAPS; k++) {
        let cone = sample_uniform_cone(hammersley_2d(k, SHADOW_TAPS), cos_angle);
        sum += visible(pos, normalize(l * cone.z + onb(l, cone.x, cone.y)), INF);
    }
    return sum / f32(SHADOW_TAPS);
}
