// Gabor noise, Cycles' `svm/gabor.h` ported whole: sparse Gabor convolution
// (Lagae 2009) with Tavernier's stratified sampling and normalization
// (2019), evaluated as Tricard's complex phasor (2019) so the node's three
// outputs — Value, Phase and Intensity — all fall out of one sum.
//
// Eight impulses per cell over a 3 × 3 (2D) or 3 × 3 × 3 (3D) neighbourhood,
// each a Gaussian-windowed cosine at the node's Frequency, turned towards
// the node's Orientation by `1 - Anisotropy`.

// `IMPULSES_COUNT`.
const GABOR_IMPULSE: i32 = 8;
// `NodeGaborType` (`SHD_GABOR_TYPE_*`).
const GABOR_2D: u32 = 0u;
const GABOR_3D: u32 = 1u;

fn polar_to_cartesian(r: f32, phi: f32) -> vec2<f32> {
    return vec2<f32>(r * cos(phi), r * sin(phi));
}

fn spherical_to_direction(theta: f32, phi: f32) -> vec3<f32> {
    return vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

// `compute_2d_gabor_kernel`, Equation (6) of the Gabor paper at unit
// magnitude and unit Gaussian width, truncated at π and windowed by a Hann
// window so the truncation stays C1 continuous. The phasor's imaginary part
// is the Gabor value (Tavernier §3.3, for a zero mean).
fn gabor_kernel_2d(position: vec2<f32>, frequency: f32, orientation: f32) -> vec2<f32> {
    let d2 = dot(position, position);
    let hann = 0.5 + 0.5 * cos(PI * d2);
    let envelope = exp(-PI * d2) * hann;
    let angle = 2.0 * PI * dot(position, polar_to_cartesian(frequency, orientation));
    return polar_to_cartesian(envelope, angle);
}

// `compute_3d_gabor_kernel`: the same, with the orientation already a unit
// frequency vector.
fn gabor_kernel_3d(position: vec3<f32>, frequency: f32, orientation: vec3<f32>) -> vec2<f32> {
    let d2 = dot(position, position);
    let hann = 0.5 + 0.5 * cos(PI * d2);
    let envelope = exp(-PI * d2) * hann;
    let angle = 2.0 * PI * dot(position, frequency * orientation);
    return polar_to_cartesian(envelope, angle);
}

// `compute_3d_orientation`: the base orientation turned by a random amount
// in spherical coordinates, `isotropy` of the way. The random range is π,
// not 2π, because the kernel is symmetric around π.
fn gabor_orientation_3d(orientation: vec3<f32>, isotropy: f32, seed: vec4<f32>) -> vec3<f32> {
    if (isotropy == 0.0) { return orientation; }
    var inclination = acos(clamp(orientation.z, -1.0, 1.0));
    // An orientation straight up the Z axis has no azimuth; Cycles divides
    // by its zero length and reads NaN, so the guard is ours.
    let xy = length(orientation.xy);
    let along = select(orientation.x / xy, 1.0, xy < 1e-8);
    var azimuth = select(1.0, -1.0, orientation.y < 0.0) * acos(clamp(along, -1.0, 1.0));
    let random = hash_float4_to_float2(seed) * PI;
    inclination += random.x * isotropy;
    azimuth += random.y * isotropy;
    return spherical_to_direction(inclination, azimuth);
}

// `compute_2d_gabor_noise_cell`: Equation (8)'s sum over one cell's
// impulses, each at a random centre with a Bernoulli ±1 weight.
fn gabor_cell_2d(cell: vec2<f32>, position: vec2<f32>, frequency: f32, isotropy: f32, base: f32) -> vec2<f32> {
    var noise = vec2<f32>(0.0);
    for (var i = 0; i < GABOR_IMPULSE; i++) {
        let k = f32(i * 3);
        let turn = (hash_float3_to_float(vec3<f32>(cell, k)) - 0.5) * PI;
        let orientation = base + turn * isotropy;
        let centre = hash_float3_to_float2(vec3<f32>(cell, k + 1.0));
        let p = position - centre;
        // The kernel is windowed past the unit distance.
        if (dot(p, p) >= 1.0) { continue; }
        let weight = select(1.0, -1.0, hash_float3_to_float(vec3<f32>(cell, k + 2.0)) < 0.5);
        noise += weight * gabor_kernel_2d(p, frequency, orientation);
    }
    return noise;
}

fn gabor_cell_3d(cell: vec3<f32>, position: vec3<f32>, frequency: f32, isotropy: f32, base: vec3<f32>) -> vec2<f32> {
    var noise = vec2<f32>(0.0);
    for (var i = 0; i < GABOR_IMPULSE; i++) {
        let k = f32(i * 3);
        let orientation = gabor_orientation_3d(base, isotropy, vec4<f32>(cell, k));
        let centre = hash_float4_to_float3(vec4<f32>(cell, k + 1.0));
        let p = position - centre;
        if (dot(p, p) >= 1.0) { continue; }
        let weight = select(1.0, -1.0, hash_float4_to_float(vec4<f32>(cell, k + 2.0)) < 0.5);
        noise += weight * gabor_kernel_3d(p, frequency, orientation);
    }
    return noise;
}

// `compute_2d_gabor_noise` / `compute_3d_gabor_noise`: the phasor summed
// over the cell the point is in and its neighbours.
fn gabor_phasor_2d(co: vec2<f32>, frequency: f32, isotropy: f32, base: f32) -> vec2<f32> {
    let cell = floor(co);
    let local = co - cell;
    var sum = vec2<f32>(0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let off = vec2<f32>(f32(i), f32(j));
            sum += gabor_cell_2d(cell + off, local - off, frequency, isotropy, base);
        }
    }
    return sum;
}

fn gabor_phasor_3d(co: vec3<f32>, frequency: f32, isotropy: f32, base: vec3<f32>) -> vec2<f32> {
    let cell = floor(co);
    let local = co - cell;
    var sum = vec2<f32>(0.0);
    for (var k = -1; k <= 1; k++) {
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let off = vec3<f32>(f32(i), f32(j), f32(k));
                sum += gabor_cell_3d(cell + off, local - off, frequency, isotropy, base);
            }
        }
    }
    return sum;
}

// `svm_node_tex_gabor`: the phasor, then the node's three outputs from it.
// `.x` Value, `.y` Phase, `.z` Intensity; `co` is already scaled.
// The standard deviations are the paper's approximations — 0.25 of the
// squared kernel's integral in 2D, `1 / (4 √2)` in 3D, each times the
// impulse count and the weights' second moment of 0.5 — and the noise is
// normalized by six of them, which Cycles says was found empirically.
fn gabor_texture(co: vec3<f32>, kind: u32, frequency0: f32, anisotropy: f32, orientation_2d: f32, orientation_3d: vec3<f32>) -> vec3<f32> {
    let isotropy = 1.0 - clamp(anisotropy, 0.0, 1.0);
    let frequency = max(0.001, frequency0);
    var phasor = vec2<f32>(0.0);
    var deviation = 1.0;
    if (kind == GABOR_3D) {
        // The socket's default is (√2, √2, 0); a file that zeroed it would
        // normalize to NaN, so the Z axis stands in.
        let dir = select(normalize(orientation_3d), vec3<f32>(0.0, 0.0, 1.0), dot(orientation_3d, orientation_3d) < 1e-12);
        phasor = gabor_phasor_3d(co, frequency, isotropy, dir);
        deviation = sqrt(f32(GABOR_IMPULSE) * 0.5 / (4.0 * sqrt(2.0)));
    } else {
        phasor = gabor_phasor_2d(co.xy, frequency, isotropy, orientation_2d);
        deviation = sqrt(f32(GABOR_IMPULSE) * 0.5 * 0.25);
    }
    let norm = 6.0 * deviation;
    return vec3<f32>(
        (phasor.y / norm) * 0.5 + 0.5,
        (atan2(phasor.y, phasor.x) + PI) / (2.0 * PI),
        length(phasor) / norm,
    );
}
