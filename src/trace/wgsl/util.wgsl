fn hash(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn rnd(seed: ptr<function, u32>) -> f32 {
    *seed = hash(*seed);
    return f32(*seed) / 4294967295.0;
}

// The denoiser's guide channels, averaged over the samples alongside the
// radiance so the pixel jitter averages out of them too: the first hit's
// albedo (words 7n..10n), its shading normal (10n..13n) and its distance
// (13n..14n, negative where the ray met nothing). A denoiser blurs radiance
// only across pixels whose guides agree, which is how it keeps an edge.
fn guide(i: u32, albedo: vec3<f32>, n: vec3<f32>, dist: f32) {
    let a = 7u * p.n + 3u * i;
    let b = 10u * p.n + 3u * i;
    let c = 13u * p.n + i;
    let keep = p.frame > 0u;
    let old_a = select(vec3<f32>(0.0), vec3<f32>(bitcast<f32>(px[a]), bitcast<f32>(px[a + 1u]), bitcast<f32>(px[a + 2u])), keep);
    let old_n = select(vec3<f32>(0.0), vec3<f32>(bitcast<f32>(px[b]), bitcast<f32>(px[b + 1u]), bitcast<f32>(px[b + 2u])), keep);
    let old_d = select(0.0, bitcast<f32>(px[c]), keep);
    let sa = old_a + albedo;
    let sn = old_n + n;
    px[a] = bitcast<u32>(sa.x);
    px[a + 1u] = bitcast<u32>(sa.y);
    px[a + 2u] = bitcast<u32>(sa.z);
    px[b] = bitcast<u32>(sn.x);
    px[b + 1u] = bitcast<u32>(sn.y);
    px[b + 2u] = bitcast<u32>(sn.z);
    px[c] = bitcast<u32>(old_d + dist);
}

// Scene-linear radiance → display with this frame's colour management.
fn view_transform(lin: vec3<f32>) -> vec3<f32> {
    return view_transform_of(lin, p.view);
}

// Blend this sample into the running averages and write the display pixel.
//
// `cover` is 1 when the camera ray met something (or the film is opaque,
// so the sky counts) and 0 when it met nothing under a transparent film;
// its sum (words 24n..25n) over the samples is the pixel's alpha.
//
// `lin` (scene-linear radiance) accumulates in words n..4n and is what
// `Tracer::radiance` reads back. The pixel shown is the view transform of
// the *mean* of those samples — transformed once, as Blender does — with the
// grid overlay (`over`, premultiplied colour + coverage, summed in words
// 20n..24n) composited on top. That transformed value is kept in words
// 4n..7n for `Tracer::display`.
//
// It used to transform every sample and average the results. That is wrong
// in a way that hides: the transform is concave and clamps, so averaging
// transformed samples is always darker than transforming the average, and a
// highlight that saturates in some samples but not others can never reach
// white. On THYMUS it cost ~30% of the highlights and, under a white
// material override, turned a figure Cycles blows out (mean 227) into a grey
// one (mean 90).
fn emit(i: u32, lin: vec3<f32>, over: vec4<f32>, cover: f32) {
    let a = p.n + 3u * i;
    let g = 20u * p.n + 3u * i;
    let gc = 23u * p.n + i;
    let fc = 24u * p.n + i;
    let keep = p.frame > 0u;
    let k = f32(p.frame + 1u);
    let film = select(0.0, bitcast<f32>(px[fc]), keep) + cover;
    px[fc] = bitcast<u32>(film);
    let l = select(vec3<f32>(0.0), vec3<f32>(bitcast<f32>(px[a]), bitcast<f32>(px[a + 1u]), bitcast<f32>(px[a + 2u])), keep) + lin;
    let gs = select(vec3<f32>(0.0), vec3<f32>(bitcast<f32>(px[g]), bitcast<f32>(px[g + 1u]), bitcast<f32>(px[g + 2u])), keep) + over.rgb;
    let cov = select(0.0, bitcast<f32>(px[gc]), keep) + over.a;
    px[a] = bitcast<u32>(l.x);
    px[a + 1u] = bitcast<u32>(l.y);
    px[a + 2u] = bitcast<u32>(l.z);
    px[g] = bitcast<u32>(gs.x);
    px[g + 1u] = bitcast<u32>(gs.y);
    px[g + 2u] = bitcast<u32>(gs.z);
    px[gc] = bitcast<u32>(cov);
    let shown = view_transform(l / k) * (1.0 - cov / k) + gs / k;
    let b = 4u * p.n + 3u * i;
    px[b] = bitcast<u32>(shown.x);
    px[b + 1u] = bitcast<u32>(shown.y);
    px[b + 2u] = bitcast<u32>(shown.z);
    px[i] = pack(shown, film / k);
}

// Blender's floor grid on z = 0 (`overlay_grid.cc`): lines every power of
// ten, the level chosen by how large a cell is on screen, with the finer
// level fading out as it shrinks below a few pixels and the coarser one taking
// over — so zooming in subdivides and zooming out merges, continuously. Lines
// are one pixel wide with an analytic anti-aliased edge (`foot` is the size of
// one pixel on the floor at that point), and fade with distance so the plane
// reads as a plane rather than a wall of lines at the horizon. X axis red,
// Y axis green, at the same width.
// The sky at pixel `i`: `p.sky.rgb`, lifted by `p.sky.a` at the left edge
// and back to itself by mid-width — the theme's sweep, when it has one
// (Blender's own background is one colour, `TH_BACKGROUND_SINGLE_COLOR`).
// The whole frame this image is of: itself, or — a tile (`p.tile`: its
// corner x, y and the frame's width and height) — the frame it is cut from.
// The pixel this invocation traces: `p.n` (past the end) outside the
// dispatch's slice (`p.slice`: first, count — all when count is 0).
fn pixel(gid: vec3<u32>) -> u32 {
    let k = item(gid, p.groups_x);
    if (p.slice.y != 0u && k >= p.slice.y) { return p.n; }
    return k + p.slice.x;
}

fn frame_size() -> vec2<f32> {
    if (p.tile.z != 0u) { return vec2<f32>(f32(p.tile.z), f32(p.tile.w)); }
    return vec2<f32>(f32(p.w), f32(p.h));
}
// Pixel `i`'s place in the whole frame.
fn frame_pixel(i: u32) -> vec2<f32> {
    return vec2<f32>(f32(p.tile.x + i % p.w), f32(p.tile.y + i / p.w));
}
// Its canvas point: `-0.5 ..= 0.5` across and down the whole frame.
fn frame_point(i: u32) -> vec2<f32> {
    return (frame_pixel(i) + 0.5) / frame_size() - 0.5;
}
fn sky_at(i: u32) -> vec3<f32> {
    let across = frame_pixel(i).x / max(frame_size().x * 0.5, 1.0);
    return p.sky.rgb + vec3<f32>(p.sky.a) * (1.0 - clamp(across, 0.0, 1.0));
}

fn floor_grid(o: vec3<f32>, d: vec3<f32>, t_hit: f32, px: f32, base: vec3<f32>) -> vec3<f32> {
    let g = grid_over(o, d, t_hit, px);
    return base * (1.0 - g.a) + g.rgb;
}

// The grid as an overlay — premultiplied colour and coverage — so the tracer
// can accumulate it per sample and composite it onto the tone-mapped mean:
// the floor and its X and Y axes, then the Z axis, each as the viewport's
// Overlays popover asks (`Camera::grid_hide`).
fn grid_over(o: vec3<f32>, d: vec3<f32>, t_hit: f32, px: f32) -> vec4<f32> {
    var g = floor_over(o, d, t_hit, px);
    if (p.grid > 0.0 && (p.grid_hide & 8u) != 0u) {
        let z = z_axis_over(o, d, t_hit, px);
        g = vec4<f32>(g.rgb * (1.0 - z.a) + z.rgb, g.a * (1.0 - z.a) + z.a);
    }
    return g;
}

// The Z axis: the vertical line through the origin, where the ray passes
// within half a pixel of it before whatever it hit.
fn z_axis_over(o: vec3<f32>, d: vec3<f32>, t_hit: f32, px: f32) -> vec4<f32> {
    let a = dot(d.xy, d.xy);
    if (a < 1e-12) { return vec4<f32>(0.0); }
    let t = -dot(o.xy, d.xy) / a;
    if (t <= 0.0 || t >= t_hit) { return vec4<f32>(0.0); }
    var foot = px;
    if (p.persp != 0u) { foot = px * t * p.units / p.dist; }
    let r = length((o + d * t).xy);
    let w = (1.0 - smoothstep(foot * 0.5, foot * 1.5, r)) * 0.85;
    return vec4<f32>(vec3<f32>(0.25, 0.53, 0.92) * w, w);
}

// The floor and its X and Y axes; bits 1, 2 and 4 of `grid_hide` leave out
// the floor's lines, the X axis and the Y axis — an axis left out is an
// ordinary floor line.
fn floor_over(o: vec3<f32>, d: vec3<f32>, t_hit: f32, px: f32) -> vec4<f32> {
    if (p.grid <= 0.0 || abs(d.z) < 1e-6) { return vec4<f32>(0.0); }
    let t = -o.z / d.z;
    if (t <= 0.0 || t >= t_hit) { return vec4<f32>(0.0); }
    let q = o + d * t;
    // One pixel's footprint on the floor, in scene units.
    var foot = px;
    if (p.persp != 0u) { foot = px * t * p.units / p.dist; }
    // Level: the fine spacing is the power of ten whose cell is at least
    // ~24 px; `f` is how far the eye has zoomed out within that decade.
    let want = foot * 24.0;
    let lg = log2(max(want, 1e-9)) / log2(10.0);
    let lvl = floor(lg);
    let f = lg - lvl;
    let g0 = pow(10.0, lvl);
    let g1 = g0 * 10.0;
    // Half a pixel each side, anti-aliased over one pixel.
    let half_w = foot * 0.5;
    let aa = foot;
    let c0 = q.xy / g0;
    let c1 = q.xy / g1;
    let d0 = abs(c0 - round(c0)) * g0;
    let d1 = abs(c1 - round(c1)) * g1;
    let l0 = max(1.0 - smoothstep(half_w, half_w + aa, d0.x), 1.0 - smoothstep(half_w, half_w + aa, d0.y));
    let l1 = max(1.0 - smoothstep(half_w, half_w + aa, d1.x), 1.0 - smoothstep(half_w, half_w + aa, d1.y));
    // The fine level fades as its cells shrink toward the pixel; the coarse
    // level is always full. Across one decade the coarse lines of this level
    // become the fine lines of the next, so the hand-over is seamless.
    let fine = (1.0 - f) * (1.0 - f);
    let line = max(l0 * fine * 0.6, l1);
    // Distance fade: the floor dims toward the horizon, and at grazing angles
    // (line density high, one line per pixel) it fades to the sky instead of
    // aliasing.
    let horizon = clamp(abs(d.z) * 4.0, 0.0, 1.0);
    var depth = 1.0;
    if (p.persp != 0u) { depth = 1.0 / (1.0 + (t * p.units / (p.dist * 6.0)) * (t * p.units / (p.dist * 6.0))); }
    let fade = horizon * depth;
    // "Over" compositing, kept premultiplied: colour and coverage together.
    let w0 = line * p.grid_color.a * fade * select(1.0, 0.0, (p.grid_hide & 1u) != 0u);
    var c = p.grid_color.rgb * w0;
    var a = w0;
    // The axes: red along X (y = 0), green along Y (x = 0).
    let ax = 1.0 - smoothstep(half_w, half_w + aa, abs(q.y));
    let ay = 1.0 - smoothstep(half_w, half_w + aa, abs(q.x));
    let w1 = ax * 0.85 * fade * select(1.0, 0.0, (p.grid_hide & 2u) != 0u);
    c = c * (1.0 - w1) + vec3<f32>(0.85, 0.3, 0.33) * w1;
    a = a * (1.0 - w1) + w1;
    let w2 = ay * 0.85 * fade * select(1.0, 0.0, (p.grid_hide & 4u) != 0u);
    c = c * (1.0 - w2) + vec3<f32>(0.47, 0.72, 0.2) * w2;
    a = a * (1.0 - w2) + w2;
    return vec4<f32>(c, a);
}

// A finished pixel: Solid mode is deterministic, so its value is written
// straight out rather than averaged into the sample buffer.
fn once(i: u32, disp: vec3<f32>) {
    px[i] = pack(disp, 1.0);
}

// Workbench's anti-aliasing: where this sample's rays are cast — the
// pixel's centre moved by the jitter, `p.taa.xy` in `−1 ..= 1`, half a
// pixel either way (`window_translate_m4` divides it by the resolution),
// in canvas units, y down.
fn taa_jitter() -> vec2<f32> {
    let size = frame_size();
    return vec2<f32>(-p.taa.x / (2.0 * size.x), p.taa.y / (2.0 * size.y));
}

// This sample's colour for pixel `i`, kept for the gather (`taa.wgsl`)
// in the scratch words the denoiser never touches in these modes.
fn taa_sample(i: u32, disp: vec3<f32>) {
    let a = 14u * p.n + 3u * i;
    px[a] = bitcast<u32>(disp.x);
    px[a + 1u] = bitcast<u32>(disp.y);
    px[a + 2u] = bitcast<u32>(disp.z);
}

// A point in the unit disk from two uniform numbers — the aperture's
// sample (`lens_offset`) and the GGX visible-normal sample's
// (`ggx_sample_vndf`), so it lives with the other primitives every kernel
// composes.
//
// Cycles ref: `camera_sample_aperture` (`kernel/camera/camera.h`) and
// `sample_uniform_disk` / `regular_polygon_sample` (`kernel/sample/mapping.h`).
// The disk mapping is Shirley's concentric one rather than the naive
// `(r = sqrt(u), phi = 2pi v)`: both are area-preserving, but the concentric
// one keeps neighbouring samples neighbours, which matters because the
// aperture is sampled with the same low-discrepancy stream as everything
// else.
fn unit_disk(ru: f32, rv: f32) -> vec2<f32> {
    let a = 2.0 * ru - 1.0;
    let b = 2.0 * rv - 1.0;
    if (a == 0.0 && b == 0.0) { return vec2<f32>(0.0); }
    var r: f32;
    var phi: f32;
    if (a * a > b * b) {
        r = a;
        phi = (PI / 4.0) * (b / a);
    } else {
        r = b;
        phi = (PI / 2.0) - (PI / 4.0) * (a / b);
    }
    return vec2<f32>(r * cos(phi), r * sin(phi));
}
