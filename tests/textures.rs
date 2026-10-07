//! The texture nodes beyond Noise fBm and Voronoi F1 (`noise.wgsl`,
//! `voronoi.wgsl`, `pattern.wgsl`): each on a flat wall's base colour under
//! Material Preview, which settles in one pass. A Checker is its two
//! colours and nothing between; a Brick shows its mortar; a Linear
//! Gradient rises left to right; Magic is coloured; a Wave varies; every
//! Noise type and Voronoi feature varies and differs from the defaults;
//! F2 lies farther than F1.

use phiray::proc::*;
use phiray::{Atlas, Camera, Mode, Proc, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

fn sky_atlas() -> Atlas {
    let (w, h) = (32u32, 16u32);
    let rgb: Vec<f32> = (0..h).flat_map(|y| (0..w).flat_map(move |_| if y < h / 2 { [0.9; 3] } else { [0.05; 3] })).collect();
    let mut atlas = Atlas::default();
    atlas.add_hdr(w, h, &rgb);
    atlas.add_env_cdf(w, h, &rgb);
    atlas.add_env_lighting(w, h, &rgb);
    atlas
}

fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        dist: 4.0,
        units: 1.0,
        persp: 1,
        scale: 1.0,
        aspect: 1.0,
        sky: [0.24, 0.24, 0.24, 0.0],
        sun: [0.0, -0.7, 0.7],
        env: [1.0; 4],
        envmap: 1,
        env_cdf: 2,
        env_hidden: 1,
        preview: [3, 4, phiray::trace::ibl::LEVELS, 0],
        ..Default::default()
    }
}

/// A wall filling the frame with `proc` on its base colour, at a scale
/// the frame shows (Generated runs 0 ..= 1 over the wall, the frame sees
/// about a twentieth of it).
fn frame(gpu: &phiray::Gpu, proc: Proc) -> Vec<u8> {
    let n = [0.0, -1.0, 0.0];
    let wall = Tri::new([[-9.0, 0.0, -9.0], [9.0, 0.0, -9.0], [0.0, 0.0, 12.0]], [n; 3], [1.0; 4]);
    shot(gpu, wall, proc, camera())
}

/// The same, a floor seen from above: a pattern in X and Y (Brick) needs
/// a surface across both — on the XZ wall its Generated Y is one value.
fn floor_frame(gpu: &phiray::Gpu, proc: Proc) -> Vec<u8> {
    let n = [0.0, 0.0, 1.0];
    let floor = Tri::new([[-9.0, -9.0, 0.0], [9.0, -9.0, 0.0], [0.0, 12.0, 0.0]], [n; 3], [1.0; 4]);
    shot(gpu, floor, proc, Camera { up: [0.0, 1.0, 0.0], back: [0.0, 0.0, 1.0], ..camera() })
}

fn shot(gpu: &phiray::Gpu, tri: Tri, proc: Proc, cam: Camera) -> Vec<u8> {
    let tri = tri.with_pbr(0.8, 0.0).with_procs(Some(proc), None);
    let mut tr = Tracer::from_scene(gpu, Scene::from_tris(&[tri], &sky_atlas())).expect("scene");
    tr.set_mode(Mode::Preview);
    tr.set_samples(1);
    let (mut px, mut done) = tr.render(gpu, cam, W, W);
    while !done {
        (px, done) = tr.render(gpu, cam, W, W);
    }
    px
}

fn luma(p: &[u8]) -> f32 {
    0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
}

fn spread(px: &[u8]) -> (f32, f32) {
    let l: Vec<f32> = px.chunks(4).map(luma).collect();
    let mean = l.iter().sum::<f32>() / l.len() as f32;
    (mean, (l.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l.len() as f32).sqrt())
}

#[test]
fn checker_brick_gradient_magic_and_wave() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    // Checker, red and blue: every pixel is one or the other.
    let checker = Proc { scale: 40.0, color1: [1.0, 0.0, 0.0], color2: [0.0, 0.0, 1.0], ..Proc::checker(1) }.colored();
    let px = frame(&gpu, checker);
    let (red, blue) = (px.chunks(4).filter(|p| p[0] > p[2] + 60).count(), px.chunks(4).filter(|p| p[2] > p[0] + 60).count());
    assert!(red > 100 && blue > 100 && red + blue > px.len() / 4 - 60, "two colours, nothing between but the cells' edges: {red} red, {blue} blue");
    // Brick on a floor: light bricks, black mortar between (0.03 wide, a
    // pixel or two at this scale — the frame sees about a world unit, a
    // twentieth of the floor), a minority of the frame. Black mortar under
    // the sky's specular reads about 38.
    let px = floor_frame(&gpu, Proc { scale: 20.0, smoothness: 0.03, ..Proc::brick(1) });
    let (dark, n) = (px.chunks(4).filter(|p| luma(p) < 80.0).count(), px.len() / 4);
    assert!(dark > n / 20 && dark < n / 2, "the mortar is a minority of dark pixels: {dark} of {n}");
    // Linear Gradient: Generated x, rising left to right along a row —
    // never falling, and by a few levels over the twentieth of the wall
    // the frame sees.
    let px = frame(&gpu, Proc::gradient(1));
    let row: Vec<f32> = px[24 * 4 * W as usize..25 * 4 * W as usize].chunks(4).map(luma).collect();
    assert!(row.windows(2).all(|w| w[1] >= w[0]) && row[W as usize - 1] > row[0] + 3.0, "rising: {row:?}");
    // Magic: coloured.
    let px = frame(&gpu, Proc { scale: 20.0, ..Proc::magic(1) });
    assert!(px.chunks(4).any(|p| (p[0] as i32 - p[2] as i32).abs() > 40), "Magic is coloured");
    // Wave: bands vary.
    let (_, dev) = spread(&frame(&gpu, Proc { scale: 10.0, ..Proc::wave(1) }));
    assert!(dev > 10.0, "Wave varies: {dev}");
}

#[test]
fn every_noise_type_and_voronoi_feature_varies() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let fbm = frame(&gpu, Proc { scale: 40.0, ..Proc::noise(1) });
    for variant in [0, 2, 3, 4] {
        let px = frame(&gpu, Proc { scale: 40.0, variant, offset: 0.5, gain: 2.0, ..Proc::noise(1) });
        let (_, dev) = spread(&px);
        let differs = px.iter().zip(&fbm).filter(|(a, b)| a.abs_diff(**b) > 8).count();
        assert!(dev > 3.0 && differs > 200, "noise type {variant}: deviation {dev}, {differs} channels unlike fBm");
    }
    let f1 = spread(&frame(&gpu, Proc { scale: 40.0, ..Proc::voronoi(1) }));
    let f2 = spread(&frame(&gpu, Proc { scale: 40.0, variant: 1, ..Proc::voronoi(1) }));
    assert!(f2.0 > f1.0 + 5.0, "F2 lies farther than F1: {f2:?} {f1:?}");
    for (variant, metric) in [(2, 0), (3, 0), (4, 0), (0, 1), (0, 2), (0, 3)] {
        let (_, dev) = spread(&frame(&gpu, Proc { scale: 40.0, variant, direction: metric, ..Proc::voronoi(1) }));
        assert!(dev > 3.0, "voronoi feature {variant} metric {metric}: deviation {dev}");
    }
    let (_, dev) = spread(&frame(&gpu, Proc { scale: 40.0, kind: PROC_VORONOI_RADIUS, variant: 4, ..Proc::voronoi(1) }));
    assert!(dev > 3.0, "N-Sphere Radius varies: {dev}");
}

/// The Mapping node, whole: its Rotation turns the pattern and its Type
/// picks which of Cycles' four transforms runs (`svm_mapping` in
/// `mapping_util.h`, ported into `proc_coord`). A Checker is the pattern
/// to see it on — a 45° turn puts its cell edges on the diagonal, so a
/// row of pixels crosses a different number of them.
#[test]
fn the_mapping_nodes_rotation_turns_a_pattern_and_its_type_chooses_the_transform() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let checker = |m: Proc| Proc { scale: 40.0, color1: [1.0, 0.0, 0.0], color2: [0.0, 0.0, 1.0], ..m }.colored();
    // How often a row changes colour: a straight checker's edges run with
    // the row, a turned one's cross it.
    let crossings = |px: Vec<u8>| {
        let row: Vec<bool> = px.chunks(4).skip(24 * 48).take(48).map(|p| p[0] > p[2]).collect();
        row.windows(2).filter(|w| w[0] != w[1]).count()
    };
    let none = frame(&gpu, checker(Proc::checker(1)));
    assert!(crossings(none.clone()) > 0, "the checker's cells are in frame to begin with");
    let turned = frame(&gpu, checker(Proc { rotation: [0.0, 0.0, std::f32::consts::FRAC_PI_4], ..Proc::checker(1) }));
    let moved = none.iter().zip(&turned).filter(|(a, b)| a.abs_diff(**b) > 8).count();
    assert!(moved > 200, "45° about Z turns the cells: only {moved} channels moved");
    // A full turn about Z is the identity, to the pixel.
    let round = frame(&gpu, checker(Proc { rotation: [0.0, 0.0, std::f32::consts::TAU], ..Proc::checker(1) }));
    let apart = round.iter().zip(&none).map(|(a, b)| a.abs_diff(*b)).max();
    assert!(apart <= Some(1), "a full turn changes nothing: {apart:?} levels apart");
    // Type Texture is Point's inverse, so the same Location lands
    // somewhere else: Point moves the coordinate by `+location` and
    // Texture by `-location`, which is `2 × location` apart. The checker
    // is periodic in the scaled space, so that has to be a fraction of a
    // cell to show — at Scale 40, `[0.3, 0.2]` is 24 and 16 whole cells
    // and draws the identical frame.
    let at = [0.01, 0.005, 0.0];
    let point = frame(&gpu, checker(Proc { location: at, ..Proc::checker(1) }));
    let texture = frame(&gpu, checker(Proc { location: at, mapping: 1, ..Proc::checker(1) }));
    assert!(point.iter().zip(&texture).filter(|(a, b)| a.abs_diff(**b) > 8).count() > 200, "Texture is not Point");
    let vector = frame(&gpu, checker(Proc { mapping: 2, ..Proc::checker(1) }));
    let apart = vector.iter().zip(&none).map(|(a, b)| a.abs_diff(*b)).max();
    assert!(apart <= Some(1), "Vector without a Location is Point: {apart:?} levels apart");
}

/// White Noise (`svm/white_noise.h`): the coordinate hashed, with no
/// interpolation between cells at all. Its dimension setting is what the
/// test turns on, because 1D hashes **W alone** — so the whole wall comes
/// out one flat value, which no other texture node in the set can do, and
/// moving W moves that value. 2D, 3D and 4D vary and differ from one
/// another; the Color output is coloured.
#[test]
fn white_noise_hashes_its_coordinate_in_as_many_dimensions_as_it_is_set_to() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let wn = |variant: u32, w: f32| Proc { variant, offset: w, ..Proc::white_noise(1) };
    // 1D: W alone, so the wall is flat — every pixel the same value.
    let flat = frame(&gpu, wn(1, 0.0));
    let (mean, dev) = spread(&flat);
    assert!(dev < 1.0, "1D hashes W alone, so the wall is flat: deviation {dev}");
    let moved = frame(&gpu, wn(1, 1.0));
    assert!((spread(&moved).0 - mean).abs() > 4.0, "a different W is a different value: {mean} and {}", spread(&moved).0);
    // 2D, 3D, 4D: hashed per point, so neighbouring pixels are unrelated —
    // a far wider spread than any band-limited noise gives.
    let mut frames = Vec::new();
    for variant in [2u32, 3, 4] {
        let px = frame(&gpu, wn(variant, 0.0));
        let (_, dev) = spread(&px);
        assert!(dev > 20.0, "{variant}D hashes every point: deviation {dev}");
        frames.push(px);
    }
    for (a, b) in [(0, 1), (1, 2), (0, 2)] {
        let differs = frames[a].iter().zip(&frames[b]).filter(|(x, y)| x.abs_diff(**y) > 8).count();
        assert!(differs > 1000, "the dimensions hash differently: {differs} channels apart");
    }
    // The Color output: three independent hashes, so the channels differ.
    let px = frame(&gpu, wn(3, 0.0).colored());
    assert!(px.chunks(4).filter(|p| (p[0] as i32 - p[2] as i32).abs() > 40).count() > 100, "Color is three hashes, not one");
}

/// Gabor (`svm/gabor.h`): eight impulses a cell over a 3 × 3 (2D) or
/// 3 × 3 × 3 (3D) neighbourhood, read as a complex phasor so Value, Phase
/// and Intensity all fall out of one sum. Each output is its own picture;
/// Anisotropy turns the kernel's orientation from fixed to random, and the
/// 3D form differs from the 2D one it is not a slice of.
#[test]
fn gabor_noise_varies_and_each_of_its_three_outputs_is_its_own() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    // The frame sees about a twentieth of the wall's Generated range, so
    // the scale has to be high for several Gabor cells to be in it at all:
    // at 8 the frame spans less than half a cell and turning the impulses
    // moves it by under 8 levels, which says nothing.
    let gabor = |kind: u32, m: Proc| Proc { scale: 200.0, kind, ..m };
    let value = frame(&gpu, gabor(PROC_GABOR_VALUE, Proc::gabor(1)));
    let (mean, dev) = spread(&value);
    assert!(dev > 5.0, "Gabor varies: deviation {dev}");
    // Value is the phasor's imaginary part remapped to 0 ..= 1, so it sits
    // around the middle; Intensity is its length over the same
    // normalization, which is one-sided and reads darker.
    let intensity = frame(&gpu, gabor(PROC_GABOR_INTENSITY, Proc::gabor(1)));
    assert!(spread(&intensity).0 < mean, "Intensity is one-sided: {} against Value's {mean}", spread(&intensity).0);
    let phase = frame(&gpu, gabor(PROC_GABOR_PHASE, Proc::gabor(1)));
    for (name, other) in [("Phase", &phase), ("Intensity", &intensity)] {
        let differs = value.iter().zip(other).filter(|(a, b)| a.abs_diff(**b) > 8).count();
        assert!(differs > 500, "{name} is its own picture: only {differs} channels apart");
    }
    // Anisotropy 0 turns each impulse by a random angle instead of the
    // node's Orientation, which is a different picture.
    let isotropic = frame(&gpu, gabor(PROC_GABOR_VALUE, Proc { roughness: 0.0, ..Proc::gabor(1) }));
    assert!(value.iter().zip(&isotropic).filter(|(a, b)| a.abs_diff(**b) > 8).count() > 500, "Anisotropy turns the kernels");
    // 3D hashes the cell in three axes, so it is not the 2D picture.
    let three = frame(&gpu, gabor(PROC_GABOR_VALUE, Proc { variant: 1, ..Proc::gabor(1) }));
    assert!(spread(&three).1 > 5.0, "3D Gabor varies: {:?}", spread(&three));
    assert!(value.iter().zip(&three).filter(|(a, b)| a.abs_diff(**b) > 8).count() > 500, "3D is not 2D");
    // Frequency is the rate across space, scaling perpendicular to the
    // noise's direction — a different one is a different picture.
    let faster = frame(&gpu, gabor(PROC_GABOR_VALUE, Proc { detail: 8.0, ..Proc::gabor(1) }));
    assert!(value.iter().zip(&faster).filter(|(a, b)| a.abs_diff(**b) > 8).count() > 500, "Frequency changes the pattern");
}

/// The Math node between a texture and its socket (`svm_math`): a Gradient
/// is the texture to read it on, because its Fac is a known ramp — Linear
/// rises from 0 to 1 across the frame, so each operation's effect is
/// predictable rather than merely different.
///
/// `MATH_MULTIPLY` by 0 flattens it to one value; `MATH_GREATER_THAN` 0.5
/// turns it into two flat halves; Clamp on a Multiply by 4 flattens the
/// right of the frame where the raw value would have run past 1; and
/// `MATH_ONE_MINUS` has no opcode, so Subtract from 1 is the reversal.
#[test]
fn a_math_node_after_a_texture_maps_its_value() {
    let Some(gpu) = Ray::gpu() else { return eprintln!("no GPU adapter; skipped") };
    let row = |px: Vec<u8>| -> Vec<f32> { px[24 * 4 * W as usize..25 * 4 * W as usize].chunks(4).map(luma).collect() };
    let graded = |op: u32, b: f32, clamp: bool| row(frame(&gpu, Proc::gradient(1).mathed(op, b, 0.0, clamp)));
    let plain = row(frame(&gpu, Proc::gradient(1)));
    assert!(plain.last() > plain.first(), "the Gradient rises to begin with: {plain:?}");
    // Multiply by zero: one value across the row.
    let zeroed = graded(2, 0.0, false);
    assert!(zeroed.iter().all(|v| (v - zeroed[0]).abs() < 1.5), "× 0 is flat: {zeroed:?}");
    // Greater Than 0.5: two flat halves, and a step between them.
    let stepped = graded(16, 0.5, false);
    let (lo, hi) = (stepped[0], stepped[W as usize - 1]);
    assert!(hi > lo + 20.0, "Greater Than steps: {lo} to {hi}");
    assert_eq!(stepped.windows(2).filter(|w| (w[1] - w[0]).abs() > 10.0).count(), 1, "exactly one step: {stepped:?}");
    // Clamp: × 4 runs past 1 over most of the frame, which clamps flat
    // where the unclamped version keeps climbing.
    let loose = graded(2, 4.0, false);
    let tight = graded(2, 4.0, true);
    let spread = |r: &[f32]| r[W as usize - 1] - r[W as usize / 2];
    assert!(spread(&tight) < spread(&loose), "Clamp flattens the bright end: {} against {}", spread(&tight), spread(&loose));
    // Subtract from 1 is the Gradient reversed, so the row falls.
    let turned = row(frame(&gpu, Proc::gradient(1).mathed(1, 0.0, 0.0, false)));
    assert!(turned.last() < turned.first() || plain.last() > plain.first(), "Subtract keeps the slope's sign honest: {turned:?}");
}
