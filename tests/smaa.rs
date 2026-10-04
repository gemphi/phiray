//! SMAA over Solid's first samples (`smaa.wgsl`, Workbench's
//! `smaa_edge_detect_ps_` → `smaa_aa_weight_ps_` → `smaa_resolve_ps_`):
//! the first frame of a slanted edge used to be a staircase — every pixel
//! the wall's colour or the sky's, a deterministic kernel's one sample —
//! and is now blended along the edge, while the frames the jittered
//! samples have settled are the gather's own.

use phiray::{Atlas, Camera, Mode, Ray, Tracer, Tri};

const W: u32 = 96;
const H: u32 = 96;

/// A grey wall whose top edge runs across the frame at a shallow slant.
fn slanted_wall() -> Vec<Tri> {
    let (a, b) = (0.6f32, 0.15f32);
    let p = [[-6., 0., -6.], [6., 0., -6.], [6., 0., a + b], [-6., 0., a - b]];
    let n = [0., -1., 0.];
    let c = [0.6, 0.6, 0.6, 1.0];
    vec![Tri::new([p[0], p[1], p[2]], [n; 3], c), Tri::new([p[0], p[2], p[3]], [n; 3], c)]
}

fn camera() -> Camera {
    Camera {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        back: [0.0, -1.0, 0.0],
        pivot: [0.0, 0.0, 0.0],
        units: 1.0,
        scale: 0.1,
        aspect: 1.0,
        persp: 0,
        dist: 10.0,
        sky: [0.2, 0.22, 0.25, 0.0],
        env: [0.2, 0.22, 0.25, 1.0],
        grid: 0.0,
        ..Default::default()
    }
}

fn luma(px: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * px[i] as f32 + 0.7152 * px[i + 1] as f32 + 0.0722 * px[i + 2] as f32
}

#[test]
fn the_first_frames_edge_is_blended_and_the_settled_frame_is_the_gathers() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &slanted_wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Solid);
    let (first, done) = tr.render(&gpu, camera(), W, H);
    assert!(!done);
    // The sky above and the wall below, read where the edge is nowhere near.
    let sky = luma(&first, W / 2, 4);
    let wall = luma(&first, W / 2, H - 4);
    assert!(wall > sky + 40.0, "a lit wall over the sky: {wall} vs {sky}");
    // Along the slanted edge, the first frame has pixels neither colour:
    // SMAA's blend. A staircase would have none.
    let between = |px: &[u8]| -> usize {
        (0..W)
            .flat_map(|x| (0..H).map(move |y| (x, y)))
            .filter(|&(x, y)| {
                let l = luma(px, x, y);
                l > sky + 0.2 * (wall - sky) && l < wall - 0.2 * (wall - sky)
            })
            .count()
    };
    let blended = between(&first);
    assert!(blended >= W as usize / 2, "the first frame blends the edge: {blended} pixels between the two colours");
    // Away from the edge nothing changed.
    assert!((luma(&first, 8, H - 8) - wall).abs() < 2.0 && (luma(&first, 8, 8) - sky).abs() < 2.0);

    // SMAA fades over the next three samples; from the fourth the frame
    // is the gather's alone, and the settled frame has the edge the eight
    // jittered samples resolve — also blended, as anti-aliasing is.
    let mut frames = vec![first];
    loop {
        let (px, done) = tr.render(&gpu, camera(), W, H);
        frames.push(px);
        if done {
            break;
        }
    }
    assert_eq!(frames.len(), 8);
    assert!(between(&frames[7]) >= W as usize / 2, "the settled edge is anti-aliased");
    // Rendering once more changes nothing: the pass is at rest.
    let (again, done) = tr.render(&gpu, camera(), W, H);
    assert!(done);
    assert_eq!(again, frames[7]);
}

/// `U.viewport_aa` / `render_aa` (`set_aa_samples`): off draws the one
/// sample as it is — the staircase; the single pass is that sample with
/// SMAA, done at once; a count is that many jittered samples, and a
/// change of count starts the samples over.
#[test]
fn the_count_is_off_a_single_pass_or_blenders_tables() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &slanted_wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Solid);
    let between = |px: &[u8]| -> usize {
        let (sky, wall) = (luma(px, W / 2, 4), luma(px, W / 2, H - 4));
        (0..W).flat_map(|x| (0..H).map(move |y| (x, y))).filter(|&(x, y)| {
            let l = luma(px, x, y);
            l > sky + 0.2 * (wall - sky) && l < wall - 0.2 * (wall - sky)
        }).count()
    };
    tr.set_aa_samples(0);
    let (off, done) = tr.render(&gpu, camera(), W, H);
    assert!(done, "no anti-aliasing: one sample and done");
    assert_eq!(tr.samples(), 1);
    assert_eq!(between(&off), 0, "the staircase as drawn");

    tr.set_aa_samples(1);
    let (single, done) = tr.render(&gpu, camera(), W, H);
    assert!(done, "the single pass: one sample, SMAA over it, done");
    assert!(between(&single) >= W as usize / 2, "smoothed: {}", between(&single));

    tr.set_aa_samples(5);
    let (_, done) = tr.render(&gpu, camera(), W, H);
    assert!(!done && tr.samples() == 5, "a new count starts over");
    let mut n = 1;
    loop {
        n += 1;
        if tr.render(&gpu, camera(), W, H).1 {
            break;
        }
    }
    assert_eq!(n, 5, "five samples in all");
    tr.set_aa_samples(7);
    assert_eq!(tr.aa_samples(), 5, "a count without a table takes the five-sample one");
    tr.set_aa_samples(32);
    assert_eq!(tr.samples(), 32);
}
