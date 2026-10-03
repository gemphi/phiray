//! The floor grid: adaptive spacing, anti-aliased lines, distance fade.
//!
//! Blender's grid subdivides as the view zooms in and merges as it zooms out,
//! with the finer level fading rather than popping; lines are one pixel with
//! soft edges; and the floor dims toward the horizon. This holds all three on
//! the tracer's own output, and writes the frames out so they can be looked
//! at (`target/grid-*.ppm` when `PHIRAY_GRID_PNG` is set).

use phiray::{Atlas, Camera, Mode, Ray, Tracer, Tri};

const W: u32 = 256;
const H: u32 = 256;

/// Looking down at the floor from a low perspective angle, like the default
/// viewport, with the grid on.
fn camera(zoom: f32) -> Camera {
    // A 35° tilt: rows of the image sweep from near (bottom) to far (top).
    let tilt = 35f32.to_radians();
    let up = [0.0, tilt.cos(), tilt.sin()];
    let back = [0.0, -tilt.sin(), tilt.cos()];
    Camera {
        right: [1.0, 0.0, 0.0],
        up,
        back,
        pivot: [0.0, 0.0, 0.0],
        units: 1.0,
        scale: zoom,
        aspect: 1.0,
        persp: 1,
        dist: 1.0 / (zoom * 0.5),
        sky: [0.2, 0.22, 0.25, 0.0],
        env: [0.2, 0.22, 0.25, 1.0],
        grid: 1.0,
        grid_color: [0.6, 0.62, 0.66, 0.5],
        ..Default::default()
    }
}

/// One tiny triangle far below the floor, out of shot: the tracer needs a
/// scene to build, and the grid is what is being looked at.
fn scene() -> Vec<Tri> {
    vec![Tri::new(
        [[100.0, 100.0, -100.0], [101.0, 100.0, -100.0], [100.0, 101.0, -100.0]],
        [[0.0, 0.0, 1.0]; 3],
        [0.5, 0.5, 0.5, 1.0],
    )]
}

fn frame(gpu: &phiray::Gpu, zoom: f32) -> Vec<u8> {
    let mut tr = Tracer::new(gpu, &scene(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Solid);
    let (px, _) = tr.render(gpu, camera(zoom), W, H);
    if std::env::var("PHIRAY_GRID_PNG").is_ok() {
        // A binary PPM needs no image crate.
        let mut out = format!("P6 {W} {H} 255\n").into_bytes();
        for p in px.chunks_exact(4) {
            out.extend_from_slice(&[p[0], p[1], p[2]]);
        }
        let _ = std::fs::create_dir_all("target");
        std::fs::write(format!("target/grid-{zoom}.ppm"), out).unwrap();
    }
    px
}

fn luma(px: &[u8], x: u32, y: u32) -> f32 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * px[i] as f32 + 0.7152 * px[i + 1] as f32 + 0.0722 * px[i + 2] as f32
}

/// How much a row's brightness varies along it — grid lines make it vary.
fn row_contrast(px: &[u8], y: u32) -> f32 {
    let vals: Vec<f32> = (0..W).map(|x| luma(px, x, y)).collect();
    let max = vals.iter().cloned().fold(0.0, f32::max);
    let min = vals.iter().cloned().fold(255.0, f32::min);
    max - min
}

/// Distinct grid lines crossed along a row, counting each rise above the
/// row's own background once.
fn lines_in_row(px: &[u8], y: u32) -> usize {
    let vals: Vec<f32> = (0..W).map(|x| luma(px, x, y)).collect();
    let base = {
        let mut s = vals.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let mut n = 0;
    let mut on = false;
    for v in vals {
        // The fading fine level counts too: it is what keeps the density up
        // as the coarse level's cells grow.
        let lit = v > base + 3.0;
        if lit && !on {
            n += 1;
        }
        on = lit;
    }
    n
}

#[test]
fn the_grid_subdivides_as_the_view_zooms_in() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    // A row in the lower half — near enough to be sharp, far enough to cross
    // several cells, and clear of the X axis, which runs along the middle.
    let y = H * 3 / 4;
    let wide = lines_in_row(&frame(&gpu, 0.05), y);
    let close = lines_in_row(&frame(&gpu, 0.5), y);
    // Ten times closer shows lines of the next level down, so the count
    // stays in the same band instead of dropping to one or two.
    assert!(wide >= 3, "zoomed out, the row crosses several lines: {wide}");
    assert!(close >= 3, "zoomed in, the grid has subdivided: {close}");
    assert!(
        (close as i32 - wide as i32).abs() <= wide as i32,
        "the line density stays of the same order across a decade: {wide} vs {close}"
    );
}

#[test]
fn the_lines_are_soft_edged_not_hard_steps() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let px = frame(&gpu, 0.2);
    let y = H * 3 / 4;
    // Walk the row: across a line the luma must pass through an intermediate
    // value at least once — a hard-stepped line goes background→peak→background
    // with nothing between.
    let vals: Vec<f32> = (0..W).map(|x| luma(&px, x, y)).collect();
    let (min, max) = (vals.iter().cloned().fold(255.0, f32::min), vals.iter().cloned().fold(0.0, f32::max));
    assert!(max - min > 8.0, "the row crosses a line at all");
    let mid_lo = min + (max - min) * 0.25;
    let mid_hi = min + (max - min) * 0.75;
    let intermediate = vals.iter().filter(|v| **v > mid_lo && **v < mid_hi).count();
    assert!(intermediate >= 2, "a line edge blends over a pixel; found {intermediate} in-between pixels");
}

#[test]
fn the_floor_fades_toward_the_horizon() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let px = frame(&gpu, 0.2);
    // Near the bottom the lines stand out; toward the horizon they fade.
    let near = row_contrast(&px, H * 7 / 8);
    let far = row_contrast(&px, H * 3 / 8);
    assert!(near > 8.0, "near the eye the grid is visible: contrast {near}");
    assert!(far < near, "toward the horizon the grid fades: near {near} vs far {far}");
}

/// A frame with the guides `hide` leaves out (`Camera::grid_hide`).
fn hiding(gpu: &phiray::Gpu, hide: u32) -> Vec<u8> {
    let mut tr = Tracer::new(gpu, &scene(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Solid);
    tr.render(gpu, Camera { grid_hide: hide, ..camera(0.2) }, W, H).0
}

#[test]
fn the_overlays_popover_leaves_out_the_floor_and_each_axis_and_adds_z() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    // Below the middle every row crosses the floor's lines and the Y axis,
    // which runs up the middle of the image.
    let y = H * 3 / 4;
    assert!(lines_in_row(&hiding(&gpu, 0), y) >= 3);
    assert_eq!(lines_in_row(&hiding(&gpu, 1), y), 1, "the floor off: the Y axis alone");
    assert!(row_contrast(&hiding(&gpu, 1 | 2 | 4), y) < 1.0, "the floor and both axes off: nothing");
    // The Z axis stands at the origin, the middle of the image: above the
    // floor's horizon too, where no floor line reaches.
    let z = hiding(&gpu, 1 | 2 | 4 | 8);
    let top = H / 8;
    assert_eq!(lines_in_row(&z, top), 1, "the Z axis alone");
    let i = ((top * W + W / 2) * 4) as usize;
    assert!(z[i + 2] > z[i] + 20, "it is blue: {:?}", &z[i..i + 3]);
}
