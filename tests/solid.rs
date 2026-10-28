//! The two deterministic modes: Workbench (Solid) and Wireframe.
//!
//! Blender's Solid viewport is a Workbench pass, not a path trace — each
//! sample is deterministic, and the frame is eight of them jittered and
//! gathered (Workbench's anti-aliasing, `taa.rs`). These tests hold that:
//! eight samples then done, a lit surface that reads brighter than the sky
//! behind it, and the same image whether it is asked for once or ten times
//! after. Wireframe is the same shape of pass, drawing edges instead of
//! surfaces.

use phiray::{Atlas, Camera, Mode, Ray, Tracer, Tri};

fn quad(p: [[f32; 3]; 4], n: [f32; 3], c: [f32; 4]) -> [Tri; 2] {
    [Tri::new([p[0], p[1], p[2]], [n; 3], c), Tri::new([p[0], p[2], p[3]], [n; 3], c)]
}

/// One grey wall square on, filling the middle of the frame.
fn wall() -> Vec<Tri> {
    quad(
        [[-3., 0., -3.], [3., 0., -3.], [3., 0., 3.], [-3., 0., 3.]],
        [0., -1., 0.],
        [0.6, 0.6, 0.6, 1.0],
    )
    .to_vec()
}

/// Looking down −Y at the wall, orthographic, no floor grid in shot.
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

const W: u32 = 64;
const H: u32 = 64;

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

fn luma(c: [u8; 3]) -> f32 {
    0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32
}

/// The frame once every sample is in.
fn settled(tr: &mut Tracer, gpu: &phiray::Gpu, cam: Camera, w: u32, h: u32) -> Vec<u8> {
    let (mut px, mut done) = tr.render(gpu, cam, w, h);
    while !done {
        (px, done) = tr.render(gpu, cam, w, h);
    }
    px
}

#[test]
fn solid_settles_in_eight_jittered_samples_and_lights_the_surface() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Solid);
    assert_eq!(tr.samples(), 8, "Workbench's `viewport_aa`: eight jittered samples make the frame");

    let (first, done) = tr.render(&gpu, camera(), W, H);
    assert!(!done, "the first sample is shown, the rest refine the edges");
    assert_eq!(first.len(), (W * H * 4) as usize);
    let mut count = 1;
    let mut px = first.clone();
    let mut finished = false;
    while !finished {
        (px, finished) = tr.render(&gpu, camera(), W, H);
        count += 1;
    }
    assert_eq!(count, 8, "done after the eighth");
    assert_eq!(pixel(&first, W / 2, H / 2), pixel(&px, W / 2, H / 2), "the wall's middle does not change with the jitter");

    // Centre is the wall, a corner is sky.
    let centre = pixel(&px, W / 2, H / 2);
    let corner = pixel(&px, 1, 1);
    assert!(
        luma(centre) > luma(corner) + 20.0,
        "a surface facing the studio rig must read brighter than the sky: \
         centre {centre:?} vs sky {corner:?}"
    );
    // The rig is neutral-to-cool on a grey surface, but it is not black and
    // not clipped — those are the two ways the port fails silently.
    assert!(luma(centre) > 30.0, "surface came out black: {centre:?}");
    assert!(luma(centre) < 250.0, "surface clipped to white: {centre:?}");
}

#[test]
fn asking_again_returns_the_same_image() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Solid);
    let first = settled(&mut tr, &gpu, camera(), W, H);
    for _ in 0..3 {
        let (again, done) = tr.render(&gpu, camera(), W, H);
        assert!(done);
        assert_eq!(first, again, "Solid must not drift between frames");
    }
    // Another tracer over the same scene settles to the very same frame:
    // the jitter table is fixed, as Blender's is.
    let mut other = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    other.set_mode(Mode::Solid);
    assert_eq!(settled(&mut other, &gpu, camera(), W, H), first);
}

#[test]
fn switching_modes_rebuilds_the_pass() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Solid);
    let solid = settled(&mut tr, &gpu, camera(), W, H);
    tr.set_mode(Mode::Rendered);
    assert_eq!(tr.samples(), Tracer::SAMPLES);
    let (traced, done) = tr.render(&gpu, camera(), W, H);
    assert!(!done, "the tracer refines, so its first frame is not the last");
    assert_ne!(solid, traced, "the two modes must not produce the same image");
    tr.set_mode(Mode::Solid);
    let back = settled(&mut tr, &gpu, camera(), W, H);
    assert_eq!(solid, back, "returning to Solid must return the same frame");
}

/// Wire tests light the wire white so it is unambiguous against the sky.
/// In `Mode::Wire` the camera's `env` carries the wire colour.
fn wire_camera() -> Camera {
    Camera { env: [1.0, 1.0, 1.0, 1.0], ..camera() }
}

/// The sky at this row. The sky is a vertical gradient, so a pixel can only
/// be compared with another on the SAME row — x = 1 is outside the wall at
/// every row, so it is always sky.
fn sky_at(px: &[u8], y: u32) -> [u8; 3] {
    pixel(px, 1, y)
}

fn is_wire(px: &[u8], x: u32, y: u32) -> bool {
    luma(pixel(px, x, y)) > luma(sky_at(px, y)) + 40.0
}

#[test]
fn wire_draws_edges_and_leaves_the_interior_alone() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut tr = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Wire);
    assert_eq!(tr.samples(), 8, "the wire pass is anti-aliased the same way");
    let px = settled(&mut tr, &gpu, wire_camera(), W, H);

    // The wall's interior is background: a wireframe shows what is behind the
    // surface, not the surface. Sampled on the wall but away from its edges
    // and its diagonal.
    assert!(
        !is_wire(&px, W / 2 - 8, H / 2 + 2),
        "the inside of a face must read as background in wireframe"
    );

    // The quad's silhouette and the diagonal where its two triangles meet do
    // draw — but as lines, not as a fill.
    let wired = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| is_wire(&px, x, y))
        .count();
    assert!(wired > 50, "almost nothing was drawn: {wired} px");
    assert!(
        wired < (W * H) as usize / 8,
        "the wire pass filled the frame instead of drawing lines: {wired} px"
    );
}

#[test]
fn a_wire_behind_a_face_does_not_draw() {
    let Some(gpu) = Ray::gpu() else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    // A band nearer the camera than the wall. The wall's diagonal runs at 45°
    // through the origin; the band's runs at a different slope, so at world
    // (1.5, 1.5) the wall's diagonal is hidden behind the band and the band's
    // own diagonal is 0.75 units away — several pixels clear.
    let mut scene = wall();
    scene.extend(quad(
        [[-3., -2., -2.], [3., -2., -2.], [3., -2., 2.], [-3., -2., 2.]],
        [0., -1., 0.],
        [0.6, 0.6, 0.6, 1.0],
    ));
    let mut tr = Tracer::new(&gpu, &scene, &Atlas::default()).expect("scene");
    tr.set_mode(Mode::Wire);
    let (px, _) = tr.render(&gpu, wire_camera(), W, H);

    // World (1.5, 1.5): the frame spans ±5 units over 64 px, so 1.5 units is
    // 9.6 px from the centre — right on the hidden diagonal.
    let (x, y) = (W / 2 + 10, H / 2 - 10);
    assert!(
        !is_wire(&px, x, y),
        "an edge behind a face must be hidden, but ({x}, {y}) drew a wire"
    );

    // Without the band in front, that same pixel IS on the wall's diagonal —
    // otherwise the test above would pass for the wrong reason.
    let mut bare = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    bare.set_mode(Mode::Wire);
    let open = settled(&mut bare, &gpu, wire_camera(), W, H);
    assert!(
        is_wire(&open, x, y),
        "the control pixel is not on an edge, so the test proves nothing"
    );
}

/// A new scene on the same device — a moved object, a played frame —
/// reuses the kernels the last one compiled (`Gpu::pipeline`): building
/// them again cost a scene change 150 ms in Solid and 700 ms in Rendered on
/// the cloud's software Vulkan (`examples/scene_rebuild`).
#[test]
fn a_new_scene_on_the_same_device_compiles_nothing() {
    let Some(gpu) = Ray::gpu() else { return };
    let mut first = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("a scene");
    first.set_mode(Mode::Solid);
    first.render(&gpu, camera(), 32, 32);
    let compiled = gpu.compiled();
    assert!(compiled >= 1);
    let mut next = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("a scene");
    next.set_mode(Mode::Solid);
    let (a, b) = (settled(&mut first, &gpu, camera(), 32, 32), settled(&mut next, &gpu, camera(), 32, 32));
    assert_eq!(gpu.compiled(), compiled, "nothing compiled again");
    assert_eq!(a, b, "and the same image");
}

/// Two walls side by side, the right one `dx` further right.
fn walls(dx: f32) -> Vec<Tri> {
    let q = |x0: f32, c: f32| quad([[x0, 0., -2.], [x0 + 2., 0., -2.], [x0 + 2., 0., 2.], [x0, 0., 2.]], [0., -1., 0.], [c, c, c, 1.0]);
    [q(-3.0, 0.3), q(0.5 + dx, 0.8)].concat()
}

/// A new pose (an object moved) goes up in place — the tree refitted, not
/// built again — and draws what a tracer built on the moved scene draws;
/// a scene with other faces is refused, for a build.
#[test]
fn a_new_pose_draws_what_a_fresh_build_draws() {
    let Some(gpu) = Ray::gpu() else { return };
    let solid = |tris: &[Tri]| {
        let mut t = Tracer::new(&gpu, tris, &Atlas::default()).expect("a scene");
        t.set_mode(Mode::Solid);
        t
    };
    let mut posed = solid(&walls(0.0));
    let before = posed.render(&gpu, camera(), 48, 48).0;
    assert!(posed.repose(&gpu, &phiray::Scene::from_tris(&walls(0.6), &Atlas::default())));
    let after = posed.render(&gpu, camera(), 48, 48).0;
    assert_ne!(before, after, "the right wall moved");
    assert_eq!(after, solid(&walls(0.6)).render(&gpu, camera(), 48, 48).0, "as a fresh build draws it");
    let more = [walls(0.6), wall()].concat();
    assert!(!posed.repose(&gpu, &phiray::Scene::from_tris(&more, &Atlas::default())), "other faces: build again");
}

/// Two walls as grids of small quads (a tree of many leaves), the right one
/// moved `dx` along X.
fn tiled(dx: f32) -> Vec<Tri> {
    let grid = |x0: f32, c: f32| {
        let cells = (0..8).flat_map(|i| (0..8).map(move |j| (x0 + i as f32 * 0.25, -2.0 + j as f32 * 0.5)));
        cells.flat_map(|(x, z)| quad([[x, 0., z], [x + 0.25, 0., z], [x + 0.25, 0., z + 0.5], [x, 0., z + 0.5]], [0., -1., 0.], [c, c, c, 1.0])).collect::<Vec<_>>()
    };
    [grid(-3.0, 0.3), grid(0.5 + dx, 0.8)].concat()
}

/// Once the motion stops the tree is built again over the pose held
/// (`retree`): the image keeps its samples — the tree decides how fast a
/// ray finds its hit, never which — and a new view draws what a tracer
/// built on the moved scene draws.
#[test]
fn a_new_tree_keeps_the_image_and_its_samples() {
    let Some(gpu) = Ray::gpu() else { return };
    let moved = phiray::Scene::from_tris(&tiled(1.5), &Atlas::default());
    let posed = || {
        let mut t = Tracer::new(&gpu, &tiled(0.0), &Atlas::default()).expect("a scene");
        assert!(t.repose(&gpu, &moved));
        (0..3).for_each(|_| _ = t.sample(&gpu, camera(), 32, 32));
        t
    };
    let (mut kept, mut retreed) = (posed(), posed());
    assert!(retreed.retree(&gpu));
    assert_eq!(retreed.sample(&gpu, camera(), 32, 32).0, 3, "the fourth sample, not the first again");
    kept.sample(&gpu, camera(), 32, 32);
    assert_eq!(retreed.pixels(&gpu), kept.pixels(&gpu), "the same image");

    let aside = Camera { pivot: [0.7, 0.0, 0.3], ..camera() };
    let mut fresh = Tracer::new(&gpu, &tiled(1.5), &Atlas::default()).expect("a scene");
    for t in [&mut retreed, &mut fresh] {
        t.set_mode(Mode::Solid);
    }
    assert_eq!(retreed.render(&gpu, aside, 40, 40).0, fresh.render(&gpu, aside, 40, 40).0, "as a fresh build draws it");
}

/// A frame too large for one binding is rendered in tiles: `Camera.tile`
/// names the tile's corner in the full frame and the frame's size, so each
/// tile's rays are the ones the whole frame would cast through those pixels.
/// Tiles are anti-aliased as the whole is (Workbench's `render_aa`), and
/// the effects and the gather each clamp at a tile's edge — so two
/// half-height tiles, each drawn two rows over into the other and laid in
/// without them, are the whole frame pixel for pixel, in Solid (outlined,
/// curvature on) and in Wireframe.
#[test]
fn tiles_laid_together_are_the_whole_frame() {
    let Some(gpu) = Ray::gpu() else { return };
    for (mode, cam) in [(Mode::Solid, Camera { fx: [1.0, 0.5, 0.7, 1.0], ..camera() }), (Mode::Wire, wire_camera())] {
        let mut whole = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
        whole.set_mode(mode);
        // Tiles trace their primary hits (`Tracer::rasterizes` draws the
        // prepass only for a whole frame), so the frame they are laid
        // against traces too: this is about the seams and the gather, not
        // about the prepass. The prepass is held to the traced image by
        // `raster.rs`'s `the_rasterized_frame_is_the_traced_frame`, within
        // a step — it carries the instance in the word the distance used
        // to have (`raster.wgsl`) and works the distance out again, so the
        // two paths round a wire's edge coverage a least significant bit
        // apart on a few dozen of this frame's 16,384 bytes.
        whole.set_raster(false);
        let full = settled(&mut whole, &gpu, cam, W, H);
        let mut tiled = Vec::new();
        let row = (W * 4) as usize;
        for (y0, pad_top, pad_bottom) in [(0, 0, 2), (H / 2, 2, 0)] {
            let mut part = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
            part.set_mode(mode);
            assert_eq!(part.samples(), 8, "{mode:?}: a tile takes Workbench's samples too");
            let px = settled(&mut part, &gpu, Camera { tile: [0, y0 - pad_top, W, H], ..cam }, W, H / 2 + pad_top + pad_bottom);
            tiled.extend_from_slice(&px[pad_top as usize * row..px.len() - pad_bottom as usize * row]);
        }
        assert_eq!(tiled.len(), full.len());
        // Say by how far when they differ: a byte or two apart is drift,
        // a wholesale difference is a seam the gather did not clamp.
        let off = tiled.iter().zip(&full).filter(|(a, b)| a != b).count();
        let worst = tiled.iter().zip(&full).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        assert_eq!(tiled, full, "{mode:?}: the tiles differ from the frame drawn whole ({off} of {} bytes, worst {worst})", full.len());
        // And the prepass draws that same frame to within a step, over the
        // instanced draws one mesh at a time (`Trees::draws`).
        let mut drawn = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
        drawn.set_mode(mode);
        let shown = settled(&mut drawn, &gpu, cam, W, H);
        let apart = shown.iter().zip(&full).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        assert!(drawn.raster() || gpu.adapter.is_software(), "{mode:?}: hardware rasterizes the prepass");
        assert!(apart <= 1, "{mode:?}: the rasterized frame is {apart} off the traced one, not a rounding step");
    }
}

/// Film ▸ Transparent (`Camera::film`): a camera ray that meets nothing
/// leaves its pixel black at alpha 0, a ray that meets the wall leaves alpha
/// 255 — so a PNG with an alpha channel has the background cut away. The
/// default film keeps the sky at alpha 255. Rendered mode, since Blender's
/// film is the path tracer's; the denoiser keeps the alpha.
#[test]
fn a_transparent_film_cuts_the_background_away() {
    let Some(gpu) = Ray::gpu() else { return };
    let mut tr = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    tr.set_samples(4);
    let clear = Camera { film: [1, 0, 0, 0], ..camera() };
    while !tr.sample(&gpu, clear, W, H).1 {}
    let px = tr.pixels(&gpu);
    let at = |x: u32, y: u32| { let i = ((y * W + x) * 4) as usize; [px[i], px[i + 1], px[i + 2], px[i + 3]] };
    assert_eq!(at(1, 1), [0, 0, 0, 0], "sky: black, alpha 0");
    assert_eq!(at(W / 2, H / 2)[3], 255, "the wall: alpha 1");
    assert!(luma([at(W / 2, H / 2)[0], at(W / 2, H / 2)[1], at(W / 2, H / 2)[2]]) > 30.0, "and lit");
    tr.denoise(&gpu);
    let px = tr.pixels(&gpu);
    assert_eq!((px[3], px[((H / 2 * W + W / 2) * 4 + 3) as usize]), (0, 255), "denoised, the alpha stays");

    let mut opaque = Tracer::new(&gpu, &wall(), &Atlas::default()).expect("scene");
    opaque.set_samples(4);
    while !opaque.sample(&gpu, camera(), W, H).1 {}
    let px = opaque.pixels(&gpu);
    assert_eq!(px[3], 255, "the default film: the sky at alpha 1");
    assert!(px[0] > 0 || px[1] > 0 || px[2] > 0, "and shown");
}
