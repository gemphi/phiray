//! Solid's Lighting and Color as the Shading popover sets them
//! (`Camera::shade`, `::single`): Color ▸ Object shades each object with
//! its own colour, ▸ Random with a hue hashed from its name
//! (`ObjectHash::hue`), ▸ Single with the one colour, each at roughness 0.4
//! and no metal (`workbench_engine.cc`'s `get_material`); Lighting ▸ Flat
//! is the colour itself, unlit (`WORKBENCH_LIGHTING_FLAT`). The default
//! words draw the frame of before.

use phiray::trace::hash::ObjectHash;
use phiray::{Camera, Mat, Mode, Ray, Scene, Tracer, Tri};

const W: u32 = 48;

fn camera() -> Camera {
    Camera { right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0], units: 1.0, scale: 0.25, aspect: 1.0, dist: 10.0, sky: [0.3, 0.3, 0.3, 0.0], env: [0.2, 0.22, 0.25, 1.0], ..Default::default() }
}

/// Two grey walls side by side, "Left" red by object colour and "Right" blue.
fn scene() -> Scene {
    let mut s = Scene::default();
    let n = [0.0, -1.0, 0.0];
    for (x, name, color) in [(-1.0f32, "Left", [1.0, 0.0, 0.0]), (1.0, "Right", [0.0, 0.0, 1.0])] {
        s.instance_of([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [x, 0.0, 0.0, 1.0]], color, ObjectHash::of(name));
        let q = [[-0.9, 0.0, -1.5], [0.9, 0.0, -1.5], [0.9, 0.0, 1.5], [-0.9, 0.0, 1.5]];
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let t = Tri::new(tri, [n; 3], [0.5, 0.5, 0.5, 1.0]);
            let m = s.material(Mat::of(&t));
            let v = tri.map(|p| s.vertex(p, n, [0.0; 2]));
            s.face(v, m);
        }
    }
    s
}

fn frame(gpu: &phiray::Gpu, cam: Camera) -> Vec<u8> {
    let mut t = Tracer::from_scene(gpu, scene()).expect("a scene");
    t.set_mode(Mode::Solid);
    let (mut px, mut done) = t.render(gpu, cam, W, W);
    while !done {
        (px, done) = t.render(gpu, cam, W, W);
    }
    px
}

fn at(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

#[test]
fn color_object_random_and_single_shade_as_blender_does() {
    let Some(gpu) = Ray::gpu() else { return };
    let (l, r) = (W / 4, 3 * W / 4);
    let plain = frame(&gpu, camera());
    assert_eq!(at(&plain, l, W / 2), at(&plain, r, W / 2), "Material: both walls the same grey");
    let object = frame(&gpu, Camera { shade: [0, 4, 0, 0], ..camera() });
    let (pl, pr) = (at(&object, l, W / 2), at(&object, r, W / 2));
    assert!(pl[0] > pl[2] + 60 && pr[2] > pr[0] + 60, "Object: red left, blue right: {pl:?} {pr:?}");
    let random = frame(&gpu, Camera { shade: [0, 1, 0, 0], ..camera() });
    let (rl, rr) = (at(&random, l, W / 2), at(&random, r, W / 2));
    assert_ne!(rl, rr, "Random: the two names hash to two hues");
    // The hue the kernel drew is the hue `ObjectHash::hue` computes: the dominant
    // channel of saturation-0.5 HSV follows the hue sextant.
    let sextant = |name: &str| (ObjectHash::of(name).hue() * 6.0).floor() as u32 % 6;
    let dominant = |c: [u8; 3]| (0..3).max_by_key(|&k| c[k]).unwrap();
    for (name, c) in [("Left", rl), ("Right", rr)] {
        let expect = match sextant(name) { 0 | 5 => 0, 1 | 2 => 1, _ => 2 };
        assert_eq!(dominant(c), expect, "{name}: hue sextant {} drew {c:?}", sextant(name));
    }
    let single = frame(&gpu, Camera { shade: [0, 2, 0, 0], single: [0.0, 0.6, 0.0, 1.0], ..camera() });
    let (sl, sr) = (at(&single, l, W / 2), at(&single, r, W / 2));
    assert_eq!(sl, sr, "Single: one colour");
    assert!(sl[1] > sl[0] + 60, "and it is the green asked: {sl:?}");
    assert_eq!(at(&single, 1, 1), at(&plain, 1, 1), "the sky is not a material");
}

#[test]
fn flat_lighting_is_the_colour_itself() {
    let Some(gpu) = Ray::gpu() else { return };
    let flat = frame(&gpu, Camera { shade: [2, 4, 0, 0], ..camera() });
    let (top, mid, low) = (at(&flat, W / 4, 8), at(&flat, W / 4, W / 2), at(&flat, W / 4, W - 8));
    assert_eq!(top, mid, "no shading across the wall");
    assert_eq!(mid, low);
    // Linear red 1.0 shows as 255, the rest 0 — the colour, untouched.
    assert_eq!(mid, [255, 0, 0], "{mid:?}");
    let lit = frame(&gpu, Camera { shade: [0, 4, 0, 0], ..camera() });
    assert_ne!(at(&lit, W / 4, W / 2), mid, "Studio lighting shades it");
}

/// Backface Culling (`V3D_SHADING_BACKFACE_CULLING`, `Camera::shade[2]`):
/// the walls face the viewer, so turning the camera round shows their backs
/// — drawn as before, or culled to the sky — and Specular Lighting off
/// (`shade[3]`) takes the highlight out of the studio lighting.
#[test]
fn backface_culling_sees_through_turned_faces_and_specular_can_go() {
    let Some(gpu) = Ray::gpu() else { return };
    let behind = Camera { back: [0.0, 1.0, 0.0], right: [-1.0, 0.0, 0.0], ..camera() };
    let drawn = frame(&gpu, behind);
    assert_ne!(at(&drawn, W / 4, W / 2), at(&drawn, 1, 1), "from behind the walls are drawn, back faces turned to the viewer");
    let culled = frame(&gpu, Camera { shade: [0, 0, 1, 0], ..behind });
    assert_eq!(at(&culled, W / 4, W / 2), at(&culled, 1, 1), "culled: the sky shows through");
    let front = frame(&gpu, Camera { shade: [0, 0, 1, 0], ..camera() });
    assert_eq!(front, frame(&gpu, camera()), "from the front nothing is culled");
    let matte = frame(&gpu, Camera { shade: [0, 0, 0, 1], ..camera() });
    let (lit, flat) = (at(&front, W / 4, W / 2), at(&matte, W / 4, W / 2));
    assert_ne!(lit, flat, "the highlight is gone");
    assert!(flat[0] <= lit[0] + 2, "and nothing brighter came of it: {flat:?} vs {lit:?}");
}

/// Color ▸ Attribute (`Camera::shade[1] == 5`): the vertices' colour
/// attribute, interpolated, at roughness 0.4 and no metal; a scene with no
/// colour attribute draws its materials (`SceneState::has_color`).
#[test]
fn attribute_colour_is_the_vertex_colour_or_the_material_without_one() {
    let Some(gpu) = Ray::gpu() else { return };
    let plain = frame(&gpu, camera());
    // No object has a colour attribute: Attribute is Material.
    assert_eq!(frame(&gpu, Camera { shade: [0, 5, 0, 0], ..camera() }), plain);
    // Paint the left wall's vertices (added first) red and the right's blue.
    let mut s = scene();
    let half = s.pos.len() as u32 / 2;
    for v in 0..s.pos.len() as u32 {
        s.paint(v, if v < half { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] });
    }
    let mut t = Tracer::from_scene(&gpu, s).expect("a scene");
    t.set_mode(Mode::Solid);
    let cam = Camera { shade: [2, 5, 0, 0], ..camera() };
    let (mut px, mut done) = t.render(&gpu, cam, W, W);
    while !done {
        (px, done) = t.render(&gpu, cam, W, W);
    }
    // Flat lighting: the colours themselves.
    assert_eq!(at(&px, W / 4, W / 2), [255, 0, 0], "{:?}", at(&px, W / 4, W / 2));
    assert_eq!(at(&px, 3 * W / 4, W / 2), [0, 0, 255]);
    // Material still draws the grey.
    let (mut px, mut done) = t.render(&gpu, Camera { shade: [2, 0, 0, 0], ..camera() }, W, W);
    while !done {
        (px, done) = t.render(&gpu, Camera { shade: [2, 0, 0, 0], ..camera() }, W, W);
    }
    assert_eq!(at(&px, W / 4, W / 2), [188, 188, 188], "half grey, Flat");
}

/// Lighting ▸ MatCap (`Camera::matcap`, `get_matcap_lighting`): the wall
/// facing the viewer reads the matcap's centre, tilted normals read toward
/// its sides (and the other way flipped), and the specular image is added
/// only while Specular Lighting is on.
#[test]
fn matcap_lighting_reads_the_image_by_the_view_normal() {
    let Some(gpu) = Ray::gpu() else { return };
    // A 4 × 4 matcap: red on the left half, blue on the right; a flat
    // specular image of 0.25 green.
    let mut s = scene();
    let mut rgb = Vec::new();
    for _ in 0..4 {
        for x in 0..4 {
            rgb.extend(if x < 2 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] });
        }
    }
    let diffuse = s.atlas.add_hdr(4, 4, &rgb);
    let specular = s.atlas.add_hdr(4, 4, &[0.0, 0.25, 0.0].repeat(16));
    let mut t = Tracer::from_scene(&gpu, s).expect("a scene");
    t.set_mode(Mode::Solid);
    let draw = |t: &mut Tracer, cam: Camera| {
        let (mut px, mut done) = t.render(&gpu, cam, W, W);
        while !done {
            (px, done) = t.render(&gpu, cam, W, W);
        }
        px
    };
    // Single white colour so the lighting is all that shows; the walls face
    // the viewer, so the normal reads the image's centre — the seam —
    // which bilinear sampling makes half red, half blue.
    let base = Camera { shade: [1, 2, 0, 0], single: [1.0; 4], matcap: [diffuse + 1, 0, 0, 0], ..camera() };
    let centre = at(&draw(&mut t, base), W / 4, W / 2);
    assert!(centre[0] > 100 && centre[2] > 100 && centre[1] == 0, "the seam, no green: {centre:?}");
    // Turn the view so the walls' normals point toward the viewer's left:
    // `uv.x` falls below the centre — red; flipped, blue.
    let turned = Camera { back: [0.6, -0.8, 0.0], right: [0.8, 0.6, 0.0], ..base };
    let left = at(&draw(&mut t, turned), W / 4, W / 2);
    assert!(left[0] > 200 && left[2] < 30, "tilted toward −x reads the red side: {left:?}");
    let flipped = at(&draw(&mut t, Camera { matcap: [diffuse + 1, 0, 1, 0], ..turned }), W / 4, W / 2);
    assert!(flipped[2] > 200 && flipped[0] < 30, "flipped reads the blue side: {flipped:?}");
    // The specular image adds green — unless Specular Lighting is off.
    let shiny = at(&draw(&mut t, Camera { matcap: [diffuse + 1, specular + 1, 0, 0], ..turned }), W / 4, W / 2);
    assert!(shiny[1] > 100, "the specular pass: {shiny:?}");
    let matte = at(&draw(&mut t, Camera { matcap: [diffuse + 1, specular + 1, 0, 0], shade: [1, 2, 0, 1], ..turned }), W / 4, W / 2);
    assert_eq!(matte[1], 0, "no specular: {matte:?}");
    // No matcap words: MatCap draws as Studio.
    assert_eq!(draw(&mut t, Camera { matcap: [0; 4], ..base }), draw(&mut t, Camera { shade: [0, 2, 0, 0], ..base }));
}

/// Color ▸ Texture (`workbench_image_lib.glsl`): a face with an image shows
/// the image — the colour, not a tint over the material's — and a face
/// without one keeps its material colour.
#[test]
fn texture_colour_is_the_image_itself() {
    let Some(gpu) = Ray::gpu() else { return };
    // The left wall gets a green 2 × 2 image; the right wall none.
    let mut s = Scene::default();
    let tex = s.atlas.add(2, 2, &[0, 255, 0, 255].repeat(4));
    let n = [0.0, -1.0, 0.0];
    for (x, textured) in [(-1.0f32, true), (1.0, false)] {
        s.instance_of([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [x, 0.0, 0.0, 1.0]], [1.0; 3], ObjectHash(0));
        let q = [[-0.9, 0.0, -1.5], [0.9, 0.0, -1.5], [0.9, 0.0, 1.5], [-0.9, 0.0, 1.5]];
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let mut t = Tri::new(tri, [n; 3], [0.5, 0.5, 0.5, 1.0]);
            if textured {
                t = t.with_texture(tex, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
            }
            let m = s.material(Mat::of(&t));
            let v = tri.map(|p| s.vertex(p, n, [0.0; 2]));
            s.face(v, m);
        }
    }
    let mut t = Tracer::from_scene(&gpu, s).expect("a scene");
    t.set_mode(Mode::Solid);
    let cam = Camera { shade: [2, 3, 0, 0], ..camera() };
    let (mut px, mut done) = t.render(&gpu, cam, W, W);
    while !done {
        (px, done) = t.render(&gpu, cam, W, W);
    }
    assert_eq!(at(&px, W / 4, W / 2), [0, 255, 0], "the image, under Flat lighting");
    assert_eq!(at(&px, 3 * W / 4, W / 2), [188, 188, 188], "no image: the material's grey");
}
