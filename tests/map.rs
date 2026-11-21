//! Textures on Roughness, Metallic and Emission Color (Principled inputs a
//! file may link an image to, like Base Color).

use phiray::{Atlas, Camera, Light, NO_TEX, Ray, Tracer, Tri};

/// A grey quad facing -y with UVs over the whole image, seen head-on with a
/// light from the right; returns the centre pixel.
struct Card {
    atlas: Atlas,
    gpu: phiray::Gpu,
    /// Direction the sun travels in.
    light: [f32; 3],
}

impl Card {
    fn new() -> Option<Self> {
        Some(Self { atlas: Atlas::default(), gpu: Ray::gpu()?, light: [-0.6, 0.6, -0.5] })
    }

    fn pixel(&self, tune: impl Fn(Tri, [[f32; 2]; 3]) -> Tri) -> [u8; 3] {
        let n = [0.0, -1.0, 0.0];
        let q = [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let grey = [0.6, 0.6, 0.6, 1.0];
        let tris = [
            tune(Tri::new([q[0], q[1], q[2]], [n; 3], grey), [uv[0], uv[1], uv[2]]),
            tune(Tri::new([q[0], q[2], q[3]], [n; 3], grey), [uv[0], uv[2], uv[3]]),
        ];
        let mut cam = Camera {
            right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], back: [0.0, -1.0, 0.0],
            units: 1.0, scale: 0.4, aspect: 1.0, count: 1, ..Default::default()
        };
        cam.lights[0] = Light { kind: 0, color: [3.0; 3], dir: self.light, ..Default::default() };
        let mut tr = Tracer::new(&self.gpu, &tris, &self.atlas).expect("scene");
        let (mut px, mut done) = tr.render(&self.gpu, cam, 16, 16);
        while !done {
            (px, done) = tr.render(&self.gpu, cam, 16, 16);
        }
        let i = (8 * 16 + 8) * 4;
        [px[i], px[i + 1], px[i + 2]]
    }
}

#[test]
fn emission_texture_colours_the_glow() {
    let Some(mut card) = Card::new() else { return };
    let green = card.atlas.add(1, 1, &[0, 255, 0, 255]);
    // White emission × a green image = green light leaving the surface.
    let lit = card.pixel(|t, _| t.with_optics([3.0; 3], 0.0, 1.5));
    let mapped = card.pixel(|t, uv| t.with_optics([3.0; 3], 0.0, 1.5).with_maps([NO_TEX, NO_TEX, green], uv));
    assert!(lit[0] > 200 && lit[1] > 200, "white emission is white {lit:?}");
    assert!(mapped[1] > mapped[0] + 60, "emission image tints the glow {mapped:?}");
}

#[test]
fn roughness_texture_sharpens_the_highlight() {
    let Some(mut card) = Card::new() else { return };
    let smooth = card.atlas.add(1, 1, &[26, 26, 26, 255]);
    // A metal card lit head-on: only the specular highlight, which roughness controls.
    card.light = [0.0, 1.0, 0.0];
    let dull = card.pixel(|t, _| t.with_pbr(1.0, 1.0));
    let shiny = card.pixel(|t, uv| t.with_pbr(1.0, 1.0).with_maps([smooth, NO_TEX, NO_TEX], uv));
    let sum = |c: [u8; 3]| c.iter().map(|&v| v as u32).sum::<u32>();
    assert_ne!(sum(shiny), sum(dull), "the roughness image changes the shading");
    assert!(sum(shiny) > sum(dull), "smoother reflects more light here: {shiny:?} vs {dull:?}");
}

#[test]
fn metallic_texture_darkens_the_diffuse() {
    let Some(mut card) = Card::new() else { return };
    let full = card.atlas.add(1, 1, &[255, 255, 255, 255]);
    // Metal has no diffuse: a white metallic image removes the matte grey.
    let matte = card.pixel(|t, _| t.with_pbr(0.6, 1.0));
    let metal = card.pixel(|t, uv| t.with_pbr(0.6, 1.0).with_maps([NO_TEX, full, NO_TEX], uv));
    let plain = card.pixel(|t, _| t.with_pbr(0.6, 0.0));
    let sum = |c: [u8; 3]| c.iter().map(|&v| v as u32).sum::<u32>();
    assert_eq!(sum(metal), sum(matte), "a white metallic image keeps metallic 1");
    assert!(sum(metal) < sum(plain), "metal loses the diffuse term: {metal:?} vs {plain:?}");
}
