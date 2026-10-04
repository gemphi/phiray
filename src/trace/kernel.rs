//! The tracer's data types (the kernels are in `shader.rs`).

use crate::bvh::Aabb;
use bytemuck::{Pod, Zeroable};

/// A triangle: vertex normals, linear base colour (alpha = Principled Alpha),
/// PBR inputs and optional texture (160 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Tri {
    pub a: [f32; 3],
    pub rough: f32,
    pub na: [f32; 3],
    /// Transmission weight (glass).
    pub trans: f32,
    pub b: [f32; 3],
    pub metal: f32,
    pub nb: [f32; 3],
    /// Index of refraction.
    pub ior: f32,
    pub c: [f32; 3],
    pub _c: f32,
    pub nc: [f32; 3],
    pub _nc: f32,
    pub color: [f32; 4],
    /// Corner UVs (u of a, b, c) and texture id (`NO_TEX` = none).
    pub us: [f32; 3],
    pub tex: u32,
    /// Corner UVs (v of a, b, c).
    pub vs: [f32; 3],
    /// Normal map texture id (`NO_TEX` = none).
    pub nmap: u32,
    /// Emitted radiance (linear).
    pub emit: [f32; 3],
    /// Normal Map node strength.
    pub bump: f32,
    /// Roughness / metallic / emission texture ids (`NO_TEX` = none); each
    /// multiplies the matching value (Blender replaces the socket, so the
    /// caller passes 1 there).
    pub maps: [u32; 3],
    /// Subsurface weight (Principled "Subsurface Weight").
    pub sss: f32,
    /// Scattering distance per channel in world units (radius × scale).
    pub sss_radius: [f32; 3],
    /// Sheen weight, its roughness and its tint (fabric retro-reflection).
    pub sheen: f32,
    pub sheen_rough: f32,
    pub sheen_tint: [f32; 3],
    /// Coat weight and roughness (clear lacquer over the base).
    pub coat: f32,
    pub coat_rough: f32,
    /// Anisotropy and its rotation; carried for parity, not yet shaded
    /// (per-triangle tangents are needed first).
    pub aniso: f32,
    pub aniso_rot: f32,
    /// A second lobe (`Mat::second`, `mix`): the id of another material
    /// in the scene and the weight it takes; `NO_MAT`, 0 for one lobe.
    pub second: u32,
    pub mix: f32,
}

/// `Tri::tex` value for untextured triangles.
pub const NO_TEX: u32 = u32::MAX;
pub use super::mat::NO_MAT;

impl Tri {
    pub fn new(p: [[f32; 3]; 3], n: [[f32; 3]; 3], color: [f32; 4]) -> Self {
        Self { a: p[0], b: p[1], c: p[2], na: n[0], nb: n[1], nc: n[2], color, rough: 0.5, ior: 1.5, tex: NO_TEX, nmap: NO_TEX, maps: [NO_TEX; 3], sheen_rough: 0.5, sheen_tint: [1.0; 3], coat_rough: 0.03, sss_radius: [0.05, 0.01, 0.005], second: NO_MAT, ..Default::default() }
    }

    /// Texture `tex` (from [`Atlas::add`]) mapped by corner UVs; it multiplies `color`.
    pub fn with_texture(mut self, tex: u32, uv: [[f32; 2]; 3]) -> Self {
        self.tex = tex;
        self.us = [uv[0][0], uv[1][0], uv[2][0]];
        self.vs = [uv[0][1], uv[1][1], uv[2][1]];
        self
    }

    /// Tangent-space normal map (atlas id, raw bytes) with the node's strength;
    /// uses the same corner UVs as [`Self::with_texture`].
    pub fn with_normal_map(mut self, id: u32, strength: f32, uv: [[f32; 2]; 3]) -> Self {
        (self.nmap, self.bump) = (id, strength);
        self.us = [uv[0][0], uv[1][0], uv[2][0]];
        self.vs = [uv[0][1], uv[1][1], uv[2][1]];
        self
    }

    /// Emission (colour × strength), transmission weight and IOR.
    pub fn with_optics(mut self, emit: [f32; 3], trans: f32, ior: f32) -> Self {
        (self.emit, self.trans, self.ior) = (emit, trans.clamp(0.0, 1.0), ior.max(1.0));
        self
    }

    /// Roughness / metallic / emission-colour textures (atlas ids, `NO_TEX` for
    /// none); each multiplies its value, with the same corner UVs as
    /// [`Self::with_texture`].
    pub fn with_maps(mut self, maps: [u32; 3], uv: [[f32; 2]; 3]) -> Self {
        self.maps = maps;
        self.us = [uv[0][0], uv[1][0], uv[2][0]];
        self.vs = [uv[0][1], uv[1][1], uv[2][1]];
        self
    }

    /// Principled roughness / metallic.
    pub fn with_pbr(mut self, rough: f32, metal: f32) -> Self {
        (self.rough, self.metal) = (rough.clamp(0.02, 1.0), metal.clamp(0.0, 1.0));
        self
    }

    /// Subsurface weight and its per-channel scattering distance in world
    /// units (Principled "Subsurface Radius" × "Subsurface Scale").
    pub fn with_subsurface(mut self, weight: f32, radius: [f32; 3]) -> Self {
        self.sss = weight.clamp(0.0, 1.0);
        self.sss_radius = radius.map(|r| r.max(0.0));
        self
    }

    /// Sheen weight, roughness and tint (velvet, cotton fuzz, brushed wool).
    pub fn with_sheen(mut self, weight: f32, rough: f32, tint: [f32; 3]) -> Self {
        (self.sheen, self.sheen_rough, self.sheen_tint) = (weight.clamp(0.0, 1.0), rough.clamp(0.07, 1.0), tint);
        self
    }

    /// Coat weight and roughness (clearcoat over the base lobes).
    pub fn with_coat(mut self, weight: f32, rough: f32) -> Self {
        (self.coat, self.coat_rough) = (weight.clamp(0.0, 1.0), rough.clamp(0.02, 1.0));
        self
    }

    /// Anisotropy and its rotation (carried for parity; not yet shaded).
    pub fn with_anisotropy(mut self, amount: f32, rotation: f32) -> Self {
        (self.aniso, self.aniso_rot) = (amount.clamp(-1.0, 1.0), rotation);
        self
    }

    /// A Mix Shader's other closure: material `second` (`Scene::material`)
    /// at weight `mix` (0 all this, 1 all the other), the UVs shared.
    pub fn with_second(mut self, second: u32, mix: f32) -> Self {
        (self.second, self.mix) = (second, mix.clamp(0.0, 1.0));
        self
    }

    pub fn bounds(&self) -> Aabb {
        Aabb::of_tri(self.a, self.b, self.c)
    }
}

/// View for one frame. Screen `u, v ∈ [-0.5, 0.5]`; view-space point
/// `x = (u − pan.x) / scale`, `y = −(v / aspect − pan.y) / scale`, placed on the
/// basis `right, up, back` around `pivot` and divided by `units` (view units per
/// world unit). Perspective puts the eye `dist` along `back`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Camera {
    pub n: u32,
    pub groups_x: u32,
    pub w: u32,
    pub h: u32,
    pub right: [f32; 3],
    pub persp: u32,
    pub up: [f32; 3],
    pub scale: f32,
    pub back: [f32; 3],
    pub units: f32,
    pub pivot: [f32; 3],
    pub aspect: f32,
    /// Direction *towards* the sun (world).
    pub sun: [f32; 3],
    pub dist: f32,
    /// Background (display space). `a` is not opacity: it is added to the
    /// sky at the left edge and fades out by mid-width (`sky_at`) — a
    /// theme's sweep; 0 for a flat sky.
    pub sky: [f32; 4],
    pub pan: [f32; 2],
    /// AO reach in world units (0 = auto).
    pub ao: f32,
    /// Ray offset (set by the tracer).
    pub eps: f32,
    /// Accumulated sample index (set by the tracer).
    pub frame: u32,
    /// Floor grid spacing in world units (0 = no grid).
    pub grid: f32,
    /// Used entries of `lights`; 0 = studio three-point rig around the view.
    pub count: u32,
    /// Emissive triangles in the scene (set by the tracer); > 0 lets
    /// indirect rays pick up light from glowing meshes.
    pub glow: u32,
    /// Grid line colour (display space, alpha = strength).
    pub grid_color: [f32; 4],
    /// World radiance (linear) for ambient light and reflections.
    pub env: [f32; 4],
    /// Where the second chunk starts for nodes, faces, vertices; atlas start in `data`
    /// (set by the tracer).
    pub split: [u32; 4],
    /// Colour management (`Scene.view_settings`): `[transform, exposure,
    /// gamma, _]` — transform 0 Standard, 1 Filmic, 2 AgX (see [`View`]);
    /// exposure in stops; gamma 0 means 1.
    pub view: [f32; 4],
    /// Equirectangular environment map: the atlas id from [`Atlas::add_hdr`]
    /// **plus one**, so the default `0` means the world is a flat colour.
    pub envmap: u32,
    /// The world Mapping node's rotation around Z, in radians.
    pub env_rot: f32,
    /// Diffuse bounces for indirect light (Blender Light Paths → Max Bounces →
    /// Diffuse). 0 takes the default, which is 1 — see `light.wgsl`'s
    /// `indirect` for why that is not yet Blender's 4.
    pub bounces: u32,
    /// Sampling table for the environment image (`Atlas::add_env_cdf`) plus
    /// one; 0 means the world is sampled only by chance.
    pub env_cdf: u32,
    /// Aperture radius in world units — `lens / (2 f-stop)`, which
    /// `phiphx::BlendDof::aperture` works out. 0 is a pinhole, and is the
    /// default, so depth of field costs nothing when it is off.
    pub aperture: f32,
    /// Distance from the eye to the plane of focus, world units. The focus
    /// surface is a *plane* perpendicular to the view axis, not a sphere.
    pub focus: f32,
    /// Aperture blades: 0 is a round iris, 3 or more gives that many edges.
    pub blades: u32,
    /// Aperture rotation, radians.
    pub blade_spin: f32,
    /// Anamorphic squeeze (`aperture_ratio`): the bokeh is divided by this
    /// across. 0 reads as 1, so a zeroed camera is not a degenerate lens.
    pub aperture_ratio: f32,
    /// The environment image lights the scene but is not shown: a camera ray
    /// that misses sees `sky` instead. Blender's Material Preview does this
    /// (`studiolight_background_alpha` defaults to 0) — the forest lights the
    /// model and the background stays the viewport grey.
    pub env_hidden: u32,
    /// X-ray (`View3DShading.show_xray` / `show_xray_wireframe`), Solid and
    /// Wireframe only: how much every surface lets through — `1 −
    /// xray_alpha` while X-ray is on, 0 (opaque: X-ray off) by default.
    pub see_through: f32,
    /// The viewport's guides as its Overlays popover asks: bits 1, 2 and 4
    /// leave out the floor's lines, the X axis and the Y axis (so the
    /// default, 0, draws all three); bit 8 adds the Z axis, which Blender
    /// ships off.
    pub grid_hide: u32,
    /// This image as a tile of a larger frame: its corner `[x, y]` in the
    /// frame and the frame's `[w, h]` — the rays, the pixel size and the
    /// sky's sweep are the frame's. All zero: the image is the frame.
    pub tile: [u32; 4],
    /// Film: `[transparent, _, _, _]`. With `transparent` set (Blender's
    /// Film ▸ Transparent, `film_transparent`), a camera ray that meets
    /// nothing leaves its pixel black with alpha 0 instead of the sky, so a
    /// PNG or EXR with an alpha channel comes out with the background cut
    /// away; the pixel's alpha is the share of its samples that met
    /// something. Zero: the sky, alpha 1.
    pub film: [u32; 4],
    /// Material Preview's environment lighting (`ibl.rs`): `[irradiance id
    /// + 1, first prefiltered id + 1, roughness levels, _]`; zero when the
    /// world has no image — the kernel then lights from the lamps or the rig.
    pub preview: [u32; 4],
    /// The pixels this dispatch traces: `[first, count, _, _]`, all of them
    /// when `count` is 0. A sample is dispatched in slices (`Tracer::
    /// sample_in`) so a newer view can take over between them instead of
    /// waiting out a whole frame.
    pub slice: [u32; 4],
    /// The instance table in `data` (set by the tracer): `[first word,
    /// count, first word of the vertices' colour attribute — 0 when no
    /// object has one (`Scene::paint`), _]` — each instance's placement
    /// and the root of its tree (`Instance::words`), then the top tree's
    /// slot order (`Trees`).
    pub insts: [u32; 4],
    /// The visibility prepass (set by the tracer): `[drawn, pixels a row
    /// of its hits spans, orthographic depth reach bits, perspective near
    /// plane bits]`; `drawn` 0 traces the primary ray instead.
    pub vis: [u32; 4],
    /// Workbench's anti-aliasing (set by the tracer, Solid and Wireframe):
    /// this sample's jitter `[x, y]` in `−1 ..= 1` (half a pixel either
    /// way), the filter weight accumulated over the samples so far, _.
    pub taa: [f32; 4],
    /// Workbench's Solid effects (`fx.wgsl`): `[object outline on,
    /// curvature ridge, curvature valley, pixel size]`. The curvature
    /// words are `CavityEffect::init`'s `0.5 / ridge_factor²` and `0.7 /
    /// valley_factor²`, 0 when Cavity is off or not Curvature; the pixel
    /// size is `U.pixelsize` (0 reads as 1), how far the neighbours sit.
    pub fx: [f32; 4],
    /// `object_outline_color`, scene-linear (Blender's default is black).
    pub outline: [f32; 4],
    /// Solid's Lighting and Color (`View3DShading.light`, `color_type`):
    /// `[light: 0 Studio, 1 MatCap (drawn as Studio), 2 Flat; color: 0
    /// Material, 1 Random, 2 Single, 3 Texture (as Material), 4 Object, 5
    /// Attribute (the vertex colour, Material where a mesh has none);
    /// backface culling on
    /// (`V3D_SHADING_BACKFACE_CULLING`: back faces are traced through);
    /// specular lighting OFF (`V3D_SHADING_SPECULAR_HIGHLIGHT` clear:
    /// `get_world_lighting` without its highlight)]`. The colour word is
    /// `eV3DShadingColorType`'s numbering; the light word puts Studio at 0
    /// (Blender's `eV3DShadingLightingMode` has Flat there) and the last
    /// two are the flags inverted where Blender's default sets them, so a
    /// zeroed camera draws as before.
    pub shade: [u32; 4],
    /// Color ▸ Single's colour (`single_color`), scene-linear.
    pub single: [f32; 4],
    /// Cavity ▸ World — Workbench's screen-space ambient occlusion
    /// (`cavity_compute`, `fx.wgsl`): `[cavity_valley_factor,
    /// cavity_ridge_factor, matcap_ssao_attenuation, matcap_ssao_distance]`
    /// (`CavityEffect::init`'s `cavity_*` words; the distance in Blender
    /// units).
    pub cavity: [f32; 4],
    /// Its sampling: `[matcap_ssao_samples — 0 when SSAO is off,
    /// samples_len (the anti-aliasing sample count, so each sample draws
    /// its own run of the Hammersley disk), _, _]`.
    pub ssao: [u32; 4],
    /// Workbench's Shadow (`workbench_shadow.cc`, `get_shadow`): the
    /// direction the light travels, world, unit — `ShadowPass::init`'s
    /// `direction_ws` — and `shadow_intensity` (`shadow_mul`; 0 turns the
    /// shadow off). Every opaque surface casts; a point is shadowed when a
    /// ray against the direction meets one (the stencil volumes' cover).
    pub shadow: [f32; 4],
    /// Its terminator: `[shadow_shift, shadow_focus', _, _]`, the edges of
    /// `get_shadow`'s `smoothstep` over `−N · direction` — `focus'` is
    /// `1 − clamp(focus) (1 − shift)`, kept above `shift`.
    pub shadow_fade: [f32; 4],
    /// MatCap lighting's images in the atlas (`get_matcap_lighting`):
    /// `[diffuse id + 1, specular id + 1 — 0 for none, flipped
    /// (`V3D_SHADING_MATCAP_FLIP_X`), _]`; a 0 diffuse word under MatCap
    /// lights as Studio.
    pub matcap: [u32; 4],
    /// Workbench's Depth of Field (`workbench_effect_dof.cc`, `DofPass::
    /// init`), Solid alone: `[aperture_size — 0 for off, distance (−focus),
    /// invsensor_size per frame pixel (the frame's width times it is
    /// `dof_invsensorsize`), far (−clip_end: the depth under a pixel with
    /// nothing drawn)]`. The bokeh's shape is `blades`, `blade_spin` and
    /// `aperture_ratio`, the camera's own.
    pub dof: [f32; 4],
    pub lights: [Light; LIGHTS],
}

/// Blender's view transforms (`view_settings.view_transform`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum View {
    #[default]
    Standard,
    Filmic,
    AgX,
}

impl View {
    /// From Blender's name; anything unknown reads as Standard.
    pub fn from_blender(name: &str) -> Self {
        match name {
            "Filmic" | "Filmic Log" => Self::Filmic,
            "AgX" => Self::AgX,
            _ => Self::Standard,
        }
    }

    /// The `Camera::view` words for this transform.
    pub fn words(self, exposure: f32, gamma: f32) -> [f32; 4] {
        [self as u32 as f32, exposure, gamma, 0.0]
    }
}

pub const LIGHTS: usize = 16;

/// Pack linear RGB into Radiance RGBE: three 8-bit mantissas and one shared
/// exponent, biased by 128 (`IMB_radiance_hdr.cc`).
fn rgbe(r: f32, g: f32, b: f32) -> u32 {
    let peak = r.max(g).max(b);
    if !(peak > 1e-32) {
        return 0;
    }
    // `peak / 2^exp` lands in (0.5, 1], so every mantissa fits in one byte.
    let exp = peak.log2().ceil().clamp(-127.0, 127.0);
    let scale = 256.0 / exp.exp2();
    let q = |c: f32| (c * scale).clamp(0.0, 255.0) as u32;
    q(r) | (q(g) << 8) | (q(b) << 16) | (((exp + 128.0) as u32).min(255) << 24)
}

/// A Blender light (`kind`: 0 sun, 1 point, 2 spot, 3 area).
/// `color` is colour × strength (sun, W/m²) or colour × power (W).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Light {
    pub pos: [f32; 3],
    pub kind: u32,
    pub color: [f32; 3],
    /// Soft-shadow size: sun angle (rad) or radius (m).
    pub size: f32,
    /// Direction the light shines (sun, spot, area).
    pub dir: [f32; 3],
    /// Spot: cosine of the half cone angle.
    pub cone: f32,
}

/// All textures in one `u32` buffer: `[count, (offset, w, h) × count, texels…]`,
/// texels packed RGBA8 (sRGB), row 0 at the top.
///
/// Each entry's head is `[first texel, width, height, levels]`: an image
/// added with [`Atlas::add`] is followed by its mip chain, one entry per
/// level, and `levels` counts them (itself included) so a kernel can sample
/// the level a pixel's footprint asks for (`sample_raw_lod`); every other
/// entry — raw words, an environment, a prefiltered copy — has one. On the
/// GPU the head is three words, the level count in the top byte of the width.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Atlas {
    heads: Vec<[u32; 4]>,
    texels: Vec<u32>,
}

/// Grid the environment's sampling table is built on. Coarse on purpose: the
/// table only has to steer rays towards the bright parts, and the estimator
/// divides by the same table's own density, so a coarse grid costs variance
/// and never accuracy.
pub const ENV_CDF_W: u32 = 128;
pub const ENV_CDF_H: u32 = 64;

impl Atlas {
    /// Add the sampling distribution of an equirectangular environment image:
    /// one conditional CDF over columns per row, then a final row holding the
    /// marginal CDF over rows. Cell weights are luminance × sin θ, so a
    /// direction is drawn in proportion to the light it actually carries.
    /// Returns an id like any other entry; read it with `bitcast<f32>`.
    pub fn add_env_cdf(&mut self, w: u32, h: u32, rgb: &[f32]) -> u32 {
        let (cw, ch) = (ENV_CDF_W.min(w.max(1)), ENV_CDF_H.min(h.max(1)));
        let lum = |x: u32, y: u32| {
            let i = ((y.min(h - 1) * w + x.min(w - 1)) * 3) as usize;
            match rgb.get(i..i + 3) {
                Some(c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2],
                None => 0.0,
            }
        };
        // Cell weights: the mean of the source pixels it covers, times the
        // solid angle its latitude band subtends.
        let mut cells = vec![0.0f32; (cw * ch) as usize];
        for gy in 0..ch {
            let sin = (std::f32::consts::PI * (gy as f32 + 0.5) / ch as f32).sin();
            for gx in 0..cw {
                let (x0, x1) = (gx * w / cw, ((gx + 1) * w / cw).max(gx * w / cw + 1));
                let (y0, y1) = (gy * h / ch, ((gy + 1) * h / ch).max(gy * h / ch + 1));
                let mut sum = 0.0f32;
                let mut n = 0u32;
                for y in y0..y1.min(h) {
                    for x in x0..x1.min(w) {
                        sum += lum(x, y);
                        n += 1;
                    }
                }
                cells[(gy * cw + gx) as usize] = sum / n.max(1) as f32 * sin;
            }
        }
        let mut words = vec![0u32; ((ch + 1) * cw) as usize];
        let mut rows = vec![0.0f32; ch as usize];
        for gy in 0..ch {
            let row = &cells[(gy * cw) as usize..((gy + 1) * cw) as usize];
            let total: f32 = row.iter().sum();
            rows[gy as usize] = total;
            let mut acc = 0.0f32;
            for gx in 0..cw {
                // A dark row still needs a usable CDF; its marginal weight is
                // zero, so it is never drawn from.
                acc += if total > 0.0 { row[gx as usize] / total } else { 1.0 / cw as f32 };
                words[(gy * cw + gx) as usize] = acc.min(1.0).to_bits();
            }
            words[(gy * cw + cw - 1) as usize] = 1.0f32.to_bits();
        }
        let total: f32 = rows.iter().sum();
        let mut acc = 0.0f32;
        for gy in 0..ch {
            acc += if total > 0.0 { rows[gy as usize] / total } else { 1.0 / ch as f32 };
            words[(ch * cw + gy) as usize] = acc.min(1.0).to_bits();
        }
        for gx in ch..cw {
            words[(ch * cw + gx) as usize] = 1.0f32.to_bits();
        }
        self.add_bits(cw, ch + 1, &words)
    }

    /// Add raw words as an entry, addressed like an image but read as data.
    pub fn add_bits(&mut self, w: u32, h: u32, words: &[u32]) -> u32 {
        let id = self.heads.len() as u32;
        self.heads.push([self.texels.len() as u32, w, h, 1]);
        self.texels.extend_from_slice(words);
        self.texels.resize(self.heads.last().expect("just pushed")[0] as usize + (w * h) as usize, 0);
        id
    }

    /// Add what Material Preview needs to light a surface from the
    /// environment `rgb` (`w × h`, linear) in one dispatch: its irradiance
    /// coefficients and its prefiltered copies (`ibl.rs`). The image itself
    /// is expected to be in the atlas already, as roughness 0.
    pub fn add_env_lighting(&mut self, w: u32, h: u32, rgb: &[f32]) -> super::ibl::EnvLighting {
        use super::ibl::{Ibl, LEVELS, Panorama};
        let env = Panorama::new(w, h, rgb);
        let coefficients = Ibl::irradiance(&env);
        let words: Vec<u32> = coefficients.iter().flatten().map(|v| v.to_bits()).collect();
        let irradiance = self.add_bits(27, 1, &words);
        let chain = Ibl::chain(&env);
        let mut prefiltered = 0;
        for level in 1..LEVELS {
            let copy = Ibl::prefiltered(&chain, level as f32 / (LEVELS - 1) as f32);
            let id = self.add_hdr(copy.w, copy.h, &copy.rgb);
            if level == 1 {
                prefiltered = id;
            }
        }
        super::ibl::EnvLighting { irradiance, prefiltered, levels: LEVELS }
    }

    /// Add a linear float RGB image as RGBE (one shared exponent per texel),
    /// which is how a 4-byte texel can still hold a sun a thousand times
    /// brighter than white. Sampled by `sample_env` in the kernel.
    pub fn add_hdr(&mut self, w: u32, h: u32, rgb: &[f32]) -> u32 {
        let mut texels = Vec::with_capacity((w * h) as usize);
        for px in rgb.chunks_exact(3) {
            texels.push(rgbe(px[0], px[1], px[2]));
        }
        texels.resize((w * h) as usize, 0);
        let id = self.heads.len() as u32;
        self.heads.push([self.texels.len() as u32, w, h, 1]);
        self.texels.extend(texels);
        id
    }

    /// Add an image, box-downsampled by a whole factor until neither side
    /// exceeds `max_side`. Blender's Preferences → Viewport → Textures →
    /// *Limit Size* (`U.glreslimit`) does the same on upload; without it a
    /// character's 4K maps put the whole atlas past one storage binding and
    /// the GPU declined the scene outright.
    pub fn add_bounded(&mut self, w: u32, h: u32, rgba: &[u8], max_side: u32) -> u32 {
        let mut f = 1u32;
        while (w / f).max(h / f) > max_side.max(1) && f < w.max(h) {
            f *= 2;
        }
        if f == 1 {
            return self.add(w, h, rgba);
        }
        let (ow, oh) = ((w / f).max(1), (h / f).max(1));
        let mut out = vec![0u8; (ow * oh * 4) as usize];
        let at = |x: u32, y: u32| ((y * w + x) * 4) as usize;
        for y in 0..oh {
            for x in 0..ow {
                let mut sum = [0u32; 4];
                let mut n = 0u32;
                for dy in 0..f {
                    for dx in 0..f {
                        let (sx, sy) = (x * f + dx, y * f + dy);
                        if sx < w && sy < h && at(sx, sy) + 4 <= rgba.len() {
                            let p = &rgba[at(sx, sy)..at(sx, sy) + 4];
                            for k in 0..4 { sum[k] += p[k] as u32; }
                            n += 1;
                        }
                    }
                }
                let o = ((y * ow + x) * 4) as usize;
                for k in 0..4 { out[o + k] = if n == 0 { 0 } else { (sum[k] / n) as u8 }; }
            }
        }
        self.add(ow, oh, &out)
    }

    /// Bytes this atlas will occupy on the GPU (`words()` × 4).
    pub fn bytes(&self) -> u64 {
        (1 + 3 * self.heads.len() + self.texels.len()) as u64 * 4
    }

    /// Add an image with its mip chain: each level half the one before,
    /// every texel the mean of four, down to one texel. The image's id is
    /// returned; its levels follow it, one id each, and its head counts them.
    pub fn add(&mut self, w: u32, h: u32, rgba: &[u8]) -> u32 {
        let (w, h) = (w.max(1), h.max(1));
        let px = (w * h) as usize;
        let mut level: Vec<u32> = rgba.chunks_exact(4).take(px).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        level.resize(px, 0xff00ffff);
        let id = self.heads.len() as u32;
        let levels = 1 + w.max(h).ilog2();
        let (mut lw, mut lh) = (w, h);
        for k in 0..levels {
            self.heads.push([self.texels.len() as u32, lw, lh, if k == 0 { levels } else { 1 }]);
            self.texels.extend_from_slice(&level);
            if k + 1 < levels {
                level = Self::halved(lw, lh, &level);
                lw = (lw / 2).max(1);
                lh = (lh / 2).max(1);
            }
        }
        id
    }

    /// `texels` (`w × h`, RGBA8 words) at half size, each the mean of the
    /// four it covers — of two or one at an odd edge.
    fn halved(w: u32, h: u32, texels: &[u32]) -> Vec<u32> {
        let (ow, oh) = ((w / 2).max(1), (h / 2).max(1));
        let at = |x: u32, y: u32| texels[(y.min(h - 1) * w + x.min(w - 1)) as usize].to_le_bytes();
        let mut out = Vec::with_capacity((ow * oh) as usize);
        for y in 0..oh {
            for x in 0..ow {
                let mut sum = [0u32; 4];
                let mut n = 0u32;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                    if sx < w && sy < h {
                        let c = at(sx, sy);
                        for k in 0..4 {
                            sum[k] += c[k] as u32;
                        }
                        n += 1;
                    }
                }
                out.push(u32::from_le_bytes([0, 1, 2, 3].map(|k| (sum[k] / n.max(1)) as u8)));
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.heads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    pub fn words(&self) -> Vec<u32> {
        let base = 1 + 3 * self.heads.len() as u32;
        let mut out = vec![self.heads.len() as u32];
        // The level count rides in the width's top byte (`head_levels`).
        out.extend(self.heads.iter().flat_map(|h| [h[0] + base, h[1] | (h[3].min(255) << 24), h[2]]));
        out.extend_from_slice(&self.texels);
        out
    }
}
