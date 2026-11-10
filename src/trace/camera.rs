//! The view a kernel is dispatched with: one uniform a frame, and the
//! display and filter choices it carries.
use super::light::Light;
use bytemuck::{Pod, Zeroable};

/// View for one frame. Screen `u, v ∈ [-0.5, 0.5]`; view-space point
/// `x = (u − pan.x) / scale`, `y = −(v / aspect − pan.y) / scale`, placed on the
/// basis `right, up, back` around `pivot` and divided by `units` (view units per
/// world unit). Perspective puts the eye `dist` along `back`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
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
    /// Film: `[transparent, world, _, _]`. `world` set: a ray that meets
    /// nothing is the world's radiance (`env`), averaged as light, not the
    /// theme's sky laid over. With `transparent` set (Blender's
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
    /// The selection outline (`outline.wgsl`, Overlays ▸ Objects ▸ Outline
    /// Selected): `[on, thick, occluded alpha, pixel size]`. `thick`
    /// checks the neighbours two pixels out as well as one
    /// (`do_thick_outlines`: `U.pixelsize > 1` or an outline wider than
    /// 2); the occluded alpha is `alpha_occlu`, which nothing reads yet
    /// (see `outline.wgsl`); the pixel size is `U.pixelsize`, 0 reading as
    /// 1. `on` 0 leaves the pass out altogether.
    pub outline_fx: [f32; 4],
    /// `theme.colors.object_select`, display space — a selected object's
    /// outline.
    pub outline_pick: [f32; 4],
    /// `theme.colors.active_object` — the active object's.
    pub outline_active: [f32; 4],
    /// `theme.colors.transform` — an object being moved.
    pub outline_move: [f32; 4],
    /// Adaptive Sampling, resolved as Cycles resolves it
    /// (`BAdaptive::of`): `[threshold, min samples, step, on]`. `on` 0
    /// leaves every pixel sampled to the end, which is what phiray did
    /// before it had this at all.
    pub adapt: [f32; 4],
    /// Light Paths ▸ Clamping (`cycles.sample_clamp_direct`, `_indirect`):
    /// `[direct, indirect, 0, 0]`, `0` for no clamp (`shade.wgsl`'s
    /// `clamp_light`).
    pub limit: [f32; 4],
    /// Light Paths ▸ Max Bounces past Diffuse (`cycles.max_bounces`,
    /// `transparent_max_bounces`, `transmission_bounces`,
    /// `glossy_bounces`): `[total, transparent, transmission, glossy]`,
    /// **each plus one**, so a zeroed Camera keeps the fixed limits phiray
    /// had before these were read (eight see-through events, glossy on).
    /// [`Camera::with_depth`] writes them from the counts.
    pub depth: [u32; 4],
    /// Film ▸ Pixel Filter and Light Paths ▸ Caustics, as Cycles takes them
    /// (`scene/film.cpp`, `scene/integrator.cpp`): `[filter, width,
    /// filter glossy, caustics]`. `filter` is 1 Box, 2 Gaussian, 3
    /// Blackman-Harris, and **0 keeps phiray's own behaviour** — a one-pixel
    /// box, the 0.3 roughness floor on secondary hits, caustics on — so a
    /// zeroed Camera renders as it always did. `caustics` is bit 0
    /// Reflective, bit 1 Refractive. [`Camera::with_integrator`] writes it.
    pub integrator: [f32; 4],
    pub lights: [Light; LIGHTS],
}

impl Camera {
    /// Max Bounces as Blender's panel holds them: Total, Transparent,
    /// Transmission and Glossy (Diffuse is [`Camera::bounces`]).
    pub fn with_depth(mut self, total: u32, transparent: u32, transmission: u32, glossy: u32) -> Self {
        self.depth = [total + 1, transparent + 1, transmission + 1, glossy + 1];
        self
    }

    /// Film ▸ Pixel Filter (`filter` and its `width` in pixels), Light
    /// Paths ▸ Caustics ▸ Filter Glossy, Reflective and Refractive
    /// (`Camera::integrator`).
    pub fn with_integrator(mut self, filter: PixelFilter, width: f32, filter_glossy: f32, reflective: bool, refractive: bool) -> Self {
        let caustics = u32::from(reflective) | (u32::from(refractive) << 1);
        self.integrator = [filter as u32 as f32, width.max(0.01), filter_glossy.max(0.0), caustics as f32];
        self
    }
}

/// Film ▸ Pixel Filter ▸ Type (`cycles.pixel_filter_type`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PixelFilter {
    Box = 1,
    Gaussian = 2,
    #[default]
    BlackmanHarris = 3,
}

impl PixelFilter {
    /// The RNA identifier, as the panel and the file name it.
    pub fn of(name: &str) -> Self {
        match name {
            "BOX" => Self::Box,
            "GAUSSIAN" => Self::Gaussian,
            _ => Self::BlackmanHarris,
        }
    }
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
        self.words_with(exposure, gamma, Look::None)
    }

    /// The words with a Look (`view_settings.look`); the kernel applies it
    /// only to Filmic, whose looks are curve swaps (`view.wgsl`'s `LOOKS`).
    pub fn words_with(self, exposure: f32, gamma: f32, look: Look) -> [f32; 4] {
        [self as u32 as f32, exposure, gamma, look as u32 as f32]
    }
}

/// Blender's Filmic looks (`view_settings.look`, `config.ocio`): each is a
/// contrast curve in Filmic Log space replacing the base
/// `filmic_to_0-70_1-03` curve, so the kernel swaps the Filmic table for
/// the look's (`view.wgsl`). The AgX looks ("AgX - Punchy", …) work
/// differently and are not drawn; they read as `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Look {
    #[default]
    None,
    VeryHighContrast,
    HighContrast,
    MediumHighContrast,
    MediumLowContrast,
    LowContrast,
    VeryLowContrast,
}

impl Look {
    /// From Blender's name; "Medium Contrast" is the base curve, so it and
    /// anything unknown read as `None`.
    pub fn from_blender(name: &str) -> Self {
        match name {
            "Very High Contrast" => Self::VeryHighContrast,
            "High Contrast" => Self::HighContrast,
            "Medium High Contrast" => Self::MediumHighContrast,
            "Medium Low Contrast" => Self::MediumLowContrast,
            "Low Contrast" => Self::LowContrast,
            "Very Low Contrast" => Self::VeryLowContrast,
            _ => Self::None,
        }
    }
}

pub const LIGHTS: usize = 64;

// Every field's default is zero, and `derive(Default)` stops at arrays of 32.
impl Default for Camera {
    fn default() -> Self {
        Self::zeroed()
    }
}
