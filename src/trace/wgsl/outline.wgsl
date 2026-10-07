// Blender's selection outline over the finished image (`overlay_outline.hh`,
// Overlays ▸ Objects ▸ Outline Selected): the id edges of the selected
// objects, in the theme's colours, drawn last — the overlay engine runs
// after the shading and its anti-aliasing, so this pass follows the gather
// and SMAA and composites into the display pixel alone. A saved render
// (`Tracer::display`) keeps none of it, as Blender's own render does not.
//
// The prepass is already done: every kernel left the instance under each
// pixel (`util.wgsl`'s `outline_store`), and `classes` holds one word an
// instance — its colour class plus one, 0 for an object that takes no
// outline (`outline.rs`). Together they are Blender's `outline_id_tx`.
//
// `overlay_outline_detect_frag.glsl` without two of its parts:
//
// * the smooth-wire path, which hands a line's endpoints to Blender's own
//   anti-aliasing resolve. There is no such pass here, so this is the
//   `do_anti_aliasing == false` line — one pixel, hard-edged, which is
//   what Blender draws with Overlay Smooth Wires off.
// * the occlusion fade (`alpha_occlu`), which wants the selected objects'
//   depth where another object hides them — a prepass that traces them
//   alone. A selected object entirely behind another therefore takes no
//   outline here, where Blender gives it one at 35%.

@group(0) @binding(0) var<uniform> p: Camera;
@group(0) @binding(1) var<storage, read_write> px: array<u32>;
// One word an instance: its outline class plus one (`Outline::word`).
@group(0) @binding(2) var<storage, read> classes: array<u32>;

// `edge_case`'s bits, Blender's order.
const XPOS: u32 = 1u;
const XNEG: u32 = 2u;
const YPOS: u32 = 4u;
const YNEG: u32 = 8u;
const ALL: u32 = 15u;

// Pixel `(x, y)` clamped to the image, as a texture fetch at its edge is.
fn edge_at(x: i32, y: i32) -> u32 {
    return u32(clamp(y, 0, i32(p.h) - 1)) * p.w + u32(clamp(x, 0, i32(p.w) - 1));
}

// Blender's `outline_id_tx` at pixel `j`: the class the host gave the
// instance shown there, packed over the instance (`outline_id_pack`).
// 0 for the background and for an object that takes no outline — which is
// why the pass needs no separate test for either.
fn outline_id_at(j: u32) -> u32 {
    let id = px[25u * p.n + j];
    if (id == 0u || id > arrayLength(&classes)) { return 0u; }
    let word = classes[id - 1u];
    if (word == 0u) { return 0u; }
    return ((word - 1u) << 14u) | (id & 0x3fffu);
}

// What the four neighbours `o` pixels out say: which sides hold another id
// (`has_edge` four times) and, for a pixel holding none itself, the id to
// take the colour from — so the line lands on the background *outside* the
// object, as Blender's "make outline bleed on the background" does.
struct BEdge { sides: u32, col: u32 }

fn edges(x: i32, y: i32, o: i32, id: u32, col: u32) -> BEdge {
    var out = BEdge(0u, col);
    let around = array<u32, 4>(edge_at(x + o, y), edge_at(x - o, y), edge_at(x, y - o), edge_at(x, y + o));
    let bits = array<u32, 4>(XPOS, XNEG, YPOS, YNEG);
    for (var k = 0u; k < 4u; k++) {
        let near = outline_id_at(around[k]);
        // Screen up is the row above, so `+y` is `y - o`.
        if (out.col == 0u) { out.col = near; }
        if (near != id) { out.sides |= bits[k]; }
    }
    return out;
}

// The theme colour for a packed id's class (`color_id = ref_col >> 14`):
// 1 selected, 3 active, anything else the transform colour — Blender's own
// fall-through, which is how class 0 reaches it.
fn outline_color(col: u32) -> vec4<f32> {
    let kind = col >> 14u;
    if (kind == 1u) { return p.outline_pick; }
    if (kind == 3u) { return p.outline_active; }
    return p.outline_move;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let x = i32(i % p.w);
    let y = i32(i / p.w);
    let id = outline_id_at(i);
    // `ui_scale` pixels away, `U.pixelsize`.
    let o = i32(max(round(p.outline_fx.w), 1.0));
    var e = edges(x, y, o, id, id);
    // `do_thick_outlines`: a pixel with no edge beside it looks two out.
    if (p.outline_fx.y > 0.0 && e.sides == 0u) { e = edges(x, y, 2 * o, id, id); }
    // Nothing outlined anywhere near, or outlined on every side (inside the
    // object, or a single-pixel gap): Blender discards the fragment.
    if (e.col == 0u || e.sides == 0u || e.sides == ALL) { return; }
    let c = outline_color(e.col);
    // `frag_color *= 254 / 255`: never quite 1, so Blender's line
    // anti-aliasing has something to work with. Kept, so the colour is the
    // same one Blender blends.
    let a = clamp(c.a, 0.0, 1.0) * (254.0 / 255.0);
    let under = unpack4x8unorm(px[i]);
    px[i] = pack(c.rgb * a + under.rgb * (1.0 - a), under.a);
}
