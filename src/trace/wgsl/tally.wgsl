// Per-pixel counters for a measurement (`trace/tally.rs`): how many rays of
// each kind a pixel's samples sent and how much of the trees they walked.
// `TALLY` is a constant (`tally_on.wgsl` with the crate's `tally` feature,
// `tally_off.wgsl` without), so a render built without it compiles every
// line below to nothing: the counters cost a measured kernel no time.
//
// Plane `k` of pixel `i` is word `34n + 1 + k n + i` of the image buffer,
// past the gather's count (`Tracer::PX_LEN`); `Tracer::TALLY_LEN` planes.
// A pixel is one thread's in a dispatch, so a plain add is race-free.
const TALLY_PATH: u32 = 0u;
const TALLY_CAMERA: u32 = 1u;
const TALLY_CUTOUT: u32 = 2u;
const TALLY_GLASS: u32 = 3u;
const TALLY_BOUNCE: u32 = 4u;
const TALLY_AO: u32 = 5u;
const TALLY_GLOSSY: u32 = 6u;
const TALLY_SHADE: u32 = 7u;
// 8..12: the shadow ray of `direct`'s light `k` (key or lamp, rim, glow, world).
const TALLY_SHADOW: u32 = 8u;
const TALLY_NODE: u32 = 12u;
const TALLY_TRI: u32 = 13u;
const TALLY_INST: u32 = 14u;
const TALLY_NODE_SHADOW: u32 = 15u;
const TALLY_TRI_SHADOW: u32 = 16u;
const TALLY_RAY: u32 = 17u;
const TALLY_REFLECT_HIT: u32 = 18u;
// 19..28: the node visits of each kind of ray, `tally_ray`'s `kind`: the
// camera's first ray, its rays on through glass or a cut-out, a diffuse
// bounce, an occlusion ray, a glossy reflection, the four shadow rays, and
// a glossy reflection off glass (a surface that also transmits).
const TALLY_NODE_BY: u32 = 19u;
const TALLY_GLOSSY_GLASS: u32 = 29u;
const TALLY_LEN: u32 = 30u;
var<private> tally_kind: u32 = 0u;
// The pixel being counted; none until the tracer's `main` names one, so the
// other kernels that share the walk (Solid, Wire, Preview) never write.
var<private> tally_px: u32 = 0xffffffffu;
// The walk's own counts, gathered across `trace_blas` and flushed by `trace`.
var<private> tally_nodes: u32 = 0u;
var<private> tally_tris: u32 = 0u;
var<private> tally_insts: u32 = 0u;
fn tally(k: u32, by: u32) {
    if (TALLY && tally_px != 0xffffffffu) {
        let at = 34u * p.n + 1u + k * p.n + tally_px;
        px[at] = px[at] + by;
    }
}
// A new pixel: its planes cleared on a new image, as the image's are.
fn tally_begin(i: u32) {
    if (TALLY) {
        tally_px = i;
        if (p.frame == 0u) {
            for (var k = 0u; k < TALLY_LEN; k++) { px[34u * p.n + 1u + k * p.n + i] = 0u; }
        }
        tally(TALLY_PATH, 1u);
    }
}
// The ray about to be walked: counted under `k`, its nodes under `kind`.
fn tally_ray(k: u32, kind: u32) {
    if (TALLY) {
        tally(k, 1u);
        tally_kind = kind;
    }
}
fn tally_walk_begin() {
    if (TALLY) {
        tally_nodes = 0u;
        tally_tris = 0u;
        tally_insts = 0u;
    }
}
fn tally_walk_end(first: bool) {
    if (TALLY) {
        tally(TALLY_RAY, 1u);
        tally(TALLY_NODE, tally_nodes);
        tally(TALLY_TRI, tally_tris);
        tally(TALLY_INST, tally_insts);
        tally(TALLY_NODE_BY + tally_kind, tally_nodes);
        if (first) {
            tally(TALLY_NODE_SHADOW, tally_nodes);
            tally(TALLY_TRI_SHADOW, tally_tris);
        }
    }
}
