//! Every kernel's WGSL parses and type-checks, with no GPU in the room.
//!
//! naga is the front end wgpu itself hands the source to, so this is the
//! same check `create_shader_module` makes — just reachable from a test
//! machine that has no adapter.

use phiray::bvh::{Clear, Count, Karras, Morton, Offset, Refit, Scatter, Seed, Spread};
use phiray::kernel::Kernel;

fn check(name: &str, src: String) {
    let module = match naga::front::wgsl::parse_str(&src) {
        Ok(m) => m,
        Err(e) => panic!("{name}: WGSL did not parse:\n{}", e.emit_to_string(&src)),
    };
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    if let Err(e) = validator.validate(&module) {
        panic!("{name}: WGSL did not validate:\n{e:?}");
    }
}

#[test]
fn every_bvh_kernel_is_valid_wgsl() {
    check(Morton::NAME, Morton::source());
    check(Karras::NAME, Karras::source());
    check(Refit::NAME, Refit::source());
    check(Seed::NAME, Seed::source());
    check(Clear::NAME, Clear::source());
    check(Count::NAME, Count::source());
    check(Offset::NAME, Offset::source());
    check(Scatter::NAME, Scatter::source());
    check(Spread::NAME, Spread::source());
}

/// The kernel's traversal stacks are as deep as the builder caps a tree
/// (`bvh::STACK`: deeper subtrees become leaves). The number lives in Rust,
/// the array in WGSL, and a tree deeper than the array walks wrong, not slow.
#[test]
fn the_walks_stacks_match_the_builders_cap() {
    let src = include_str!("../src/trace/wgsl/bvh.wgsl");
    let decl = format!("var stack: array<u32, {}>;", phiray::bvh::STACK);
    assert_eq!(src.matches(&decl).count(), 2, "one stack each for the top tree's walk and an instance's");
}

/// Every tracer kernel, as `create_shader_module` would see it: the
/// prelude, the camera, the bindings and the body concatenated.
#[test]
fn every_trace_kernel_is_valid_wgsl() {
    use phiray::trace::{Adapt, AdaptClear, AdaptDilate, AdaptGather, Edge, Fx, Preview, Preview4, Show, Smaa, Solid, Solid4, Taa, Trace, Trace4, Wire, Wire4};
    check(Trace::NAME, Trace::source());
    check(Trace4::NAME, Trace4::source());
    check(Solid::NAME, Solid::source());
    check(Solid4::NAME, Solid4::source());
    check(Wire::NAME, Wire::source());
    check(Wire4::NAME, Wire4::source());
    check(Preview::NAME, Preview::source());
    check(Preview4::NAME, Preview4::source());
    check(Fx::NAME, Fx::source());
    check(Taa::NAME, Taa::source());
    check(Smaa::NAME, Smaa::source());
    check(Edge::NAME, Edge::source());
    check(Show::NAME, Show::source());
    check(Adapt::NAME, Adapt::source());
    check(AdaptDilate::NAME, AdaptDilate::source());
    check(AdaptClear::NAME, AdaptClear::source());
    check(AdaptGather::NAME, AdaptGather::source());
}
