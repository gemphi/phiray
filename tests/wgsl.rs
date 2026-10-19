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
