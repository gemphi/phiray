//! What this machine's GPU is and what it allows:
//! `cargo run --release -p phiray --example limits`

fn main() {
    println!("adapters, best first:");
    for a in phiray::GpuAdapter::distinct(phiray::GpuAdapters::all()) {
        println!("  {} ({}, {}) — {} MB per binding", a.name, a.kind_label(), a.backend_label(), a.binding >> 20);
    }
    let Some(gpu) = phiray::Gpu::new() else {
        return println!("no GPU adapter — phiray would fall back to the mesh painter");
    };
    let info = gpu.info.clone();
    println!("GPU      : {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    println!("driver   : {} {}", info.driver, info.driver_info);
    let l = gpu.device.limits();
    println!("storage binding : {} MB", l.max_storage_buffer_binding_size >> 20);
    println!("buffer          : {} MB", l.max_buffer_size >> 20);
    println!("storage buffers : {} per stage", l.max_storage_buffers_per_shader_stage);
    println!("workgroup       : {} invocations", l.max_compute_invocations_per_workgroup);
    // What that allows: faces are 16 bytes, vertices 32.
    let chunks = phiray::Tracer::chunks_for(&gpu) as u64;
    let faces = (l.max_storage_buffer_binding_size as u64 * chunks) / 16;
    println!("\nphiray can hold {} bindings per array → about {:.1} M triangles", chunks, faces as f64 / 1e6);
}
