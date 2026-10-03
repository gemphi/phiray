//! Built-in kernels. Each has a WGSL body and a CPU twin used for tests and fallback.

mod denoise;
mod skin;

pub use denoise::{Denoise, DenoiseParams};
pub use skin::{Pair, Skin, SkinParams};
