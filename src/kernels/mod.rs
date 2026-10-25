//! Built-in kernels. Each has a WGSL body and a CPU twin used for tests and fallback.

mod denoise;
mod image;
mod skin;

pub use denoise::{Denoise, DenoiseParams};
pub use image::{Bilateral, BilateralParams, Convolve, ConvolveParams, ImageBilateralFilter, ImageConvolution, MatMul, MatMulParams, MatrixMultiplication};
pub use skin::{Pair, Skin, SkinParams};
