//! The image and matrix kernels Apple ships as MetalPerformanceShaders —
//! `MPSImageConvolution`, `MPSImageBilateralFilter`,
//! `MPSMatrixMultiplication` — as WGSL, so they run wherever wgpu does:
//! Vulkan on the Windows machine, Metal on a Mac, DX12, WebGPU.
//!
//! `crates/apple/src/metal_performance_shaders` carries the three as structs
//! with the same fields and no bodies: binding shapes for a framework that
//! exists only on Apple hardware. These are the bodies. Each struct here has
//! the fields the Apple one has, a `run` that records the dispatch, and a
//! `cpu` twin the tests check the kernel against.
//!
//! Images are `w × h` of `vec4<f32>` (RGBA, linear), row-major, as
//! `Tracer`'s scratch images are, so the bilateral filter can stand in as a
//! denoiser pass over the accumulated radiance (`Tracer::denoise`) without
//! a copy. Edges clamp (`MPSImageEdgeModeClamp`, the MPS default).

use crate::gpu::{Gpu, GpuArray, Pass};
use crate::kernel::{Bind, Kernel};
use bytemuck::{Pod, Zeroable};

// ── Convolution ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ConvolveParams {
    pub n: u32,
    pub groups_x: u32,
    pub w: u32,
    pub h: u32,
    /// Kernel width and height (odd; the centre tap is `(kw / 2, kh / 2)`).
    pub kw: u32,
    pub kh: u32,
    pub _pad: [u32; 2],
}

/// `MPSImageConvolution`: every output pixel is the weighted sum of its
/// `kw × kh` neighbourhood, weights row-major, edges clamped.
pub struct Convolve;

impl Kernel for Convolve {
    const NAME: &'static str = "image.convolve";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Read, Bind::Write];
    type Params = ConvolveParams;
    const WGSL: &'static str = r#"
struct Params { n: u32, groups_x: u32, w: u32, h: u32, kw: u32, kh: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;

fn at(x: i32, y: i32) -> vec4<f32> {
    let cx = clamp(x, 0, i32(p.w) - 1);
    let cy = clamp(y, 0, i32(p.h) - 1);
    return src[u32(cy) * p.w + u32(cx)];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let x = i32(i % p.w);
    let y = i32(i / p.w);
    let hx = i32(p.kw / 2u);
    let hy = i32(p.kh / 2u);
    var acc = vec4<f32>(0.0);
    for (var ky = 0u; ky < p.kh; ky++) {
        for (var kx = 0u; kx < p.kw; kx++) {
            let wgt = weights[ky * p.kw + kx];
            acc += wgt * at(x + i32(kx) - hx, y + i32(ky) - hy);
        }
    }
    out[i] = acc;
}
"#;
}

/// The Apple struct's shape (`MPSImageConvolution { width, height, weights }`),
/// with a body.
#[derive(Debug, Clone)]
pub struct ImageConvolution {
    pub width: usize,
    pub height: usize,
    pub weights: Vec<f32>,
}

impl ImageConvolution {
    pub fn new(width: usize, height: usize, weights: &[f32]) -> Self {
        assert_eq!(weights.len(), width * height, "one weight per tap");
        Self { width, height, weights: weights.to_vec() }
    }

    /// Record the dispatch over a `w × h` image: `src` → `out`.
    pub fn run(&self, gpu: &Gpu, pass: &mut Pass, w: u32, h: u32, src: &GpuArray<[f32; 4]>, out: &GpuArray<[f32; 4]>) -> GpuArray<f32> {
        let weights = GpuArray::from(gpu, &self.weights);
        let params = ConvolveParams { n: w * h, w, h, kw: self.width as u32, kh: self.height as u32, ..Default::default() };
        pass.add::<Convolve>(gpu, w * h, params, &[weights.buffer(), src.buffer(), out.buffer()]);
        weights
    }

    /// CPU twin.
    pub fn cpu(&self, w: usize, h: usize, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let at = |x: i64, y: i64| src[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize];
        let (hx, hy) = ((self.width / 2) as i64, (self.height / 2) as i64);
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let mut acc = [0.0f32; 4];
                for ky in 0..self.height {
                    for kx in 0..self.width {
                        let wgt = self.weights[ky * self.width + kx];
                        let s = at(x + kx as i64 - hx, y + ky as i64 - hy);
                        for c in 0..4 {
                            acc[c] += wgt * s[c];
                        }
                    }
                }
                acc
            })
            .collect()
    }
}

// ── Bilateral filter ─────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BilateralParams {
    pub n: u32,
    pub groups_x: u32,
    pub w: u32,
    pub h: u32,
    pub radius: u32,
    pub _pad: u32,
    /// `1 / (2 σ²)` for the spatial and the colour Gaussians.
    pub inv2_spatial: f32,
    pub inv2_color: f32,
}

/// `MPSImageBilateralFilter`: a blur that weights each neighbour by how far
/// it is AND how different its colour is, so noise is averaged out and
/// edges are not — the same idea as the à-trous denoiser without the
/// albedo / normal guides, which makes it the right pass for an image with
/// no guide buffers (a loaded picture, a viewport without passes).
pub struct Bilateral;

impl Kernel for Bilateral {
    const NAME: &'static str = "image.bilateral";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Write];
    type Params = BilateralParams;
    const WGSL: &'static str = r#"
struct Params { n: u32, groups_x: u32, w: u32, h: u32, radius: u32, _a: u32, inv2_spatial: f32, inv2_color: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let x = i32(i % p.w);
    let y = i32(i / p.w);
    let centre = src[i];
    let r = i32(p.radius);
    var acc = vec4<f32>(0.0);
    var total = 0.0;
    for (var dy = -r; dy <= r; dy++) {
        let sy = clamp(y + dy, 0, i32(p.h) - 1);
        for (var dx = -r; dx <= r; dx++) {
            let sx = clamp(x + dx, 0, i32(p.w) - 1);
            let s = src[u32(sy) * p.w + u32(sx)];
            let d2 = f32(dx * dx + dy * dy);
            let dc = s.rgb - centre.rgb;
            let wgt = exp(-d2 * p.inv2_spatial - dot(dc, dc) * p.inv2_color);
            acc += wgt * s;
            total += wgt;
        }
    }
    out[i] = acc / max(total, 1e-12);
}
"#;
}

/// The Apple struct's shape (`MPSImageBilateralFilter { filter_radius,
/// spatial_sigma, color_sigma }`), with a body.
#[derive(Debug, Clone)]
pub struct ImageBilateralFilter {
    pub filter_radius: usize,
    pub spatial_sigma: f32,
    pub color_sigma: f32,
}

impl ImageBilateralFilter {
    pub fn new(radius: usize, spatial: f32, color: f32) -> Self {
        Self { filter_radius: radius, spatial_sigma: spatial, color_sigma: color }
    }

    fn params(&self, w: u32, h: u32) -> BilateralParams {
        let inv2 = |s: f32| 1.0 / (2.0 * s.max(1e-6) * s.max(1e-6));
        BilateralParams { n: w * h, w, h, radius: self.filter_radius as u32, inv2_spatial: inv2(self.spatial_sigma), inv2_color: inv2(self.color_sigma), ..Default::default() }
    }

    /// Record the dispatch over a `w × h` image: `src` → `out`.
    pub fn run(&self, gpu: &Gpu, pass: &mut Pass, w: u32, h: u32, src: &GpuArray<[f32; 4]>, out: &GpuArray<[f32; 4]>) {
        pass.add::<Bilateral>(gpu, w * h, self.params(w, h), &[src.buffer(), out.buffer()]);
    }

    /// CPU twin.
    pub fn cpu(&self, w: usize, h: usize, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let p = self.params(w as u32, h as u32);
        let r = self.filter_radius as i64;
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                let c = src[i];
                let (mut acc, mut total) = ([0.0f32; 4], 0.0f32);
                for dy in -r..=r {
                    let sy = (y + dy).clamp(0, h as i64 - 1) as usize;
                    for dx in -r..=r {
                        let sx = (x + dx).clamp(0, w as i64 - 1) as usize;
                        let s = src[sy * w + sx];
                        let d2 = (dx * dx + dy * dy) as f32;
                        let dc: f32 = (0..3).map(|k| (s[k] - c[k]).powi(2)).sum();
                        let wgt = (-d2 * p.inv2_spatial - dc * p.inv2_color).exp();
                        for k in 0..4 {
                            acc[k] += wgt * s[k];
                        }
                        total += wgt;
                    }
                }
                let t = total.max(1e-12);
                [acc[0] / t, acc[1] / t, acc[2] / t, acc[3] / t]
            })
            .collect()
    }
}

// ── Matrix multiplication ────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MatMulParams {
    pub n: u32,
    pub groups_x: u32,
    /// `out[m × cols] = a[m × k] · b[k × cols]`, all row-major.
    pub m: u32,
    pub cols: u32,
    pub k: u32,
    pub _pad: [u32; 3],
}

/// `MPSMatrixMultiplication`: one thread per element of the result.
/// Row-major `f32`, no tiling — the sizes this app multiplies (bone
/// palettes, colour transforms, small solves) do not reach where tiling
/// pays; the kernel is correct first and measured later.
pub struct MatMul;

impl Kernel for MatMul {
    const NAME: &'static str = "matrix.multiply";
    const BINDS: &'static [Bind] = &[Bind::Read, Bind::Read, Bind::Write];
    type Params = MatMulParams;
    const WGSL: &'static str = r#"
struct Params { n: u32, groups_x: u32, m: u32, cols: u32, k: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = item(gid, p.groups_x);
    if (i >= p.n) { return; }
    let row = i / p.cols;
    let col = i % p.cols;
    var acc = 0.0;
    for (var t = 0u; t < p.k; t++) {
        acc += a[row * p.k + t] * b[t * p.cols + col];
    }
    out[i] = acc;
}
"#;
}

/// The Apple struct's shape (`MPSMatrixMultiplication { m, n, k }`), with a
/// body. `n` keeps Apple's name for the result's column count.
#[derive(Debug, Clone)]
pub struct MatrixMultiplication {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl MatrixMultiplication {
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }

    /// Record `out = a · b`.
    pub fn run(&self, gpu: &Gpu, pass: &mut Pass, a: &GpuArray<f32>, b: &GpuArray<f32>, out: &GpuArray<f32>) {
        let (m, cols, k) = (self.m as u32, self.n as u32, self.k as u32);
        let params = MatMulParams { n: m * cols, m, cols, k, ..Default::default() };
        pass.add::<MatMul>(gpu, m * cols, params, &[a.buffer(), b.buffer(), out.buffer()]);
    }

    /// CPU twin.
    pub fn cpu(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!((a.len(), b.len()), (self.m * self.k, self.k * self.n));
        (0..self.m * self.n)
            .map(|i| {
                let (row, col) = (i / self.n, i % self.n);
                (0..self.k).map(|t| a[row * self.k + t] * b[t * self.n + col]).sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize) -> Vec<[f32; 4]> {
        (0..w * h).map(|i| { let v = (i % w) as f32 / (w - 1) as f32; [v, v * 0.5, 1.0 - v, 1.0] }).collect()
    }

    #[test]
    fn an_identity_convolution_returns_the_image() {
        let (w, h) = (5, 4);
        let img = gradient(w, h);
        let id = ImageConvolution::new(3, 3, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(id.cpu(w, h, &img), img);
    }

    #[test]
    fn a_box_blur_flattens_and_edges_clamp() {
        let (w, h) = (4, 1);
        let img: Vec<[f32; 4]> = [0.0, 0.0, 1.0, 1.0].iter().map(|&v| [v, v, v, 1.0]).collect();
        let b = ImageConvolution::new(3, 1, &[1.0 / 3.0; 3]);
        let out = b.cpu(w, h, &img);
        // Left edge: (0, 0, 0) clamped → 0. Middle: (0, 0, 1) / 3 and (0, 1, 1) / 3.
        assert!((out[0][0] - 0.0).abs() < 1e-6);
        assert!((out[1][0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((out[2][0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((out[3][0] - 1.0).abs() < 1e-6, "right edge clamps to 1: {}", out[3][0]);
    }

    #[test]
    fn the_bilateral_filter_smooths_noise_and_keeps_an_edge() {
        let (w, h) = (16, 1);
        // Two flat halves at 0.2 and 0.8 with a little noise on each.
        let img: Vec<[f32; 4]> = (0..w)
            .map(|x| {
                let base = if x < 8 { 0.2 } else { 0.8 };
                let noise = if x % 2 == 0 { 0.02 } else { -0.02 };
                let v = base + noise;
                [v, v, v, 1.0]
            })
            .collect();
        let f = ImageBilateralFilter::new(3, 2.0, 0.1);
        let out = f.cpu(w, h, &img);
        // Inside a half the noise is averaged away…
        assert!((out[3][0] - 0.2).abs() < 0.01, "{}", out[3][0]);
        assert!((out[12][0] - 0.8).abs() < 0.01, "{}", out[12][0]);
        // …and the step between halves survives: the pixels either side of
        // it stay on their own side.
        assert!(out[7][0] < 0.3 && out[8][0] > 0.7, "edge kept: {} | {}", out[7][0], out[8][0]);
    }

    #[test]
    fn matrix_multiply_matches_by_hand() {
        // [1 2; 3 4] · [5 6; 7 8] = [19 22; 43 50]
        let mm = MatrixMultiplication::new(2, 2, 2);
        assert_eq!(mm.cpu(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]), vec![19.0, 22.0, 43.0, 50.0]);
        // Non-square: [2×3] · [3×1].
        let mm = MatrixMultiplication::new(2, 1, 3);
        assert_eq!(mm.cpu(&[1.0, 0.0, 2.0, 0.0, 1.0, 3.0], &[1.0, 2.0, 3.0]), vec![7.0, 11.0]);
    }

    /// The WGSL bodies parse and type-check against the prelude, so a kernel
    /// that fails to compile fails here and not at the first dispatch.
    #[test]
    fn the_kernels_compile() {
        for src in [Convolve::source(), Bilateral::source(), MatMul::source()] {
            naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        }
    }
}
