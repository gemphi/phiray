//! `GpuSort`: the radix passes, recorded once and replayed.
//!
//! This is the hop `bvh/gpu.rs` used to make to the CPU. The build now goes
//! Morton → sort → Karras links without leaving the device in the middle;
//! what still comes back is the packing (`Bvh::assemble`), and that is the
//! one hop left in the LBVH build.

use super::radix::{BINS, Clear, Count, DIGIT, Offset, PASSES, RadixParams, Scatter, Seed};
use super::spread::Spread;
use crate::gpu::{Gpu, GpuArray, Pass};

/// Keys being sorted, the buffer they ping-pong into, and the 512-slot
/// tally the passes share.
pub struct GpuSort {
    key: GpuArray<[u32; 2]>,
    alt: GpuArray<[u32; 2]>,
    tally: GpuArray<u32>,
    /// The sorted codes on their own, for `Karras` (`spread.rs`).
    code: GpuArray<u32>,
    pass: Pass,
    n: u32,
}

impl GpuSort {
    /// Record a sort of `n` keys seeded from a buffer of Morton codes.
    ///
    /// [`PASSES`] is even, so the ping-pong always ends in `key` — there is
    /// no "which buffer holds the answer" to get wrong, and the same
    /// recorded list can be replayed after new codes are written.
    pub fn record(gpu: &Gpu, codes: &wgpu::Buffer, n: u32) -> Self {
        let key = GpuArray::<[u32; 2]>::zeroed(gpu, n as usize);
        let alt = GpuArray::<[u32; 2]>::zeroed(gpu, n as usize);
        let tally = GpuArray::<u32>::zeroed(gpu, (BINS * 2) as usize);
        let code = GpuArray::<u32>::zeroed(gpu, n as usize);
        let mut pass = Pass::default();
        let base = RadixParams::default();
        pass.add::<Seed>(gpu, n, base, &[codes, key.buffer()]);
        pass.add::<Clear>(gpu, BINS * 2, base, &[tally.buffer()]);
        for d in 0..PASSES {
            let p = RadixParams { shift: d * DIGIT, ..base };
            let (src, dst) = if d % 2 == 0 { (&key, &alt) } else { (&alt, &key) };
            pass.add::<Count>(gpu, n, p, &[src.buffer(), tally.buffer()]);
            // One invocation over the 256 bins; `n` is 1 so the grid is one block.
            pass.add::<Offset>(gpu, 1, p, &[tally.buffer()]);
            pass.add::<Scatter>(gpu, n, p, &[src.buffer(), dst.buffer(), tally.buffer()]);
        }
        pass.add::<Spread>(gpu, n, base, &[key.buffer(), code.buffer()]);
        Self { key, alt, tally, code, pass, n }
    }

    /// Compile every kernel on the device so the first sort does not pay for
    /// it (`Tracer::warm`).
    pub fn warm(gpu: &Gpu) {
        Pass::compile::<Seed>(gpu);
        Pass::compile::<Clear>(gpu);
        Pass::compile::<Count>(gpu);
        Pass::compile::<Offset>(gpu);
        Pass::compile::<Scatter>(gpu);
        Pass::compile::<Spread>(gpu);
    }

    /// Replay the recorded passes.
    pub fn run(&self, gpu: &Gpu) {
        self.pass.run(gpu);
    }

    /// The sorted keys, on the device: `.0` the original slot, `.1` the code.
    pub fn keys(&self) -> &GpuArray<[u32; 2]> {
        &self.key
    }

    /// The sorted codes alone, on the device — what `Karras` reads.
    pub fn code(&self) -> &GpuArray<u32> {
        &self.code
    }

    /// Keys sorted.
    pub fn len(&self) -> u32 {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Dispatches recorded: seed, clear, then three a digit.
    pub fn steps(&self) -> usize {
        self.pass.len()
    }

    /// Scratch the sort owns, in bytes — what it costs to keep recorded.
    pub fn bytes(&self) -> u64 {
        (self.key.len() + self.alt.len()) as u64 * 8
            + (self.tally.len() + self.code.len()) as u64 * 4
    }

    /// The order `Bvh::assemble` wants: original slot per sorted position.
    ///
    /// The one readback the build still makes, and it is here because
    /// packing is still on the CPU. The codes do **not** come with it —
    /// `Karras` reads [`GpuSort::code`] where it is.
    pub fn order(&self, gpu: &Gpu) -> Vec<u32> {
        self.key.read(gpu).into_iter().map(|k| k[0]).collect()
    }

    /// `(order, sorted)` for a caller that wants both on the CPU (tests).
    pub fn read(&self, gpu: &Gpu) -> (Vec<u32>, Vec<u32>) {
        let keys = self.key.read(gpu);
        (keys.iter().map(|k| k[0]).collect(), keys.iter().map(|k| k[1]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::super::build::Bvh;
    use super::super::radix::{BINS, DIGIT, PASSES};

    /// The same digit the kernels read, in Rust.
    fn digit(k: [u32; 2], shift: u32) -> u32 {
        if shift < 32 { (k[0] >> shift) & 0xFF } else { (k[1] >> (shift - 32)) & 0xFF }
    }

    /// What the recorded passes do, step for step: seed, then for each digit
    /// a histogram, an exclusive prefix sum, and a scatter through the
    /// cursor. The GPU raced on the scatter; nothing here can, which is the
    /// point — if the two agree, the race was harmless.
    fn model(codes: &[u32]) -> Vec<[u32; 2]> {
        let mut key: Vec<[u32; 2]> = codes.iter().enumerate().map(|(i, &c)| [i as u32, c]).collect();
        let mut alt = key.clone();
        for d in 0..PASSES {
            let shift = d * DIGIT;
            let mut cursor = vec![0u32; BINS as usize];
            for k in &key {
                cursor[digit(*k, shift) as usize] += 1;
            }
            let mut running = 0;
            for slot in cursor.iter_mut() {
                let count = *slot;
                *slot = running;
                running += count;
            }
            for k in &key {
                let b = digit(*k, shift) as usize;
                alt[cursor[b] as usize] = *k;
                cursor[b] += 1;
            }
            std::mem::swap(&mut key, &mut alt);
        }
        key
    }

    /// The order has to be the CPU sort's, value for value — the GPU sort is
    /// a move, not a change. Duplicates are the case that matters: Morton
    /// codes collide constantly, and a radix sort that is not stable would
    /// agree on the codes and disagree on the order.
    #[test]
    fn the_radix_order_is_the_cpu_sorts() {
        let sets: [Vec<u32>; 5] = [
            vec![],
            vec![7],
            vec![5, 5, 5, 5, 5, 5, 5, 5],
            vec![9, 1, 8, 1, 7, 1, 0, u32::MAX, 0, 3],
            // Knuth's multiplicative hash mod 97: a thousand codes over 97
            // distinct values, so every code has about ten collisions.
            (0..1000u32).map(|i| i.wrapping_mul(2_654_435_761) % 97).collect(),
        ];
        for codes in sets {
            let (order, sorted) = Bvh::sort_codes(&codes);
            let got = model(&codes);
            assert_eq!(order, got.iter().map(|k| k[0]).collect::<Vec<_>>(), "order for {codes:?}");
            assert_eq!(sorted, got.iter().map(|k| k[1]).collect::<Vec<_>>(), "codes for {codes:?}");
        }
    }

    /// Every bit of the key is covered, and the ping-pong ends where
    /// `GpuSort::record` says it does.
    #[test]
    fn the_passes_cover_the_key_and_land_back_in_key() {
        assert_eq!(PASSES * DIGIT, 64, "the key is two u32");
        assert_eq!(BINS, 1 << DIGIT, "a bin per digit value");
        assert_eq!(PASSES % 2, 0, "an odd pass count would leave the answer in `alt`");
    }
}
