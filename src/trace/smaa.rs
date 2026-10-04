//! SMAA's two precomputed tables (`BLI_smaa_textures.h`, Jimenez et al.,
//! MIT): the area texture, 160 × 560 RG8 — for each pair of crossing edges
//! and each distance to them, the coverage of the two pixels an edge line
//! passes through — and the search texture, 64 × 16 R8, how many pixels
//! a bilinearly fetched pair of edges adds to a search. Workbench uploads
//! them once (`AntiAliasingPass::init`, `smaa_area_tx_`, `smaa_search_tx_`,
//! both filtered) and the kernel samples them in `smaa.wgsl`. Here they
//! are one storage buffer of bytes packed into words, the area table first.

pub struct SmaaTables;

impl SmaaTables {
    const AREA: &'static [u8] = include_bytes!("smaa/area.bin");
    const SEARCH: &'static [u8] = include_bytes!("smaa/search.bin");
    pub const AREA_SIZE: (usize, usize) = (160, 560);
    pub const SEARCH_SIZE: (usize, usize) = (64, 16);
    /// Where the search table's bytes begin (`AREA_BYTES` in the kernel).
    pub const AREA_BYTES: usize = Self::AREA_SIZE.0 * Self::AREA_SIZE.1 * 2;

    /// The two tables as the kernel's `tables` binding: byte `b` is
    /// `(tables[b / 4] >> (8 (b % 4))) & 255`.
    pub fn words() -> Vec<u32> {
        let bytes: Vec<u8> = Self::AREA.iter().chain(Self::SEARCH).copied().collect();
        bytes.chunks(4).map(|c| c.iter().enumerate().fold(0u32, |w, (k, &b)| w | (b as u32) << (8 * k))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_are_blenders_bytes() {
        assert_eq!(SmaaTables::AREA.len(), SmaaTables::AREA_BYTES);
        assert_eq!(SmaaTables::SEARCH.len(), SmaaTables::SEARCH_SIZE.0 * SmaaTables::SEARCH_SIZE.1);
        // `areaTexBytes[33] == 31`, `searchTexBytes[0] == 254`, `[24] == 127`.
        assert_eq!((SmaaTables::AREA[33], SmaaTables::AREA[160]), (31, 64));
        assert_eq!((SmaaTables::SEARCH[0], SmaaTables::SEARCH[2], SmaaTables::SEARCH[24]), (254, 0, 127));
        let words = SmaaTables::words();
        assert_eq!(words.len(), (SmaaTables::AREA_BYTES + 1024).div_ceil(4));
        let byte = |b: usize| (words[b / 4] >> (8 * (b % 4))) & 255;
        assert_eq!(byte(33), 31);
        assert_eq!(byte(SmaaTables::AREA_BYTES), 254);
        assert_eq!(byte(SmaaTables::AREA_BYTES + 24), 127);
    }
}
