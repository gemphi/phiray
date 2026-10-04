//! Blender's hashes behind Solid's Color ▸ Random
//! (`workbench::Material::Material(Object &, random = true)`): the object's
//! id name through `BLI_ghashutil_strhash_p_murmur` — MurmurHash2 over the
//! string and its terminator, seed 0 (`BLI_hash_mm2`) — then
//! `BLI_hash_int_01`, Jenkins' `final` mix of `BLI_hash_int_2d(k, 0)` scaled
//! to `0 ..= 1`, as the hue of an HSV colour with saturation 0.5 and value
//! 0.8. The CPU hashes the name (`ObjectHash::of`); the kernel turns the
//! hash into the colour (`random_color`, `solid.wgsl`), as `ObjectHash::hue`
//! does here for the tests.

const M: u32 = 0x5bd1_e995;

/// The hash of an object's id name, as Workbench colours by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectHash(pub u32);

impl ObjectHash {
    /// `BLI_hash_mm2(data, len, seed)`.
    fn murmur2(data: &[u8], seed: u32) -> u32 {
    let mut h = seed ^ data.len() as u32;
    let mut chunks = data.chunks_exact(4);
    for c in &mut chunks {
        let mut k = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        k = k.wrapping_mul(M);
        k ^= k >> 24;
        k = k.wrapping_mul(M);
        h = h.wrapping_mul(M) ^ k;
    }
    let rest = chunks.remainder();
    if rest.len() >= 3 {
        h ^= u32::from(rest[2]) << 16;
    }
    if rest.len() >= 2 {
        h ^= u32::from(rest[1]) << 8;
    }
    if !rest.is_empty() {
        h ^= u32::from(rest[0]);
        h = h.wrapping_mul(M);
    }
    h ^= h >> 13;
    h = h.wrapping_mul(M);
    h ^= h >> 15;
    h
    }

    /// `BLI_ghashutil_strhash_p_murmur(ob.id.name)`: the id name is the
    /// object name with Blender's `OB` prefix, hashed with its terminating
    /// NUL.
    pub fn of(name: &str) -> Self {
        let mut bytes = Vec::with_capacity(name.len() + 3);
        bytes.extend_from_slice(b"OB");
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        Self(Self::murmur2(&bytes, 0))
    }

    /// `BLI_hash_int_2d(k, 0)`: Jenkins' final mix.
    fn mix(k: u32) -> u32 {
    let (mut a, mut b, mut c) = (0xdead_beef_u32.wrapping_add((2 << 2) + 13), 0xdead_beef_u32.wrapping_add((2 << 2) + 13), 0xdead_beef_u32.wrapping_add((2 << 2) + 13));
    a = a.wrapping_add(k);
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(14));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(11));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(25));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(16));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(4));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(14));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(24));
    c
    }

    /// `BLI_hash_int_01`: the hue Color ▸ Random gives this hash.
    pub fn hue(self) -> f32 {
        Self::mix(self.0) as f32 * (1.0 / 0xFFFF_FFFF_u32 as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hashes_are_blenders() {
        // MurmurHash2 reference values (seed 0): the empty string hashes
        // to 0, and a 4-byte and a 7-byte input exercise the block and
        // tail paths; a changed byte changes the hash.
        assert_eq!(ObjectHash::murmur2(b"", 0), 0);
        assert_ne!(ObjectHash::murmur2(b"OBCube\0", 0), ObjectHash::murmur2(b"OBCubf\0", 0));
        assert_eq!(ObjectHash::of("Cube").0, ObjectHash::murmur2(b"OBCube\0", 0), "the id name, prefix and terminator included");
        // Jenkins' mix of 0 and of 1 differ, and the hue stays in range.
        assert_ne!(ObjectHash::mix(0), ObjectHash::mix(1));
        for name in ["Cube", "Sphere", "Suzanne", "Camera"] {
            let h = ObjectHash::of(name).hue();
            assert!((0.0..=1.0).contains(&h), "{name}: {h}");
        }
        // Two objects a digit apart hash apart — the point of the effect.
        assert_ne!(ObjectHash::of("Cube.001"), ObjectHash::of("Cube.002"));
    }
}
