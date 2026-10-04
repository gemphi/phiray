//! An instance: one object's place in the world. Its faces and vertices are
//! kept in the object's own space, under a tree of their own (`blas`), and
//! the world holds the instances under a tree over their world bounds
//! (`tlas`, `Tracer::with_chunks`). Moving, turning or scaling an object
//! then changes one matrix — the instance table and the small top tree go
//! up again, nothing else (`Tracer::reinstance`), where it used to mean
//! every vertex of the scene transformed on the CPU and uploaded again.
//!
//! A face names its instance in the top half of its material word
//! (`Scene::face`); the kernel reads the world matrix from there
//! (`tri_at`) and traces each instance in its own space (`trace_blas`).

use crate::bvh::Aabb;

/// How many words one instance takes in the data table (`inst_word`).
pub const INSTANCE_WORDS: usize = 32;
/// Instances a scene may hold: the face word keeps 16 bits for the id.
pub const MAX_INSTANCES: usize = 1 << 16;

/// One object's placement and its slice of the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// Object → world, Blender's layout: `m[col][row]`, translation in `m[3]`.
    pub world: [[f32; 4]; 4],
    /// Its faces, `[lo, hi)`, in the scene's face array.
    pub face_lo: u32,
    pub face_hi: u32,
    /// Its vertices, `[lo, hi)`.
    pub vert_lo: u32,
    pub vert_hi: u32,
    /// The object's own colour (`Object.col`, scene-linear) — what Solid's
    /// Color ▸ Object shades with. White until told.
    pub color: [f32; 3],
    /// `BLI_ghashutil_strhash_p_murmur(ob.id.name)`: the seed of Solid's
    /// Color ▸ Random (`hash.rs`). 0 until told.
    pub hash: u32,
    /// Whether a shadow ray may meet it — Workbench's Viewport Display ▸
    /// Shadow (`OB_DRAW_NO_SHADOW_CAST`) in Solid, Cycles' Ray Visibility ▸
    /// Shadow (`OB_HIDE_SHADOW`) in Rendered. Casts until told.
    pub casts_shadow: bool,
}

pub const IDENTITY: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

impl Instance {
    pub fn at(world: [[f32; 4]; 4], face_lo: u32, vert_lo: u32) -> Self {
        Self { world, face_lo, face_hi: face_lo, vert_lo, vert_hi: vert_lo, color: [1.0; 3], hash: 0, casts_shadow: true }
    }

    /// The object's colour and the hash of its id name, for Solid's Color
    /// ▸ Object and ▸ Random (`workbench::Material::Material(Object &, bool)`).
    pub fn colored(mut self, color: [f32; 3], hash: u32) -> Self {
        self.color = color;
        self.hash = hash;
        self
    }

    /// A translation.
    pub fn translation(t: [f32; 3]) -> [[f32; 4]; 4] {
        let mut m = IDENTITY;
        m[3] = [t[0], t[1], t[2], 1.0];
        m
    }

    pub fn faces(&self) -> std::ops::Range<usize> {
        self.face_lo as usize..self.face_hi as usize
    }

    /// The point `p` in the world.
    pub fn to_world(&self, p: [f32; 3]) -> [f32; 3] {
        let m = &self.world;
        std::array::from_fn(|r| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r])
    }

    /// The upper 3 × 3 inverted and the translation undone: world → object.
    /// A singular matrix (a zero scale) inverts to zero: nothing is hit.
    pub fn inverse(&self) -> [[f32; 4]; 3] {
        let m = &self.world;
        // Rows of the 3 × 3: a[r][c] = m[c][r].
        let a = |r: usize, c: usize| m[c][r];
        let det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
        if det.abs() < 1e-20 {
            return [[0.0; 4]; 3];
        }
        let d = 1.0 / det;
        let inv = [
            [(a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * d, (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * d, (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * d],
            [(a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * d, (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * d, (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * d],
            [(a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * d, (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * d, (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * d],
        ];
        let t = [m[3][0], m[3][1], m[3][2]];
        std::array::from_fn(|r| [inv[r][0], inv[r][1], inv[r][2], -(inv[r][0] * t[0] + inv[r][1] * t[1] + inv[r][2] * t[2])])
    }

    /// The box `b` (object space) in the world: its eight corners placed.
    pub fn world_box(&self, b: Aabb) -> Aabb {
        let mut out = Aabb::default();
        for k in 0..8 {
            let c = [if k & 1 == 0 { b.lo[0] } else { b.hi[0] }, if k & 2 == 0 { b.lo[1] } else { b.hi[1] }, if k & 4 == 0 { b.lo[2] } else { b.hi[2] }];
            let p = self.to_world(c);
            out = out.union(Aabb::new(p, p));
        }
        out
    }

    /// The instance's words for the kernel: the world's three rows, the
    /// inverse's three rows, then the root of its tree, its face range,
    /// whether it sits at the identity (the kernel then places nothing)
    /// and whether it casts a shadow,
    /// its object colour and its name hash.
    pub fn words(&self, blas_root: u32) -> [u32; INSTANCE_WORDS] {
        let mut w = [0u32; INSTANCE_WORDS];
        let m = &self.world;
        for r in 0..3 {
            for c in 0..4 {
                w[r * 4 + c] = m[c][r].to_bits();
            }
        }
        let inv = self.inverse();
        for r in 0..3 {
            for c in 0..4 {
                w[12 + r * 4 + c] = inv[r][c].to_bits();
            }
        }
        w[24] = blas_root;
        w[25] = self.face_lo;
        w[26] = self.face_hi;
        // Bit 0: at the identity; bit 1: casts no shadow (`trace`'s any-hit
        // rays pass it by).
        w[27] = u32::from(self.world == IDENTITY) | (u32::from(!self.casts_shadow) << 1);
        for k in 0..3 {
            w[28 + k] = self.color[k].to_bits();
        }
        w[31] = self.hash;
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_inverse_undoes_the_world() {
        let mut world = IDENTITY;
        // A scale of 2, a quarter turn about Z, moved to (1, 2, 3).
        world[0] = [0.0, 2.0, 0.0, 0.0];
        world[1] = [-2.0, 0.0, 0.0, 0.0];
        world[2] = [0.0, 0.0, 2.0, 0.0];
        world[3] = [1.0, 2.0, 3.0, 1.0];
        let inst = Instance::at(world, 0, 0);
        let p = [0.5, -1.0, 4.0];
        let w = inst.to_world(p);
        let inv = inst.inverse();
        let back: [f32; 3] = std::array::from_fn(|r| inv[r][0] * w[0] + inv[r][1] * w[1] + inv[r][2] * w[2] + inv[r][3]);
        for k in 0..3 {
            assert!((back[k] - p[k]).abs() < 1e-5, "{back:?} vs {p:?}");
        }
        let b = inst.world_box(Aabb::new([0.0; 3], [1.0; 3]));
        assert_eq!(b.lo, [-1.0, 2.0, 3.0]);
        assert_eq!(b.hi, [1.0, 4.0, 5.0]);
        assert_eq!((inst.words(7)[24], inst.words(7)[27], Instance::at(IDENTITY, 0, 0).words(0)[27]), (7, 0, 1));
        let mut shadowless = Instance::at(IDENTITY, 0, 0);
        shadowless.casts_shadow = false;
        assert_eq!(shadowless.words(0)[27], 0b11, "the flags share the word");
    }
}
