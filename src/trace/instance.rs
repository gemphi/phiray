//! An instance: one placement of a mesh in the world. The mesh keeps its
//! faces and vertices in its own space under a tree of its own
//! (`mesh.rs`, `Trees::blas`), and the world holds the instances under a
//! tree over their world bounds (`tlas`, `Tracer::with_chunks`). Many
//! instances may name one mesh — a collection instanced a thousand times
//! uploads its triangles once (Cycles' `Object` over a shared `Geometry`,
//! `scene/object.cpp`; roadmap §4.4, Step 1). Moving, turning or scaling
//! an object then changes one matrix — the instance table and the small
//! top tree go up again, nothing else (`Tracer::reinstance`).
//!
//! A face names only its material; the traversal carries the instance
//! from the top-tree leaf into the hit (`Hit.inst`, `bvh.wgsl`), and the
//! kernel reads the world matrix from the instance's record (`inst_at`).

use super::mesh::Mesh;
use crate::bvh::Aabb;

/// How many words one instance takes in the data table (`inst_word`).
pub const INSTANCE_WORDS: usize = 40;

/// One placement of a mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// Object → world, Blender's layout: `m[col][row]`, translation in `m[3]`.
    pub world: [[f32; 4]; 4],
    /// The mesh it places (`Scene::meshes`).
    pub mesh: u32,
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
    /// Mesh `mesh` placed by `world`.
    pub fn of(world: [[f32; 4]; 4], mesh: u32) -> Self {
        Self { world, mesh, color: [1.0; 3], hash: 0, casts_shadow: true }
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
        let det = self.det();
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

    /// The determinant of the upper 3 × 3.
    fn det(&self) -> f32 {
        let m = &self.world;
        let a = |r: usize, c: usize| m[c][r];
        a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0))
    }

    /// Whether the matrix mirrors — a negative determinant, a scale of −1 on
    /// one axis (or on three). Placing a face then turns its winding over:
    /// the world's `(b − a) × (c − a)` points into the surface its normals
    /// point out of, since normals go under the inverse-transpose and are
    /// not turned with it. The kernel turns the face normal back
    /// (`world_ng`), as Cycles does for a transform it applies
    /// (`object_negative_scale_applied`, `kernel/geom/triangle.h`).
    pub fn mirrors(&self) -> bool {
        self.det() < 0.0
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
    /// inverse's three rows, then the root of its mesh's tree, the mesh's
    /// face range, whether it sits at the identity (the kernel then places
    /// nothing), whether it casts a shadow and whether it mirrors, its
    /// object colour and its name hash, the mesh's texture space, and the
    /// mesh's id. The mesh's words are copied in so a hit reads one record.
    pub fn words(&self, blas_root: u32, mesh: &Mesh) -> [u32; INSTANCE_WORDS] {
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
        w[25] = mesh.face_lo;
        w[26] = mesh.face_hi;
        // Bit 0: at the identity; bit 1: casts no shadow (`trace`'s any-hit
        // rays pass it by); bit 2: mirrors, its faces' winding turned over
        // in the world (`mirrors`, `world_ng`).
        w[27] = u32::from(self.world == IDENTITY) | (u32::from(!self.casts_shadow) << 1) | (u32::from(self.mirrors()) << 2);
        for k in 0..3 {
            w[28 + k] = self.color[k].to_bits();
        }
        w[31] = self.hash;
        // 32..35 the texture space's corner, 35..38 its size (object space).
        let (lo, size) = mesh.texspace();
        for k in 0..3 {
            w[32 + k] = lo[k].to_bits();
            w[35 + k] = size[k].to_bits();
        }
        w[38] = self.mesh;
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit() -> Mesh {
        Mesh::from(0, 0)
    }

    #[test]
    fn the_inverse_undoes_the_world() {
        let mut world = IDENTITY;
        // A scale of 2, a quarter turn about Z, moved to (1, 2, 3).
        world[0] = [0.0, 2.0, 0.0, 0.0];
        world[1] = [-2.0, 0.0, 0.0, 0.0];
        world[2] = [0.0, 0.0, 2.0, 0.0];
        world[3] = [1.0, 2.0, 3.0, 1.0];
        let inst = Instance::of(world, 3);
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
        let w = inst.words(7, &unit());
        assert_eq!((w[24], w[27], w[38], Instance::of(IDENTITY, 0).words(0, &unit())[27]), (7, 0, 3, 1));
        let mut shadowless = Instance::of(IDENTITY, 0);
        shadowless.casts_shadow = false;
        assert_eq!(shadowless.words(0, &unit())[27], 0b11, "the flags share the word");
    }

    /// A scale of −1 on one axis mirrors (bit 2); on two it is a half turn,
    /// which does not; on all three it mirrors again.
    #[test]
    fn a_negative_determinant_is_flagged_as_mirroring() {
        let scaled = |s: [f32; 3]| {
            let mut m = IDENTITY;
            (0..3).for_each(|k| m[k][k] = s[k]);
            m[3] = [0.5, 0.0, 0.0, 1.0];
            Instance::of(m, 0)
        };
        assert!(scaled([-1.0, 1.0, 1.0]).mirrors());
        assert_eq!(scaled([-1.0, 1.0, 1.0]).words(0, &unit())[27], 0b100);
        assert!(!scaled([-1.0, -1.0, 1.0]).mirrors(), "a half turn about z");
        assert!(scaled([-2.0, -0.5, -1.0]).mirrors());
        assert_eq!(Instance::of(IDENTITY, 0).words(0, &unit())[27] & 0b100, 0);
    }

    /// The record carries its mesh's texture space, so a hit reads one record.
    #[test]
    fn the_record_copies_the_meshs_texture_space() {
        let mut wall = unit();
        wall.bounds = [[-9.0, 0.0, -9.0], [9.0, 0.0, 12.0]];
        let w = Instance::of(IDENTITY, 0).words(0, &wall);
        assert_eq!((f32::from_bits(w[33]), f32::from_bits(w[36])), (-1.0, 2.0));
    }
}
