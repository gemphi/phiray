//! A mesh: geometry uploaded once — a run of faces and vertices in the
//! scene's arrays, kept in its own space under a tree of its own
//! (`Trees::blas`) — that any number of instances place in the world
//! (`instance.rs`). Cycles keeps the same split: one `Geometry`
//! (`scene/geometry.cpp`) under many `Object`s (`scene/object.cpp`), each
//! with its own transform, so a collection instanced a thousand times
//! holds its triangles and its BVH once. Roadmap §4.4, Step 1.

/// Geometry, uploaded once: a range of faces and vertices, and the box
/// they fill. Its tree's root is the trees' business (`Trees::roots`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mesh {
    /// Its faces, `[lo, hi)`, in the scene's face array.
    pub face_lo: u32,
    pub face_hi: u32,
    /// Its vertices, `[lo, hi)`.
    pub vert_lo: u32,
    pub vert_hi: u32,
    /// The bounding box of its vertices in its own space, `[min, max]`:
    /// what Generated texture coordinates normalize by (`ATTR_STD_GENERATED`).
    pub bounds: [[f32; 3]; 2],
}

impl Mesh {
    /// A mesh beginning at `face_lo`, `vert_lo`, nothing in it yet.
    pub fn from(face_lo: u32, vert_lo: u32) -> Self {
        Self { face_lo, face_hi: face_lo, vert_lo, vert_hi: vert_lo, bounds: [[0.0; 3], [1.0; 3]] }
    }

    pub fn faces(&self) -> std::ops::Range<usize> {
        self.face_lo as usize..self.face_hi as usize
    }

    pub fn verts(&self) -> std::ops::Range<usize> {
        self.vert_lo as usize..self.vert_hi as usize
    }

    /// The mesh's texture space (`BKE_mesh_texspace_calc`), the corner and
    /// size that Generated coordinates divide by: the bounding box — but an
    /// axis the box is flat on takes a half-extent of 1 about its centre,
    /// so a plane's Generated coordinate across it is a half, not the float
    /// noise of the hit divided by nothing.
    pub fn texspace(&self) -> ([f32; 3], [f32; 3]) {
        let (mut lo, mut size) = ([0.0; 3], [0.0; 3]);
        for k in 0..3 {
            let half = 0.5 * (self.bounds[1][k] - self.bounds[0][k]);
            let centre = 0.5 * (self.bounds[1][k] + self.bounds[0][k]);
            let half = if half == 0.0 { 1.0 } else { half.max(0.00001) };
            lo[k] = centre - half;
            size[k] = 2.0 * half;
        }
        (lo, size)
    }

    /// The box its vertices `pos` fill; a mesh without any keeps the unit box.
    pub fn fit(&mut self, pos: &[[f32; 4]]) {
        let verts = &pos[self.verts()];
        if verts.is_empty() {
            return;
        }
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in verts {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        self.bounds = [lo, hi];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wall on the XZ plane: its texture space spans the box on X and Z,
    /// and two units about y = 0 where the box is flat, so a hit's Generated
    /// coordinate there is a half.
    #[test]
    fn a_flat_axis_takes_a_half_extent_of_one() {
        let mut wall = Mesh::from(0, 0);
        wall.bounds = [[-9.0, 0.0, -9.0], [9.0, 0.0, 12.0]];
        let (lo, size) = wall.texspace();
        assert_eq!((lo, size), ([-9.0, -1.0, -9.0], [18.0, 2.0, 21.0]));
    }

    #[test]
    fn the_box_fits_the_vertices_of_the_range() {
        let pos = [[0.0, 0.0, 0.0, 1.0], [2.0, -1.0, 3.0, 1.0], [9.0, 9.0, 9.0, 1.0]];
        let mut m = Mesh { face_hi: 1, vert_hi: 2, ..Mesh::from(0, 0) };
        m.fit(&pos);
        assert_eq!(m.bounds, [[0.0, -1.0, 0.0], [2.0, 0.0, 3.0]]);
        assert_eq!((m.faces(), m.verts()), (0..1, 0..2));
    }
}
