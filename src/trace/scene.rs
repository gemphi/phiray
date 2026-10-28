//! Compact GPU scene: shared vertices (position + packed normal/UV, 32 bytes),
//! faces (3 vertex ids + material, 16 bytes) and a material table — about a
//! quarter of the memory of one 160-byte [`Tri`] per triangle. The faces
//! and vertices belong to meshes (`mesh.rs`), each placed by any number of
//! instances (`instance.rs`): one `Geometry`, many `Object`s, as Cycles'
//! `scene/object.cpp` has it (roadmap §4.4, Step 1).

use super::instance::{IDENTITY, Instance};
use super::kernel::{Atlas, Tri};
use super::mat::Mat;
use super::mesh::Mesh;
use crate::bvh::Aabb;
use std::collections::HashMap;

/// A face's material id: its fourth word, whole.
pub fn face_mat(word: u32) -> u32 {
    word
}

/// Octahedral encoding of a unit normal into two floats.
fn oct(n: [f32; 3]) -> [f32; 2] {
    let l = n[0].abs() + n[1].abs() + n[2].abs();
    if l < 1e-12 {
        return [0.0, 0.0];
    }
    let (x, y, z) = (n[0] / l, n[1] / l, n[2] / l);
    if z >= 0.0 {
        [x, y]
    } else {
        [(1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum()]
    }
}

/// Packed normal; outside the octahedron (2, 2) marks flat.
pub(super) fn pack(n: [f32; 3]) -> [f32; 2] {
    if n == [0.0; 3] { [2.0, 2.0] } else { oct(n) }
}

#[derive(Default)]
pub struct Scene {
    /// Object-space positions, mesh by mesh (`meshes`).
    pub pos: Vec<[f32; 4]>,
    /// Octahedral normal (xy) + UV (zw). A zero normal packs to (0, 0) → flat.
    pub attr: Vec<[f32; 4]>,
    /// Each vertex's colour attribute, sRGB bytes packed `r | g << 8 | b <<
    /// 16 | a << 24` (`paint`); empty when no object has one — Solid's
    /// Color ▸ Attribute.
    pub colors: Vec<u32>,
    /// Vertex ids + material (`face_mat`).
    pub faces: Vec<[u32; 4]>,
    pub mats: Vec<Mat>,
    pub atlas: Atlas,
    /// The meshes, each the owner of a run of faces and vertices; the one
    /// being added to is the last while `open`.
    pub meshes: Vec<Mesh>,
    /// The objects, each a placement of one mesh. A scene built from
    /// world-space triangles (`add`, `push`) has one, at the identity.
    pub insts: Vec<Instance>,
    /// Whether the last mesh is still taking faces.
    open: bool,
    verts: HashMap<[u32; 7], u32>,
    mat_ids: HashMap<[u32; 164], u32>,
}

impl Scene {
    pub fn new(atlas: Atlas) -> Self {
        Self { atlas, ..Default::default() }
    }

    pub fn from_tris(tris: &[Tri], atlas: &Atlas) -> Self {
        let mut s = Self::new(atlas.clone());
        tris.iter().for_each(|t| s.add(t));
        s
    }

    /// Begin an object placed by `world` (Blender's `m[col][row]`): a new
    /// mesh, which the vertices and faces added next join, in its own
    /// space, and one instance of it. Returns the instance's id.
    pub fn instance(&mut self, world: [[f32; 4]; 4]) -> u32 {
        self.instance_of(world, [1.0; 3], super::hash::ObjectHash(0))
    }

    /// [`Self::instance`] with the object's own colour (`Object.col`,
    /// linear) and the hash of its id name (`ObjectHash::of`), for Solid's
    /// Color ▸ Object and ▸ Random.
    pub fn instance_of(&mut self, world: [[f32; 4]; 4], color: [f32; 3], hash: super::hash::ObjectHash) -> u32 {
        self.close_mesh();
        self.meshes.push(Mesh::from(self.faces.len() as u32, self.pos.len() as u32));
        self.open = true;
        self.insts.push(Instance::of(world, self.meshes.len() as u32 - 1).colored(color, hash.0));
        self.insts.len() as u32 - 1
    }

    /// Another object showing mesh `mesh` — one already begun, a
    /// collection instance's member, say — placed by `world`: its faces
    /// go up once and this draws them again (`make_duplis_collection`).
    /// Closes the mesh being added to; a mesh that does not exist names
    /// the last. Returns the instance's id.
    pub fn place(&mut self, mesh: u32, world: [[f32; 4]; 4], color: [f32; 3], hash: super::hash::ObjectHash) -> u32 {
        self.close_mesh();
        let mesh = mesh.min(self.meshes.len().saturating_sub(1) as u32);
        self.insts.push(Instance::of(world, mesh).colored(color, hash.0));
        self.insts.len() as u32 - 1
    }

    /// The mesh instance `inst` places.
    pub fn mesh_of(&self, inst: u32) -> u32 {
        self.insts.get(inst as usize).map_or(0, |i| i.mesh)
    }

    /// Instance `inst` casts no shadow: Workbench's Viewport Display ▸
    /// Shadow off (`OB_DRAW_NO_SHADOW_CAST`), or Cycles' Ray Visibility ▸
    /// Shadow off (`OB_HIDE_SHADOW`) — the shadow rays pass it by.
    pub fn hide_shadow(&mut self, inst: u32) {
        if let Some(i) = self.insts.get_mut(inst as usize) {
            i.casts_shadow = false;
        }
    }

    /// Make sure a mesh is taking faces: the open one, else a new one at
    /// the identity (a scene of world-space triangles).
    fn current(&mut self) {
        if !self.open {
            self.instance(IDENTITY);
        }
    }

    /// The meshes as a sealed scene would have them: the open one's ranges
    /// ended here (a pose handed over unsealed, `Tracer::repose`).
    pub fn placed(&self) -> Vec<Mesh> {
        let mut meshes = self.meshes.clone();
        match (self.open, meshes.last_mut()) {
            (true, Some(last)) => {
                last.face_hi = self.faces.len() as u32;
                last.vert_hi = self.pos.len() as u32;
                last.fit(&self.pos);
            }
            (_, None) => meshes.push(Mesh { face_hi: self.faces.len() as u32, vert_hi: self.pos.len() as u32, ..Mesh::from(0, 0) }),
            _ => {}
        }
        meshes
    }

    /// The open mesh's ranges end here, and its box is measured.
    fn close_mesh(&mut self) {
        if !self.open {
            return;
        }
        self.open = false;
        if let Some(last) = self.meshes.last_mut() {
            last.face_hi = self.faces.len() as u32;
            last.vert_hi = self.pos.len() as u32;
            last.fit(&self.pos);
        }
    }

    /// Done adding: free the de-duplication maps and close the last mesh.
    pub fn seal(&mut self) {
        self.verts = HashMap::new();
        self.mat_ids = HashMap::new();
        if self.meshes.is_empty() {
            self.current();
        }
        self.close_mesh();
    }

    pub fn len(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    fn corner(&self, f: usize, k: usize) -> [f32; 3] {
        let p = self.pos[self.faces[f][k] as usize];
        [p[0], p[1], p[2]]
    }

    /// Each face's box in its mesh's space.
    pub fn boxes(&self) -> Vec<Aabb> {
        (0..self.len()).map(|f| Aabb::of_tri(self.corner(f, 0), self.corner(f, 1), self.corner(f, 2))).collect()
    }

    /// Put faces in BVH leaf order (`order[slot]` = old face). An order over
    /// the whole array that keeps each mesh's faces within its range.
    pub fn reorder(&mut self, order: &[u32]) {
        self.faces = order.iter().map(|&i| self.faces[i as usize]).collect();
    }

    /// Largest single buffer in bytes.
    pub fn biggest(&self) -> u64 {
        [self.pos.len() * 32, self.faces.len() * 16, self.atlas.words().len() * 4, self.mats.len() * std::mem::size_of::<Mat>() + 64]
            .into_iter()
            .max()
            .unwrap_or(0) as u64
    }

    /// `[lamp count, face ids…, instance ids…, cdf bits…, materials…]`
    /// (see `wgsl/layout.wgsl`).
    pub fn table(&self) -> Vec<u32> {
        super::lamps::table_of(&self.faces, &self.pos, &self.mats, &self.placed(), &self.insts)
    }
}

#[cfg(test)]
mod tests;

mod add;
