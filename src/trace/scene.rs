//! Compact GPU scene: shared vertices (position + packed normal/UV, 32 bytes),
//! faces (3 vertex ids + material, 16 bytes) and a material table — about a
//! quarter of the memory of one 160-byte [`Tri`] per triangle.

use super::instance::{IDENTITY, Instance, MAX_INSTANCES};
use super::kernel::{Atlas, Tri};
use super::mat::Mat;
use crate::bvh::Aabb;
use std::collections::HashMap;

/// A face's material id and instance id, packed in its fourth word.
pub fn face_mat(word: u32) -> u32 {
    word & 0xffff
}
pub fn face_inst(word: u32) -> u32 {
    word >> 16
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
fn pack(n: [f32; 3]) -> [f32; 2] {
    if n == [0.0; 3] { [2.0, 2.0] } else { oct(n) }
}

#[derive(Default)]
pub struct Scene {
    /// Object-space positions, instance by instance (`insts`).
    pub pos: Vec<[f32; 4]>,
    /// Octahedral normal (xy) + UV (zw). A zero normal packs to (0, 0) → flat.
    pub attr: Vec<[f32; 4]>,
    /// Each vertex's colour attribute, sRGB bytes packed `r | g << 8 | b <<
    /// 16 | a << 24` (`paint`); empty when no object has one — Solid's
    /// Color ▸ Attribute.
    pub colors: Vec<u32>,
    /// Vertex ids + `material | instance << 16` (`face_mat`, `face_inst`).
    pub faces: Vec<[u32; 4]>,
    pub mats: Vec<Mat>,
    pub atlas: Atlas,
    /// The objects, each the owner of a run of faces and vertices; the one
    /// being added to is the last. A scene built from world-space triangles
    /// (`add`, `push`) has one, at the identity.
    pub insts: Vec<Instance>,
    verts: HashMap<[u32; 7], u32>,
    mat_ids: HashMap<[u32; 112], u32>,
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

    /// Begin an object placed by `world` (Blender's `m[col][row]`): the
    /// vertices and faces added next are its, in its own space. Returns its
    /// id. Past `MAX_INSTANCES` the last instance keeps taking faces — a
    /// scene of sixty-five thousand objects places the rest together.
    pub fn instance(&mut self, world: [[f32; 4]; 4]) -> u32 {
        self.instance_of(world, [1.0; 3], super::hash::ObjectHash(0))
    }

    /// [`Self::instance`] with the object's own colour (`Object.col`,
    /// linear) and the hash of its id name (`ObjectHash::of`), for Solid's
    /// Color ▸ Object and ▸ Random.
    pub fn instance_of(&mut self, world: [[f32; 4]; 4], color: [f32; 3], hash: super::hash::ObjectHash) -> u32 {
        self.close_instance();
        if self.insts.len() >= MAX_INSTANCES {
            return self.insts.len() as u32 - 1;
        }
        self.insts.push(Instance::at(world, self.faces.len() as u32, self.pos.len() as u32).colored(color, hash.0));
        self.insts.len() as u32 - 1
    }

    /// Instance `inst` casts no shadow: Workbench's Viewport Display ▸
    /// Shadow off (`OB_DRAW_NO_SHADOW_CAST`), or Cycles' Ray Visibility ▸
    /// Shadow off (`OB_HIDE_SHADOW`) — the shadow rays pass it by.
    pub fn hide_shadow(&mut self, inst: u32) {
        if let Some(i) = self.insts.get_mut(inst as usize) {
            i.casts_shadow = false;
        }
    }

    /// The instance faces are being added to, the identity one when none
    /// was begun.
    fn current(&mut self) -> u32 {
        if self.insts.is_empty() {
            self.insts.push(Instance::at(IDENTITY, 0, 0));
        }
        self.insts.len() as u32 - 1
    }

    /// The instances as a sealed scene would have them: the open one's
    /// ranges ended here (a pose handed over unsealed, `Tracer::repose`).
    pub fn placed(&self) -> Vec<Instance> {
        let mut insts = self.insts.clone();
        match insts.last_mut() {
            Some(last) => {
                last.face_hi = self.faces.len() as u32;
                last.vert_hi = self.pos.len() as u32;
            }
            None => insts.push(Instance { face_hi: self.faces.len() as u32, vert_hi: self.pos.len() as u32, ..Instance::at(IDENTITY, 0, 0) }),
        }
        insts
    }

    /// The open instance's ranges end here.
    fn close_instance(&mut self) {
        if let Some(last) = self.insts.last_mut() {
            last.face_hi = self.faces.len() as u32;
            last.vert_hi = self.pos.len() as u32;
            // The bounding box of its vertices, for Generated coordinates.
            let verts = &self.pos[last.vert_lo as usize..last.vert_hi as usize];
            if !verts.is_empty() {
                let mut lo = [f32::INFINITY; 3];
                let mut hi = [f32::NEG_INFINITY; 3];
                for v in verts {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k]);
                        hi[k] = hi[k].max(v[k]);
                    }
                }
                last.bounds = [lo, hi];
            }
        }
    }

    /// Add one triangle record.
    pub fn add(&mut self, t: &Tri) {
        let uv = [[t.us[0], t.vs[0]], [t.us[1], t.vs[1]], [t.us[2], t.vs[2]]];
        self.push([t.a, t.b, t.c], [t.na, t.nb, t.nc], uv, Mat::of(t));
    }

    fn vert(&mut self, p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> u32 {
        let o = pack(n);
        let key = [p[0], p[1], p[2], o[0], o[1], uv[0], uv[1]].map(f32::to_bits);
        let next = self.pos.len() as u32;
        *self.verts.entry(key).or_insert_with(|| {
            self.pos.push([p[0], p[1], p[2], 1.0]);
            self.attr.push([o[0], o[1], uv[0], uv[1]]);
            next
        })
    }

    /// Append one vertex with no de-duplication (callers that already share
    /// vertices, e.g. per mesh); returns its id. A zero normal = flat.
    pub fn vertex(&mut self, p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> u32 {
        let o = pack(n);
        self.pos.push([p[0], p[1], p[2], 1.0]);
        self.attr.push([o[0], o[1], uv[0], uv[1]]);
        self.pos.len() as u32 - 1
    }

    /// Give vertex `v` its colour attribute, scene-linear `rgb` — kept as
    /// sRGB bytes, Blender's byte colour; vertices never painted are white.
    pub fn paint(&mut self, v: u32, rgb: [f32; 3]) {
        if self.colors.len() < self.pos.len() {
            self.colors.resize(self.pos.len(), 0xffff_ffff);
        }
        let byte = |c: f32| u32::from(super::srgb::Srgb::linear_to_byte(c));
        if let Some(slot) = self.colors.get_mut(v as usize) {
            *slot = byte(rgb[0]) | byte(rgb[1]) << 8 | byte(rgb[2]) << 16 | 0xff << 24;
        }
    }

    /// Shared id of a material.
    pub fn material(&mut self, mat: Mat) -> u32 {
        let next = self.mats.len() as u32;
        *self.mat_ids.entry(mat.key()).or_insert_with(|| {
            self.mats.push(mat);
            next
        })
    }

    /// Append a face over existing vertex ids, in the current instance.
    pub fn face(&mut self, v: [u32; 3], mat: u32) {
        let inst = self.current();
        self.faces.push([v[0], v[1], v[2], (mat & 0xffff) | (inst << 16)]);
    }

    pub fn push(&mut self, p: [[f32; 3]; 3], n: [[f32; 3]; 3], uv: [[f32; 2]; 3], mat: Mat) {
        let m = self.material(mat);
        let v = [0, 1, 2].map(|k| self.vert(p[k], n[k], uv[k]));
        self.face(v, m);
    }

    /// Done adding: free the de-duplication maps and close the last instance.
    pub fn seal(&mut self) {
        self.verts = HashMap::new();
        self.mat_ids = HashMap::new();
        self.current();
        self.close_instance();
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

    /// Each face's box in its object's space.
    pub fn boxes(&self) -> Vec<Aabb> {
        (0..self.len()).map(|f| Aabb::of_tri(self.corner(f, 0), self.corner(f, 1), self.corner(f, 2))).collect()
    }

    /// Put faces in BVH leaf order (`order[slot]` = old face). An order over
    /// the whole array that keeps each instance's faces within its range.
    pub fn reorder(&mut self, order: &[u32]) {
        self.faces = order.iter().map(|&i| self.faces[i as usize]).collect();
    }

    /// Largest single buffer in bytes.
    pub fn biggest(&self) -> u64 {
        [self.pos.len() * 32, self.faces.len() * 16, self.atlas.words().len() * 4, self.mats.len() * 448 + 64]
            .into_iter()
            .max()
            .unwrap_or(0) as u64
    }

    /// `[lamp count, face ids…, cdf bits…, materials…]` (see `wgsl/layout.wgsl`).
    pub fn table(&self) -> Vec<u32> {
        Self::table_of(&self.faces, &self.pos, &self.mats, &self.insts)
    }

    /// [`Scene::table`] over `faces` at `pos` placed by `insts` — a new pose
    /// of the same faces (`trace/pose.rs`). An emitter's power is its area
    /// in the world, so each corner is placed first.
    pub fn table_of(faces: &[[u32; 4]], pos: &[[f32; 4]], mats: &[Mat], insts: &[Instance]) -> Vec<u32> {
        let corner = |f: usize, k: usize| {
            let p = pos[faces[f][k] as usize];
            match insts.get(face_inst(faces[f][3]) as usize) {
                Some(inst) => inst.to_world([p[0], p[1], p[2]]),
                None => [p[0], p[1], p[2]],
            }
        };
        let area = |f: usize| {
            let (a, b, c) = (corner(f, 0), corner(f, 1), corner(f, 2));
            let (e1, e2) = ([0, 1, 2].map(|k| b[k] - a[k]), [0, 1, 2].map(|k| c[k] - a[k]));
            let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            0.5 * cr.iter().map(|v| v * v).sum::<f32>().sqrt()
        };
        let power: Vec<f32> = mats.iter().map(|m| m.power_in(mats)).collect();
        let (ids, pw): (Vec<u32>, Vec<f32>) = (0..faces.len())
            .filter(|&f| power[face_mat(faces[f][3]) as usize] > 0.0)
            .map(|f| (f as u32, area(f) * power[face_mat(faces[f][3]) as usize]))
            .filter(|(_, p)| *p > 0.0)
            .unzip();
        let total: f32 = pw.iter().sum();
        let mut run = 0.0;
        let mut w = vec![ids.len() as u32];
        w.extend(&ids);
        w.extend(pw.iter().map(|p| {
            run += p;
            (run / total).to_bits()
        }));
        w.extend(mats.iter().flat_map(|m| m.key()));
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::NO_TEX;

    #[test]
    fn shares_vertices_and_materials() {
        let n = [[0.0, 0.0, 1.0]; 3];
        let c = [0.5; 4];
        let q = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let tris = [Tri::new([q[0], q[1], q[2]], n, c), Tri::new([q[0], q[2], q[3]], n, c)];
        let mut s = Scene::from_tris(&tris, &Atlas::default());
        assert_eq!((s.pos.len(), s.faces.len(), s.mats.len()), (4, 2, 1));
        assert_eq!(s.mats[0].tex, NO_TEX);
        s.seal();
        assert_eq!(s.insts.len(), 1, "world-space triangles: one instance, at the identity");
        assert_eq!((s.insts[0].faces(), face_inst(s.faces[1][3]), face_mat(s.faces[1][3])), (0..2, 0, 0));
        assert_eq!(oct([0.0, 0.0, -1.0]).map(|v| v.abs()), [1.0, 1.0]);
    }
}

#[cfg(test)]
mod instance_tests {
    use super::*;

    #[test]
    fn each_instance_owns_a_run_of_faces_and_names_itself_in_them() {
        let mut s = Scene::default();
        let n = [0.0, 0.0, 1.0];
        let a = s.instance(Instance::translation([5.0, 0.0, 0.0]));
        let v = [s.vertex([0.0; 3], n, [0.0; 2]), s.vertex([1.0, 0.0, 0.0], n, [0.0; 2]), s.vertex([0.0, 1.0, 0.0], n, [0.0; 2])];
        let plain = Tri::new([[0.0; 3]; 3], [n; 3], [0.5; 4]);
        let m = s.material(Mat::of(&plain));
        s.face(v, m);
        let b = s.instance(IDENTITY);
        let w = [s.vertex([0.0; 3], n, [0.0; 2]), s.vertex([2.0, 0.0, 0.0], n, [0.0; 2]), s.vertex([0.0, 2.0, 0.0], n, [0.0; 2])];
        s.face(w, m);
        s.face(w, m);
        s.seal();
        assert_eq!((a, b), (0, 1));
        assert_eq!(s.insts[0].faces(), 0..1);
        assert_eq!(s.insts[1].faces(), 1..3);
        assert_eq!((s.insts[0].vert_lo, s.insts[0].vert_hi, s.insts[1].vert_lo, s.insts[1].vert_hi), (0, 3, 3, 6));
        assert_eq!(face_inst(s.faces[0][3]), 0);
        assert_eq!(face_inst(s.faces[2][3]), 1);
        assert_eq!(face_mat(s.faces[2][3]), 0);
        // The light table places an emitter's corners in the world.
        let mut glow = Scene::default();
        glow.instance(Instance::translation([0.0, 0.0, 9.0]));
        let lit = Mat::of(&plain.with_optics([4.0, 4.0, 4.0], 0.0, 1.5));
        let v = [glow.vertex([0.0; 3], n, [0.0; 2]), glow.vertex([1.0, 0.0, 0.0], n, [0.0; 2]), glow.vertex([0.0, 1.0, 0.0], n, [0.0; 2])];
        let lm = glow.material(lit);
        glow.face(v, lm);
        glow.seal();
        let table = glow.table();
        assert_eq!(table[0], 1, "one emitter, its area unchanged by a translation");
    }
}
