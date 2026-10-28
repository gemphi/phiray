//! Adding geometry to a [`Scene`]: vertices (shared or appended as they
//! come), their colour attribute, materials and faces. The faces join the
//! mesh being built (`Scene::instance_of`), in its own space.

use super::{Scene, pack};
use crate::trace::kernel::Tri;
use crate::trace::mat::Mat;

impl Scene {
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
        let byte = |c: f32| u32::from(crate::trace::srgb::Srgb::linear_to_byte(c));
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

    /// Append a face over existing vertex ids, in the current mesh.
    pub fn face(&mut self, v: [u32; 3], mat: u32) {
        self.current();
        self.faces.push([v[0], v[1], v[2], mat]);
    }

    pub fn push(&mut self, p: [[f32; 3]; 3], n: [[f32; 3]; 3], uv: [[f32; 2]; 3], mat: Mat) {
        let m = self.material(mat);
        let v = [0, 1, 2].map(|k| self.vert(p[k], n[k], uv[k]));
        self.face(v, m);
    }

}
