//! A new pose of the same scene without building the tracer again.
//!
//! Objects moved, turned or scaled whole (`reinstance`): each is an
//! instance of a mesh with a tree of its own over its faces in its own
//! space, so a new matrix changes nothing under it — the instance table
//! and the small top tree go up again (`Trees::retlas`), a few kilobytes,
//! and the image starts over. Vertices moved (`repose`: a frame of an
//! animation, a mesh edited): the vertices go up in place and each mesh's
//! tree keeps its shape, its bounds refitted over the moved faces
//! (`Bvh::refit_slots`). No sort, no new buffers, no new bind groups, and
//! the atlas (images, the world) stays where it is.
//!
//! A refitted tree is looser than a fresh one once things move far, so once
//! the motion stops the caller asks for new trees (`retree`, phiber's
//! render thread does): built over the pose held, the faces swapped into
//! their order in their buffers — and the image keeps its samples, since a
//! tree only decides how fast a ray finds what it hits, never what.

use super::instance::Instance;
use super::mesh::Mesh;
use super::trees::Trees;
use super::{Scene, Tracer};
use crate::bvh::Aabb;
use crate::gpu::{Gpu, GpuArray};
use crate::trace::Mat;
use rayon::prelude::*;

/// What a new pose keeps of the build: the vertex positions (object
/// space), the faces in leaf order and which face of the scene each slot
/// holds, the meshes and their trees, the instances, the materials (for
/// the light table) and the byte limit the arrays were split by.
pub(super) struct Pose {
    pub pos: Vec<[f32; 4]>,
    pub faces: Vec<[u32; 4]>,
    pub order: Vec<u32>,
    pub mats: Vec<Mat>,
    pub meshes: Vec<Mesh>,
    pub insts: Vec<Instance>,
    pub trees: Trees,
    pub limit: u64,
}

impl Pose {
    /// Each face's bounds in its mesh's space, in leaf order.
    fn boxes(&self) -> Vec<Aabb> {
        let corner = |f: &[u32; 4], k: usize| {
            let p = self.pos[f[k] as usize];
            [p[0], p[1], p[2]]
        };
        self.faces.par_iter().map(|f| Aabb::of_tri(corner(f, 0), corner(f, 1), corner(f, 2))).collect()
    }

    /// Whether any material emits (a light table to keep).
    fn glows(&self) -> bool {
        self.mats.iter().any(|m| m.power() > 0.0)
    }

    /// Whether `scene` is this pose's scene with its vertices elsewhere:
    /// the same meshes over the same faces, corner for corner, under the
    /// same instances.
    fn same_as(&self, scene: &Scene, meshes: &[Mesh]) -> bool {
        scene.pos.len() == self.pos.len()
            && scene.attr.len() == self.pos.len()
            && scene.faces.len() == self.faces.len()
            && meshes.len() == self.meshes.len()
            && scene.insts.len() == self.insts.len()
            && meshes.iter().zip(&self.meshes).all(|(a, b)| a.faces() == b.faces() && a.verts() == b.verts())
            && scene.insts.iter().zip(&self.insts).all(|(a, b)| a.mesh == b.mesh)
            && self.order.par_iter().zip(self.faces.par_iter()).all(|(&o, f)| scene.faces.get(o as usize).is_some_and(|g| g[..3] == f[..3]))
    }
}

impl Tracer {
    /// Take `scene` as a new pose of the one built: the same vertices and
    /// faces, each vertex somewhere else. `false` — build the tracer again —
    /// when it is not (a face or a vertex more or less, other corners).
    pub fn repose(&mut self, gpu: &Gpu, scene: &Scene) -> bool {
        let Some(pose) = self.pose.as_mut() else { return false };
        let meshes = scene.placed();
        if !pose.same_as(scene, &meshes) {
            return false;
        }
        pose.pos.copy_from_slice(&scene.pos);
        pose.meshes = meshes;
        pose.insts = scene.insts.clone();
        let boxes = pose.boxes();
        for (k, mesh) in pose.meshes.iter().enumerate() {
            pose.trees.refit(k, &boxes[mesh.faces()]);
        }
        pose.trees.retlas(&pose.insts, &pose.meshes);
        self.extent = pose.trees.extent();
        self.bounds = pose.trees.bounds();
        let verts: Vec<[f32; 8]> = scene.pos.par_iter().zip(&scene.attr).map(|(p, a)| [p[0], p[1], p[2], p[3], a[0], a[1], a[2], a[3]]).collect();
        let [n_at, _, v_at, _] = self.split;
        let fits = Self::refill(gpu, &mut self.nodes, &pose.trees.nodes(&pose.meshes), n_at) && Self::refill(gpu, &mut self.verts, &verts, v_at);
        // The image starts again: what was accumulated shows the old pose.
        self.last = None;
        self.frame = 0;
        fits && self.reinst(gpu) && self.relight(gpu)
    }

    /// Place the instances by `worlds` (one matrix each, Blender's
    /// `m[col][row]`), the meshes under them unchanged: what moving,
    /// turning or scaling objects takes. `false` when the count differs
    /// (build the tracer again) or the device cannot take the table.
    pub fn reinstance(&mut self, gpu: &Gpu, worlds: &[[[f32; 4]; 4]]) -> bool {
        let Some(pose) = self.pose.as_mut() else { return false };
        if worlds.len() != pose.insts.len() {
            return false;
        }
        for (inst, world) in pose.insts.iter_mut().zip(worlds) {
            inst.world = *world;
        }
        pose.trees.retlas(&pose.insts, &pose.meshes);
        self.extent = pose.trees.extent();
        self.bounds = pose.trees.bounds();
        let top = pose.trees.tlas_nodes();
        let fits = Self::refill(gpu, &mut self.nodes, &top, self.split[0]);
        self.last = None;
        self.frame = 0;
        fits && self.reinst(gpu) && self.relight(gpu)
    }

    /// Build each mesh's tree again over the pose held, keeping the image
    /// and its samples. `false` when there is no pose or the device cannot
    /// hold it.
    pub fn retree(&mut self, gpu: &Gpu) -> bool {
        let Some(pose) = self.pose.as_mut() else { return false };
        let boxes = pose.boxes();
        for k in 0..pose.meshes.len() {
            let range = pose.meshes[k].faces();
            let fresh = pose.trees.rebuild(k, &boxes[range.clone()]);
            // Slot → old slot: the faces move into the new tree's order, and
            // the order still names each face as the scene gave it.
            let (faces, order) = (pose.faces[range.clone()].to_vec(), pose.order[range.clone()].to_vec());
            for (slot, &old) in fresh.iter().enumerate() {
                pose.faces[range.start + slot] = faces[old as usize];
                pose.order[range.start + slot] = order[old as usize];
            }
        }
        pose.trees.retlas(&pose.insts, &pose.meshes);
        let nodes = pose.trees.nodes(&pose.meshes);
        let [n_at, f_at, ..] = self.split;
        if !Self::refill(gpu, &mut self.nodes, &nodes, n_at) {
            // More nodes than the arrays hold: new arrays, and the pass is
            // recorded again over them (the image buffer stays).
            let Some((arrays, at)) = GpuArray::split(gpu, &nodes, pose.limit, self.nodes.len()) else { return false };
            (self.nodes, self.split[0]) = (arrays, at);
            self.pass.clear();
        }
        Self::refill(gpu, &mut self.faces, &pose.faces, f_at) && self.reinst(gpu) && self.relight(gpu)
    }

    /// The instance table as the instances stand: their matrices and the
    /// roots of their meshes' trees.
    fn reinst(&mut self, gpu: &Gpu) -> bool {
        let Some(pose) = self.pose.as_ref() else { return false };
        self.data.write_at(gpu, self.insts[0] as usize, &pose.trees.inst_words(&pose.insts, &pose.meshes))
    }

    /// Lamps: the light table follows the faces — each emissive face's
    /// place under each instance, and its share of the light by area.
    /// `false` when it would reach into the atlas behind it.
    fn relight(&mut self, gpu: &Gpu) -> bool {
        let Some(pose) = self.pose.as_ref().filter(|p| p.glows()) else { return true };
        let table = super::lamps::table_of(&pose.faces, &pose.pos, &pose.mats, &pose.meshes, &pose.insts);
        if table.len() > self.split[3] as usize || !self.data.write_front(gpu, &table) {
            return false;
        }
        self.glow = table[0];
        true
    }

    /// Write `items` over `arrays` as `GpuArray::split` laid them, `per` to
    /// a binding, the items past `items.len()` kept; `false` when one had
    /// to grow (its binding changed).
    fn refill<T: bytemuck::Pod>(gpu: &Gpu, arrays: &mut [GpuArray<T>], items: &[T], per: u32) -> bool {
        let per = per.max(1) as usize;
        arrays.iter_mut().enumerate().all(|(k, array)| match items.get(k * per..((k + 1) * per).min(items.len())) {
            Some(part) if !part.is_empty() => part.len() <= array.len() && array.write_at(gpu, 0, part) || !array.write(gpu, part),
            _ => true,
        })
    }
}
