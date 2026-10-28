//! The scene's trees: one over each mesh's faces in its own space
//! (`blas`), and one over the instances' world bounds (`tlas`), laid out
//! for the kernel as a single node array — the top tree first, so the root
//! is node 0, then each mesh's tree shifted to its place (`Trees::nodes`).
//! A top-tree leaf names instance slots and carries the `INST` bit
//! (`bvh.wgsl`); an instance's record says where its mesh's tree starts
//! (`Instance::words`). A mesh shown by many instances has one tree, as
//! Cycles builds one BVH per `Geometry` and an instance BVH over the
//! `Object`s (`scene/object.cpp`; roadmap §4.4, Step 1).

use super::instance::{INSTANCE_WORDS, Instance};
use super::mesh::Mesh;
use super::scene::Scene;
use crate::bvh::{Aabb, Bvh, Node};
use rayon::prelude::*;

/// Marks a top-tree leaf, whose range is instance slots, not faces.
pub const INST: u32 = 1 << 30;
/// Faces per leaf of a mesh's tree.
const LEAF: u32 = 4;

pub struct Trees {
    /// One per mesh; empty for a mesh with no faces.
    pub blas: Vec<Bvh>,
    pub tlas: Bvh,
    /// Top-tree slot → instance id.
    pub inst_order: Vec<u32>,
    /// The instances grouped by mesh: what the raster prepass draws, one
    /// instanced draw a mesh (`draws`).
    pub draw_order: Vec<u32>,
    /// Per mesh: `[first vertex, vertex count, first draw slot, instance
    /// count]` — one instanced draw (`Visibility::draw`); a mesh with no
    /// faces or no instance has none.
    pub draws: Vec<[u32; 4]>,
    /// Instances placed (the top tree's leaves).
    n_insts: usize,
}

impl Trees {
    /// Build every mesh's tree over `scene`'s faces (in its own space) and
    /// the top tree over the instances' placed bounds. Returns the trees
    /// and the face order the scene takes (`Scene::reorder`): each mesh's
    /// faces permuted within its own range.
    pub fn build(scene: &Scene) -> (Self, Vec<u32>) {
        let boxes = scene.boxes();
        let blas: Vec<Bvh> = scene
            .meshes
            .par_iter()
            .map(|mesh| if mesh.faces().is_empty() { Bvh::default() } else { Bvh::build_sah(&boxes[mesh.faces()], LEAF) })
            .collect();
        let mut order: Vec<u32> = (0..scene.len() as u32).collect();
        for (mesh, tree) in scene.meshes.iter().zip(&blas) {
            for (slot, &face) in tree.order.iter().enumerate() {
                order[mesh.face_lo as usize + slot] = mesh.face_lo + face;
            }
        }
        let mut trees = Self { blas, tlas: Bvh::default(), inst_order: Vec::new(), draw_order: Vec::new(), draws: Vec::new(), n_insts: scene.insts.len() };
        trees.regroup(&scene.insts, &scene.meshes);
        trees.retlas(&scene.insts, &scene.meshes);
        (trees, order)
    }

    /// Node slots kept for the top tree: the most a tree over the instances
    /// takes, so a top tree built again (`retlas`) with a node more or less
    /// never moves the mesh trees behind it.
    pub fn tlas_slots(&self) -> usize {
        (2 * self.n_insts).saturating_sub(1).max(1)
    }

    /// Each mesh's tree root in the node array, `u32::MAX` for one without
    /// faces.
    pub fn roots(&self) -> Vec<u32> {
        let mut at = self.tlas_slots() as u32;
        self.blas
            .iter()
            .map(|tree| {
                if tree.nodes.is_empty() {
                    return u32::MAX;
                }
                let root = at;
                at += tree.nodes.len() as u32;
                root
            })
            .collect()
    }

    /// The instances grouped by mesh, and one draw a mesh over its group.
    fn regroup(&mut self, insts: &[Instance], meshes: &[Mesh]) {
        let mut by_mesh: Vec<Vec<u32>> = vec![Vec::new(); meshes.len()];
        for (i, inst) in insts.iter().enumerate() {
            if let Some(list) = by_mesh.get_mut(inst.mesh as usize) {
                list.push(i as u32);
            }
        }
        self.draw_order.clear();
        self.draws.clear();
        for (mesh, group) in meshes.iter().zip(&by_mesh) {
            if !mesh.faces().is_empty() && !group.is_empty() {
                self.draws.push([mesh.face_lo * 3, (mesh.face_hi - mesh.face_lo) * 3, self.draw_order.len() as u32, group.len() as u32]);
            }
            self.draw_order.extend(group);
        }
        self.draw_order.resize(insts.len(), 0);
    }

    /// The top tree again over the instances as placed now.
    pub fn retlas(&mut self, insts: &[Instance], meshes: &[Mesh]) {
        self.n_insts = insts.len();
        let placed: Vec<(u32, Aabb)> = insts
            .iter()
            .enumerate()
            .filter_map(|(i, inst)| {
                let tree = self.blas.get(inst.mesh as usize).filter(|t| !t.nodes.is_empty())?;
                meshes.get(inst.mesh as usize)?;
                Some((i as u32, inst.world_box(tree.nodes[0].bounds())))
            })
            .collect();
        let boxes: Vec<Aabb> = placed.iter().map(|(_, b)| *b).collect();
        self.tlas = Bvh::build_sah(&boxes, 1);
        self.inst_order = self.tlas.order.iter().map(|&slot| placed[slot as usize].0).collect();
    }

    /// The whole node array for the kernel: the top tree in its slots, its
    /// leaves flagged and ranging over instance slots, then each mesh's
    /// tree with its children and face slots shifted into place.
    pub fn nodes(&self, meshes: &[Mesh]) -> Vec<Node> {
        let mut out: Vec<Node> = self
            .tlas
            .nodes
            .iter()
            .map(|n| {
                let mut n = *n;
                if n.is_leaf() {
                    n.lo.i |= INST;
                }
                n
            })
            .collect();
        out.resize(self.tlas_slots(), Node::default());
        for (mesh, tree) in meshes.iter().zip(&self.blas) {
            let base = out.len() as u32;
            out.extend(tree.nodes.iter().map(|n| {
                let mut n = *n;
                match n.is_leaf() {
                    true => {
                        n.lo.i = (n.lo.index() + mesh.face_lo) | crate::bvh::LEAF;
                        n.hi.i += mesh.face_lo;
                    }
                    false => {
                        n.lo.i += base;
                        n.hi.i += base;
                    }
                }
                n
            }));
        }
        out
    }

    /// The top tree's slots alone, as `nodes` lays them out: what a move
    /// uploads.
    pub fn tlas_nodes(&self) -> Vec<Node> {
        self.nodes(&[])
    }

    /// The instance table for the kernel: each record, then the top-tree
    /// slot order, then the draw order (`layout.wgsl`: `inst_order`,
    /// `draw_inst`).
    pub fn inst_words(&self, insts: &[Instance], meshes: &[Mesh]) -> Vec<u32> {
        let roots = self.roots();
        let unit = Mesh::from(0, 0);
        let mut words: Vec<u32> = insts
            .iter()
            .flat_map(|inst| {
                let k = inst.mesh as usize;
                inst.words(roots.get(k).copied().unwrap_or(u32::MAX), meshes.get(k).unwrap_or(&unit))
            })
            .collect();
        words.extend(&self.inst_order);
        words.resize(insts.len() * (INSTANCE_WORDS + 1), 0);
        words.extend(&self.draw_order);
        words.resize(insts.len() * (INSTANCE_WORDS + 2), 0);
        words
    }

    /// The world's bounds: the top tree's root.
    pub fn bounds(&self) -> Aabb {
        self.tlas.nodes.first().map_or(Aabb::new([0.0; 3], [0.0; 3]), Node::bounds)
    }

    /// The world's diagonal (ray offsets and AO reach scale by it).
    pub fn extent(&self) -> f32 {
        match self.tlas.nodes.first() {
            Some(root) => {
                let b = root.bounds();
                (0..3).map(|k| (b.hi[k] - b.lo[k]).powi(2)).sum::<f32>().sqrt().max(1e-3)
            }
            None => 1e-3,
        }
    }

    /// Mesh `k`'s tree refitted over `boxes`, its faces' boxes in slot
    /// order (a new pose of the same faces).
    pub fn refit(&mut self, k: usize, boxes: &[Aabb]) {
        if !boxes.is_empty() {
            self.blas[k].refit_slots(boxes);
        }
    }

    /// Mesh `k`'s tree built again over `boxes` (slot order); the slot →
    /// old slot order of the new tree, for the faces to follow.
    pub fn rebuild(&mut self, k: usize, boxes: &[Aabb]) -> Vec<u32> {
        if boxes.is_empty() {
            return Vec::new();
        }
        let fresh = Bvh::build_sah(boxes, LEAF);
        let order = fresh.order.clone();
        self.blas[k] = Bvh { order: (0..boxes.len() as u32).collect(), ..fresh };
        order
    }
}

#[cfg(test)]
mod tests;
