//! The scene's trees: one over each instance's faces in its own space
//! (`blas`), and one over the instances' world bounds (`tlas`), laid out
//! for the kernel as a single node array — the top tree first, so the root
//! is node 0, then each instance's tree shifted to its place
//! (`Trees::nodes`). A top-tree leaf names instance slots and carries the
//! `INST` bit (`bvh.wgsl`); an instance's record says where its tree starts
//! (`Instance::words`).

use super::instance::Instance;
use super::scene::Scene;
use crate::bvh::{Aabb, Bvh, Node, Wide, WIDTH};
use rayon::prelude::*;

/// Marks a top-tree leaf, whose range is instance slots, not faces.
pub const INST: u32 = 1 << 30;
/// Faces per leaf of an instance's tree.
const LEAF: u32 = 4;

pub struct Trees {
    /// One per instance; empty for an instance with no faces.
    pub blas: Vec<Bvh>,
    pub tlas: Bvh,
    /// Top-tree slot → instance id.
    pub inst_order: Vec<u32>,
}

impl Trees {
    /// Build every instance's tree over `scene`'s faces (in its own space)
    /// and the top tree over their placed bounds. Returns the trees and the
    /// face order the scene takes (`Scene::reorder`): each instance's faces
    /// permuted within its own range.
    pub fn build(scene: &Scene) -> (Self, Vec<u32>) {
        let boxes = scene.boxes();
        let blas: Vec<Bvh> = scene
            .insts
            .par_iter()
            .map(|inst| if inst.faces().is_empty() { Bvh::default() } else { Bvh::build_compact(&boxes[inst.faces()], LEAF) })
            .collect();
        let mut order: Vec<u32> = (0..scene.len() as u32).collect();
        for (inst, tree) in scene.insts.iter().zip(&blas) {
            for (slot, &face) in tree.order.iter().enumerate() {
                order[inst.face_lo as usize + slot] = inst.face_lo + face;
            }
        }
        let mut trees = Self { blas, tlas: Bvh::default(), inst_order: Vec::new() };
        trees.retlas(&scene.insts);
        (trees, order)
    }

    /// Node slots kept for the top tree: the most a tree over the instances
    /// takes, so a top tree built again (`retlas`) with a node more or less
    /// never moves the instance trees behind it.
    pub fn tlas_slots(&self) -> usize {
        (2 * self.blas.len()).saturating_sub(1).max(1)
    }

    /// Each instance's tree root in the node array, `u32::MAX` for one
    /// without faces.
    pub fn roots(&self) -> Vec<u32> {
        let mut at = self.tlas_slots() as u32;
        self.blas
            .iter()
            .map(|tree| {
                if tree.wide.is_empty() {
                    return u32::MAX;
                }
                let root = at;
                at += tree.wide.len() as u32;
                root
            })
            .collect()
    }

    /// The top tree again over the instances as placed now.
    pub fn retlas(&mut self, insts: &[Instance]) {
        let placed: Vec<(u32, Aabb)> = insts
            .iter()
            .zip(&self.blas)
            .enumerate()
            .filter(|(_, (_, tree))| !tree.nodes.is_empty())
            .map(|(i, (inst, tree))| (i as u32, inst.world_box(tree.nodes[0].bounds())))
            .collect();
        let boxes: Vec<Aabb> = placed.iter().map(|(_, b)| *b).collect();
        self.tlas = Bvh::build_compact(&boxes, 1);
        self.inst_order = self.tlas.order.iter().map(|&slot| placed[slot as usize].0).collect();
    }

    /// The whole node array for the kernel — the four-wide view
    /// (`bvh::Wide`), since that is what the kernel walks: the top tree in
    /// its slots, its leaves flagged and ranging over instance slots, then
    /// each instance's tree with its children and face slots shifted into
    /// place. The binary trees stay as they are; this is a view of them.
    pub fn nodes(&self, insts: &[Instance]) -> Vec<Wide> {
        let mut out: Vec<Wide> = self
            .tlas
            .wide
            .iter()
            .map(|w| {
                let mut w = *w;
                for k in 0..WIDTH {
                    if w.filled(k) && w.is_leaf_child(k) {
                        w.c[k] |= INST;
                    }
                }
                w
            })
            .collect();
        out.resize(self.tlas_slots(), Wide::default());
        for (inst, tree) in insts.iter().zip(&self.blas) {
            let base = out.len() as u32;
            out.extend(tree.wide.iter().map(|w| {
                let mut w = *w;
                for k in 0..WIDTH {
                    if !w.filled(k) {
                        continue;
                    }
                    if w.is_leaf_child(k) {
                        w.c[k] = ((w.c[k] & !crate::bvh::LEAF) + inst.face_lo) | crate::bvh::LEAF;
                        w.e[k] += inst.face_lo;
                    } else {
                        w.c[k] += base;
                    }
                }
                w
            }));
        }
        out
    }

    /// The top tree's slots alone, as `nodes` lays them out: what a move
    /// uploads.
    pub fn tlas_nodes(&self) -> Vec<Wide> {
        self.nodes(&[])
    }

    /// The instance table for the kernel: each record, then the top-tree
    /// slot order.
    pub fn inst_words(&self, insts: &[Instance]) -> Vec<u32> {
        let roots = self.roots();
        let mut words: Vec<u32> = insts.iter().zip(roots).flat_map(|(inst, root)| inst.words(root)).collect();
        words.extend(&self.inst_order);
        words.resize(insts.len() * super::instance::INSTANCE_WORDS + insts.len(), 0);
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

    /// Instance `k`'s tree refitted over `boxes`, its faces' boxes in slot
    /// order (a new pose of the same faces).
    pub fn refit(&mut self, k: usize, boxes: &[Aabb]) {
        if !boxes.is_empty() {
            self.blas[k].refit_slots(boxes);
        }
    }

    /// Instance `k`'s tree built again over `boxes` (slot order); the
    /// slot → old slot order of the new tree, for the faces to follow.
    pub fn rebuild(&mut self, k: usize, boxes: &[Aabb]) -> Vec<u32> {
        if boxes.is_empty() {
            return Vec::new();
        }
        let fresh = Bvh::build_compact(boxes, LEAF);
        let order = fresh.order.clone();
        self.blas[k] = Bvh { order: (0..boxes.len() as u32).collect(), ..fresh };
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::Tri;

    fn two_cubes() -> Scene {
        let mut s = Scene::default();
        let n = [0.0, 0.0, 1.0];
        let tri = Tri::new([[0.0; 3]; 3], [n; 3], [0.5; 4]);
        for (k, at) in [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]].into_iter().enumerate() {
            s.instance(Instance::translation(at));
            let m = s.material(crate::trace::Mat::of(&tri));
            for f in 0..(k + 1) * 6 {
                let z = f as f32 * 0.1;
                let v = [s.vertex([0.0, 0.0, z], n, [0.0; 2]), s.vertex([1.0, 0.0, z], n, [0.0; 2]), s.vertex([0.0, 1.0, z], n, [0.0; 2])];
                s.face(v, m);
            }
        }
        s.seal();
        s
    }

    #[test]
    fn the_top_tree_leads_and_each_instance_tree_is_shifted_into_place() {
        let mut s = two_cubes();
        let (trees, order) = Trees::build(&s);
        assert_eq!(order.len(), 18);
        assert!(order[..6].iter().all(|&f| f < 6) && order[6..].iter().all(|&f| (6..18).contains(&f)), "each instance's faces stay in its range: {order:?}");
        s.reorder(&order);
        let nodes = trees.nodes(&s.insts);
        let t = trees.tlas_slots();
        assert_eq!((trees.tlas.nodes.len(), t), (3, 3));
        let filled = |w: &Wide| (0..WIDTH).filter(|&k| w.filled(k)).collect::<Vec<_>>();
        assert!(nodes[..t].iter().all(|w| filled(w).iter().all(|&k| !w.is_leaf_child(k) || w.c[k] & INST != 0)), "top leaves are flagged");
        let roots = trees.roots();
        assert_eq!(roots[0], t as u32, "the first instance's tree follows the top tree's slots");
        assert_eq!(roots[1], t as u32 + trees.blas[0].wide.len() as u32);
        // Every instance leaf ranges within its instance's faces.
        for (k, inst) in s.insts.iter().enumerate() {
            let (lo, hi) = (roots[k] as usize, roots[k] as usize + trees.blas[k].wide.len());
            for w in &nodes[lo..hi] {
                for j in filled(w) {
                    if w.is_leaf_child(j) {
                        let (a, b) = ((w.c[j] & !(crate::bvh::LEAF | INST)) as usize, w.e[j] as usize);
                        assert!(a >= inst.faces().start && b <= inst.faces().end, "{a}..{b} in {:?}", inst.faces());
                    } else {
                        assert!((w.c[j] as usize) >= lo && (w.c[j] as usize) < hi, "children stay in the tree");
                    }
                }
            }
        }
        // The world spans both cubes; the instance table names both roots.
        assert!(trees.extent() > 10.0);
        let words = trees.inst_words(&s.insts);
        let w = super::super::instance::INSTANCE_WORDS;
        assert_eq!(words[24], roots[0]);
        assert_eq!(words[w + 24], roots[1]);
        assert_eq!(&words[2 * w..2 * w + 2], &trees.inst_order[..]);
    }

    #[test]
    fn moving_an_instance_moves_only_the_top_tree() {
        let mut s = two_cubes();
        let (mut trees, order) = Trees::build(&s);
        s.reorder(&order);
        let before = trees.nodes(&s.insts);
        s.insts[1].world = Instance::translation([0.0, 50.0, 0.0]);
        trees.retlas(&s.insts);
        let after = trees.nodes(&s.insts);
        let t = trees.tlas_slots();
        assert_eq!(&before[t..], &after[t..], "the instance trees are untouched");
        assert!((0..WIDTH).any(|k| after[0].filled(k) && after[0].hiy[k] >= 50.0), "the root reaches the moved cube");
        assert_eq!((trees.tlas_nodes().len(), &trees.tlas_nodes()[..], &after[..t]), (t, &after[..t], &after[..t]));
    }
}
