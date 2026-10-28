use super::*;
use crate::trace::Tri;

/// Two meshes: a small cube and a bigger one, each under one instance.
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
fn the_top_tree_leads_and_each_mesh_tree_is_shifted_into_place() {
    let mut s = two_cubes();
    let (trees, order) = Trees::build(&s);
    assert_eq!(order.len(), 18);
    assert!(order[..6].iter().all(|&f| f < 6) && order[6..].iter().all(|&f| (6..18).contains(&f)), "each mesh's faces stay in its range: {order:?}");
    s.reorder(&order);
    let nodes = trees.nodes(&s.meshes);
    let t = trees.tlas_slots();
    assert_eq!((trees.tlas.nodes.len(), t), (3, 3));
    assert!(nodes[..t].iter().filter(|n| n.is_leaf()).all(|n| n.lo.i & INST != 0), "top leaves are flagged");
    let roots = trees.roots();
    assert_eq!(roots[0], t as u32, "the first mesh's tree follows the top tree's slots");
    assert_eq!(roots[1], t as u32 + trees.blas[0].nodes.len() as u32);
    // Every mesh leaf ranges within its mesh's faces.
    for (k, mesh) in s.meshes.iter().enumerate() {
        let (lo, hi) = (roots[k] as usize, roots[k] as usize + trees.blas[k].nodes.len());
        for n in &nodes[lo..hi] {
            match n.is_leaf() {
                true => {
                    let (a, b) = ((n.lo.i & !(crate::bvh::LEAF | INST)) as usize, n.hi.i as usize);
                    assert!(a >= mesh.faces().start && b <= mesh.faces().end, "{a}..{b} in {:?}", mesh.faces());
                }
                false => assert!((n.lo.i as usize) >= lo && (n.hi.i as usize) < hi, "children stay in the tree"),
            }
        }
    }
    // The world spans both cubes; the instance table names both roots.
    assert!(trees.extent() > 10.0);
    let words = trees.inst_words(&s.insts, &s.meshes);
    let w = INSTANCE_WORDS;
    assert_eq!(words[24], roots[0]);
    assert_eq!(words[w + 24], roots[1]);
    assert_eq!(&words[2 * w..2 * w + 2], &trees.inst_order[..]);
    assert_eq!(&words[2 * w + 2..2 * w + 4], &[0, 1], "the draw order: each mesh's one instance");
    assert_eq!(trees.draws, vec![[0, 18, 0, 1], [18, 36, 1, 1]]);
}

#[test]
fn moving_an_instance_moves_only_the_top_tree() {
    let mut s = two_cubes();
    let (mut trees, order) = Trees::build(&s);
    s.reorder(&order);
    let before = trees.nodes(&s.meshes);
    s.insts[1].world = Instance::translation([0.0, 50.0, 0.0]);
    trees.retlas(&s.insts, &s.meshes);
    let after = trees.nodes(&s.meshes);
    let t = trees.tlas_slots();
    assert_eq!(&before[t..], &after[t..], "the mesh trees are untouched");
    assert!(after[0].bounds().hi[1] >= 50.0, "the root reaches the moved cube");
    assert_eq!((trees.tlas_nodes().len(), &trees.tlas_nodes()[..], &after[..t]), (t, &after[..t], &after[..t]));
}

/// The prepass's draw list covers every instance of every mesh with faces,
/// exactly once, and a move leaves it alone: `draws` and `draw_order` name
/// which mesh each instance shows, never where it stands, so the three
/// calls that place instances again (`repose`, `reinstance`, `retree`,
/// through `retlas`) cannot stale it. A scene whose mesh assignment did
/// change fails `Pose::same_as` and is built afresh, which regroups.
#[test]
fn the_draw_list_covers_every_instance_and_a_move_leaves_it_alone() {
    let mut s = two_cubes();
    let hash = crate::trace::hash::ObjectHash(0);
    s.place(0, Instance::translation([0.0, 20.0, 0.0]), [1.0; 3], hash);
    s.seal();
    let (mut trees, _) = Trees::build(&s);
    let covered = |t: &Trees| {
        let mut seen: Vec<u32> = t.draws.iter().flat_map(|&[_, _, at, n]| t.draw_order[at as usize..(at + n) as usize].to_vec()).collect();
        seen.sort_unstable();
        seen
    };
    assert!(!trees.draws.is_empty(), "a scene with faces draws something");
    assert_eq!(covered(&trees), vec![0, 1, 2], "every instance is drawn once");
    assert_eq!(trees.draws.iter().map(|d| d[3]).sum::<u32>(), s.insts.len() as u32);
    // A move: the top tree is built again, the draw list is not touched.
    let (draws, order) = (trees.draws.clone(), trees.draw_order.clone());
    s.insts[2].world = Instance::translation([0.0, 90.0, 0.0]);
    trees.retlas(&s.insts, &s.meshes);
    assert_eq!((trees.draws, trees.draw_order), (draws, order));
}

/// One mesh under three instances: one tree, three top-tree leaves, and
/// one instanced draw of three.
#[test]
fn a_shared_mesh_has_one_tree_and_one_draw_of_many() {
    let mut s = two_cubes();
    let hash = crate::trace::hash::ObjectHash(0);
    s.place(0, Instance::translation([0.0, 20.0, 0.0]), [1.0; 3], hash);
    s.place(0, Instance::translation([0.0, 40.0, 0.0]), [1.0; 3], hash);
    s.seal();
    let (trees, _) = Trees::build(&s);
    assert_eq!((trees.blas.len(), s.insts.len(), trees.tlas_slots()), (2, 4, 7));
    assert_eq!(trees.inst_order.len(), 4, "every instance is a top-tree leaf");
    assert_eq!(trees.draws, vec![[0, 18, 0, 3], [18, 36, 3, 1]]);
    assert_eq!(trees.draw_order, vec![0, 2, 3, 1]);
    let words = trees.inst_words(&s.insts, &s.meshes);
    let w = INSTANCE_WORDS;
    assert_eq!((words[2 * w + 24], words[2 * w + 38]), (words[24], 0), "the third instance names the first mesh's root and id");
    assert!(trees.bounds().hi[1] >= 41.0);
}
