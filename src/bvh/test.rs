//! CPU builder invariants (GPU equivalence lives in `tests/bvh.rs`).

use super::*;

/// Deterministic pseudo-random triangles, with duplicates to force equal codes.
pub fn prims(n: usize) -> Vec<Aabb> {
    let mut x = 0x9E37_79B9u32;
    let mut r = move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        (x % 10_000) as f32 * 0.01
    };
    (0..n)
        .map(|i| {
            if i % 10 == 9 {
                return Aabb::new([1.0; 3], [1.5; 3]);
            }
            let a = [r(), r(), r()];
            Aabb::of_tri(a, [a[0] + 1.0, a[1], a[2]], [a[0], a[1] + 1.0, a[2] + r() * 0.1])
        })
        .collect()
}

pub fn check(b: &Bvh, prims: &[Aabb]) {
    let mut seen = vec![0u32; prims.len()];
    for i in b.reachable() {
        let nd = b.nodes[i];
        if nd.is_leaf() {
            for s in nd.lo.index()..nd.hi.index() {
                let p = b.order[s as usize] as usize;
                seen[p] += 1;
                assert!(nd.bounds().contains(&prims[p]));
            }
        } else {
            for c in [nd.lo.index(), nd.hi.index()] {
                assert!(nd.bounds().contains(&b.nodes[c as usize].bounds()));
            }
        }
    }
    assert!(seen.iter().all(|&c| c == 1), "every prim exactly once");
    assert!(b.depth < STACK as u32);
}

#[test]
fn builds_valid_trees() {
    for n in [1, 2, 3, 7, 100, 5_000] {
        let p = prims(n);
        for leaf in [1, 4] {
            check(&Bvh::build(&p, leaf), &p);
        }
    }
    assert!(Bvh::build(&[], 4).nodes.is_empty());
}

#[test]
fn all_equal_codes_stay_shallow() {
    let p = vec![Aabb::new([0.0; 3], [1.0; 3]); 1000];
    let b = Bvh::build(&p, 1);
    check(&b, &p);
}

#[test]
fn compaction_keeps_the_tree() {
    let p = prims(5_000);
    let mut b = Bvh::build(&p, 4);
    let before = b.nodes.len();
    b.compact();
    assert!(b.nodes.len() < before / 2 + 2, "{} of {before}", b.nodes.len());
    assert_eq!(b.reachable().len(), b.nodes.len());
    check(&b, &p);
    assert_eq!(Bvh::build_compact(&p, 4), b, "direct compact build == build + compact");
    let flat = vec![Aabb::new([0.0; 3], [1.0; 3]); 1000];
    let mut deep = Bvh::build(&flat, 1);
    deep.compact();
    assert_eq!(Bvh::build_compact(&flat, 1), deep);
}

#[test]
fn sah_builds_valid_trees() {
    for n in [1, 2, 3, 7, 100, 5_000, 70_000] {
        let p = prims(n);
        for leaf in [1, 4] {
            check(&Bvh::build_sah(&p, leaf), &p);
        }
    }
    assert!(Bvh::build_sah(&[], 4).nodes.is_empty());
    // Every centre in one point: no plane to split on, still a valid tree.
    let flat = vec![Aabb::new([0.0; 3], [1.0; 3]); 1000];
    check(&Bvh::build_sah(&flat, 4), &flat);
}

/// What the SAH is for: a tree whose nodes, weighted by how likely a ray
/// is to enter them (their area against the root's), cost less than the
/// Morton tree's over the same primitives.
#[test]
fn sah_trees_are_cheaper_to_walk_than_morton_ones() {
    fn area(b: &Aabb) -> f32 {
        let d: [f32; 3] = std::array::from_fn(|k| (b.hi[k] - b.lo[k]).max(0.0));
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }
    fn cost(b: &Bvh) -> f32 {
        let root = area(&b.nodes[0].bounds());
        b.reachable()
            .into_iter()
            .map(|i| {
                let nd = b.nodes[i];
                let held = if nd.is_leaf() { (nd.hi.index() - nd.lo.index()) as f32 } else { 1.0 };
                area(&nd.bounds()) / root * held
            })
            .sum()
    }
    let p = prims(20_000);
    let (sah, morton) = (cost(&Bvh::build_sah(&p, 4)), cost(&Bvh::build_compact(&p, 4)));
    assert!(sah < morton * 0.9, "SAH {sah:.1} against Morton {morton:.1}");
}
