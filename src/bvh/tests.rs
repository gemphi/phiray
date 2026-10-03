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
