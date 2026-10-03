//! Tree shape: compaction and reachability.

use super::build::Bvh;

impl Bvh {
    /// Drop nodes the packing made unreachable and renumber the rest
    /// breadth-first (root stays 0): about half of `2n - 1` with 4-prim leaves.
    pub fn compact(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let mut map = vec![u32::MAX; self.nodes.len()];
        let mut keep = vec![0usize];
        map[0] = 0;
        let mut i = 0;
        while i < keep.len() {
            let nd = self.nodes[keep[i]];
            if !nd.is_leaf() {
                for c in [nd.lo.index() as usize, nd.hi.index() as usize] {
                    map[c] = keep.len() as u32;
                    keep.push(c);
                }
            }
            i += 1;
        }
        self.nodes = keep
            .iter()
            .map(|&k| {
                let mut nd = self.nodes[k];
                if !nd.is_leaf() {
                    nd.lo.i = map[nd.lo.index() as usize];
                    nd.hi.i = map[nd.hi.index() as usize];
                }
                nd
            })
            .collect();
    }

    /// Reachable node indices (root first).
    pub fn reachable(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = if self.nodes.is_empty() { vec![] } else { vec![0usize] };
        while let Some(i) = stack.pop() {
            out.push(i);
            let nd = self.nodes[i];
            if !nd.is_leaf() {
                stack.extend([nd.lo.index() as usize, nd.hi.index() as usize]);
            }
        }
        out
    }
}
