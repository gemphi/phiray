//! Emitter sampling table: the light table the kernel's next-event
//! estimation draws from (`glow_query`, `lamp_chance` in `light.wgsl`),
//! and the test-side `Lamps` over world-space triangles.

use super::instance::Instance;
use super::kernel::Tri;
use super::mat::Mat;
use super::mesh::Mesh;
use super::scene::face_mat;

/// Emissive triangles chosen in proportion to their power (area × luminance).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lamps {
    pub ids: Vec<u32>,
    /// Normalised cumulative power, `cdf.last() == 1`.
    pub cdf: Vec<f32>,
}

impl Lamps {
    pub fn of(tris: &[Tri]) -> Self {
        let power = |t: &Tri| {
            let (e1, e2) = ([0, 1, 2].map(|k| t.b[k] - t.a[k]), [0, 1, 2].map(|k| t.c[k] - t.a[k]));
            let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let area = 0.5 * cr.iter().map(|v| v * v).sum::<f32>().sqrt();
            area * (0.2126 * t.emit[0] + 0.7152 * t.emit[1] + 0.0722 * t.emit[2])
        };
        let (ids, pw): (Vec<u32>, Vec<f32>) =
            tris.iter().enumerate().map(|(i, t)| (i as u32, power(t))).filter(|(_, p)| *p > 0.0).unzip();
        let total: f32 = pw.iter().sum();
        let mut run = 0.0;
        let cdf = pw.iter().map(|p| { run += p; run / total }).collect();
        Self { ids, cdf }
    }

    /// GPU layout: `[count, ids…, cdf bits…]`.
    pub fn words(&self) -> Vec<u32> {
        let mut w = vec![self.ids.len() as u32];
        w.extend(&self.ids);
        w.extend(self.cdf.iter().map(|c| c.to_bits()));
        w
    }
}

/// `[lamp count, face ids…, instance ids…, cdf bits…, materials…]` over
/// `faces` at `pos`, the meshes placed by `insts` — a new pose of the same
/// faces (`trace/pose.rs`). An emissive face is one emitter under each
/// instance of its mesh (Cycles' `light_tree`/`distribution` list an
/// emissive triangle once an object): its power is its area in the world,
/// so each corner is placed first. The pairs come out sorted by face, then
/// instance — `lamp_chance` searches them so.
pub fn table_of(faces: &[[u32; 4]], pos: &[[f32; 4]], mats: &[Mat], meshes: &[Mesh], insts: &[Instance]) -> Vec<u32> {
    let power: Vec<f32> = mats.iter().map(|m| m.power_in(mats)).collect();
    let mut by_mesh: Vec<Vec<u32>> = vec![Vec::new(); meshes.len()];
    for (i, inst) in insts.iter().enumerate() {
        if let Some(list) = by_mesh.get_mut(inst.mesh as usize) {
            list.push(i as u32);
        }
    }
    let corner = |f: usize, k: usize, inst: &Instance| {
        let p = pos[faces[f][k] as usize];
        inst.to_world([p[0], p[1], p[2]])
    };
    let area = |f: usize, inst: &Instance| {
        let (a, b, c) = (corner(f, 0, inst), corner(f, 1, inst), corner(f, 2, inst));
        let (e1, e2) = ([0, 1, 2].map(|k| b[k] - a[k]), [0, 1, 2].map(|k| c[k] - a[k]));
        let cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        0.5 * cr.iter().map(|v| v * v).sum::<f32>().sqrt()
    };
    let (mut ids, mut owners, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    for (mesh, placed) in meshes.iter().zip(&by_mesh) {
        for f in mesh.faces() {
            let Some(&glow) = faces.get(f).and_then(|face| power.get(face_mat(face[3]) as usize)) else { continue };
            if glow <= 0.0 {
                continue;
            }
            for &i in placed {
                let p = area(f, &insts[i as usize]) * glow;
                if p > 0.0 {
                    ids.push(f as u32);
                    owners.push(i);
                    pw.push(p);
                }
            }
        }
    }
    let total: f32 = pw.iter().sum();
    let mut run = 0.0;
    let mut w = vec![ids.len() as u32];
    w.extend(&ids);
    w.extend(&owners);
    w.extend(pw.iter().map(|p| {
        run += p;
        (run / total).to_bits()
    }));
    w.extend(mats.iter().flat_map(|m| m.key()));
    w
}
