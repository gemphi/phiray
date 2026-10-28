mod shared {
    use super::super::*;
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
        assert_eq!((s.meshes.len(), s.insts.len()), (1, 1), "world-space triangles: one mesh, one instance at the identity");
        assert_eq!((s.meshes[0].faces(), s.insts[0].mesh, face_mat(s.faces[1][3])), (0..2, 0, 0));
        assert_eq!(oct([0.0, 0.0, -1.0]).map(|v| v.abs()), [1.0, 1.0]);
    }
}


mod instances {
    use super::super::*;

    #[test]
    fn each_mesh_owns_a_run_of_faces_and_an_instance_may_share_one() {
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
        // A third object showing the first mesh again, elsewhere.
        let c = s.place(0, Instance::translation([0.0, 7.0, 0.0]), [1.0; 3], crate::trace::hash::ObjectHash(9));
        s.seal();
        assert_eq!((a, b, c), (0, 1, 2));
        assert_eq!((s.meshes.len(), s.insts.len()), (2, 3));
        assert_eq!(s.meshes[0].faces(), 0..1);
        assert_eq!(s.meshes[1].faces(), 1..3);
        assert_eq!((s.meshes[0].verts(), s.meshes[1].verts()), (0..3, 3..6));
        assert_eq!((s.mesh_of(0), s.mesh_of(1), s.mesh_of(2), s.insts[2].hash), (0, 1, 0, 9));
        assert_eq!(face_mat(s.faces[2][3]), 0);
        assert_eq!(s.len(), 3, "the shared mesh's faces are held once");
        // The light table places an emitter's corners in the world, once
        // an instance.
        let mut glow = Scene::default();
        glow.instance(Instance::translation([0.0, 0.0, 9.0]));
        let lit = Mat::of(&plain.with_optics([4.0, 4.0, 4.0], 0.0, 1.5));
        let v = [glow.vertex([0.0; 3], n, [0.0; 2]), glow.vertex([1.0, 0.0, 0.0], n, [0.0; 2]), glow.vertex([0.0, 1.0, 0.0], n, [0.0; 2])];
        let lm = glow.material(lit);
        glow.face(v, lm);
        glow.place(0, IDENTITY, [1.0; 3], crate::trace::hash::ObjectHash(0));
        glow.seal();
        let table = glow.table();
        assert_eq!(&table[..5], &[2, 0, 0, 0, 1], "two emitters: the one face under each instance");
        assert_eq!(f32::from_bits(table[5]), 0.5, "the same area each, so the first holds half the power");
    }
}
