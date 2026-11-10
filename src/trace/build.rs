//! Building a tracer: a scene laid into the device's bindings, the arrays
//! split over as many bindings as each needs, and the kernels compiled ahead
//! of the first frame of a mode (`Tracer::warm`).
use super::{Atlas, Preview, Preview4, Scene, Solid, Solid4, Trace, Trace4, Tracer, Tri, Wire, Wire4};
use super::{blue_noise, forest, instance, pose, smaa, taa};
use crate::gpu::{Clock, Gpu, GpuArray, Pass, Time};
use crate::mode::Mode;

impl Tracer {
    /// `None` for an empty scene.
    pub fn new(gpu: &Gpu, tris: &[Tri], atlas: &Atlas) -> Option<Self> {
        Self::from_scene(gpu, Scene::from_tris(tris, atlas))
    }

    /// Build from a compact scene; `None` when empty or larger than the device allows.
    pub fn from_scene(gpu: &Gpu, scene: Scene) -> Option<Self> {
        Self::with_limit(gpu, scene, gpu.max_binding())
    }

    /// Like [`Self::from_scene`] with an explicit per-binding byte limit, over as
    /// many bindings per array as the device allows ([`Self::chunks_for`]).
    pub fn with_limit(gpu: &Gpu, scene: Scene, limit: u64) -> Option<Self> {
        Self::with_chunks(gpu, scene, limit, Self::chunks_for(gpu))
    }

    /// Compile every viewport kernel on `gpu` — the ones a scene of this
    /// device's binding count will use — so switching shading does not wait
    /// a second for the first frame of a mode while its kernel compiles.
    /// Each compiles once per device; a kernel already compiled costs
    /// nothing here. Meant for a thread of its own once the first image is
    /// up: the compiles hold the device's pipeline lock one at a time.
    /// Returns each kernel's name and how long it took, in milliseconds (a
    /// first run on a driver with no shader cache can take seconds each).
    pub fn warm(gpu: &Gpu) -> Vec<(&'static str, f64)> {
        let timed = |name, compile: fn(&Gpu)| {
            // `web_time`, not `std::time`: `Instant::now()` panics on
            // wasm32-unknown-unknown, and warming the kernels is the first
            // thing a browser build would do.
            let at = web_time::Instant::now();
            compile(gpu);
            (name, at.elapsed().as_secs_f64() * 1000.0)
        };
        // Preview goes LAST, so the kernels anyone is likely to want are
        // ready first. A compile now holds only its own kernel's slot
        // (`Gpu::pipeline`), so nothing else waits behind it either; before
        // that it held the device's whole pipeline lock:
        //
        // These compiles held the device's pipeline lock one at a time, and
        // on this driver Material Preview takes **248 seconds** while the
        // path tracer takes 145 ms (session log, 2026-10-18: `Solid 0,
        // Preview 247997, Wire 3006, Trace 145`). Compiled second, as it
        // was, Preview stood in front of every kernel anything else needed
        // for four minutes: a render started in that window did not trace
        // at all, it blocked on the lock, and its first sample timed at
        // 143 s. Compiled last, Solid, Wire and Trace are all ready in
        // about three seconds and nothing the user asked for waits behind
        // a mode they may never open.
        //
        // Why Preview is three orders of magnitude slower than the tracer
        // over the same bindings is its own question (`preview.wgsl` pulls
        // in the procedural, Voronoi, Gabor and noise evaluators); this
        // only stops it blocking everything else meanwhile.
        if Self::chunks_for(gpu) == 4 {
            vec![timed("Solid", Pass::compile::<Solid4>), timed("Wire", Pass::compile::<Wire4>), timed("Trace", Pass::compile::<Trace4>), timed("Preview", Pass::compile::<Preview4>)]
        } else {
            vec![timed("Solid", Pass::compile::<Solid>), timed("Wire", Pass::compile::<Wire>), timed("Trace", Pass::compile::<Trace>), timed("Preview", Pass::compile::<Preview>)]
        }
    }

    /// 4 bindings per array when the device has 14 storage bindings, else 2.
    pub fn chunks_for(gpu: &Gpu) -> usize {
        if gpu.device.limits().max_storage_buffers_per_shader_stage >= 14 { 4 } else { 2 }
    }

    /// Each instance's tree is built on the CPU (all cores) over its faces
    /// in its own space, and a top tree over the instances as placed
    /// (`forest.rs`); nodes, faces and vertices are each split over `chunks`
    /// (2 or 4) bindings, so a scene may use `chunks ×` the limit per array
    /// (as Blender's GPU backends chunk large buffers).
    /// Why this scene will not build on a device with this binding limit,
    /// in the words the Info report shows — `None` when it fits.
    ///
    /// Every `None` out of [`Self::with_chunks`] used to look the same from
    /// outside: the app said "no adapter, or a scene the GPU declined" and
    /// the Render Result was blank. A frame that does not render has one
    /// specific reason and the user is entitled to it, so the checks state
    /// theirs and `with_chunks` reads from here rather than keeping a
    /// second copy of the arithmetic.
    pub fn why_not(scene: &Scene, limit: u64, chunks: usize) -> Option<String> {
        let chunks = if chunks > 2 { 4 } else { 2 } as u64;
        let mib = |b: u64| format!("{:.0} MB", b as f64 / 1048576.0);
        if scene.is_empty() {
            return Some("the scene has nothing in it".into());
        }
        let biggest = scene.biggest();
        if biggest > limit * chunks {
            return Some(format!(
                "its largest array is {} and this device binds {} over {chunks} chunks",
                mib(biggest),
                mib(limit * chunks)
            ));
        }
        // The one array that is NOT chunked: the scene table, the texture
        // atlas, the instance words and — when any object carries a colour
        // attribute — one word per vertex in the whole scene.
        let data = Self::data_words(scene) * 4;
        if data > limit {
            return Some(format!(
                "its material/texture table is {} and this device binds {} in one piece (it is not chunked: the atlas is {}, and a colour attribute adds one word per vertex over {} vertices)",
                mib(data),
                mib(limit),
                mib(scene.atlas.words().len() as u64 * 4),
                scene.pos.len()
            ));
        }
        for (what, items, each) in [
            ("vertices", scene.pos.len() as u64, 32u64),
            ("faces", scene.faces.len() as u64, 16),
        ] {
            let cap = (limit / each).max(1) * chunks;
            if items > cap {
                return Some(format!("it has {items} {what} and this device binds {cap} over {chunks} chunks"));
            }
        }
        None
    }

    /// The length, in words, of the unchunked `data` buffer `with_chunks`
    /// assembles — the figure [`Self::why_not`] checks and the one the
    /// builder itself lays out below.
    fn data_words(scene: &Scene) -> u64 {
        let colors = if scene.colors.is_empty() { 0 } else { scene.pos.len() as u64 };
        scene.table().len() as u64 + scene.atlas.words().len() as u64 + scene.insts.len() as u64 * (instance::INSTANCE_LEN as u64 + 2) + colors
    }

    pub fn with_chunks(gpu: &Gpu, mut scene: Scene, limit: u64, chunks: usize) -> Option<Self> {
        let chunks = if chunks > 2 { 4 } else { 2 };
        if scene.biggest() > limit * chunks as u64 {
            return None;
        }
        if scene.is_empty() {
            return None;
        }
        scene.seal();
        let (trees, order) = forest::Forest::build(&scene);
        // Faces in leaf order, so leaves index faces directly.
        scene.reorder(&order);
        let extent = trees.extent();
        let bounds = trees.bounds();
        let mut data = scene.table();
        let (glow, atlas_at) = (data[0], data.len() as u32);
        data.extend(scene.atlas.words());
        let insts_at = data.len() as u32;
        data.extend(trees.inst_words(&scene.insts, &scene.meshes));
        // The vertices' colour attribute, when any object has one (`Scene::paint`).
        let colors_at = if scene.colors.is_empty() { 0 } else { data.len() as u32 };
        if colors_at != 0 {
            data.extend(&scene.colors);
            data.resize(colors_at as usize + scene.pos.len(), 0xffff_ffff);
        }
        if data.len() as u64 * 4 > limit {
            return None;
        }
        let verts: Vec<[f32; 8]> = scene.pos.iter().zip(&scene.attr).map(|(p, a)| [p[0], p[1], p[2], p[3], a[0], a[1], a[2], a[3]]).collect();
        let (nodes, n_at) = GpuArray::split(gpu, &trees.nodes(&scene.meshes), limit, chunks)?;
        let (faces, f_at) = GpuArray::split(gpu, &scene.faces, limit, chunks)?;
        let (verts, v_at) = GpuArray::split(gpu, &verts, limit, chunks)?;
        let draws = trees.draws.clone();
        let pose = pose::Pose {
            pos: std::mem::take(&mut scene.pos),
            faces: std::mem::take(&mut scene.faces),
            mats: std::mem::take(&mut scene.mats),
            meshes: std::mem::take(&mut scene.meshes),
            insts: std::mem::take(&mut scene.insts),
            order,
            trees,
            limit,
        };
        Some(Self {
            glow,
            nodes,
            faces,
            verts,
            data: GpuArray::from(gpu, &data),
            px: None,
            vis: None,
            blank: GpuArray::zeroed(gpu, 4),
            noise: GpuArray::from(gpu, &blue_noise::BlueNoise::words()),
            smaa: GpuArray::from(gpu, &smaa::SmaaTableList::words()),
            class: GpuArray::zeroed(gpu, pose.insts.len().max(1)),
            classes: vec![0; pose.insts.len()],
            reshow: false,
            // Hardware rasterizes its primary hits (Workbench's prepass); a
            // software adapter traces them, where walking the tree is the
            // cheaper of the two (`visible.rs`).
            raster: !gpu.adapter.is_software(),
            draws,
            bounds,
            split: [n_at, f_at, v_at, atlas_at],
            insts: [insts_at, pose.insts.len() as u32, colors_at, 0],
            last: None,
            frame: 0,
            pass: Pass::default(),
            clock: Clock::new(gpu),
            time: Time::default(),
            size: (0, 0),
            extent,
            mode: Mode::Rendered,
            target: Self::SAMPLE_COUNT,
            aa: taa::SAMPLE_COUNT,
            pose: Some(pose),
            // Off until the render path resolves the scene's rows
            // (`Tracer::set_adaptive`).
            adapt: [0.0; 4],
        })
    }
}
