//! The solver's state arrays (ROADMAP.md §2.2 "State layout"): `Pod`, so
//! each `Vec<T>` here is the CPU twin of a `GpuArray<T>` with the same
//! bytes, and the WGSL structs that follow can be laid over them.

use bytemuck::{Pod, Zeroable};

/// One rigid body, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Body {
    /// Centre of mass, world.
    pub x: [f32; 3],
    /// 0 for passive and kinematic bodies.
    pub inv_mass: f32,
    /// Orientation, xyzw.
    pub q: [f32; 4],
    /// Linear velocity.
    pub v: [f32; 3],
    /// Index into the shapes.
    pub shape: u32,
    /// Angular velocity, world.
    pub w: [f32; 3],
    /// `Kind` in bits 0..2, then `SLEEPING` and `DEACTIVATES`.
    pub flags: u32,
}

/// Per-body constants, 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct BodyConst {
    /// Principal inverse inertia, body frame.
    pub inv_inertia: [f32; 3],
    pub friction: f32,
    /// Centre of mass in the scaled object space (object origin != COM).
    pub com_local: [f32; 3],
    pub restitution: f32,
    pub lin_damp: f32,
    pub ang_damp: f32,
    pub margin: f32,
    /// Blender's 20 collision collections as a bit mask.
    pub groups: u32,
}

/// The pose at the start of a substep (XPBD derives velocity from it).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct BodyPrev {
    pub x: [f32; 3],
    pub _a: f32,
    pub q: [f32; 4],
}

/// 32 bytes; `data` points into the hull points or names the mesh for
/// `Hull` and `Mesh`. `half` is the box's half extents, the capsule's and
/// cylinder's half length on z, the cone's half height on z.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Shape {
    pub kind: u32,
    pub data: u32,
    pub count: u32,
    pub radius: f32,
    pub half: [f32; 3],
    pub _a: f32,
}

/// A contact found by the narrowphase, re-evaluated every substep from
/// its body-local anchors (Muller 2020). `normal` points from `a` to `b`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Contact {
    pub a: u32,
    pub b: u32,
    pub anchor_a: [f32; 3],
    pub lambda_n: f32,
    pub anchor_b: [f32; 3],
    pub lambda_t: f32,
    pub normal: [f32; 3],
    /// Penetration when found (negative: a speculative contact inside the margin).
    pub depth0: f32,
}

/// `Body.flags` bit: the body sleeps until a contact wakes it.
pub const SLEEPING: u32 = 1 << 2;
/// `Body.flags` bit: the body may fall asleep (Blender's Deactivation).
pub const DEACTIVATES: u32 = 1 << 3;
const KIND_MASK: u32 = 0b11;

/// Blender's `RigidBodyOb.type` plus the animated case.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    #[default]
    Active = 0,
    Passive = 1,
    Kinematic = 2,
}

impl Kind {
    pub fn of(flags: u32) -> Self {
        match flags & KIND_MASK {
            1 => Self::Passive,
            2 => Self::Kinematic,
            _ => Self::Active,
        }
    }
}

/// Blender's `RigidBodyOb.shape`, in its numbering.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShapeKind {
    #[default]
    Box = 0,
    Sphere = 1,
    Capsule = 2,
    Cylinder = 3,
    Cone = 4,
    Hull = 5,
    Mesh = 6,
}

impl ShapeKind {
    pub fn of(kind: u32) -> Self {
        match kind {
            1 => Self::Sphere,
            2 => Self::Capsule,
            3 => Self::Cylinder,
            4 => Self::Cone,
            5 => Self::Hull,
            6 => Self::Mesh,
            _ => Self::Box,
        }
    }
}

impl Body {
    pub fn kind(&self) -> Kind {
        Kind::of(self.flags)
    }

    pub fn asleep(&self) -> bool {
        self.flags & SLEEPING != 0
    }

    pub fn set_asleep(&mut self, asleep: bool) {
        self.flags = match asleep {
            true => self.flags | SLEEPING,
            false => self.flags & !SLEEPING,
        };
    }

    /// Whether the body moves under the solver: active and awake.
    pub fn dynamic(&self) -> bool {
        self.kind() == Kind::Active && !self.asleep()
    }

    /// Whether the body's pose changes this frame (active or kinematic).
    pub fn moves(&self) -> bool {
        self.kind() != Kind::Passive && !self.asleep()
    }
}

impl Shape {
    pub fn kind(&self) -> ShapeKind {
        ShapeKind::of(self.kind)
    }

    pub fn sphere(radius: f32) -> Self {
        Self { kind: ShapeKind::Sphere as u32, radius, ..Default::default() }
    }

    pub fn boxed(half: [f32; 3]) -> Self {
        Self { kind: ShapeKind::Box as u32, half, ..Default::default() }
    }

    /// A capsule along z: `radius` and the half length of its straight part.
    pub fn capsule(radius: f32, half: f32) -> Self {
        Self { kind: ShapeKind::Capsule as u32, radius, half: [radius, radius, half], ..Default::default() }
    }

    /// A cylinder along z: `radius` and half height.
    pub fn cylinder(radius: f32, half: f32) -> Self {
        Self { kind: ShapeKind::Cylinder as u32, radius, half: [radius, radius, half], ..Default::default() }
    }

    /// A cone along z, apex at `+half`, base disk of `radius` at `-half`.
    pub fn cone(radius: f32, half: f32) -> Self {
        Self { kind: ShapeKind::Cone as u32, radius, half: [radius, radius, half], ..Default::default() }
    }

    /// `count` hull points from `data` in the world's hull table; `half`
    /// is their bounds' half extents (for the inertia).
    pub fn hull(data: u32, count: u32, half: [f32; 3]) -> Self {
        Self { kind: ShapeKind::Hull as u32, data, count, half, ..Default::default() }
    }

    /// Mesh number `data` in the world's mesh table, `count` triangles.
    pub fn mesh(data: u32, count: u32, half: [f32; 3]) -> Self {
        Self { kind: ShapeKind::Mesh as u32, data, count, half, ..Default::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_the_roadmap() {
        assert_eq!(std::mem::size_of::<Body>(), 64);
        assert_eq!(std::mem::size_of::<BodyConst>(), 48);
        assert_eq!(std::mem::size_of::<BodyPrev>(), 32);
        assert_eq!(std::mem::size_of::<Shape>(), 32);
        assert_eq!(std::mem::size_of::<Contact>(), 56);
        let b = Body { flags: Kind::Kinematic as u32 | SLEEPING, ..Default::default() };
        assert_eq!((b.kind(), b.asleep(), b.moves()), (Kind::Kinematic, true, false));
    }
}
