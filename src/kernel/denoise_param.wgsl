struct P {
    n: u32, groups_x: u32, w: u32, h: u32,
    step: u32, round: u32, last: u32, keep: u32,
    strength: vec4<f32>,
    view: vec4<f32>,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> px: array<u32>;
