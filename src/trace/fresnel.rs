//! The view-dependent slots' arithmetic, CPU twin of `procedural.wgsl`'s
//! `fresnel_slot` and `layer_weight_slot` — Cycles' `svm_node_fresnel` and
//! `svm_node_layer_weight` (`kernel/svm/fresnel.h`) over
//! `fresnel_dielectric_cos` (`kernel/closure/bsdf_util.h`).
//!
//! A Fresnel or a Layer Weight is not a texture: its value is the angle
//! between the shading normal and the ray, so it can only be evaluated at
//! a hit. The baker cannot fold one and instead stands in with the mean
//! over the hemisphere ([`fresnel_fss`], [`facing_mean`]; phiphx's
//! `blend/shade/fold.rs` carries the same two). The test below holds the
//! two sides together: the quadrature of the per-pixel function is the
//! mean the fold uses.

/// `fresnel_dielectric_cos`: the dielectric reflectance at `cosi` for a
/// relative IOR `eta`; 1 past total internal reflection.
pub fn fresnel_dielectric_cos(cosi: f32, eta: f32) -> f32 {
    let c = cosi.abs();
    let g = eta * eta - 1.0 + c * c;
    if g <= 0.0 {
        return 1.0;
    }
    let g = g.sqrt();
    let a = (g - c) / (g + c);
    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    0.5 * a * a * (1.0 + b * b)
}

/// `svm_node_fresnel` at a hit whose normal meets the ray at `cos`.
pub fn fresnel_at(cos: f32, ior: f32) -> f32 {
    fresnel_dielectric_cos(cos, ior.max(1e-5))
}

/// `svm_node_layer_weight` at `cos`: the Fresnel output at `1 / (1 −
/// blend)`, or the Facing output, `1 − cosᵖ` with the Blend remapped to
/// the power.
pub fn layer_weight_at(cos: f32, blend: f32, facing: bool) -> f32 {
    if !facing {
        return fresnel_dielectric_cos(cos, 1.0 / (1.0 - blend).max(1e-5));
    }
    let f = cos.abs();
    let p = facing_power(blend);
    1.0 - f.powf(p)
}

/// The power the Facing output raises the cosine to, as the node remaps
/// its Blend (`svm_node_layer_weight`).
fn facing_power(blend: f32) -> f32 {
    match blend == 0.5 {
        true => 1.0,
        false => {
            let b = blend.clamp(0.0, 1.0 - 1e-5);
            match b < 0.5 {
                true => 2.0 * b,
                false => 0.5 / (1.0 - b),
            }
        }
    }
}

/// `fresnel_dielectric_Fss` (`kernel/closure/bsdf_util.h`): the mean of
/// the dielectric Fresnel over the hemisphere, Kulla–Conty's fit. What a
/// baker that cannot see the hit uses in its place.
pub fn fresnel_fss(eta: f32) -> f32 {
    match eta < 1.0 {
        true => 0.997118 + eta * (0.1014 - eta * (0.965241 + eta * 0.130607)),
        false => (eta - 1.0) / (4.08567 + 1.00071 * eta),
    }
}

/// The same mean for Layer Weight ▸ Facing: `∫2·cos·(1 − cosᵖ) dcos`,
/// which is `1 − 2 / (p + 2)`.
pub fn facing_mean(blend: f32) -> f32 {
    1.0 - 2.0 / (facing_power(blend) + 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `∫₀¹ 2·cos·f(cos) dcos` by the midpoint rule — the average a
    /// surface shows a camera over all orientations.
    fn hemisphere_mean(f: impl Fn(f32) -> f32) -> f32 {
        let n = 20_000;
        (0..n)
            .map(|i| {
                let c = (i as f32 + 0.5) / n as f32;
                2.0 * c * f(c) / n as f32
            })
            .sum()
    }

    /// At normal incidence a dielectric reflects `((η−1)/(η+1))²`; at
    /// grazing it reflects everything.
    #[test]
    fn the_dielectric_fresnel_is_cycles_own() {
        let f0 = |eta: f32| ((eta - 1.0) / (eta + 1.0)).powi(2);
        for eta in [1.05f32, 1.33, 1.45, 1.8, 2.4] {
            assert!((fresnel_dielectric_cos(1.0, eta) - f0(eta)).abs() < 1e-5, "{eta}");
            assert!(fresnel_dielectric_cos(0.0, eta) > 0.999, "grazing at {eta}");
            assert!(fresnel_dielectric_cos(0.5, eta) > f0(eta), "the edge is brighter than the face");
        }
        // Under 1 the ray can run out of refracted component entirely.
        assert_eq!(fresnel_dielectric_cos(0.1, 0.5), 1.0);
    }

    /// The fold's stand-in is the per-pixel function's own mean: what the
    /// baker writes when it cannot see the hit is what the kernel averages
    /// to over the hemisphere (`blend/shade/fold.rs` carries the same two
    /// constants).
    #[test]
    fn the_folds_stand_in_is_the_mean_of_the_per_pixel_value() {
        for ior in [1.1f32, 1.45, 1.8, 2.4] {
            let mean = hemisphere_mean(|c| fresnel_at(c, ior));
            assert!((mean - fresnel_fss(ior)).abs() < 0.01, "IOR {ior}: quadrature {mean}, fit {}", fresnel_fss(ior));
        }
        for blend in [0.0f32, 0.1, 0.3, 0.5, 0.7, 0.9] {
            let mean = hemisphere_mean(|c| layer_weight_at(c, blend, true));
            assert!((mean - facing_mean(blend)).abs() < 1e-3, "Blend {blend}: quadrature {mean}, closed form {}", facing_mean(blend));
        }
    }

    /// Facing is 0 head on and 1 at the edge, and sharper the higher the
    /// Blend — except at Blend 0, where the node's own remapping takes the
    /// power to 0 and the output is 0 over the whole surface. The Fresnel
    /// output of a Layer Weight at Blend 0.5 is a Fresnel at IOR 2.
    #[test]
    fn layer_weight_runs_from_the_face_to_the_edge() {
        for blend in [0.25f32, 0.5, 0.75, 0.95] {
            assert!(layer_weight_at(1.0, blend, true).abs() < 1e-4, "head on at {blend}");
            assert!((layer_weight_at(0.0, blend, true) - 1.0).abs() < 1e-4, "at the edge at {blend}");
        }
        // Sharper with the Blend: at 0.9 the power is 5, at 0.5 it is 1.
        assert!(layer_weight_at(0.7, 0.9, true) > layer_weight_at(0.7, 0.5, true));
        assert_eq!((layer_weight_at(1.0, 0.0, true), layer_weight_at(0.0, 0.0, true)), (0.0, 0.0), "Blend 0 is no facing at all");
        assert_eq!(facing_mean(0.0), 0.0);
        let at = |c: f32| layer_weight_at(c, 0.5, false);
        assert!((at(1.0) - fresnel_dielectric_cos(1.0, 2.0)).abs() < 1e-6);
        assert!(at(0.2) > at(1.0), "the edge reflects more");
    }
}
