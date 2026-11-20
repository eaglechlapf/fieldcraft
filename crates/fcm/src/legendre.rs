//! One-dimensional hierarchic shape functions on [−1, 1]: the two linear nodal functions and the
//! integrated Legendre polynomials (B. Szabó, I. Babuška, *Finite Element Analysis*, Wiley 1991):
//!
//! - φ₀(ξ) = (1 − ξ)/2, φ₁(ξ) = (1 + ξ)/2;
//! - φᵢ(ξ) = √((2i − 1)/2) ∫ Pᵢ₋₁(t) dt over [−1, ξ] = (Pᵢ(ξ) − Pᵢ₋₂(ξ)) / √(2(2i − 1)) for i ≥ 2,
//!
//! with Pₖ the Legendre polynomials. The functions of degree 2 and up vanish at both ends, so a
//! higher degree adds unknowns without changing the ones already there, and their derivatives
//! √((2i − 1)/2) Pᵢ₋₁ are orthonormal on [−1, 1].

/// The highest polynomial degree.
pub const MAX_DEGREE: usize = 8;
/// Shape functions of the highest degree.
pub const MAX_MODES: usize = MAX_DEGREE + 1;

/// The shape functions of one degree at one point: values and first derivatives of φ₀ … φ_p
/// (entries past p are zero).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub n: [f64; MAX_MODES],
    pub d: [f64; MAX_MODES],
}

/// The shape functions of degree `p` (clamped to 1..=[`MAX_DEGREE`]) at `xi`.
pub fn shape(p: usize, xi: f64) -> Shape {
    let p = p.clamp(1, MAX_DEGREE);
    // Legendre polynomials by the three-term recurrence (k + 1) P_{k+1} = (2k + 1) ξ P_k − k P_{k−1}.
    let mut legendre = [0.0; MAX_MODES];
    legendre[0] = 1.0;
    legendre[1] = xi;
    for k in 1..p {
        let kf = k as f64;
        legendre[k + 1] = ((2.0 * kf + 1.0) * xi * legendre[k] - kf * legendre[k - 1]) / (kf + 1.0);
    }
    let mut s = Shape { n: [0.0; MAX_MODES], d: [0.0; MAX_MODES] };
    s.n[0] = 0.5 * (1.0 - xi);
    s.d[0] = -0.5;
    s.n[1] = 0.5 * (1.0 + xi);
    s.d[1] = 0.5;
    for i in 2..=p {
        let c = (2 * i - 1) as f64;
        s.n[i] = (legendre[i] - legendre[i - 2]) / (2.0 * c).sqrt();
        s.d[i] = (0.5 * c).sqrt() * legendre[i - 1];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauss::gauss_legendre;

    #[test]
    fn linear_modes_sum_to_one_and_higher_modes_vanish_at_the_ends() {
        for p in 1..=MAX_DEGREE {
            for xi in [-1.0, -0.3, 0.0, 0.71, 1.0] {
                let s = shape(p, xi);
                assert!((s.n[0] + s.n[1] - 1.0).abs() < 1e-15);
            }
            for end in [-1.0, 1.0] {
                let s = shape(p, end);
                for i in 2..=p {
                    assert!(s.n[i].abs() < 1e-14, "degree {i} at {end}: {}", s.n[i]);
                }
            }
        }
    }

    #[test]
    fn derivatives_match_differences() {
        let step = 1e-6;
        for xi in [-0.9, -0.2, 0.4, 0.85] {
            let (lo, hi, at) = (shape(MAX_DEGREE, xi - step), shape(MAX_DEGREE, xi + step), shape(MAX_DEGREE, xi));
            for i in 0..=MAX_DEGREE {
                let slope = (hi.n[i] - lo.n[i]) / (2.0 * step);
                assert!((slope - at.d[i]).abs() < 1e-8, "degree {i} at {xi}: {slope} vs {}", at.d[i]);
            }
        }
    }

    #[test]
    fn higher_mode_derivatives_are_orthonormal() {
        let rule = gauss_legendre(MAX_DEGREE + 1);
        for i in 2..=MAX_DEGREE {
            for j in 2..=MAX_DEGREE {
                let sum: f64 = rule.iter().map(|(x, w)| w * shape(MAX_DEGREE, *x).d[i] * shape(MAX_DEGREE, *x).d[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 1e-12, "({i}, {j}): {sum}");
            }
        }
    }
}
