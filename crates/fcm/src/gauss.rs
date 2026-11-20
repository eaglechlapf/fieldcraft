//! Gauss–Legendre quadrature on [−1, 1] of any order.
//!
//! The abscissae are the roots of the Legendre polynomial Pₙ, found by Newton's iteration from
//! the estimate cos(π (i + ¾) / (n + ½)), and the weights are 2 / ((1 − x²) Pₙ′(x)²) (P. J. Davis,
//! P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed., Academic Press 1984). A rule of n
//! points integrates polynomials of degree 2n − 1 exactly. The estimate uses the portable cosine,
//! and the iteration only adds, multiplies and divides, so the rule is the same bit for bit on
//! every platform.

/// The most points a rule may have.
pub const MAX_POINTS: usize = 32;

/// Newton steps per root: the estimate is good to a few digits and the iteration converges
/// quadratically, so a handful suffice.
const MAX_STEPS: usize = 32;

/// Pₙ(x) and Pₙ₋₁(x), by the three-term recurrence.
fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut previous, mut current) = (1.0, x);
    for k in 1..n {
        let kf = k as f64;
        let next = ((2.0 * kf + 1.0) * x * current - kf * previous) / (kf + 1.0);
        previous = current;
        current = next;
    }
    (current, previous)
}

/// Pₙ′(x) = n (x Pₙ − Pₙ₋₁) / (x² − 1), for |x| < 1.
fn slope(n: usize, x: f64) -> (f64, f64) {
    let (p, below) = legendre(n, x);
    (p, n as f64 * (x * p - below) / (x * x - 1.0))
}

/// The `n`-point rule (`n` clamped to 1..=[`MAX_POINTS`]): abscissae in increasing order with
/// their weights.
pub fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let n = n.clamp(1, MAX_POINTS);
    let mut rule = vec![(0.0, 0.0); n];
    for i in 0..n.div_ceil(2) {
        // The i-th root from the right end; the middle one of an odd rule is exactly zero.
        let middle = n % 2 == 1 && i == n / 2;
        let mut x = if middle { 0.0 } else { libm::cos(std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)) };
        if !middle {
            for _ in 0..MAX_STEPS {
                let (p, dp) = slope(n, x);
                let step = p / dp;
                x -= step;
                if step.abs() <= 1e-16 {
                    break;
                }
            }
        }
        let (_, dp) = slope(n, x);
        let weight = 2.0 / ((1.0 - x * x) * dp * dp);
        if let Some(right) = rule.get_mut(n - 1 - i) {
            *right = (x, weight);
        }
        if let Some(left) = rule.get_mut(i) {
            *left = (-x, weight);
        }
    }
    rule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_integrate_monomials_exactly() {
        for n in 1..=MAX_POINTS {
            let rule = gauss_legendre(n);
            assert_eq!(rule.len(), n);
            assert!(rule.windows(2).all(|w| w[0].0 < w[1].0), "{n} points are not increasing");
            for degree in 0..2 * n {
                let sum: f64 = rule.iter().map(|(x, w)| w * x.powi(degree as i32)).sum();
                let exact = if degree % 2 == 0 { 2.0 / (degree as f64 + 1.0) } else { 0.0 };
                assert!((sum - exact).abs() < 1e-13, "{n} points, degree {degree}: {sum} vs {exact}");
            }
        }
    }

    #[test]
    fn low_orders_match_their_closed_forms() {
        let one = gauss_legendre(1);
        assert_eq!(one, vec![(0.0, 2.0)]);
        let two = gauss_legendre(2);
        assert!((two[1].0 - 1.0 / 3f64.sqrt()).abs() < 1e-15 && (two[1].1 - 1.0).abs() < 1e-15);
        let three = gauss_legendre(3);
        assert_eq!(three[1].0, 0.0);
        assert!((three[1].1 - 8.0 / 9.0).abs() < 1e-15 && (three[2].0 - 0.6f64.sqrt()).abs() < 1e-15);
    }
}
