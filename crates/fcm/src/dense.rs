//! Small dense symmetric matrices, row-major: the Cholesky factorization of a cell matrix and the
//! largest eigenvalue of a symmetric pencil, for the Nitsche parameter of a cell.

/// Power iterations, at most. The Rayleigh quotient converges with the square of the ratio of
/// the two largest eigenvalues, so even a 1 % gap is resolved long before this.
const MAX_ITERATIONS: usize = 1000;
/// The iteration stops when the Rayleigh quotient changes by less than this fraction.
const CONVERGED: f64 = 1e-9;

/// Factor the symmetric positive definite `a` (n × n) in place: its lower triangle becomes L
/// with A = L Lᵀ (the upper triangle is left as it was). False when a pivot is not positive.
pub(crate) fn cholesky(a: &mut [f64], n: usize) -> bool {
    if a.len() != n * n {
        return false;
    }
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 0.0) || !d.is_finite() {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    true
}

/// Solve L y = b in place.
fn forward(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Solve Lᵀ x = b in place.
fn backward(l: &[f64], n: usize, b: &mut [f64]) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The largest eigenvalue λ of H x = λ (L Lᵀ) x for a symmetric positive semidefinite `h` and
/// the Cholesky factor `l` of a positive definite matrix (both n × n): power iteration on
/// L⁻¹ H L⁻ᵀ from a fixed start vector. The value returned is a Rayleigh quotient, so it never
/// exceeds the eigenvalue.
pub(crate) fn largest_eigenvalue(h: &[f64], l: &[f64], n: usize) -> f64 {
    if n == 0 || h.len() != n * n || l.len() != n * n {
        return 0.0;
    }
    // A fixed pseudo-random start (a linear congruential sequence), so no eigenvector is
    // orthogonal to it by construction and the result is the same on every run.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut y: Vec<f64> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    let norm = dot(&y, &y).sqrt();
    y.iter_mut().for_each(|x| *x /= norm);
    let mut lambda = 0.0;
    let mut z = vec![0.0; n];
    for _ in 0..MAX_ITERATIONS {
        let mut x = y.clone();
        backward(l, n, &mut x);
        for (i, zi) in z.iter_mut().enumerate() {
            *zi = dot(&h[i * n..(i + 1) * n], &x);
        }
        forward(l, n, &mut z);
        let next = dot(&y, &z);
        let norm = dot(&z, &z).sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            return if next.is_finite() { next.max(0.0) } else { 0.0 };
        }
        for (yi, zi) in y.iter_mut().zip(&z) {
            *yi = zi / norm;
        }
        let converged = (next - lambda).abs() <= CONVERGED * next.abs();
        lambda = next;
        if converged {
            break;
        }
    }
    lambda.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_and_the_largest_eigenvalue_of_a_pencil() {
        // B = diag(1, 4, 9), H = diag(2, 4, 27): eigenvalues 2, 1, 3.
        let mut b = vec![1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0];
        let h = vec![2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 27.0];
        assert!(cholesky(&mut b, 3));
        assert!((largest_eigenvalue(&h, &b, 3) - 3.0).abs() < 1e-7);
        // A full pencil: H = B scaled by 5 has the single eigenvalue 5.
        let full = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let scaled: Vec<f64> = full.iter().map(|x| 5.0 * x).collect();
        let mut l = full.clone();
        assert!(cholesky(&mut l, 3));
        assert!((largest_eigenvalue(&scaled, &l, 3) - 5.0).abs() < 1e-9);
        // Not positive definite, a zero pencil and wrong sizes are refused without a panic.
        assert!(!cholesky(&mut [1.0, 2.0, 2.0, 1.0], 2));
        assert!(!cholesky(&mut [1.0, 2.0, 2.0], 2));
        assert_eq!(largest_eigenvalue(&[0.0; 9], &l, 3), 0.0);
        assert_eq!(largest_eigenvalue(&[1.0], &l, 3), 0.0);
    }
}
