//! Small dense symmetric matrices, row-major: the Cholesky factorization of a cell matrix and the
//! largest eigenvalue of a symmetric pencil, for the Nitsche parameter of a cell.
//!
//! The pencil is H x = λ V x with H the traction matrix of the cell's supported pieces and V its
//! stiffness. H has a low rank: the tractions on a flat face are polynomials on that face, far
//! fewer than the cell's modes. So H is factored first, H = FᵀF with as many rows as its rank,
//! and the eigenvalue is found on a matrix of that size.

/// Power iterations, at most. The Rayleigh quotient converges with the square of the ratio of
/// the two largest eigenvalues, so even a 1 % gap is resolved long before this.
const MAX_ITERATIONS: usize = 1000;
/// The iteration stops when the Rayleigh quotient changes by less than this fraction.
const CONVERGED: f64 = 1e-9; // unit-factors: a relative change of the eigenvalue

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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A factor F (r rows of n, row-major) of the symmetric positive semidefinite `h` (n × n) with
/// FᵀF = H, by the Cholesky factorization with diagonal pivoting: each step takes the largest
/// diagonal entry left as its pivot, and the factorization stops at the numerical rank r, when
/// no diagonal entry left exceeds n ε times the largest of H (N. J. Higham, *Accuracy and
/// Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, §10.3). A step costs n times the
/// steps before it, so a matrix of low rank is factored in n r² operations.
pub(crate) fn low_rank_factor(h: &[f64], n: usize) -> Vec<f64> {
    let mut factor: Vec<f64> = Vec::new();
    if n == 0 || h.len() != n * n {
        return factor;
    }
    let mut diagonal: Vec<f64> = (0..n).map(|i| h[i * n + i]).collect();
    let largest = diagonal.iter().copied().fold(0.0, f64::max);
    if !(largest > 0.0) || !largest.is_finite() {
        return factor;
    }
    let small = n as f64 * f64::EPSILON * largest;
    let mut pivots: Vec<usize> = Vec::new();
    for _ in 0..n {
        // The first of the largest entries, so the order never depends on rounding ties.
        let (mut pivot, mut value) = (0, 0.0);
        for (i, d) in diagonal.iter().enumerate() {
            if *d > value {
                (pivot, value) = (i, *d);
            }
        }
        if !(value > small) {
            break;
        }
        // The pivot's row of what is left of H: its row of H less the rows found so far.
        let mut row = h[pivot * n..(pivot + 1) * n].to_vec();
        for earlier in factor.chunks_exact(n) {
            let scale = earlier[pivot];
            for (x, e) in row.iter_mut().zip(earlier) {
                *x -= scale * e;
            }
        }
        if !(row[pivot] > 0.0) || !row[pivot].is_finite() {
            break;
        }
        let root = row[pivot].sqrt();
        for x in &mut row {
            *x /= root;
        }
        // What is left is zero in the rows and columns of the pivots.
        for &done in &pivots {
            row[done] = 0.0;
        }
        for (d, x) in diagonal.iter_mut().zip(&row) {
            *d -= x * x;
        }
        diagonal[pivot] = 0.0;
        pivots.push(pivot);
        factor.extend_from_slice(&row);
    }
    factor
}

/// The largest eigenvalue λ of H x = λ (L Lᵀ) x for a symmetric positive semidefinite `h` and
/// the Cholesky factor `l` of a positive definite matrix (both n × n). With H = FᵀF
/// ([`low_rank_factor`]) it is the largest eigenvalue of Y Yᵀ, Y = F L⁻ᵀ, a matrix of the size of
/// H's rank: power iteration on it from a fixed start vector. The value returned is a Rayleigh
/// quotient, so it never exceeds the eigenvalue.
pub(crate) fn largest_eigenvalue(h: &[f64], l: &[f64], n: usize) -> f64 {
    if n == 0 || h.len() != n * n || l.len() != n * n {
        return 0.0;
    }
    let mut rows = low_rank_factor(h, n);
    let r = rows.len() / n;
    if r == 0 {
        return 0.0;
    }
    for row in rows.chunks_exact_mut(n) {
        forward(l, n, row);
    }
    let mut small = vec![0.0; r * r];
    for (i, a) in rows.chunks_exact(n).enumerate() {
        for (j, b) in rows.chunks_exact(n).enumerate().take(i + 1) {
            let value = dot(a, b);
            small[i * r + j] = value;
            small[j * r + i] = value;
        }
    }
    drop(rows);
    // A fixed pseudo-random start (a linear congruential sequence), so no eigenvector is
    // orthogonal to it by construction and the result is the same on every run.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut y: Vec<f64> = (0..r)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    let norm = dot(&y, &y).sqrt();
    y.iter_mut().for_each(|x| *x /= norm);
    let mut lambda = 0.0;
    let mut z = vec![0.0; r];
    for _ in 0..MAX_ITERATIONS {
        for (zi, row) in z.iter_mut().zip(small.chunks_exact(r)) {
            *zi = dot(row, &y);
        }
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
        // A matrix of rank one, u uᵀ with u = (1, 2, 2), against the identity: |u|² = 9.
        let u = [1.0, 2.0, 2.0];
        let outer: Vec<f64> = u.iter().flat_map(|a| u.iter().map(move |b| a * b)).collect();
        let factor = low_rank_factor(&outer, 3);
        assert_eq!(factor.len(), 3, "rank one: {factor:?}");
        assert!(factor.iter().zip(&u).all(|(f, x)| (f - x).abs() < 1e-15));
        let mut identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(cholesky(&mut identity, 3));
        assert!((largest_eigenvalue(&outer, &identity, 3) - 9.0).abs() < 1e-12);
        // The factor of a full matrix gives it back.
        let factor = low_rank_factor(&full, 3);
        assert_eq!(factor.len(), 9);
        for i in 0..3 {
            for j in 0..3 {
                let sum: f64 = (0..3).map(|k| factor[k * 3 + i] * factor[k * 3 + j]).sum();
                assert!((sum - full[i * 3 + j]).abs() < 1e-14);
            }
        }
        assert!(low_rank_factor(&[0.0; 9], 3).is_empty() && low_rank_factor(&[1.0], 3).is_empty());
        // Not positive definite, a zero pencil and wrong sizes are refused without a panic.
        assert!(!cholesky(&mut [1.0, 2.0, 2.0, 1.0], 2));
        assert!(!cholesky(&mut [1.0, 2.0, 2.0], 2));
        assert_eq!(largest_eigenvalue(&[0.0; 9], &l, 3), 0.0);
        assert_eq!(largest_eigenvalue(&[1.0], &l, 3), 0.0);
    }
}
