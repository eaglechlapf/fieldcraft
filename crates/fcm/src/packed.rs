//! Small dense symmetric matrices kept as their lower triangle, row by row (row i starts at
//! i (i + 1) / 2): the matrix of a few nodes of the stiffness, its Cholesky factor and the two
//! substitutions with it. The blocks of the iteration ([`crate::iterative`]) and the unknowns a
//! cell has to itself ([`crate::condense`]) are solved with them.

use crate::blocked::Blocked;

/// The lower triangle of the part of `matrix` on the nodes `nodes` (rising, each once; three
/// unknowns each). None when there is no memory for it.
pub(crate) fn lower(matrix: &Blocked, nodes: &[usize]) -> Option<Vec<f64>> {
    let n = 3 * nodes.len();
    let mut triangle = Vec::new();
    triangle.try_reserve_exact(n * (n + 1) / 2).ok()?;
    triangle.resize(n * (n + 1) / 2, 0.0);
    for (i, node) in nodes.iter().enumerate() {
        let (others, blocks) = matrix.row(*node);
        // Both lists rise: one pass over them.
        let mut j = 0;
        for (other, block) in others.iter().zip(blocks) {
            let other = *other as usize;
            while nodes.get(j).is_some_and(|node| *node < other) {
                j += 1;
            }
            if j > i {
                break;
            }
            if nodes.get(j) != Some(&other) {
                continue;
            }
            for (r, entries) in block.as_chunks::<3>().0.iter().enumerate() {
                let row = 3 * i + r;
                for (column, value) in (3 * j..=row.min(3 * j + 2)).zip(entries) {
                    if let Some(slot) = triangle.get_mut(row * (row + 1) / 2 + column) {
                        *slot = *value;
                    }
                }
            }
        }
    }
    Some(triangle)
}

/// Factor the symmetric positive definite matrix whose lower triangle is `a`, in place:
/// A = L Lᵀ. False when a pivot is not positive.
pub(crate) fn cholesky(a: &mut [f64], n: usize) -> bool {
    if a.len() != n * (n + 1) / 2 {
        return false;
    }
    for i in 0..n {
        let row = i * (i + 1) / 2;
        let (done, rest) = a.split_at_mut(row);
        let Some(own) = rest.get_mut(..i + 1) else { return false };
        for j in 0..i {
            let other = j * (j + 1) / 2;
            let Some(entries) = done.get(other..other + j + 1) else { return false };
            let Some((diagonal, before)) = entries.split_last() else { return false };
            let sum: f64 = before.iter().zip(own.iter()).map(|(x, y)| x * y).sum();
            if let Some(slot) = own.get_mut(j) {
                *slot = (*slot - sum) / diagonal;
            }
        }
        let Some((diagonal, before)) = own.split_last_mut() else { return false };
        let pivot = *diagonal - before.iter().map(|x| x * x).sum::<f64>();
        if !(pivot > 0.0) || !pivot.is_finite() {
            return false;
        }
        *diagonal = pivot.sqrt();
    }
    true
}

/// `x` = L⁻¹ `x`, row by row.
pub(crate) fn forward(l: &[f64], x: &mut [f64]) {
    for i in 0..x.len() {
        let row = i * (i + 1) / 2;
        let (Some(entries), Some((done, rest))) = (l.get(row..row + i + 1), x.split_at_mut_checked(i)) else { return };
        let Some((diagonal, before)) = entries.split_last() else { return };
        let sum: f64 = before.iter().zip(done.iter()).map(|(a, b)| a * b).sum();
        if let Some(slot) = rest.first_mut() {
            *slot = (*slot - sum) / diagonal;
        }
    }
}

/// `x` = L⁻ᵀ `x`: the last unknown first, each taken out of those before it.
pub(crate) fn backward(l: &[f64], x: &mut [f64]) {
    for i in (0..x.len()).rev() {
        let row = i * (i + 1) / 2;
        let (Some(entries), Some((before, rest))) = (l.get(row..row + i + 1), x.split_at_mut_checked(i)) else { return };
        let (Some((diagonal, above)), Some(slot)) = (entries.split_last(), rest.first_mut()) else { return };
        *slot /= diagonal;
        let value = *slot;
        for (target, a) in before.iter_mut().zip(above) {
            *target -= a * value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_factor_solves_its_matrix() {
        // A = B Bᵀ + 4 I for a B of our own.
        let n = 7;
        let b: Vec<f64> = (0..n * n).map(|k| ((k * k) as f64 * 0.37).sin()).collect();
        let entry = |i: usize, j: usize| (0..n).map(|k| b[i * n + k] * b[j * n + k]).sum::<f64>() + if i == j { 4.0 } else { 0.0 };
        let mut l = Vec::new();
        for i in 0..n {
            for j in 0..=i {
                l.push(entry(i, j));
            }
        }
        assert!(cholesky(&mut l, n));
        let x: Vec<f64> = (0..n).map(|i| 0.5 + i as f64).collect();
        let mut r: Vec<f64> = (0..n).map(|i| (0..n).map(|j| entry(i, j) * x[j]).sum()).collect();
        forward(&l, &mut r);
        backward(&l, &mut r);
        for (a, b) in r.iter().zip(&x) {
            assert!((a - b).abs() < 1e-12, "{a} against {b}");
        }
        assert!(!cholesky(&mut [1.0, 2.0, 1.0], 2), "not positive");
        assert!(!cholesky(&mut [1.0], 2), "the wrong size");
        // Lists of the wrong lengths change nothing they should not; none panics.
        let mut short = [1.0, 2.0, 3.0];
        forward(&l[..2], &mut short);
        backward(&l[..2], &mut short);
    }
}
