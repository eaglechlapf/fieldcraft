//! The unknowns by iteration: for systems whose factorization would take far more work than
//! the system has entries, or more memory than there is.
//!
//! A factorization of the stiffness of a compact part takes work that grows with the square of
//! the unknowns and memory that grows faster than they do ([`fieldcraft_solver::cholesky`]). An
//! iteration takes work in proportion to the entries of the matrix, each time round, and its
//! memory is the matrix's; what it needs is a count of rounds that does not grow with the grid.
//!
//! # The method
//! Conjugate gradients (M. R. Hestenes, E. Stiefel, *Methods of conjugate gradients for solving
//! linear systems*, J. Res. Natl. Bur. Stand. 49(6):409–436, 1952), preconditioned by an
//! additive Schwarz method of two levels (A. Toselli, O. Widlund, *Domain Decomposition
//! Methods: Algorithms and Theory*, Springer 2005, ch. 2 and 3): the approximate inverse is the
//! sum of exact inverses on overlapping pieces of the unknowns,
//!
//! M⁻¹ = R₀ᵀ K₀⁻¹ R₀ + Σₑ W Rₑᵀ Kₑ⁻¹ Rₑ W, each K = R K Rᵀ.
//!
//! **The coarse level** (index 0) is the linear shape functions of every level of the grid: the
//! trilinear finite elements of the refined grid, a thirtieth of the unknowns at degree 3. The
//! basis is hierarchic, so they are a subset of the unknowns and their matrix a part of the
//! stiffness; it is factored ([`Factor`]). They carry what is smooth over many cells from one
//! end of the part to the other in one step, which is what keeps the count of rounds from
//! growing with the number of cells (for shape functions of high degree: L. F. Pavarino,
//! *Additive Schwarz methods for the p-version finite element method*, Numer. Math.
//! 66:493–516, 1993).
//!
//! **The local level** is one block for every element: all the unknowns of the functions that
//! reach it, the higher ones of its own cell with those it shares with the cells around. Each
//! block's matrix is factored densely. An unknown is in as many blocks as cells share its
//! function, and the blocks' answers for it are summed: W, diagonal, weights each unknown by
//! one over the root of that number on the way in and on the way out, so that the sum is their
//! mean and M⁻¹ stays symmetric (measured: a third to a half fewer rounds than unweighted). In
//! a cell the surface cuts, the functions are nearly
//! dependent on the little material the cell holds, which is what makes the stiffness of a cut
//! grid ill-conditioned; within the cell's block that is solved exactly, and the iteration does
//! not see it (F. de Prenter, C. V. Verhoosel, G. J. van Zwieten, E. H. van Brummelen,
//! *Condition number analysis and preconditioning of the finite cell method*, Comput. Methods
//! Appl. Mech. Engrg. 316:297–327, 2017; with a coarse level, J. N. Jomo, F. de Prenter,
//! M. Elhaddad, D. D'Angella, C. V. Verhoosel, S. Kollmannsberger, J. S. Kirschke, V. Nübel,
//! E. H. van Brummelen, E. Rank, *Robust and parallel scalable iterative solutions for
//! large-scale finite cell analyses*, Finite Elem. Anal. Des. 163:14–30, 2019).
//!
//! # What it is not good at
//! The count of rounds does not grow with the grid, but it does with what the two levels do
//! not see: a material that hardly changes its volume (Poisson's ratio near one half), whose
//! stiffness against that one kind of strain dwarfs the rest. Such a system, and any other the
//! iteration does not settle in [`MOST`] rounds, is left to the factorization
//! ([`crate::assembly`]).
//!
//! # Determinism
//! Every sum runs in a fixed order: the products of the matrix row by row, each over its
//! columns in order; the blocks one by one into the result in their order; the inner products
//! from the first unknown to the last. The result is the same on any number of threads.

use fieldcraft_parallel as parallel;
use fieldcraft_solver::Limits;
use fieldcraft_solver::cholesky::{Factor, UpperColumns};
use fieldcraft_solver::sparse::Csr;

use crate::{Error, Result};

/// The residual at which the iteration is done, relative to the load: the unknowns are then
/// right to about as many digits, which is more than the shape functions resolve of the part.
pub(crate) const SETTLED: f64 = 1e-9; // unit-factors: a ratio of two forces

/// How far above the tolerance the true residual may end, when the recurrence's is below it:
/// the sums of a stiff system (a thin part on coarse cells) round at about this many times the
/// tolerance, and no further round brings the residual under it.
const ROUNDING: f64 = 10.0;

/// Rounds after which a system is left to the factorization: three times what a system the
/// two levels suit takes at the tolerance (about a hundred).
pub(crate) const MOST: usize = 400;

/// Rounds between two looks at whether the caller has cancelled.
const LOOK: usize = 8;

/// A solved system.
pub(crate) struct Outcome {
    /// The unknowns.
    pub u: Vec<f64>,
    /// The rounds taken.
    pub iterations: usize,
    /// The residual they ended at, relative to the load.
    pub residual: f64,
    /// Entries of the coarse level's factor.
    pub coarse_entries: usize,
}

/// One block of the local level: its unknowns in rising order, and the Cholesky factor of
/// their matrix, the lower triangle row by row.
struct Block {
    unknowns: Vec<usize>,
    factor: Vec<f64>,
}

impl Block {
    /// The block of the nodes `nodes` (three unknowns each) of `matrix`.
    fn new(matrix: &Csr, nodes: &[usize]) -> Block {
        let mut unknowns: Vec<usize> = nodes.iter().flat_map(|node| [3 * node, 3 * node + 1, 3 * node + 2]).collect();
        unknowns.sort_unstable();
        unknowns.dedup();
        let n = unknowns.len();
        // The lower triangle, row by row: row i starts at i (i + 1) / 2.
        let mut factor = vec![0.0; n * (n + 1) / 2];
        for (i, unknown) in unknowns.iter().enumerate() {
            let (columns, values) = matrix.row(*unknown);
            let row = i * (i + 1) / 2;
            // Both lists rise: one pass over them.
            let mut j = 0;
            for (column, value) in columns.iter().zip(values) {
                while unknowns.get(j).is_some_and(|other| other < column) {
                    j += 1;
                }
                if j > i {
                    break;
                }
                if unknowns.get(j) == Some(column)
                    && let Some(slot) = factor.get_mut(row + j)
                {
                    *slot = *value;
                }
            }
        }
        if !cholesky(&mut factor, n) {
            // A block that is not positive to working precision (a sliver of material on its
            // fictitious stiffness): its diagonal alone, which is.
            for (i, unknown) in unknowns.iter().enumerate() {
                let (columns, values) = matrix.row(*unknown);
                let diagonal = columns.binary_search(unknown).ok().and_then(|at| values.get(at)).copied().unwrap_or(1.0);
                let row = i * (i + 1) / 2;
                if let Some(entries) = factor.get_mut(row..row + i + 1) {
                    entries.fill(0.0);
                }
                if let Some(slot) = factor.get_mut(row + i) {
                    *slot = if diagonal > 0.0 && diagonal.is_finite() { diagonal.sqrt() } else { 1.0 };
                }
            }
        }
        Block { unknowns, factor }
    }

    /// `out` = K_block⁻¹ of the block's part of `r`.
    fn solve(&self, r: &[f64], out: &mut Vec<f64>) {
        let n = self.unknowns.len();
        out.clear();
        out.extend(self.unknowns.iter().map(|unknown| r.get(*unknown).copied().unwrap_or(0.0)));
        let l = &self.factor;
        // L y = r, row by row.
        for i in 0..n {
            let row = i * (i + 1) / 2;
            let (Some(entries), Some((done, rest))) = (l.get(row..row + i + 1), out.split_at_mut_checked(i)) else { return };
            let Some((diagonal, before)) = entries.split_last() else { return };
            let sum: f64 = before.iter().zip(done.iter()).map(|(a, b)| a * b).sum();
            if let Some(slot) = rest.first_mut() {
                *slot = (*slot - sum) / diagonal;
            }
        }
        // Lᵀ x = y: the last unknown first, each taken out of those before it.
        for i in (0..n).rev() {
            let row = i * (i + 1) / 2;
            let (Some(entries), Some((before, rest))) = (l.get(row..row + i + 1), out.split_at_mut_checked(i)) else { return };
            let (Some((diagonal, above)), Some(slot)) = (entries.split_last(), rest.first_mut()) else { return };
            *slot /= diagonal;
            let x = *slot;
            for (target, a) in before.iter_mut().zip(above) {
                *target -= a * x;
            }
        }
    }
}

/// Factor the symmetric positive definite matrix whose lower triangle is `a`, row by row, in
/// place: A = L Lᵀ. False when a pivot is not positive.
fn cholesky(a: &mut [f64], n: usize) -> bool {
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

/// The sum of the products of two lists, from the first to the last.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve `matrix` u = `load` by the iteration of the module's head, to the residual `settled`
/// of the load in at most `most` rounds. `linear` says which nodes (three unknowns each) carry
/// a linear shape function, and `blocks` lists the nodes of every block of the local level
/// (every node in one at least). `Ok(None)` when the rounds did not settle it, or a step broke
/// down: the system is then the factorization's.
pub(crate) fn solve(
    matrix: &Csr,
    load: &[f64],
    linear: &[bool],
    blocks: &[Vec<usize>],
    whole: &[bool],
    limits: &Limits,
    settled: f64,
    most: usize,
    cancelled: &dyn Fn() -> bool,
) -> Result<Option<Outcome>> {
    let n = matrix.n;
    if load.len() != n || linear.len() * 3 != n {
        return Err(Error::Internal("the iteration's system and its load differ in size".into()));
    }
    let scale = dot(load, load).sqrt();
    if !(scale > 0.0) || !scale.is_finite() {
        // No load: nothing moves.
        return Ok(scale.is_finite().then(|| Outcome { u: vec![0.0; n], iterations: 0, residual: 0.0, coarse_entries: 0 }));
    }

    // The coarse level: the linear functions' part of the matrix, factored.
    let mut place = vec![usize::MAX; n];
    let mut coarse = Vec::new();
    for (node, _) in linear.iter().enumerate().filter(|(_, linear)| **linear) {
        for unknown in 3 * node..3 * node + 3 {
            if let Some(slot) = place.get_mut(unknown) {
                *slot = coarse.len();
            }
            coarse.push(unknown);
        }
    }
    let (mut ptr, mut idx, mut val) = (vec![0usize], Vec::new(), Vec::new());
    for unknown in &coarse {
        let (columns, values) = matrix.row(*unknown);
        for (column, value) in columns.iter().zip(values) {
            if let Some(at) = place.get(*column).copied().filter(|at| *at != usize::MAX) {
                idx.push(at);
                val.push(*value);
            }
        }
        ptr.push(idx.len());
    }
    drop(place);
    let coarse_matrix = Csr { n: coarse.len(), ptr, idx, val };
    let coarse_factor = match Factor::new(&UpperColumns::from_symmetric(&coarse_matrix), limits.max_factor_entries) {
        Ok(factor) => Some(factor),
        // Without linear functions there is no coarse level; one that cannot be factored is
        // left out, and the rounds show whether the blocks are enough.
        Err(_) if coarse.is_empty() => None,
        Err(_) => return Ok(None),
    };
    drop(coarse_matrix);
    if cancelled() {
        return Err(Error::Internal("cancelled".into()));
    }

    // The local level, and each unknown's weight: one over the root of the blocks it is in.
    let blocks: Vec<Block> = parallel::map(blocks.len(), |block| Block::new(matrix, blocks.get(block).map(Vec::as_slice).unwrap_or_default()));
    let mut weight = vec![0.0f64; n];
    for unknown in blocks.iter().flat_map(|block| &block.unknowns) {
        if let Some(count) = weight.get_mut(*unknown) {
            *count += 1.0;
        }
    }
    for w in &mut weight {
        *w = if *w > 0.0 { 1.0 / w.sqrt() } else { 1.0 };
    }
    // A block that is taken whole has one weight for all its unknowns, the mean of theirs: its
    // answer is scaled, not bent.
    let even: Vec<Option<f64>> = blocks
        .iter()
        .enumerate()
        .map(|(at, block)| {
            if !whole.get(at).copied().unwrap_or(false) {
                return None;
            }
            let sum: f64 = block.unknowns.iter().map(|unknown| weight.get(*unknown).map_or(1.0, |w| w * w)).sum();
            Some((sum / block.unknowns.len().max(1) as f64).sqrt())
        })
        .collect();
    let mut locals: Vec<Vec<f64>> = vec![Vec::new(); blocks.len()];
    let mut held = Vec::with_capacity(coarse.len());
    let mut weighted = vec![0.0; n];
    // z = M⁻¹ r.
    let mut apply = |r: &[f64], z: &mut [f64]| -> Result<()> {
        z.fill(0.0);
        if let Some(factor) = &coarse_factor {
            held.clear();
            held.extend(coarse.iter().map(|unknown| r.get(*unknown).copied().unwrap_or(0.0)));
            factor.solve(&mut held)?;
            for (unknown, value) in coarse.iter().zip(&held) {
                if let Some(slot) = z.get_mut(*unknown) {
                    *slot = *value;
                }
            }
        }
        for ((scaled, residual), w) in weighted.iter_mut().zip(r).zip(&weight) {
            *scaled = residual * w;
        }
        let scaled = &weighted;
        parallel::for_each_mut(&mut locals, |block, out| {
            if let Some(found) = blocks.get(block) {
                match even.get(block).copied().flatten() {
                    Some(_) => found.solve(r, out),
                    None => found.solve(scaled, out),
                }
            }
        });
        for ((block, local), even) in blocks.iter().zip(&locals).zip(&even) {
            for (unknown, value) in block.unknowns.iter().zip(local) {
                if let (Some(slot), Some(w)) = (z.get_mut(*unknown), weight.get(*unknown)) {
                    *slot += match even {
                        // In and out: the weight twice.
                        Some(even) => even * even * value,
                        None => w * value,
                    };
                }
            }
        }
        Ok(())
    };

    let mut u = vec![0.0; n];
    let mut r = load.to_vec();
    let mut z = vec![0.0; n];
    apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0; n];
    let mut residual = 1.0;
    for iteration in 1..=most {
        if iteration % LOOK == 0 && cancelled() {
            return Err(Error::Internal("cancelled".into()));
        }
        matrix.mul(&p, &mut ap);
        let curvature = dot(&p, &ap);
        // A direction with no stiffness, or a number that is none: not this method's system.
        if !(curvature > 0.0 && rz > 0.0) || !curvature.is_finite() || !rz.is_finite() {
            return Ok(None);
        }
        let alpha = rz / curvature;
        for ((x, d), (res, a)) in u.iter_mut().zip(&p).zip(r.iter_mut().zip(&ap)) {
            *x += alpha * d;
            *res -= alpha * a;
        }
        residual = dot(&r, &r).sqrt() / scale;
        if !residual.is_finite() {
            return Ok(None);
        }
        if residual <= settled {
            // The residual of the recurrence drifts from the true one over many rounds: the
            // true one decides.
            matrix.mul(&u, &mut ap);
            for ((res, b), a) in r.iter_mut().zip(load).zip(&ap) {
                *res = b - a;
            }
            residual = dot(&r, &r).sqrt() / scale;
            if residual <= ROUNDING * settled {
                return Ok(Some(Outcome { u, iterations: iteration, residual, coarse_entries: coarse_factor.as_ref().map_or(0, Factor::entries) }));
            }
            // Further off than rounding explains: on from the true residual, with a fresh
            // direction.
            apply(&r, &mut z)?;
            p.copy_from_slice(&z);
            rz = dot(&r, &z);
            continue;
        }
        apply(&r, &mut z)?;
        let next = dot(&r, &z);
        let beta = next / rz;
        rz = next;
        for (d, zi) in p.iter_mut().zip(&z) {
            *d = zi + beta * *d;
        }
    }
    let _ = residual;
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chain of `nodes` nodes, three unknowns each: springs between neighbours along every
    /// component, a little coupling between the components, the first node held.
    fn chain(nodes: usize) -> (Csr, Vec<f64>) {
        let n = 3 * nodes;
        let mut dense = vec![vec![0.0f64; n]; n];
        for node in 0..nodes {
            for c in 0..3 {
                let i = 3 * node + c;
                dense[i][i] += if node == 0 { 2.0 } else { 1.0 };
                if node + 1 < nodes {
                    let j = 3 * (node + 1) + c;
                    dense[i][i] += 1.0;
                    dense[i][j] -= 1.0;
                    dense[j][i] -= 1.0;
                }
                // The components of a node pull on each other a little.
                let other = 3 * node + (c + 1) % 3;
                dense[i][other] += 0.1;
                dense[other][i] += 0.1;
                dense[i][i] += 0.2;
            }
        }
        let (mut ptr, mut idx, mut val) = (vec![0], Vec::new(), Vec::new());
        for row in &dense {
            for (j, v) in row.iter().enumerate() {
                if *v != 0.0 {
                    idx.push(j);
                    val.push(*v);
                }
            }
            ptr.push(idx.len());
        }
        let load: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
        (Csr { n, ptr, idx, val }, load)
    }

    #[test]
    fn the_iteration_solves_what_the_factorization_does() {
        let nodes = 60;
        let (matrix, load) = chain(nodes);
        // Every fourth node is of the coarse level; the blocks are runs of five nodes that
        // overlap by one.
        let linear: Vec<bool> = (0..nodes).map(|node| node % 4 == 0).collect();
        let blocks: Vec<Vec<usize>> = (0..nodes).step_by(4).map(|first| (first..(first + 5).min(nodes)).collect()).collect();
        let solved = solve(&matrix, &load, &linear, &blocks, &[], &Limits::default(), SETTLED, MOST, &|| false).unwrap().expect("settled");
        let mut direct = load.clone();
        Factor::new(&UpperColumns::from_symmetric(&matrix), usize::MAX).unwrap().solve(&mut direct).unwrap();
        let largest = direct.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (a, b) in solved.u.iter().zip(&direct) {
            assert!((a - b).abs() < 1e-7 * largest, "{a} against {b}");
        }
        assert!(solved.iterations > 0 && solved.iterations < 60 && solved.residual <= SETTLED, "{} rounds, {}", solved.iterations, solved.residual);
        assert!(solved.coarse_entries > 0);
        // The same to the bit on one thread and on many.
        let one = parallel::with_threads(1, || solve(&matrix, &load, &linear, &blocks, &[], &Limits::default(), SETTLED, MOST, &|| false))
            .unwrap()
            .unwrap();
        assert!(one.u.iter().zip(&solved.u).all(|(a, b)| a.to_bits() == b.to_bits()) && one.iterations == solved.iterations);

        // No load moves nothing; too few rounds, and a matrix that is not positive, are the
        // factorization's; a caller that cancels ends it.
        let still = solve(&matrix, &vec![0.0; 3 * nodes], &linear, &blocks, &[], &Limits::default(), SETTLED, MOST, &|| false).unwrap().unwrap();
        assert!(still.iterations == 0 && still.u.iter().all(|x| *x == 0.0));
        assert!(solve(&matrix, &load, &linear, &blocks, &[], &Limits::default(), SETTLED, 2, &|| false).unwrap().is_none());
        let mut indefinite = matrix.clone();
        for value in &mut indefinite.val {
            *value = -*value;
        }
        assert!(solve(&indefinite, &load, &linear, &blocks, &[], &Limits::default(), SETTLED, MOST, &|| false).unwrap().is_none());
        assert!(solve(&matrix, &load, &linear, &blocks, &[], &Limits::default(), SETTLED, MOST, &|| true).is_err());
        assert!(solve(&matrix, &load[1..], &linear, &blocks, &[], &Limits::default(), SETTLED, MOST, &|| false).is_err());
        // Without a coarse level the blocks alone still solve it.
        let none = vec![false; nodes];
        let alone = solve(&matrix, &load, &none, &blocks, &[], &Limits::default(), SETTLED, MOST, &|| false).unwrap().expect("settled");
        assert!(alone.coarse_entries == 0 && alone.residual <= SETTLED);
        for (a, b) in alone.u.iter().zip(&direct) {
            assert!((a - b).abs() < 1e-7 * largest, "{a} against {b}");
        }
    }

    #[test]
    fn a_block_is_solved_by_its_own_factor() {
        let (matrix, _) = chain(6);
        let block = Block::new(&matrix, &[4, 1, 2]);
        assert_eq!(block.unknowns, [3, 4, 5, 6, 7, 8, 12, 13, 14]);
        // K_block x = r for a known x.
        let x: Vec<f64> = (0..9).map(|i| 0.5 + i as f64).collect();
        let mut r = vec![0.0; matrix.n];
        for (i, unknown) in block.unknowns.iter().enumerate() {
            let (columns, values) = matrix.row(*unknown);
            let _ = i;
            for (column, value) in columns.iter().zip(values) {
                if let Ok(j) = block.unknowns.binary_search(column) {
                    r[*unknown] += value * x[j];
                }
            }
        }
        let mut out = Vec::new();
        block.solve(&r, &mut out);
        for (a, b) in out.iter().zip(&x) {
            assert!((a - b).abs() < 1e-12, "{a} against {b}");
        }
        assert!(!cholesky(&mut [1.0, 2.0, 1.0], 2), "not positive");
        assert!(!cholesky(&mut [1.0], 2), "the wrong size");
    }
}
