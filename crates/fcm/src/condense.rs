//! The unknowns a cell has to itself, taken out of the system before it is iterated.
//!
//! A shape function that only one cell carries couples with nothing outside that cell: the
//! functions inside it (the products of the higher polynomials along all three axes), and at
//! the surface of the part those on the faces and edges whose other cells hold no material.
//! Their unknowns u_I follow from the others' u_B by the cell's own equations,
//!
//! K_II u_I = f_I − K_IB u_B,
//!
//! so the system is solved for u_B alone, on the stiffness and the load they leave,
//!
//! S = K_BB − Σ K_BI K_II⁻¹ K_IB, g = f_B − Σ K_BI K_II⁻¹ f_I,
//!
//! each sum over the cells: static condensation (R. J. Guyan, *Reduction of stiffness and mass
//! matrices*, AIAA J. 3(2):380, 1965), which elements of high degree are solved with as a
//! rule, for the functions inside each. Nothing is approximated: with u_B solved, u_I is
//! exact.
//!
//! # Why
//! An iteration is slow on what its blocks do not hold together ([`crate::iterative`]). Where
//! cells hold little material, a function they share is nearly a sum of functions each has to
//! itself, on the material there is: that difference has almost no stiffness, and no block of
//! shared functions holds it. In S it is not there to be found: a function's stiffness in S is
//! what is left of it once every cell's own functions have taken their part.
//! It also leaves fewer unknowns to iterate on: seven eighths of them at degree 3 inside a
//! part, and about half where the part is one or two cells thick.
//!
//! # In place
//! S has the pattern of K on the unknowns that are left, so its blocks take the place of
//! K's ([`Blocked::change_rows`]); the rows of the unknowns taken out stay as they are, for
//! [`Condensed::recover`]. The products of the iteration leave those rows and columns out
//! ([`Blocked::mul_kept`]).
//!
//! # Determinism
//! A block of S is changed by the cells that hold both its nodes, in their order; each cell's
//! part is computed from that cell alone. The result is the same on any number of threads.

use fieldcraft_parallel as parallel;
use fieldcraft_solver::cholesky::{Factor, FactorError, UpperColumns};

use crate::blocked::Blocked;
use crate::packed;
use crate::{Error, Result};

/// Cells a task takes.
const GRAIN: usize = 16;

/// The nodes one cell has to itself, rising, and the Cholesky factor of their matrix.
struct Cell {
    inner: Vec<usize>,
    factor: Vec<f64>,
}

/// What a cell's own unknowns take out of its others', until it is taken.
struct Share {
    cell: Cell,
    /// The cell's other nodes, rising.
    outer: Vec<usize>,
    /// L⁻¹ K_IB, column by column.
    columns: Vec<f64>,
    /// L⁻¹ f_I.
    load: Vec<f64>,
}

impl Share {
    /// The column of `unknown` among the other nodes' unknowns.
    fn column(&self, unknown: usize) -> &[f64] {
        let n = 3 * self.cell.inner.len();
        self.columns.get(unknown * n..(unknown + 1) * n).unwrap_or_default()
    }
}

/// A system with the cells' own unknowns taken out.
pub(crate) struct Condensed {
    /// The nodes that were taken out.
    pub own: Vec<bool>,
    /// The load on the nodes that are left, zero on the others.
    pub load: Vec<f64>,
    cells: Vec<Cell>,
}

/// The sum of the products of two lists, from the first to the last.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// What the own nodes of the cell of `list` (rising, each once) take out of its others; `count`
/// says in how many lists each node is. None when it has none to itself, their matrix is not
/// positive (they are then left in the system) or there is no memory.
fn share(matrix: &Blocked, load: &[f64], list: &[usize], count: &[u32]) -> Option<Share> {
    let own = |node: &usize| count.get(*node) == Some(&1);
    let inner: Vec<usize> = list.iter().copied().filter(own).collect();
    if inner.is_empty() {
        return None;
    }
    let outer: Vec<usize> = list.iter().copied().filter(|node| !own(node)).collect();
    let n = 3 * inner.len();
    let mut factor = packed::lower(matrix, &inner)?;
    if !packed::cholesky(&mut factor, n) {
        return None;
    }
    let mut columns = Vec::new();
    columns.try_reserve_exact(3 * outer.len() * n).ok()?;
    columns.resize(3 * outer.len() * n, 0.0);
    for (i, node) in inner.iter().enumerate() {
        let (others, blocks) = matrix.row(*node);
        // Both lists rise: one pass over them.
        let mut j = 0;
        for (other, block) in others.iter().zip(blocks) {
            let other = *other as usize;
            while outer.get(j).is_some_and(|node| *node < other) {
                j += 1;
            }
            if outer.get(j) != Some(&other) {
                continue;
            }
            for (r, entries) in block.as_chunks::<3>().0.iter().enumerate() {
                for (c, value) in entries.iter().enumerate() {
                    if let Some(slot) = columns.get_mut((3 * j + c) * n + 3 * i + r) {
                        *slot = *value;
                    }
                }
            }
        }
    }
    for column in columns.chunks_exact_mut(n) {
        packed::forward(&factor, column);
    }
    let mut own_load: Vec<f64> =
        inner.iter().flat_map(|node| 3 * node..3 * node + 3).map(|unknown| load.get(unknown).copied().unwrap_or(0.0)).collect();
    packed::forward(&factor, &mut own_load);
    Some(Share { cell: Cell { inner, factor }, outer, columns, load: own_load })
}

/// Take the unknowns of the nodes that are in one of `lists` only out of `matrix` and `load`:
/// the lists are the nodes of every cell, and of every two cells a connection couples. The
/// blocks of `matrix` between the nodes that are left become S; the others stay.
pub(crate) fn condense(matrix: &mut Blocked, load: &[f64], lists: &[Vec<usize>]) -> Result<Condensed> {
    let nodes = matrix.nodes();
    if load.len() != 3 * nodes {
        return Err(Error::Internal("the system and its load differ in size".into()));
    }
    let lists: Vec<Vec<usize>> = parallel::map_grained(lists.len(), GRAIN, |at| {
        let mut list = lists.get(at).cloned().unwrap_or_default();
        list.sort_unstable();
        list.dedup();
        list
    });
    let mut count = vec![0u32; nodes];
    for node in lists.iter().flatten() {
        if let Some(count) = count.get_mut(*node) {
            *count = count.saturating_add(1);
        }
    }
    let shares: Vec<Option<Share>> = {
        let matrix = &*matrix;
        parallel::map_grained(lists.len(), GRAIN, |at| share(matrix, load, lists.get(at)?, &count))
    };
    drop(lists);
    drop(count);

    // For every node that is left, the cells that take from it and where it is among their
    // other nodes; the cells in their order.
    let mut own = vec![false; nodes];
    let mut takers: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nodes];
    for (at, share) in shares.iter().enumerate() {
        let Some(share) = share else { continue };
        for node in &share.cell.inner {
            if let Some(flag) = own.get_mut(*node) {
                *flag = true;
            }
        }
        for (position, node) in share.outer.iter().enumerate() {
            if let (Some(list), Ok(at), Ok(position)) = (takers.get_mut(*node), u32::try_from(at), u32::try_from(position)) {
                list.push((at, position));
            }
        }
    }
    matrix.change_rows(|node, others, blocks| {
        for (at, position) in takers.get(node).map(Vec::as_slice).unwrap_or_default() {
            let Some(Some(share)) = shares.get(*at as usize) else { continue };
            let mine = [0, 1, 2].map(|r| share.column(3 * *position as usize + r));
            // Both lists rise: one pass over them.
            let mut j = 0;
            for (other, block) in others.iter().zip(blocks.iter_mut()) {
                let other = *other as usize;
                while share.outer.get(j).is_some_and(|node| *node < other) {
                    j += 1;
                }
                if share.outer.get(j) != Some(&other) {
                    continue;
                }
                for (mine, entries) in mine.iter().zip(block.as_chunks_mut::<3>().0) {
                    for (c, entry) in entries.iter_mut().enumerate() {
                        *entry -= dot(mine, share.column(3 * j + c));
                    }
                }
            }
        }
    });
    drop(takers);

    let mut left = load.to_vec();
    for share in shares.iter().flatten() {
        for (position, node) in share.outer.iter().enumerate() {
            for c in 0..3 {
                if let Some(entry) = left.get_mut(3 * node + c) {
                    *entry -= dot(share.column(3 * position + c), &share.load);
                }
            }
        }
        for node in &share.cell.inner {
            if let Some(entries) = left.get_mut(3 * node..3 * node + 3) {
                entries.fill(0.0);
            }
        }
    }
    let cells = shares.into_iter().flatten().map(|share| share.cell).collect();
    Ok(Condensed { own, load: left, cells })
}

impl Condensed {
    /// The unknowns of the nodes that were taken out, into `u`, which has those of the nodes
    /// that were left: of the system `matrix` was made of, with the load it had.
    pub(crate) fn recover(&self, matrix: &Blocked, load: &[f64], u: &mut [f64]) {
        let solved: Vec<Vec<f64>> = {
            let u = &*u;
            parallel::map_grained(self.cells.len(), GRAIN, |at| {
                let Some(cell) = self.cells.get(at) else { return Vec::new() };
                let mut x = Vec::with_capacity(3 * cell.inner.len());
                for node in &cell.inner {
                    let mut rest = [0, 1, 2].map(|c| load.get(3 * node + c).copied().unwrap_or(0.0));
                    let (others, blocks) = matrix.row(*node);
                    for (other, a) in others.iter().zip(blocks) {
                        let at = 3 * *other as usize;
                        if self.own.get(*other as usize).copied().unwrap_or(false) {
                            continue;
                        }
                        let Some(&[u0, u1, u2]) = u.get(at..at + 3).and_then(|u| u.first_chunk::<3>()) else { continue };
                        rest[0] -= a[0] * u0 + a[1] * u1 + a[2] * u2;
                        rest[1] -= a[3] * u0 + a[4] * u1 + a[5] * u2;
                        rest[2] -= a[6] * u0 + a[7] * u1 + a[8] * u2;
                    }
                    x.extend(rest);
                }
                packed::forward(&cell.factor, &mut x);
                packed::backward(&cell.factor, &mut x);
                x
            })
        };
        for (cell, x) in self.cells.iter().zip(&solved) {
            for (node, values) in cell.inner.iter().zip(x.as_chunks::<3>().0) {
                if let Some(entries) = u.get_mut(3 * node..3 * node + 3) {
                    entries.copy_from_slice(values);
                }
            }
        }
    }

    /// The unknowns of the nodes that are left by factorization of S (`matrix` after
    /// [`condense`]), zero on the others, and the entries of the factor: for what the
    /// iteration does not settle.
    pub(crate) fn factored(&self, matrix: &Blocked, max_entries: usize) -> std::result::Result<(Vec<f64>, usize), FactorError> {
        let rows = matrix.rows_kept(&self.own);
        let upper = UpperColumns::from_symmetric(&rows);
        drop(rows);
        let factor = Factor::new(&upper, max_entries)?;
        drop(upper);
        let kept = || self.own.iter().enumerate().filter(|(_, own)| !**own).flat_map(|(node, _)| 3 * node..3 * node + 3);
        let mut x: Vec<f64> = kept().map(|unknown| self.load.get(unknown).copied().unwrap_or(0.0)).collect();
        factor.solve(&mut x).map_err(|e| FactorError::Internal(e.to_string()))?;
        let mut u = vec![0.0; self.load.len()];
        for (unknown, value) in kept().zip(&x) {
            if let Some(slot) = u.get_mut(unknown) {
                *slot = *value;
            }
        }
        Ok((u, factor.entries()))
    }
}

#[cfg(test)]
mod tests {
    use fieldcraft_solver::sparse::NodeGraph;

    use super::*;

    /// Cells of five nodes along a chain, each sharing its first and last node with the cells
    /// beside it and keeping three to itself, with a matrix of its own (symmetric, positive).
    fn cells(count: usize) -> Vec<(Vec<usize>, Vec<f64>)> {
        (0..count)
            .map(|cell| {
                let list = vec![4 * cell + 4, 4 * cell + 2, 4 * cell, 4 * cell + 3, 4 * cell + 1];
                let width = 3 * list.len();
                let mut data = vec![0.0; width * width];
                for i in 0..width {
                    for j in 0..=i {
                        let value = ((cell * 131 + i * 17 + j * 5) as f64 * 0.37).sin() + if i == j { 20.0 } else { 0.0 };
                        data[i * width + j] = value;
                        data[j * width + i] = value;
                    }
                }
                (list, data)
            })
            .collect()
    }

    fn system(count: usize) -> (Blocked, Vec<f64>, Vec<Vec<usize>>) {
        let cells = cells(count);
        let nodes = 4 * count + 1;
        let graph = NodeGraph::new(nodes, cells.iter().map(|(list, _)| list));
        let mut matrix = Blocked::zero(&graph.adjacency).expect("small");
        for (list, data) in &cells {
            matrix.add(list, data).unwrap();
        }
        let load: Vec<f64> = (0..3 * nodes).map(|i| ((i * i) as f64 * 0.11).cos()).collect();
        (matrix, load, cells.into_iter().map(|(list, _)| list).collect())
    }

    #[test]
    fn what_is_left_and_what_was_taken_out_solve_the_system() {
        let count = 9;
        let (mut matrix, load, lists) = system(count);
        let whole = matrix.rows();
        let mut direct = load.clone();
        Factor::new(&UpperColumns::from_symmetric(&whole), usize::MAX).unwrap().solve(&mut direct).unwrap();

        let condensed = condense(&mut matrix, &load, &lists).unwrap();
        // Three nodes of every cell are its own; the nodes between two cells, and the two ends
        // (one list each, but an end's cell has them too: also own), are not all left.
        let own = condensed.own.iter().filter(|own| **own).count();
        assert_eq!(own, 3 * count + 2, "the cells' three, and the chain's two ends");
        assert!(condensed.load.iter().zip(&condensed.own.iter().flat_map(|own| [*own; 3]).collect::<Vec<_>>()).all(|(g, own)| !own || *g == 0.0));
        // S is symmetric to the bit.
        let rows = matrix.rows_kept(&condensed.own);
        for r in 0..rows.n {
            let (columns, values) = rows.row(r);
            for (c, value) in columns.iter().zip(values) {
                let (back, mirrored) = rows.row(*c);
                let at = back.binary_search(&r).expect("a symmetric pattern");
                assert_eq!(value.to_bits(), mirrored[at].to_bits());
            }
        }
        let (mut u, entries) = condensed.factored(&matrix, usize::MAX).unwrap();
        assert!(entries > 0);
        condensed.recover(&matrix, &load, &mut u);
        let largest = direct.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (a, b) in u.iter().zip(&direct) {
            assert!((a - b).abs() < 1e-12 * largest, "{a} against {b}");
        }
        // The products that leave the own unknowns out are S's.
        let x: Vec<f64> =
            condensed.own.iter().flat_map(|own| [*own; 3]).enumerate().map(|(i, own)| if own { 0.0 } else { (i as f64 * 0.7).sin() }).collect();
        let mut y = vec![0.0; x.len()];
        matrix.mul_kept(&condensed.own, &x, &mut y);
        let compact: Vec<f64> = x.iter().zip(condensed.own.iter().flat_map(|own| [*own; 3])).filter(|(_, own)| !own).map(|(x, _)| *x).collect();
        let mut by_rows = vec![0.0; rows.n];
        rows.mul(&compact, &mut by_rows);
        let kept: Vec<f64> = y.iter().zip(condensed.own.iter().flat_map(|own| [*own; 3])).filter(|(_, own)| !own).map(|(y, _)| *y).collect();
        for (a, b) in kept.iter().zip(&by_rows) {
            assert!((a - b).abs() < 1e-12, "{a} against {b}");
        }
        assert!(y.iter().zip(condensed.own.iter().flat_map(|own| [*own; 3])).all(|(y, own)| !own || *y == 0.0));

        // The same to the bit on one thread.
        let (mut again, load_again, _) = system(count);
        let one = parallel::with_threads(1, || condense(&mut again, &load_again, &lists)).unwrap();
        assert!(one.load.iter().zip(&condensed.load).all(|(a, b)| a.to_bits() == b.to_bits()));
        let (a, b) = (again.rows(), matrix.rows());
        assert!(a.val.iter().zip(&b.val).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn what_cannot_be_taken_out_is_left_in() {
        // A node in two lists is no cell's own; a cell whose own matrix is not positive keeps
        // its unknowns in the system; a load of the wrong size is refused.
        let (mut matrix, load, mut lists) = system(3);
        lists.push(vec![1, 2]);
        let condensed = condense(&mut matrix, &load, &lists).unwrap();
        assert!(!condensed.own[1] && !condensed.own[2] && condensed.own[3] && condensed.own[5]);
        let (mut matrix, load, lists) = system(3);
        matrix.add(&[5], &[-100.0, 0.0, 0.0, 0.0, -100.0, 0.0, 0.0, 0.0, -100.0]).unwrap();
        let condensed = condense(&mut matrix, &load, &lists).unwrap();
        assert!(!condensed.own[5] && !condensed.own[6] && !condensed.own[7], "the second cell's own");
        assert!(condensed.own[1] && condensed.own[9]);
        assert!(condense(&mut matrix, &load[1..], &lists).is_err());
        let (mut matrix, load, _) = system(3);
        let none = condense(&mut matrix, &load, &[]).unwrap();
        assert!(none.own.iter().all(|own| !own) && none.load == load);
    }
}
