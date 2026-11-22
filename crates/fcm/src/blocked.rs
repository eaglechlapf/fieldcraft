//! The stiffness as the iteration keeps it: by blocks of three by three.
//!
//! A shape function has three unknowns, the displacements along x, y and z, and they couple
//! with all three of every function it couples with: the entries of the stiffness come in
//! blocks of three by three. Kept as blocks with one column each they take 76 bytes for nine
//! entries, where rows of entries with a column each take 144.
//!
//! An iteration's memory is its matrix's, and a round takes its time to read it
//! ([`crate::iterative`]): the product is bound by memory, not by arithmetic. So the cells'
//! matrices are added up into the blocks directly, and the largest system the iteration takes
//! and the time of its rounds both follow from the bytes here. The factorization reads rows
//! ([`Blocked::rows`]).
//!
//! # Determinism
//! The product's sums run node by node, each row's over its blocks in order; the tasks write
//! disjoint parts of the result. It is the same on any number of threads.

use fieldcraft_parallel as parallel;
use fieldcraft_solver::sparse::Csr;

use crate::{Error, Result};

/// Nodes a task of the product takes.
const ROWS: usize = 1024;

/// A symmetric matrix of three unknowns to a node, by blocks.
pub(crate) struct Blocked {
    /// Where each node's blocks start, and where the last one's end.
    start: Vec<usize>,
    /// The node each block is for, rising within a node's blocks.
    node: Vec<u32>,
    /// The blocks, row by row.
    value: Vec<[f64; 9]>,
}

fn outside() -> Error {
    Error::Internal("an entry outside the stiffness's pattern".into())
}

impl Blocked {
    /// The zero matrix on the pattern of `adjacency`: for every node the nodes it couples
    /// with, rising, itself among them ([`fieldcraft_solver::sparse::NodeGraph`]). None when
    /// there is no memory for it, or more nodes than a column counts.
    pub(crate) fn zero(adjacency: &[Vec<usize>]) -> Option<Blocked> {
        u32::try_from(adjacency.len()).ok()?;
        let total = adjacency.iter().try_fold(0usize, |sum, row| sum.checked_add(row.len()))?;
        let (mut start, mut node, mut value) = (Vec::new(), Vec::new(), Vec::new());
        start.try_reserve_exact(adjacency.len() + 1).ok()?;
        node.try_reserve_exact(total).ok()?;
        value.try_reserve_exact(total).ok()?;
        start.push(0);
        for row in adjacency {
            node.extend(row.iter().map(|other| u32::try_from(*other).unwrap_or(u32::MAX)));
            start.push(node.len());
        }
        value.resize(total, [0.0; 9]);
        Some(Blocked { start, node, value })
    }

    /// The nodes: a third of the unknowns.
    pub(crate) fn nodes(&self) -> usize {
        self.start.len().saturating_sub(1)
    }

    /// The entries: nine a block.
    pub(crate) fn entries(&self) -> usize {
        9 * self.node.len()
    }

    /// The blocks of `node`'s three rows: the nodes they are for, rising, and the blocks.
    pub(crate) fn row(&self, node: usize) -> (&[u32], &[[f64; 9]]) {
        let (from, to) = (self.start.get(node).copied().unwrap_or(0), self.start.get(node + 1).copied().unwrap_or(0));
        (self.node.get(from..to).unwrap_or_default(), self.value.get(from..to).unwrap_or_default())
    }

    /// The diagonal entry of `unknown`.
    pub(crate) fn diagonal(&self, unknown: usize) -> Option<f64> {
        let (others, blocks) = self.row(unknown / 3);
        let at = others.binary_search(&u32::try_from(unknown / 3).ok()?).ok()?;
        blocks.get(at)?.get(4 * (unknown % 3)).copied()
    }

    /// Add the matrix `data` of an element: by rows, over the three unknowns of each of
    /// `nodes` in turn. Every two of the nodes must couple in the pattern.
    pub(crate) fn add(&mut self, nodes: &[usize], data: &[f64]) -> Result<()> {
        let width = 3 * nodes.len();
        if data.len() != width * width {
            return Err(Error::Internal("element matrix size".into()));
        }
        for (a, node) in nodes.iter().enumerate() {
            let from = self.start.get(*node).copied().ok_or_else(outside)?;
            let to = self.start.get(node + 1).copied().ok_or_else(outside)?;
            let others = self.node.get(from..to).ok_or_else(outside)?;
            for (b, other) in nodes.iter().enumerate() {
                let other = u32::try_from(*other).map_err(|_| outside())?;
                let at = others.binary_search(&other).map_err(|_| outside())?;
                let block = self.value.get_mut(from + at).ok_or_else(outside)?;
                for (i, sums) in block.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let first = (3 * a + i) * width + 3 * b;
                    for (sum, value) in sums.iter_mut().zip(data.get(first..first + 3).ok_or_else(outside)?) {
                        *sum += value;
                    }
                }
            }
        }
        Ok(())
    }

    /// y = A x: the nodes in parallel, each row's sum over its blocks in order.
    pub(crate) fn mul(&self, x: &[f64], y: &mut [f64]) {
        parallel::chunks_mut(y, 3 * ROWS, |task, out| {
            for (i, sums) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let (others, blocks) = self.row(task * ROWS + i);
                let mut sum = [0.0f64; 3];
                for (other, a) in others.iter().zip(blocks) {
                    let at = 3 * *other as usize;
                    let Some(&[x0, x1, x2]) = x.get(at..at + 3).and_then(|x| x.first_chunk::<3>()) else { continue };
                    sum[0] += a[0] * x0 + a[1] * x1 + a[2] * x2;
                    sum[1] += a[3] * x0 + a[4] * x1 + a[5] * x2;
                    sum[2] += a[6] * x0 + a[7] * x1 + a[8] * x2;
                }
                *sums = sum;
            }
        });
    }

    /// The matrix by rows of entries, each with its column: what the factorization reads.
    pub(crate) fn rows(&self) -> Csr {
        let (n, entries) = (3 * self.nodes(), self.entries());
        let (mut ptr, mut idx, mut val) = (Vec::with_capacity(n + 1), Vec::with_capacity(entries), Vec::with_capacity(entries));
        ptr.push(0);
        for node in 0..self.nodes() {
            let (others, blocks) = self.row(node);
            for i in 0..3 {
                for (other, block) in others.iter().zip(blocks) {
                    let first = 3 * *other as usize;
                    idx.extend(first..first + 3);
                    val.extend_from_slice(block.get(3 * i..3 * i + 3).unwrap_or_default());
                }
                ptr.push(idx.len());
            }
        }
        Csr { n, ptr, idx, val }
    }

    /// The matrix `rows` by blocks (three unknowns to a node), with a zero where a block has
    /// an entry the rows have none for.
    #[cfg(test)]
    pub(crate) fn from_rows(rows: &Csr) -> Blocked {
        let nodes = rows.n / 3;
        let adjacency: Vec<Vec<usize>> = (0..nodes)
            .map(|node| {
                let mut others: Vec<usize> = (3 * node..3 * node + 3).flat_map(|row| rows.row(row).0.iter().map(|column| column / 3)).collect();
                others.sort_unstable();
                others.dedup();
                others
            })
            .collect();
        let mut blocked = Blocked::zero(&adjacency).expect("a test's matrix");
        for row in 0..rows.n {
            let from = blocked.start[row / 3];
            let (columns, values) = rows.row(row);
            for (column, value) in columns.iter().zip(values) {
                let at = adjacency[row / 3].binary_search(&(column / 3)).expect("in the pattern");
                blocked.value[from + at][3 * (row % 3) + column % 3] = *value;
            }
        }
        blocked
    }
}

#[cfg(test)]
mod tests {
    use fieldcraft_fem::{DofLayout, DofSet};
    use fieldcraft_solver::sparse::{NodeGraph, add_block};

    use super::*;

    /// Elements of four nodes along a chain of `nodes`, overlapping by two, each with a
    /// matrix of its own (symmetric, every entry different).
    fn elements(nodes: usize) -> Vec<(Vec<usize>, Vec<f64>)> {
        (0..nodes.saturating_sub(3))
            .step_by(2)
            .map(|first| {
                // Not in rising order: the element's own order of its nodes.
                let list = vec![first + 2, first, first + 3, first + 1];
                let width = 3 * list.len();
                let mut data = vec![0.0; width * width];
                for i in 0..width {
                    for j in 0..=i {
                        let value = ((first * 131 + i * 17 + j * 5) as f64 * 0.37).sin() + if i == j { 30.0 } else { 0.0 };
                        data[i * width + j] = value;
                        data[j * width + i] = value;
                    }
                }
                (list, data)
            })
            .collect()
    }

    #[test]
    fn the_blocks_hold_what_the_rows_hold() {
        let nodes = 40;
        let elements = elements(nodes);
        let graph = NodeGraph::new(nodes, elements.iter().map(|(list, _)| list));
        let layout = DofLayout::uniform(0..nodes as u32, DofSet::TRANSLATIONS);
        let mut rows = graph.zero_matrix(&layout);
        let mut blocked = Blocked::zero(&graph.adjacency).expect("small");
        for (list, data) in &elements {
            add_block(&mut rows, &layout, list, DofSet::TRANSLATIONS, data).unwrap();
            blocked.add(list, data).unwrap();
        }
        assert_eq!((blocked.nodes(), blocked.entries()), (nodes, rows.val.len()));
        // The same sums in the same order: the rows to the bit.
        let back = blocked.rows();
        assert_eq!((back.n, &back.ptr, &back.idx), (rows.n, &rows.ptr, &rows.idx));
        assert!(back.val.iter().zip(&rows.val).all(|(a, b)| a.to_bits() == b.to_bits()));
        for unknown in [0, 7, 3 * nodes - 1] {
            assert_eq!(blocked.diagonal(unknown), Some(rows.diagonal()[unknown]));
        }
        assert_eq!(blocked.diagonal(3 * nodes), None);
        let again = Blocked::from_rows(&rows);
        assert!(again.node == blocked.node && again.value == blocked.value);

        // The product, on one thread and on many.
        let x: Vec<f64> = (0..rows.n).map(|i| ((i * i) as f64 * 0.37).sin()).collect();
        let (mut by_rows, mut by_blocks, mut one) = (vec![0.0; rows.n], vec![1.0; rows.n], vec![0.0; rows.n]);
        rows.mul(&x, &mut by_rows);
        blocked.mul(&x, &mut by_blocks);
        for (a, b) in by_rows.iter().zip(&by_blocks) {
            assert!((a - b).abs() < 1e-12, "{a} against {b}");
        }
        parallel::with_threads(1, || blocked.mul(&x, &mut one));
        assert!(one.iter().zip(&by_blocks).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn what_is_not_in_the_pattern_is_refused() {
        let graph = NodeGraph::new(4, [[0usize, 1], [2, 3]].iter());
        let mut blocked = Blocked::zero(&graph.adjacency).expect("small");
        assert!(blocked.add(&[0, 1], &[1.0; 36]).is_ok());
        // A vector too short for a block's node leaves the block out; none panics.
        let mut y = vec![0.0; 12];
        blocked.mul(&[1.0; 5], &mut y);
        assert_eq!(&y[..3], &[3.0, 3.0, 3.0]);
        assert!(blocked.add(&[1, 2], &[1.0; 36]).is_err(), "two nodes no element joins");
        assert!(blocked.add(&[0, 9], &[1.0; 36]).is_err(), "no such node");
        assert!(blocked.add(&[0, 1], &[1.0; 35]).is_err(), "the wrong size");
        assert!(blocked.add(&[], &[]).is_ok());
        assert_eq!(Blocked::zero(&[]).map(|empty| (empty.nodes(), empty.entries())), Some((0, 0)));
    }
}
