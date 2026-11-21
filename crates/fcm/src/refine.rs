//! Local refinement: cells divided where they are too coarse, and the shape functions of the
//! cells of every level joined into one conforming basis.
//!
//! This is the multi-level hp method (N. Zander, T. Bog, S. Kollmannsberger, D. Schillinger,
//! E. Rank, *Multi-level hp-adaptivity: high-order mesh adaptivity without the difficulties of
//! constraining hanging nodes*, Comput. Mech. 55(3):499–517, 2015; in three dimensions N. Zander,
//! T. Bog, M. Elhaddad, F. Frischmann, S. Kollmannsberger, E. Rank, *The multi-level hp-method
//! for three-dimensional problems: dynamically changing high-order mesh refinement with
//! arbitrary hanging nodes*, Comput. Methods Appl. Mech. Engrg. 310:252–277, 2016), which grew
//! out of the hp-d finite cell method (D. Schillinger, E. Rank, *An unfitted hp-adaptive finite
//! element method based on hierarchical B-splines for interface problems of complex geometry*,
//! Comput. Methods Appl. Mech. Engrg. 200(47–48):3358–3380, 2011).
//!
//! # The cells
//! A cell of the grid that is too coarse is divided into eight cells of half its size, and any
//! of those again: a tree under every cell of the grid, of which the cells that are not divided
//! (the *leaves*) are the ones the part is integrated over. A divided cell is not replaced: it
//! stays, and its children are laid over it. Neighbouring leaves may differ by any number of
//! levels.
//!
//! # Which shape functions are active
//! Every cell of every level has the shape functions of a cell of the grid ([`crate::space`]),
//! on its vertices, edges, faces and interior, which the cells of its own level share. Two rules
//! pick the ones that are unknowns, in the form of P. Kopp, E. Rank, V. M. Calo,
//! S. Kollmannsberger, *Efficient multi-level hp-finite elements in arbitrary dimensions*,
//! Comput. Methods Appl. Mech. Engrg. 401:115575, 2022. The functions of a vertex, an edge, a
//! face or an interior of level k are active when
//! 1. a leaf of level k touches it (holds it in its closure): functions live on the level of
//!    the leaves they serve, so a divided cell keeps the functions it shares with a neighbour
//!    that is not divided and hands all others down to its children (this keeps the functions
//!    linearly independent: a function of a divided cell that its children could represent is
//!    not there);
//! 2. no leaf of a lower level touches it: the functions of the finer cells vanish on the
//!    boundary of the zone they refine, where only the coarser neighbour's functions are left
//!    (this keeps the field continuous without constraining hanging nodes).
//!
//! A cell that holds no material does not count: the boundary of the cells that hold material
//! is the boundary of the grid, as it is without refinement, and the finest functions are
//! active on it. Such a cell is not integrated either, also where it is a part of a divided cut
//! cell that a function of the cut cell reaches: there the functions of its own level are
//! missing, the ones left could not move rigidly without straining, and a fictitious stiffness
//! on them would pull at the part (the reactions were off by a hundred times the fictitious
//! fraction when it was tried). Every such function has a whole leaf of its own level to hold
//! it. The span is that of Zander et al.; the difference is that a vertex all of whose cells
//! are divided has its function on the finer level, with the smaller support.
//!
//! # A leaf's functions on its own modes
//! A function of a divided cell is, on one of the leaves under it, a polynomial of degree p
//! along each axis in the leaf's coordinates, so it is a combination of the leaf's own shape
//! functions, active or not, with coefficients that are products of three one-dimensional ones
//! ([`Transfer`]). The functions that reach a leaf (its own active ones and the active ones of
//! the cells above it) are therefore the columns of a matrix B on the leaf's modes ([`Map`]).
//! Everything a leaf integrates (stiffness, loads, supports) is integrated in its own modes,
//! exactly as for a cell of the grid, and then becomes Bᵀ K B and Bᵀ f; the field in a leaf is
//! that of the mode coefficients B u. The trunk space works the same way: the span of its modes
//! is a sum of full tensor-product spaces of lower degrees, each of which holds its own
//! functions on a part of the cell.
//!
//! The Nitsche parameter of a leaf stays the one of its own modes ([`crate::cell`]): the
//! functions that reach the leaf span a subspace of the span of its modes, and the largest
//! eigenvalue of the pencil on a subspace is no larger, so the form stays positive.

use crate::gauss::gauss_legendre;
use crate::grid::{Grid, MAX_REFINE};
use crate::legendre::{MAX_DEGREE, shape};
use crate::space::{Kind, NONE};
use crate::{Error, Result};

/// A cell of the grid or of a refined level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Node {
    /// 0 for a cell of the grid.
    pub level: u8,
    pub kind: Kind,
    /// Its position among the cells of its level ([`Grid::fine_lo`]).
    pub ijk: [u32; 3],
    /// The first of its eight children (child c is in the upper half along axis d where bit d
    /// of c is set), or [`NONE`] for a leaf.
    pub children: u32,
    /// The cell it is a part of, or [`NONE`] for a cell of the grid.
    pub parent: u32,
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        self.children == NONE
    }

    /// Whether the cell holds material.
    pub fn material(&self) -> bool {
        self.kind != Kind::Outside
    }

    pub fn position(&self) -> [usize; 3] {
        self.ijk.map(|i| i as usize)
    }
}

/// The cells: those of the grid that hold material, and the cells they are divided into.
#[derive(Debug, Clone)]
pub(crate) struct Tree {
    /// The cells of the grid along each axis.
    cells: [usize; 3],
    /// For each cell of the grid, its node, or [`NONE`] where it holds no material.
    pub root: Vec<u32>,
    /// The cells of the grid that hold material in the grid's order, then the children in the
    /// order their parents were divided.
    pub nodes: Vec<Node>,
}

/// Every one of the 27 parts of a cell (its vertices, edges, faces and interior).
pub(crate) const ALL_PARTS: u32 = (1 << 27) - 1;

/// The part of a cell a mode sits on: along each axis its lower plane (0), its upper plane (1) or
/// its span (2), at `x + 3 (y + 3 z)`.
pub(crate) fn part_of(mode: &[u8; 3]) -> usize {
    usize::from(mode[0].min(2)) + 3 * (usize::from(mode[1].min(2)) + 3 * usize::from(mode[2].min(2)))
}

impl Tree {
    /// The cells of the grid that hold material, none of them divided.
    pub fn new(grid: &Grid, kinds: &[Kind]) -> Tree {
        let mut tree = Tree { cells: grid.cells, root: vec![NONE; kinds.len()], nodes: Vec::new() };
        for (cell, kind) in kinds.iter().enumerate() {
            if *kind == Kind::Outside {
                continue;
            }
            if let Some(root) = tree.root.get_mut(cell) {
                *root = tree.nodes.len() as u32;
            }
            let [i, j, k] = grid.coords(cell);
            tree.nodes.push(Node { level: 0, kind: *kind, ijk: [i as u32, j as u32, k as u32], children: NONE, parent: NONE });
        }
        tree
    }

    /// Divide the leaf `node` into eight cells of the next level with the `kinds`; the first of
    /// them is returned.
    pub fn divide(&mut self, node: usize, kinds: [Kind; 8]) -> Result<usize> {
        let broken = || Error::Internal("a cell that cannot be divided".into());
        let parent = *self.nodes.get(node).ok_or_else(broken)?;
        let first = self.nodes.len();
        if !parent.is_leaf() || !parent.material() || usize::from(parent.level) >= MAX_REFINE || first + 8 >= NONE as usize {
            return Err(broken());
        }
        for (child, kind) in kinds.into_iter().enumerate() {
            let bit = |axis: usize| ((child >> axis) & 1) as u32;
            let ijk = [2 * parent.ijk[0] + bit(0), 2 * parent.ijk[1] + bit(1), 2 * parent.ijk[2] + bit(2)];
            self.nodes.push(Node { level: parent.level + 1, kind, ijk, children: NONE, parent: node as u32 });
        }
        if let Some(parent) = self.nodes.get_mut(node) {
            parent.children = first as u32;
        }
        Ok(first)
    }

    /// The deepest level of any cell.
    pub fn deepest(&self) -> usize {
        self.nodes.iter().map(|node| usize::from(node.level)).max().unwrap_or(0)
    }

    /// The deepest cell of a level up to `level` that covers the place of cell `at` of `level`;
    /// `None` where no cell of the grid holds material.
    pub fn covering(&self, level: usize, at: [i64; 3]) -> Option<usize> {
        let mut base = [0usize; 3];
        for axis in 0..3 {
            if at[axis] < 0 {
                return None;
            }
            base[axis] = (at[axis] >> level) as usize;
            if base[axis] >= self.cells[axis] {
                return None;
            }
        }
        let root = self.root.get(base[0] + self.cells[0] * (base[1] + self.cells[1] * base[2])).copied()?;
        if root == NONE {
            return None;
        }
        let mut node = root as usize;
        for down in (0..level).rev() {
            let cell = self.nodes.get(node)?;
            if cell.is_leaf() {
                break;
            }
            let octant = (0..3).map(|axis| (((at[axis] >> down) & 1) as usize) << axis).sum::<usize>();
            node = cell.children as usize + octant;
        }
        Some(node)
    }

    /// The leaf that covers the place of cell `at` of `level`, or of the leaves under that cell
    /// the one nearest to the point `t` (in cells of the grid from its origin).
    pub fn leaf_near(&self, level: usize, at: [i64; 3], t: [f64; 3]) -> Option<usize> {
        let mut node = self.covering(level, at)?;
        for _ in 0..=MAX_REFINE {
            let cell = self.nodes.get(node)?;
            if cell.is_leaf() {
                return Some(node);
            }
            // The half of the cell the point is in or beside, along each axis.
            let steps = (2u64 << cell.level) as f64;
            let octant = (0..3).map(|axis| usize::from(t[axis] * steps >= (2 * cell.ijk[axis] + 1) as f64) << axis).sum::<usize>();
            node = cell.children as usize + octant;
        }
        None
    }

    /// The cell `at` of `level`, when there is one.
    pub fn find(&self, level: usize, at: [i64; 3]) -> Option<usize> {
        let node = self.covering(level, at)?;
        (usize::from(self.nodes.get(node)?.level) == level).then_some(node)
    }

    /// Which parts of cell `node` carry active shape functions (bit [`part_of`]): the ones a
    /// leaf of the cell's level touches and no leaf of a lower level does, of the cells that
    /// hold material (the module's two rules).
    pub fn active_parts(&self, node: usize) -> u32 {
        let Some(cell) = self.nodes.get(node) else { return 0 };
        if !cell.material() {
            return 0;
        }
        let level = usize::from(cell.level);
        let at = cell.ijk.map(i64::from);
        // The cells of the level around a part, along one axis.
        let around = |class: usize, i: i64| match class {
            0 => [Some(i - 1), Some(i)],
            1 => [Some(i), Some(i + 1)],
            _ => [Some(i), None],
        };
        let mut parts = 0;
        for part in 0..27 {
            let (mut leaf, mut coarser) = (false, false);
            for z in around(part / 9, at[2]).into_iter().flatten() {
                for y in around((part / 3) % 3, at[1]).into_iter().flatten() {
                    for x in around(part % 3, at[0]).into_iter().flatten() {
                        let Some(other) = self.covering(level, [x, y, z]).and_then(|other| self.nodes.get(other)) else { continue };
                        if !other.material() {
                            continue;
                        }
                        if usize::from(other.level) < level {
                            coarser = true;
                        } else if other.is_leaf() {
                            leaf = true;
                        }
                    }
                }
            }
            if leaf && !coarser {
                parts |= 1 << part;
            }
        }
        parts
    }

    /// The leaves: under the cells of the grid in the grid's order, the children of a cell in
    /// theirs.
    pub fn leaves(&self) -> Vec<u32> {
        let mut leaves = Vec::new();
        let mut stack = Vec::new();
        for &root in &self.root {
            if root == NONE {
                continue;
            }
            stack.push(root);
            while let Some(node) = stack.pop() {
                let Some(cell) = self.nodes.get(node as usize) else { continue };
                if cell.is_leaf() {
                    leaves.push(node);
                } else {
                    stack.extend((0..8).rev().map(|child| cell.children + child));
                }
            }
        }
        leaves
    }
}

/// The one-dimensional shape functions of a cell ([`crate::legendre`]) on a part of it, in the
/// part's own shape functions. `d` levels down a cell's span is 2ᵈ parts; on part `o` of them,
/// which is [s₀, s₁] of the cell's [−1, 1], φₐ(s₀ + (ξ + 1)(s₁ − s₀)/2) = Σ_b T[a][b] φ_b(ξ) with
/// T[a][0] = φₐ(s₀), T[a][1] = φₐ(s₁) and, for b ≥ 2, T[a][b] = (s₁ − s₀)/2 ∫ φₐ′ φ_b′ dξ: the
/// derivatives of the higher functions are orthonormal and orthogonal to the constant, the
/// derivative of the linear ones. T[a][b] is zero for b above the degree of φₐ.
#[derive(Debug, Clone)]
pub(crate) struct Transfer {
    q: usize,
    levels: usize,
    /// The table of part `o` of the 2ᵈ parts `d` levels down at `(2ᵈ − 2 + o) q²`, row by row.
    tables: Vec<f64>,
}

impl Transfer {
    /// The tables of degree `p` for up to `levels` levels down.
    pub fn new(p: usize, levels: usize) -> Transfer {
        let p = p.clamp(1, MAX_DEGREE);
        let levels = levels.min(MAX_REFINE);
        let q = p + 1;
        // φₐ′ φ_b′ has degree 2p − 2 at most.
        let rule = gauss_legendre(p + 1);
        let mut tables = vec![0.0; ((1usize << (levels + 1)) - 2) * q * q];
        for d in 1..=levels {
            let parts = 1usize << d;
            // Half the length of a part: the ends of the parts are dyadic, so they are exact.
            let half = 1.0 / parts as f64;
            for o in 0..parts {
                let lower = -1.0 + 2.0 * o as f64 * half;
                let (at_lower, at_upper) = (shape(p, lower), shape(p, lower + 2.0 * half));
                let at = (parts - 2 + o) * q * q;
                for a in 0..q {
                    tables[at + a * q] = at_lower.n[a];
                    tables[at + a * q + 1] = at_upper.n[a];
                }
                for (x, w) in &rule {
                    let (coarse, fine) = (shape(p, lower + (x + 1.0) * half), shape(p, *x));
                    for a in 2..q {
                        for b in 2..=a {
                            tables[at + a * q + b] += w * half * coarse.d[a] * fine.d[b];
                        }
                    }
                }
            }
        }
        Transfer { q, levels, tables }
    }

    /// The coefficients of the functions of part `o`, `d` levels down, in function `a` of the
    /// cell (empty for a part or a function there is not).
    pub fn row(&self, d: usize, o: usize, a: usize) -> &[f64] {
        if d == 0 || d > self.levels || o >= (1 << d) || a >= self.q {
            return &[];
        }
        let at = (((1usize << d) - 2 + o) * self.q + a) * self.q;
        self.tables.get(at..at + self.q).unwrap_or_default()
    }
}

/// The functions that reach a leaf on the leaf's own modes: the matrix B of the module, one
/// sparse column a function.
#[derive(Debug, Clone, Default)]
pub(crate) struct Map {
    /// Where each column starts, and the end of the last.
    start: Vec<u32>,
    /// The mode of each entry.
    rows: Vec<u16>,
    values: Vec<f64>,
}

impl Map {
    /// The number of functions.
    pub fn functions(&self) -> usize {
        self.start.len().saturating_sub(1)
    }

    /// Add a function by its coefficients on the modes.
    pub fn push(&mut self, entries: impl Iterator<Item = (usize, f64)>) {
        if self.start.is_empty() {
            self.start.push(0);
        }
        for (row, value) in entries {
            self.rows.push(row as u16);
            self.values.push(value);
        }
        self.start.push(self.rows.len() as u32);
    }

    fn column(&self, function: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = match (self.start.get(function), self.start.get(function + 1)) {
            (Some(&from), Some(&to)) => from as usize..to as usize,
            _ => 0..0,
        };
        let rows = self.rows.get(range.clone()).unwrap_or_default();
        let values = self.values.get(range).unwrap_or_default();
        rows.iter().zip(values).map(|(row, value)| (usize::from(*row), *value))
    }

    /// B u: the coefficients of the cell's `m` modes (three components each) from those of the
    /// functions.
    pub fn local(&self, m: usize, u: &[f64]) -> Vec<f64> {
        let mut local = vec![0.0; 3 * m];
        for (function, u) in u.as_chunks::<3>().0.iter().enumerate().take(self.functions()) {
            for (row, weight) in self.column(function) {
                if let Some(out) = local.get_mut(3 * row..3 * row + 3) {
                    for (out, u) in out.iter_mut().zip(u) {
                        *out += weight * u;
                    }
                }
            }
        }
        local
    }

    /// Bᵀ f for a vector on the cell's `m` modes (three components each).
    pub fn vector(&self, m: usize, f: &[f64]) -> Vec<f64> {
        let mut projected = vec![0.0; 3 * self.functions()];
        if f.len() != 3 * m {
            return projected;
        }
        for (function, out) in projected.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            for (row, weight) in self.column(function) {
                if let Some(f) = f.get(3 * row..3 * row + 3) {
                    for (out, f) in out.iter_mut().zip(f) {
                        *out += weight * f;
                    }
                }
            }
        }
        projected
    }

    /// Bᵀ K B for a symmetric matrix on the cell's `m` modes (3m × 3m, row-major): symmetric to
    /// the last bit.
    pub fn matrix(&self, m: usize, k: &[f64]) -> Vec<f64> {
        let (n, functions) = (3 * m, self.functions());
        let width = 3 * functions;
        let mut projected = vec![0.0; width * width];
        if k.len() != n * n || self.rows.iter().any(|row| usize::from(*row) >= m) {
            return projected;
        }
        // K B, one row a function's component (K is symmetric).
        let mut half = vec![0.0; width * n];
        for (function, out) in half.chunks_exact_mut(3 * n).enumerate() {
            for (row, weight) in self.column(function) {
                for (out, k) in out.chunks_exact_mut(n).zip(k[3 * row * n..(3 * row + 3) * n].chunks_exact(n)) {
                    for (out, k) in out.iter_mut().zip(k) {
                        *out += weight * k;
                    }
                }
            }
        }
        // Bᵀ (K B): the upper triangle, then its mirror.
        for function in 0..functions {
            for (row, weight) in self.column(function) {
                for component in 0..3 {
                    let (at, mode) = (3 * function + component, 3 * row + component);
                    for column in 3 * function..width {
                        projected[at * width + column] += weight * half[column * n + mode];
                    }
                }
            }
        }
        for row in 0..width {
            for column in row + 1..width {
                projected[column * width + row] = projected[row * width + column];
            }
        }
        projected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{self, Face, Lame, Shapes, Surface};
    use crate::dense;
    use crate::grid::{Piece, QuadPoint};
    use crate::moments::Scratch;
    use crate::space::{Space, local_modes};
    use std::collections::BTreeMap;

    #[test]
    fn a_function_on_a_part_of_its_cell_is_a_sum_of_the_parts_functions() {
        for p in [1, 2, 3, 5, MAX_DEGREE] {
            let transfer = Transfer::new(p, 3);
            for d in 1..=3usize {
                for o in 0..1usize << d {
                    let (lower, length) = (-1.0 + 2.0 * o as f64 / (1 << d) as f64, 2.0 / (1 << d) as f64);
                    for a in 0..=p {
                        let row = transfer.row(d, o, a);
                        assert_eq!(row.len(), p + 1);
                        // Nothing above the function's own degree.
                        assert!(row.iter().skip(a.max(1) + 1).all(|x| *x == 0.0), "degree {p}, function {a}: {row:?}");
                        for xi in [-1.0, -0.37, 0.0, 0.52, 1.0] {
                            let (coarse, fine) = (shape(p, lower + 0.5 * (xi + 1.0) * length), shape(p, xi));
                            let (value, slope): (f64, f64) =
                                (row.iter().zip(&fine.n).map(|(t, n)| t * n).sum(), row.iter().zip(&fine.d).map(|(t, d)| t * d).sum());
                            assert!((value - coarse.n[a]).abs() < 1e-13, "degree {p}, {d} down, part {o}, function {a} at {xi}");
                            assert!((slope - 0.5 * length * coarse.d[a]).abs() < 1e-12, "degree {p}, {d} down, part {o}, slope of {a} at {xi}");
                        }
                    }
                }
            }
            assert!(transfer.row(0, 0, 0).is_empty() && transfer.row(4, 0, 0).is_empty() && transfer.row(1, 2, 0).is_empty());
        }
    }

    /// Two cells of the grid, the first divided, and one of its children next to the second
    /// divided again.
    fn two_levels(divide_again: bool) -> (Grid, Tree) {
        let grid = Grid { origin: [0.0; 3], spacing: 1.0, cells: [2, 1, 1] };
        let mut tree = Tree::new(&grid, &[Kind::Inside, Kind::Inside]);
        let first = tree.divide(0, [Kind::Inside; 8]).unwrap();
        if divide_again {
            // The child at the upper x, lower y and z corner: it touches the undivided cell.
            tree.divide(first + 1, [Kind::Inside; 8]).unwrap();
        }
        (grid, tree)
    }

    /// The values of an element's modes at a point of the grid, or `None` outside its cell.
    fn values(grid: &Grid, tree: &Tree, space: &Space, shapes: &Shapes, element: usize, x: [f64; 3]) -> Option<Vec<f64>> {
        let node = tree.nodes[space.elements[element] as usize];
        let (level, size) = (usize::from(node.level), grid.step(usize::from(node.level)));
        let lo = grid.fine_lo(level, node.position());
        let xi = [0, 1, 2].map(|a| 2.0 * (x[a] - lo[a]) / size - 1.0);
        if xi.iter().any(|t| t.abs() > 1.0 + 1e-12) {
            return None;
        }
        let (mut n, mut d) = (vec![0.0; shapes.len()], vec![[0.0; 3]; shapes.len()]);
        cell::eval(shapes, xi, 2.0 / size, &mut n, &mut d);
        Some(n)
    }

    /// The values at `x` of the field with the coefficients `u`, one for each element whose
    /// cell holds the point.
    fn fields(grid: &Grid, tree: &Tree, space: &Space, shapes: &Shapes, u: &[f64], x: [f64; 3]) -> Vec<f64> {
        let mut found = Vec::new();
        for element in 0..space.elements.len() {
            let Some(n) = values(grid, tree, space, shapes, element, x) else { continue };
            let spread: Vec<f64> = u.iter().flat_map(|x| [*x, 0.0, 0.0]).collect();
            let local = space.local(element, &spread);
            found.push(n.iter().enumerate().map(|(mode, n)| n * local[3 * mode]).sum());
        }
        found
    }

    /// The mass matrix of the functions and their products with `f`, both exact for a
    /// polynomial `f` of the cells' degree.
    fn mass(grid: &Grid, tree: &Tree, space: &Space, shapes: &Shapes, f: impl Fn([f64; 3]) -> f64) -> (Vec<f64>, Vec<f64>) {
        let (m, count) = (shapes.len(), space.count);
        let rule = gauss_legendre(shapes.p + 1);
        let (mut matrix, mut load) = (vec![0.0; count * count], vec![0.0; count]);
        let (mut n, mut d) = (vec![0.0; m], vec![[0.0; 3]; m]);
        for element in 0..space.elements.len() {
            let node = tree.nodes[space.elements[element] as usize];
            let (level, size) = (usize::from(node.level), grid.step(usize::from(node.level)));
            let lo = grid.fine_lo(level, node.position());
            let functions = space.functions(element);
            // Each function on the modes, by a unit coefficient.
            let columns: Vec<Vec<f64>> = (0..functions.len())
                .map(|function| {
                    let mut unit = vec![0.0; space.count * 3];
                    unit[3 * functions[function]] = 1.0;
                    space.local(element, &unit).as_chunks::<3>().0.iter().map(|c| c[0]).collect()
                })
                .collect();
            for (z, wz) in &rule {
                for (y, wy) in &rule {
                    for (x, wx) in &rule {
                        let xi = [*x, *y, *z];
                        cell::eval(shapes, xi, 2.0 / size, &mut n, &mut d);
                        let weight = wx * wy * wz * 0.125 * size * size * size;
                        let at = [0, 1, 2].map(|a| lo[a] + 0.5 * (xi[a] + 1.0) * size);
                        let here: Vec<f64> = columns.iter().map(|column| column.iter().zip(&n).map(|(c, n)| c * n).sum()).collect();
                        for (i, a) in functions.iter().enumerate() {
                            load[*a] += weight * here[i] * f(at);
                            for (j, b) in functions.iter().enumerate() {
                                matrix[a * count + b] += weight * here[i] * here[j];
                            }
                        }
                    }
                }
            }
        }
        (matrix, load)
    }

    /// Solve A x = b for a symmetric positive definite A; `None` when it is not.
    fn solve(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
        let n = b.len();
        let mut l = a.to_vec();
        if !dense::cholesky(&mut l, n) {
            return None;
        }
        let mut x = b.to_vec();
        for i in 0..n {
            for k in 0..i {
                x[i] -= l[i * n + k] * x[k];
            }
            x[i] /= l[i * n + i];
        }
        for i in (0..n).rev() {
            for k in i + 1..n {
                x[i] -= l[k * n + i] * x[k];
            }
            x[i] /= l[i * n + i];
        }
        Some(x)
    }

    /// One of two cells divided: the functions are the undivided cell's (p + 1)³ and those of
    /// the eight children, a grid of 2 × 2 × 2 cells with (2p + 1)³ functions, less the
    /// (2p + 1)² on the face to the undivided cell, where the finer functions vanish.
    #[test]
    fn the_unknowns_of_a_divided_cell_next_to_an_undivided_one_are_counted_by_hand() {
        for p in 1..=4 {
            let (_, tree) = two_levels(false);
            let space = Space::new(&tree, &local_modes(p, false), p);
            assert_eq!(space.elements.len(), 9);
            assert_eq!(space.count, (p + 1) * (p + 1) * (p + 1) + 2 * p * (2 * p + 1) * (2 * p + 1), "degree {p}");
            // The undivided cell has its own modes and nothing else.
            let undivided = space.element_of[1] as usize;
            assert!(space.map(undivided).is_none());
            assert_eq!(space.functions(undivided).len(), (p + 1) * (p + 1) * (p + 1));
        }
        // Without refinement the cells of the grid are the elements, with their own modes.
        let grid = Grid { origin: [0.0; 3], spacing: 1.0, cells: [2, 1, 1] };
        let tree = Tree::new(&grid, &[Kind::Inside, Kind::Cut]);
        assert_eq!(tree.leaves(), vec![0, 1]);
        assert!(tree.nodes.iter().enumerate().all(|(node, _)| tree.active_parts(node) == ALL_PARTS));
    }

    /// The functions of cells of three levels are linearly independent (their mass matrix is
    /// positive definite), continuous across every face, hanging or not, and hold every
    /// polynomial of the cells' degree.
    #[test]
    fn the_functions_of_several_levels_are_independent_continuous_and_complete() {
        for (p, trunk) in [(1, false), (2, false), (3, false), (2, true), (4, true)] {
            let (grid, tree) = two_levels(true);
            let shapes = Shapes::new(p, local_modes(p, trunk)).unwrap();
            let space = Space::new(&tree, &shapes.modes, p);
            assert_eq!(space.elements.len(), 16);
            // A polynomial of the degree along x, and of degree one along y and z: in the trunk
            // space too.
            let f = |x: [f64; 3]| (0.3 + x[0]).powi(p as i32) * (1.0 - 0.7 * x[1]) + 0.4 * x[2] - 2.0;
            let (matrix, load) = mass(&grid, &tree, &space, &shapes, f);
            let u = solve(&matrix, &load).unwrap_or_else(|| panic!("degree {p}, trunk {trunk}: the functions are dependent"));
            // Points inside cells, on faces between cells of the same and of different levels,
            // and on edges and vertices of the finest cells.
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            let mut next = || {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let random: Vec<f64> = (0..space.count).map(|_| next() - 0.5).collect();
            let mut points = vec![[1.0, 0.3, 0.6], [1.0, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.25, 0.1], [0.75, 0.5, 0.3], [1.0, 0.125, 0.4]];
            points.extend((0..200).map(|_| [2.0 * next(), next(), next()]));
            // On the planes of the finest cells.
            points.extend((0..200).map(|i| {
                let mut x = [2.0 * next(), next(), next()];
                x[i % 3] = (4.0 * x[i % 3]).round() / 4.0;
                x
            }));
            for x in points {
                let exact = fields(&grid, &tree, &space, &shapes, &u, x);
                assert!(!exact.is_empty());
                for value in exact {
                    assert!((value - f(x)).abs() < 1e-9, "degree {p}, trunk {trunk} at {x:?}: {value} vs {}", f(x));
                }
                let any = fields(&grid, &tree, &space, &shapes, &random, x);
                for value in &any {
                    assert!((value - any[0]).abs() < 1e-12, "degree {p}, trunk {trunk}: not continuous at {x:?}: {any:?}");
                }
            }
        }
    }

    /// The projected matrices: Bᵀ K B and Bᵀ f are what B u gives, entry by entry.
    #[test]
    fn a_map_projects_matrices_and_vectors() {
        let (m, mut map) = (3, Map::default());
        map.push([(0, 1.0)].into_iter());
        map.push([(1, 0.5), (2, -2.0)].into_iter());
        assert_eq!(map.functions(), 2);
        let n = 3 * m;
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                k[i * n + j] = 1.0 / (1.0 + i as f64 + j as f64) + if i == j { 2.0 } else { 0.0 };
            }
        }
        let f: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        // B as a dense matrix on the components.
        let mut b = vec![0.0; n * 6];
        for c in 0..3 {
            b[c * 6 + c] = 1.0;
            b[(3 + c) * 6 + 3 + c] = 0.5;
            b[(6 + c) * 6 + 3 + c] = -2.0;
        }
        let projected = map.matrix(m, &k);
        let vector = map.vector(m, &f);
        for i in 0..6 {
            let exact: f64 = (0..n).map(|r| b[r * 6 + i] * f[r]).sum();
            assert!((vector[i] - exact).abs() < 1e-14);
            for j in 0..6 {
                let exact: f64 = (0..n).map(|r| (0..n).map(|c| b[r * 6 + i] * k[r * n + c] * b[c * 6 + j]).sum::<f64>()).sum();
                assert!((projected[i * 6 + j] - exact).abs() < 1e-13, "({i}, {j}): {} vs {exact}", projected[i * 6 + j]);
                assert_eq!(projected[i * 6 + j], projected[j * 6 + i]);
            }
        }
        let u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(map.local(m, &u), vec![1.0, 2.0, 3.0, 2.0, 2.5, 3.0, -8.0, -10.0, -12.0]);
        // Wrong sizes give zeros, not a panic.
        assert!(map.matrix(2, &k).iter().all(|x| *x == 0.0) && map.vector(2, &f).iter().all(|x| *x == 0.0));
    }

    /// The Nitsche parameter of a leaf's own modes holds for the functions that reach it: on
    /// their span, a subspace, the largest eigenvalue of H x = Λ V x is no larger.
    #[test]
    fn the_nitsche_bound_of_a_leaf_holds_for_the_functions_that_reach_it() {
        let lame = Lame { lambda: 1.2, mu: 0.8 };
        for p in [1, 2, 3] {
            let (grid, tree) = two_levels(true);
            let shapes = Shapes::new(p, local_modes(p, false)).unwrap();
            let space = Space::new(&tree, &shapes.modes, p);
            let (m, n) = (shapes.len(), 3 * shapes.len());
            let rule = gauss_legendre(p + 1);
            let mut whole = Vec::new();
            for (z, wz) in &rule {
                for (y, wy) in &rule {
                    for (x, wx) in &rule {
                        whole.push(QuadPoint { xi: [*x, *y, *z], weight: wx * wy * wz, inside: true });
                    }
                }
            }
            let faces = BTreeMap::from([(7u32, Face { fixed: [true; 3], ..Face::default() })]);
            let surface = Surface { normals: &[[0.0, -1.0, 0.0]], tags: &[7], faces: &faces };
            let mut checked = 0;
            for element in 0..space.elements.len() {
                let Some(map) = space.map(element) else { continue };
                let node = tree.nodes[space.elements[element] as usize];
                let (level, size) = (usize::from(node.level), grid.step(usize::from(node.level)));
                let lo = grid.fine_lo(level, node.position());
                // Fixed on its whole lower y face.
                let corner = |x: f64, z: f64| [lo[0] + x * size, lo[1], lo[2] + z * size];
                let pieces = [
                    Piece { corners: [corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0)], triangle: 0 },
                    Piece { corners: [corner(0.0, 0.0), corner(1.0, 1.0), corner(0.0, 1.0)], triangle: 0 },
                ];
                let mut scratch = Scratch::default();
                let k = cell::stiffness(&shapes, size, &cell::integrate(&shapes, &whole).inside, &lame, &mut scratch);
                let b = cell::boundary(&shapes, lo, size, &pieces, &surface, &lame, &gauss_legendre(2 * p + 1), &mut scratch);
                let own = 0.5 * cell::nitsche_parameter(&shapes, &k, &b.traction).unwrap();
                // Rayleigh quotients of the functions' span: random combinations, and the power
                // iteration on the projected pencil, which a shift keeps positive definite on
                // what the functions do not span and on the rigid motions.
                let (reached_k, reached_h) = (map.matrix(m, &k), map.matrix(m, &b.traction));
                let width = 3 * map.functions();
                let scale = (0..width).map(|i| reached_k[i * width + i]).fold(0.0, f64::max);
                let mut shifted = reached_k.clone();
                for i in 0..width {
                    shifted[i * width + i] += scale;
                }
                assert!(dense::cholesky(&mut shifted, width));
                // (K + s I) x = (1/Λ') H x has Λ' ≤ Λ of the unshifted pencil; the quotients
                // below test the unshifted one directly.
                let below = dense::largest_eigenvalue(&reached_h, &shifted, width);
                assert!(below <= own * (1.0 + 1e-9), "degree {p}, element {element}: {below} above {own}");
                let mut state = 0x2545_F491_4F6C_DD1Du64 + element as u64;
                let mut next = || {
                    state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                };
                for _ in 0..50 {
                    let u: Vec<f64> = (0..width).map(|_| next()).collect();
                    let x = map.local(m, &u);
                    let form = |a: &[f64]| -> f64 { (0..n).map(|i| x[i] * (0..n).map(|j| a[i * n + j] * x[j]).sum::<f64>()).sum() };
                    let (traction, energy) = (form(&b.traction), form(&k));
                    assert!(traction <= own * energy * (1.0 + 1e-9), "degree {p}, element {element}: {traction} above {own} × {energy}");
                    // The projected matrices are the forms of B u.
                    let projected: f64 = (0..width).map(|i| u[i] * (0..width).map(|j| reached_k[i * width + j] * u[j]).sum::<f64>()).sum();
                    assert!((projected - energy).abs() < 1e-10 * energy.abs().max(1e-30));
                }
                checked += 1;
            }
            assert!(checked >= 8, "degree {p}: {checked} elements with functions of other cells");
        }
    }
}
