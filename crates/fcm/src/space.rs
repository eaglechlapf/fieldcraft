//! The unknowns of the grid: which shape function of which cell is which global mode.
//!
//! A cell of degree p carries the tensor product of the one-dimensional shape functions
//! ([`crate::legendre`]): (p + 1)³ modes, mode `(a, b, c)` at local index
//! `a + (p + 1)(b + (p + 1) c)`. Along one axis, index 0 belongs to the cell's lower grid plane,
//! index 1 to its upper one and an index of 2 or more to the cell's own span, so a mode with no
//! such index sits on a grid vertex, with one on an edge, with two on a face and with three in
//! the cell's interior. Two cells share the modes of the vertices, edges and faces they share:
//! the grid is Cartesian, so both see such a mode with the same orientation and no sign is
//! needed. A mode is named by its level, where it sits (in half cell steps of that level) and
//! its degree along each axis.
//!
//! Every mode carries the three displacement components, so to the sparse matrix it is a node
//! with three translations.
//!
//! # The trunk space
//! The full tensor product holds every monomial ξⁱ ηʲ ζᵏ with i, j, k ≤ p, far more than the
//! polynomials of total degree p that set the rate of convergence. The *trunk space* (B. Szabó,
//! I. Babuška, *Finite Element Analysis*, Wiley 1991, the hexahedral element) keeps the vertex
//! modes, the edge modes up to degree p, the face modes whose two degrees sum to at most p and
//! the interior modes whose three degrees sum to at most p: it still holds every polynomial of
//! total degree p, with a fraction of the modes (74 instead of 216 at p = 5). Two cells keep the
//! same modes of a shared face, so the sharing above is unchanged.
//!
//! # Refined cells
//! Where cells are divided ([`crate::refine`]) the cells of every level have such modes, and
//! the unknowns are the active ones of all levels. The cells the part is integrated over, the
//! *elements*, are the leaves; the functions that reach an element are its own active modes
//! and the active modes of the cells above it, given on the element's modes by its
//! [`Map`]. An element of the grid that no refinement touches has its own modes and no map.

use std::collections::BTreeMap;

use crate::legendre::MAX_DEGREE;
use crate::refine::{ALL_PARTS, Map, Transfer, Tree, part_of};
use crate::{Error, Result};

/// What a cell is to the part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Outside,
    Inside,
    Cut,
}

/// The slot of a cell that carries no unknowns.
pub(crate) const NONE: u32 = u32::MAX;

/// Cells whose active parts are found per parallel task.
const GRAIN: usize = 1024;

#[derive(Debug, Clone)]
pub(crate) struct Space {
    /// Modes per cell.
    pub modes: usize,
    /// The elements: the leaves of the tree that hold material, by node, in the tree's order.
    pub elements: Vec<u32>,
    /// For each node of the tree, its element, or [`NONE`].
    pub element_of: Vec<u32>,
    /// Where the functions of each element start in `functions`, and the end of the last.
    start: Vec<usize>,
    /// The global mode of every function of every element.
    functions: Vec<usize>,
    /// For each element, its functions on the modes of its cell; `None` where they are those
    /// modes themselves, in their order.
    maps: Vec<Option<Map>>,
    /// Global modes.
    pub count: usize,
    /// Where each mode sits, in half cell steps of the grid from its origin.
    pub xyz: Vec<[f64; 3]>,
}

/// The modes of a cell of degree `p`, each by its index along the three axes: the full tensor
/// product in the order `a + (p + 1)(b + (p + 1) c)`, or its trunk space in the same order.
pub(crate) fn local_modes(p: usize, trunk: bool) -> Vec<[u8; 3]> {
    let q = p.min(MAX_DEGREE) + 1;
    // The linear functions do not count towards the degree of a mode.
    let degree = |a: usize| if a < 2 { 0 } else { a };
    let mut modes = Vec::with_capacity(q * q * q);
    for c in 0..q {
        for b in 0..q {
            for a in 0..q {
                if !trunk || degree(a) + degree(b) + degree(c) <= p {
                    modes.push([a as u8, b as u8, c as u8]);
                }
            }
        }
    }
    modes
}

impl Space {
    /// The unknowns of the cells of `tree`, each cell carrying the modes `local`
    /// ([`local_modes`]) of degree `p`.
    pub fn new(tree: &Tree, local: &[[u8; 3]], p: usize) -> Space {
        let mut space = Space {
            modes: local.len(),
            elements: Vec::new(),
            element_of: vec![NONE; tree.nodes.len()],
            start: vec![0],
            functions: Vec::new(),
            maps: Vec::new(),
            count: 0,
            xyz: Vec::new(),
        };
        let transfer = Transfer::new(p, tree.deepest());
        let parts: Vec<u32> = if tree.deepest() == 0 {
            vec![ALL_PARTS; tree.nodes.len()]
        } else {
            fieldcraft_parallel::map_grained(tree.nodes.len(), GRAIN, |node| tree.active_parts(node))
        };
        let parts = |node: u32| parts.get(node as usize).copied().unwrap_or(0);
        let active = |parts: u32, mode: &[u8; 3]| (parts >> part_of(mode)) & 1 == 1;
        // Where along one axis local index `a` of cell `i` sits, and its degree there (0 for the
        // two linear functions, which differ by where they sit).
        let place = |i: u32, a: usize| -> (u32, u32) {
            match a {
                0 => (2 * i, 0),
                1 => (2 * i + 2, 0),
                _ => (2 * i + 1, a as u32),
            }
        };
        let mut names: BTreeMap<[u32; 7], usize> = BTreeMap::new();
        let mut name = |space: &mut Space, level: u8, ijk: [u32; 3], mode: &[u8; 3]| {
            let [a, b, c] = mode.map(usize::from);
            let ((x, dx), (y, dy), (z, dz)) = (place(ijk[0], a), place(ijk[1], b), place(ijk[2], c));
            let next = names.len();
            let found = *names.entry([u32::from(level), x, y, z, dx, dy, dz]).or_insert(next);
            if found == next {
                let steps = (1u64 << level) as f64;
                space.xyz.push([x as f64 / steps, y as f64 / steps, z as f64 / steps]);
            }
            space.functions.push(found);
        };
        for leaf in tree.leaves() {
            // A cell that holds no material is no element: a function of a cell above it that
            // reaches it is not integrated there.
            let Some(cell) = tree.nodes.get(leaf as usize).filter(|cell| cell.material()) else { continue };
            let own = parts(leaf);
            // The cells above that have active modes.
            let mut above = Vec::new();
            let mut parent = cell.parent;
            while let Some(ancestor) = tree.nodes.get(parent as usize) {
                if parts(parent) != 0 {
                    above.push((parts(parent), *ancestor));
                }
                parent = ancestor.parent;
            }
            if own == ALL_PARTS && above.is_empty() {
                for mode in local {
                    name(&mut space, cell.level, cell.ijk, mode);
                }
                space.maps.push(None);
            } else {
                let mut map = Map::default();
                for (index, mode) in local.iter().enumerate() {
                    if active(own, mode) {
                        name(&mut space, cell.level, cell.ijk, mode);
                        map.push([(index, 1.0)].into_iter());
                    }
                }
                for (reached, ancestor) in above {
                    let down = usize::from(cell.level.saturating_sub(ancestor.level));
                    let part = cell.ijk.map(|i| (i & ((1u32 << down) - 1)) as usize);
                    for mode in local {
                        if !active(reached, mode) {
                            continue;
                        }
                        let rows = [0, 1, 2].map(|axis| transfer.row(down, part[axis], usize::from(mode[axis])));
                        name(&mut space, ancestor.level, ancestor.ijk, mode);
                        map.push(local.iter().enumerate().filter_map(|(index, fine)| {
                            let weight = (0..3).map(|axis| rows[axis].get(usize::from(fine[axis])).copied().unwrap_or(0.0)).product::<f64>();
                            (weight != 0.0).then_some((index, weight))
                        }));
                    }
                }
                space.maps.push(Some(map));
            }
            if let Some(element) = space.element_of.get_mut(leaf as usize) {
                *element = space.elements.len() as u32;
            }
            space.elements.push(leaf);
            space.start.push(space.functions.len());
        }
        space.count = names.len();
        space
    }

    /// The global modes of the functions of `element`.
    pub fn functions(&self, element: usize) -> &[usize] {
        match (self.start.get(element), self.start.get(element + 1)) {
            (Some(&from), Some(&to)) => self.functions.get(from..to).unwrap_or_default(),
            _ => &[],
        }
    }

    /// The functions of `element` on the modes of its cell; `None` where they are the modes.
    pub fn map(&self, element: usize) -> Option<&Map> {
        self.maps.get(element).and_then(Option::as_ref)
    }

    /// The coefficients of the modes of the cell of `element`, three per mode, in the solution
    /// `u` (three per global mode).
    pub fn local(&self, element: usize, u: &[f64]) -> Vec<f64> {
        let own: Vec<f64> =
            self.functions(element).iter().flat_map(|mode| (0..3).map(move |i| u.get(3 * mode + i).copied().unwrap_or(0.0))).collect();
        match self.map(element) {
            Some(map) => map.local(self.modes, &own),
            None => own,
        }
    }

    /// Renumber the modes so that `order[k]` becomes mode `k`.
    pub fn renumber(&mut self, order: &[usize]) -> Result<()> {
        let broken = || Error::Internal("the mode order is not a permutation".into());
        if order.len() != self.count {
            return Err(broken());
        }
        let mut position = vec![usize::MAX; self.count];
        let mut xyz = Vec::with_capacity(self.count);
        for (k, &mode) in order.iter().enumerate() {
            let slot = position.get_mut(mode).ok_or_else(broken)?;
            if *slot != usize::MAX {
                return Err(broken());
            }
            *slot = k;
            xyz.push(self.xyz.get(mode).copied().ok_or_else(broken)?);
        }
        for mode in &mut self.functions {
            *mode = position.get(*mode).copied().ok_or_else(broken)?;
        }
        self.xyz = xyz;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn neighbours_share_the_modes_of_their_common_face() {
        let grid = Grid { origin: [0.0; 3], spacing: 1.0, cells: [2, 1, 1] };
        for p in 1..=4 {
            let q = p + 1;
            let mut space = Space::new(&Tree::new(&grid, &[Kind::Inside, Kind::Cut]), &local_modes(p, false), p);
            // Two cells of q³ modes each, sharing the q² modes of one face.
            assert_eq!(space.count, 2 * q * q * q - q * q, "degree {p}");
            assert_eq!(space.xyz.len(), space.count);
            assert!(space.map(0).is_none() && space.map(1).is_none());
            let (left, right) = (space.functions(0).to_vec(), space.functions(1).to_vec());
            for c in 0..q {
                for b in 0..q {
                    // The upper x plane of the left cell is the lower x plane of the right one.
                    assert_eq!(left[1 + q * (b + q * c)], right[q * (b + q * c)]);
                }
            }
            // Reversing the numbering keeps the sharing.
            let order: Vec<usize> = (0..space.count).rev().collect();
            space.renumber(&order).unwrap();
            assert_eq!(space.functions(0)[1], space.functions(1)[0]);
            assert_eq!(space.functions(0)[0], space.count - 1 - left[0]);
            assert!(space.renumber(&[0, 0]).is_err());
        }
        let tree = Tree::new(&grid, &[Kind::Outside, Kind::Inside]);
        let space = Space::new(&tree, &local_modes(2, false), 2);
        assert_eq!((space.elements.clone(), tree.root.clone(), space.count), (vec![0], vec![NONE, 0], 27));
    }

    /// The trunk space has the modes Szabó and Babuška count: 8 vertex modes, p − 1 an edge,
    /// (p − 2)(p − 3)/2 a face and (p − 3)(p − 4)(p − 5)/6 inside, and two cells share the ones
    /// of their common face.
    #[test]
    fn the_trunk_space_keeps_the_modes_of_total_degree() {
        for p in 1..=MAX_DEGREE {
            let trunk = local_modes(p, true);
            let (face, inside) = (if p >= 4 { (p - 2) * (p - 3) / 2 } else { 0 }, if p >= 6 { (p - 3) * (p - 4) * (p - 5) / 6 } else { 0 });
            assert_eq!(trunk.len(), 8 + 12 * (p - 1) + 6 * face + inside, "degree {p}");
            assert_eq!(local_modes(p, false).len(), (p + 1) * (p + 1) * (p + 1));
            // Every vertex mode, and in the order of the tensor product.
            assert!((0..8u8).all(|corner| trunk.contains(&[corner & 1, (corner >> 1) & 1, (corner >> 2) & 1])));
            assert!(trunk.windows(2).all(|w| (w[0][2], w[0][1], w[0][0]) < (w[1][2], w[1][1], w[1][0])));
            let grid = Grid { origin: [0.0; 3], spacing: 1.0, cells: [2, 1, 1] };
            let space = Space::new(&Tree::new(&grid, &[Kind::Inside, Kind::Cut]), &trunk, p);
            let on_face = trunk.iter().filter(|mode| mode[0] == 1).count();
            assert_eq!(space.count, 2 * trunk.len() - on_face, "degree {p}");
        }
    }
}
