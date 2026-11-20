//! The unknowns of the grid: which shape function of which cell is which global mode.
//!
//! A cell of degree p carries the tensor product of the one-dimensional shape functions
//! ([`crate::legendre`]): (p + 1)³ modes, mode `(a, b, c)` at local index
//! `a + (p + 1)(b + (p + 1) c)`. Along one axis, index 0 belongs to the cell's lower grid plane,
//! index 1 to its upper one and an index of 2 or more to the cell's own span, so a mode with no
//! such index sits on a grid vertex, with one on an edge, with two on a face and with three in
//! the cell's interior. Two cells share the modes of the vertices, edges and faces they share:
//! the grid is Cartesian, so both see such a mode with the same orientation and no sign is
//! needed. A mode is named by where it sits (in half cell steps) and its degree along each axis.
//!
//! Every mode carries the three displacement components, so to the sparse matrix it is a node
//! with three translations.

use std::collections::BTreeMap;

use crate::grid::Grid;
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

pub(crate) struct Space {
    /// Modes per cell.
    pub modes: usize,
    /// The cells that carry unknowns, by grid index, in increasing order.
    pub active: Vec<usize>,
    /// For each grid cell, its position in `active`, or [`NONE`].
    pub slot: Vec<u32>,
    /// For each active cell, the global mode of each of its local modes.
    pub cell_modes: Vec<usize>,
    /// Global modes.
    pub count: usize,
    /// Where each mode sits, in half cell steps from the grid's origin.
    pub xyz: Vec<[f64; 3]>,
}

/// The number of modes of a cell of degree `p`.
pub(crate) fn modes(p: usize) -> usize {
    (p + 1) * (p + 1) * (p + 1)
}

impl Space {
    pub fn new(grid: &Grid, kinds: &[Kind], p: usize) -> Space {
        let q = p + 1;
        let mut space =
            Space { modes: modes(p), active: Vec::new(), slot: vec![NONE; kinds.len()], cell_modes: Vec::new(), count: 0, xyz: Vec::new() };
        // Where along one axis local index `a` of cell `i` sits, and its degree there (0 for the
        // two linear functions, which differ by where they sit).
        let place = |i: usize, a: usize| -> (u32, u32) {
            match a {
                0 => (2 * i as u32, 0),
                1 => (2 * i as u32 + 2, 0),
                _ => (2 * i as u32 + 1, a as u32),
            }
        };
        let mut names: BTreeMap<[u32; 6], usize> = BTreeMap::new();
        for (cell, kind) in kinds.iter().enumerate() {
            if *kind == Kind::Outside {
                continue;
            }
            if let Some(slot) = space.slot.get_mut(cell) {
                *slot = space.active.len() as u32;
            }
            space.active.push(cell);
            let [i, j, k] = grid.coords(cell);
            for c in 0..q {
                for b in 0..q {
                    for a in 0..q {
                        let ((x, dx), (y, dy), (z, dz)) = (place(i, a), place(j, b), place(k, c));
                        let next = names.len();
                        let mode = *names.entry([x, y, z, dx, dy, dz]).or_insert(next);
                        if mode == next {
                            space.xyz.push([x as f64, y as f64, z as f64]);
                        }
                        space.cell_modes.push(mode);
                    }
                }
            }
        }
        space.count = names.len();
        space
    }

    /// The global modes of active cell `slot`.
    pub fn cell(&self, slot: usize) -> &[usize] {
        self.cell_modes.get(slot * self.modes..(slot + 1) * self.modes).unwrap_or_default()
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
        for mode in &mut self.cell_modes {
            *mode = position.get(*mode).copied().ok_or_else(broken)?;
        }
        self.xyz = xyz;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_share_the_modes_of_their_common_face() {
        let grid = Grid { origin: [0.0; 3], spacing: 1.0, cells: [2, 1, 1] };
        for p in 1..=4 {
            let q = p + 1;
            let mut space = Space::new(&grid, &[Kind::Inside, Kind::Cut], p);
            // Two cells of q³ modes each, sharing the q² modes of one face.
            assert_eq!(space.count, 2 * q * q * q - q * q, "degree {p}");
            assert_eq!(space.xyz.len(), space.count);
            let (left, right) = (space.cell(0).to_vec(), space.cell(1).to_vec());
            for c in 0..q {
                for b in 0..q {
                    // The upper x plane of the left cell is the lower x plane of the right one.
                    assert_eq!(left[1 + q * (b + q * c)], right[q * (b + q * c)]);
                }
            }
            // Reversing the numbering keeps the sharing.
            let order: Vec<usize> = (0..space.count).rev().collect();
            space.renumber(&order).unwrap();
            assert_eq!(space.cell(0)[1], space.cell(1)[0]);
            assert_eq!(space.cell(0)[0], space.count - 1 - left[0]);
            assert!(space.renumber(&[0, 0]).is_err());
        }
        let space = Space::new(&grid, &[Kind::Outside, Kind::Inside], 2);
        assert_eq!((space.active.clone(), space.slot.clone(), space.count), (vec![1], vec![NONE, 0], 27));
    }
}
