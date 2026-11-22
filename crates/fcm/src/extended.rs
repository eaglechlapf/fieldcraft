//! Shape functions taken, in the blocks of the iteration, together with what follows them:
//! those of a cell beside a divided one, and those of a body another is connected to.
//!
//! # Divided cells
//! Where a cell is divided and the one beside it is not, the functions the two share are those
//! of the larger cell ([`crate::refine`]): they reach over every cell the divided one was
//! divided into, with all the finer functions of those cells inside their reach. Such a
//! function is nearly a sum of the finer ones, but next to the larger cell, and a block that
//! is to solve it exactly has to hold it with all of them ([`crate::iterative`]): thousands of
//! unknowns where cells are divided twice.
//!
//! So the blocks see such a function F not as it is but with what follows it,
//!
//! F̃ = F − Σ xᵢ φᵢ, K_II x = K_IF,
//!
//! the sum over the functions φᵢ within its reach: the x that leave F̃ with the least energy,
//! its discrete harmonic extension into them (A. Toselli, O. Widlund, *Domain Decomposition
//! Methods: Algorithms and Theory*, Springer 2005, ch. 4). F̃ is orthogonal in energy to every
//! function within its reach, so no block needs to hold it with them: it is in a block with
//! the other far-reaching functions around its vertex, a few tens of unknowns, and the
//! functions within its reach keep their own small blocks.
//!
//! This changes the blocks only. With T the matrix whose columns are the F̃ (and the unit
//! vectors of all other unknowns), the local level of the approximate inverse is
//! T (Σ w² Rᵀ (R Tᵀ K T Rᵀ)⁻¹ R) Tᵀ, in place of the same sum without T: symmetric and positive
//! whatever the x are, so how well they are chosen decides the rounds and never the result.
//! The system that is iterated and the coarse level stay as they are: the coarse level has
//! the linear functions themselves, which hold a rigid motion of the part exactly; with the F̃
//! in their place it would not, and the rounds are those of a grid with a few very soft modes
//! (measured: 215 rounds for 40).
//!
//! # Connected bodies
//! Two bodies that a connection holds together have each its own functions on its own cells,
//! coupled stiffly where they touch: there a function of one is, to the connection, nearly a
//! sum of functions of the other. It is the same trouble and has the same answer. Across a
//! connection the cells of one body lead (the body with fewer cells there, so the larger
//! cells) and those of the other follow; a function of a leading cell is taken with the
//! functions of the other body whose cells all follow its cells. Measured on three blocks,
//! 188,000 unknowns at degree 3: bonded 85 rounds without and 53 with, sliding 78 and 51; with
//! cells three times as large on one side 165 and 85.
//!
//! # What it costs
//! For every set of functions that follow others, one sparse factorization of their
//! stiffness, and a column x for every unknown of those they follow; each round, one product
//! with the columns on the way into the blocks and one on the way out. Functions that the
//! same ones follow share the factorization.
//!
//! Where cells are divided three times, the linear function of a vertex of the largest cells
//! reaches hundreds of cells and has ten thousand functions within its reach, and their
//! factorizations take longer than the rounds they save. A linear function on a divided cell
//! is the sum of the linear functions of the smaller cells but at the faces towards larger
//! ones, so with more than [`MOST_OF_ALL`] functions within its reach it is taken with the
//! linear ones among them alone, a thirtieth of them at degree 3. Functions that more than
//! [`MOST_INSIDE`] would follow are left as they are, and go into blocks of single cells.
//!
//! # Determinism
//! What follows a function is worked out from the grid and the matrix alone, each set in
//! parallel, and the columns are applied in the order of their functions. The result is the
//! same on any number of threads.

use fieldcraft_parallel as parallel;
use fieldcraft_solver::cholesky::{Factor, UpperColumns};
use fieldcraft_solver::sparse::Csr;

use crate::blocked::Blocked;
use crate::packed;

/// The functions within the reach of linear ones that are all taken to follow them; with
/// more, the linear ones among them alone. Measured at degree 3 on the oracle's bracket with
/// its cells of 8 mm divided three times (447,000 unknowns; the functions of 57 vertices have
/// more within their reach than this, up to 11,300): 39 rounds and 16 s of factorizations
/// with all of them, 66 rounds and 5 s with this many, 70 and 5 s with half; and divided twice
/// 38 rounds and 1.4 s, 63 and 0.9 s, 63 and 0.8 s. Divided once, no reach holds as many.
const MOST_OF_ALL: usize = 2000;

/// The functions that follow others, at most: the factorization of more takes as long as the
/// iteration it is to save.
const MOST_INSIDE: usize = 8192;

/// Sets of cells a task takes.
const GRAIN: usize = 4;

/// The nodes that the same nodes follow, and those.
struct Group {
    /// The nodes that are taken with what follows them, rising.
    members: Vec<usize>,
    /// What follows them, rising.
    inside: Vec<usize>,
    /// The nodes outside `inside` that K F̃ may be nonzero at, rising (the members among them).
    rows: Vec<usize>,
    /// x: for the inside node at p and the member at j, at `p * members + j`, the coefficient
    /// of the node's unknown d for the member's unknown c at `3 d + c`.
    x: Vec<[f64; 9]>,
    /// K F̃ on `rows`, likewise: at `q * members + j`, row unknown r at `3 r + c`.
    w: Vec<[f64; 9]>,
}

impl Group {
    fn x(&self, p: usize, j: usize) -> &[f64; 9] {
        self.x.get(p * self.members.len() + j).unwrap_or(&[0.0; 9])
    }

    /// K F̃ of member `j` at `node`: zero where it is none of the rows.
    fn w(&self, node: usize, j: usize) -> [f64; 9] {
        self.rows.binary_search(&node).ok().and_then(|q| self.w.get(q * self.members.len() + j)).copied().unwrap_or([0.0; 9])
    }
}

/// The nodes that the blocks take with what follows them.
pub(crate) struct Extended {
    /// For every node: its group and its place among the group's members.
    place: Vec<Option<(u32, u32)>>,
    groups: Vec<Group>,
}

fn within(small: &[u32], large: &[u32]) -> bool {
    small.iter().all(|cell| large.binary_search(cell).is_ok())
}

/// What follows the nodes `members`, which reach the cells `set` (none when they do not reach
/// far) and which the cells `led` of another body follow, rising: the nodes within the reach
/// and those of the cells that follow; of linear functions with more than [`MOST_OF_ALL`]
/// within their reach, the linear ones alone. None when there are more than [`MOST_INSIDE`].
fn inside(lists: &[Vec<usize>], reach: &[Vec<u32>], linear: &[bool], set: &[u32], led: &[u32], members: &[usize]) -> Vec<usize> {
    let reach_of = |node: usize| reach.get(node).map(Vec::as_slice).unwrap_or_default();
    let is_linear = |node: &usize| linear.get(*node) == Some(&true);
    let mut inside: Vec<usize> = set.iter().chain(led).filter_map(|cell| lists.get(*cell as usize)).flatten().copied().collect();
    inside.sort_unstable();
    inside.dedup();
    inside.retain(|node| {
        let cells = reach_of(*node);
        !cells.is_empty() && ((cells != set && within(cells, set)) || within(cells, led))
    });
    if inside.len() > MOST_OF_ALL && led.is_empty() && members.iter().all(is_linear) {
        inside.retain(is_linear);
    }
    if inside.len() > MOST_INSIDE {
        inside.clear();
    }
    inside
}

/// The group of the nodes `members`, which the nodes `inside` follow (both rising): None when
/// the stiffness of those is not positive.
fn group(matrix: &Blocked, reach: &[Vec<u32>], inside: &[usize], members: Vec<usize>) -> Option<Group> {
    let reach_of = |node: usize| reach.get(node).map(Vec::as_slice).unwrap_or_default();
    // The stiffness of what lies within, by rows, and its factor.
    let (mut ptr, mut idx, mut val) = (vec![0usize], Vec::new(), Vec::new());
    for node in inside {
        let (others, blocks) = matrix.row(*node);
        for r in 0..3 {
            for (other, block) in others.iter().zip(blocks) {
                if let Ok(at) = inside.binary_search(&(*other as usize)) {
                    idx.extend(3 * at..3 * at + 3);
                    val.extend_from_slice(block.get(3 * r..3 * r + 3).unwrap_or_default());
                }
            }
            ptr.push(idx.len());
        }
    }
    let rows_of_inside = Csr { n: 3 * inside.len(), ptr, idx, val };
    let inside = inside.to_vec();
    let factor = Factor::new(&UpperColumns::from_symmetric(&rows_of_inside), usize::MAX / 2).ok()?;
    drop(rows_of_inside);
    let m = members.len();
    let mut x = vec![[0.0f64; 9]; inside.len() * m];
    let mut column = vec![0.0; 3 * inside.len()];
    for (j, member) in members.iter().enumerate() {
        let (others, blocks) = matrix.row(*member);
        for c in 0..3 {
            column.fill(0.0);
            // K_IF by the member's own rows: the matrix is symmetric.
            for (other, block) in others.iter().zip(blocks) {
                if let (Ok(p), Some(entries)) = (inside.binary_search(&(*other as usize)), block.get(3 * c..3 * c + 3))
                    && let Some(slots) = column.get_mut(3 * p..3 * p + 3)
                {
                    slots.copy_from_slice(entries);
                }
            }
            factor.solve(&mut column).ok()?;
            for (p, values) in column.as_chunks::<3>().0.iter().enumerate() {
                if let Some(slot) = x.get_mut(p * m + j) {
                    for (d, value) in values.iter().enumerate() {
                        slot[3 * d + c] = *value;
                    }
                }
            }
        }
    }
    if x.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    // Where K F̃ may be nonzero: the nodes coupled with a member or with a node within, that
    // are in the system and not within.
    let mut rows: Vec<usize> = members.iter().chain(&inside).flat_map(|node| matrix.row(*node).0.iter().map(|other| *other as usize)).collect();
    rows.sort_unstable();
    rows.dedup();
    rows.retain(|node| !reach_of(*node).is_empty() && inside.binary_search(node).is_err());
    let mut w = vec![[0.0f64; 9]; rows.len() * m];
    for (q, node) in rows.iter().enumerate() {
        let Some(out) = w.get_mut(q * m..(q + 1) * m) else { continue };
        let (others, blocks) = matrix.row(*node);
        for (other, k) in others.iter().zip(blocks) {
            let other = *other as usize;
            if let Ok(j) = members.binary_search(&other) {
                if let Some(slot) = out.get_mut(j) {
                    for (sum, value) in slot.iter_mut().zip(k) {
                        *sum += value;
                    }
                }
            } else if let Ok(p) = inside.binary_search(&other) {
                // Less K times x, three by three.
                for (slot, xs) in out.iter_mut().zip(x.get(p * m..(p + 1) * m).unwrap_or_default()) {
                    for r in 0..3 {
                        for c in 0..3 {
                            slot[3 * r + c] -= k[3 * r] * xs[c] + k[3 * r + 1] * xs[3 + c] + k[3 * r + 2] * xs[6 + c];
                        }
                    }
                }
            }
        }
    }
    Some(Group { members, inside, rows, x, w })
}
impl Extended {
    /// None of the `nodes` nodes far-reaching.
    #[cfg(test)]
    pub(crate) fn none(nodes: usize) -> Extended {
        Extended { place: vec![None; nodes], groups: Vec::new() }
    }

    /// The nodes of `matrix` that reach more than `around` cells, and those that cells of
    /// another body follow, each with what follows it: `lists` has the nodes of every cell and
    /// `reach` the cells of every node that is in the system, rising (none for a node that is
    /// not); `linear` says which nodes carry a linear shape function; `pairs` has, for every
    /// two cells a connection couples, the one that leads and the one that follows.
    pub(crate) fn new(matrix: &Blocked, lists: &[Vec<usize>], reach: &[Vec<u32>], linear: &[bool], around: usize, pairs: &[(u32, u32)]) -> Extended {
        // The cells that follow each cell, and with them each node, rising.
        let mut pairs = pairs.to_vec();
        pairs.sort_unstable();
        pairs.dedup();
        let followers = |cell: u32| {
            let from = pairs.partition_point(|(leader, _)| *leader < cell);
            pairs.get(from..).unwrap_or_default().iter().take_while(move |(leader, _)| *leader == cell).map(|(_, follower)| *follower)
        };
        let led: Vec<Vec<u32>> = if pairs.is_empty() {
            Vec::new()
        } else {
            parallel::map_grained(reach.len(), 1024, |node| {
                let mut cells: Vec<u32> = reach.get(node).into_iter().flatten().flat_map(|cell| followers(*cell)).collect();
                cells.sort_unstable();
                cells.dedup();
                cells
            })
        };
        let led_of = |node: usize| led.get(node).map(Vec::as_slice).unwrap_or_default();
        // The far-reaching and the leading nodes by what they reach and lead.
        let far_reach = |node: usize| reach.get(node).map(Vec::as_slice).filter(|reach| reach.len() > around).unwrap_or_default();
        let mut far: Vec<(&[u32], &[u32], usize)> = (0..reach.len())
            .filter(|node| !far_reach(*node).is_empty() || !led_of(*node).is_empty())
            .map(|node| (far_reach(node), led_of(node), node))
            .collect();
        far.sort_unstable();
        let mut sets: Vec<(&[u32], &[u32], Vec<usize>)> = Vec::new();
        for (set, led, node) in far {
            match sets.last_mut() {
                Some((last, last_led, members)) if *last == set && *last_led == led => members.push(node),
                _ => sets.push((set, led, vec![node])),
            }
        }
        // What follows each: where it is the same nodes for several, they are one group with
        // one factorization.
        let insides: Vec<Vec<usize>> = parallel::map_grained(sets.len(), GRAIN, |at| {
            let Some((set, led, members)) = sets.get(at) else { return Vec::new() };
            inside(lists, reach, linear, set, led, members)
        });
        let inside_of = |at: usize| insides.get(at).map(Vec::as_slice).unwrap_or_default();
        let mut order: Vec<usize> = (0..sets.len()).filter(|at| !inside_of(*at).is_empty()).collect();
        order.sort_by(|a, b| inside_of(*a).cmp(inside_of(*b)).then(a.cmp(b)));
        let mut merged: Vec<(usize, Vec<usize>)> = Vec::new();
        for at in order {
            let members = sets.get(at).map(|(_, _, members)| members.as_slice()).unwrap_or_default();
            match merged.last_mut() {
                Some((last, all)) if inside_of(*last) == inside_of(at) => all.extend_from_slice(members),
                _ => merged.push((at, members.to_vec())),
            }
        }
        let groups: Vec<Group> = parallel::map_grained(merged.len(), 1, |at| {
            let (first, members) = merged.get(at)?;
            let mut members = members.clone();
            members.sort_unstable();
            group(matrix, reach, inside_of(*first), members)
        })
        .into_iter()
        .flatten()
        .collect();
        let mut place = vec![None; reach.len()];
        for (g, group) in groups.iter().enumerate() {
            for (j, member) in group.members.iter().enumerate() {
                if let (Some(slot), Ok(g), Ok(j)) = (place.get_mut(*member), u32::try_from(g), u32::try_from(j)) {
                    *slot = Some((g, j));
                }
            }
        }
        Extended { place, groups }
    }

    /// Whether any node is taken with what follows it.
    pub(crate) fn any(&self) -> bool {
        !self.groups.is_empty()
    }

    /// The group and the place in it of a node that is taken with what follows it.
    fn of(&self, node: usize) -> Option<(&Group, usize)> {
        let (g, j) = self.place.get(node).copied().flatten()?;
        Some((self.groups.get(g as usize)?, j as usize))
    }

    /// Whether `node` is taken with what follows it.
    pub(crate) fn has(&self, node: usize) -> bool {
        self.place.get(node).copied().flatten().is_some()
    }

    /// The blocks of the far-reaching nodes (those that reach more than `around` cells): for
    /// every reach among them that is not part of a larger one, the far-reaching nodes
    /// within it, rising.
    pub(crate) fn blocks(&self, lists: &[Vec<usize>], reach: &[Vec<u32>], around: usize) -> Vec<Vec<usize>> {
        let reach_of = |node: usize| reach.get(node).map(Vec::as_slice).unwrap_or_default();
        let mut sets: Vec<&[u32]> =
            self.groups.iter().flat_map(|group| &group.members).map(|node| reach_of(*node)).filter(|reach| reach.len() > around).collect();
        sets.sort_unstable();
        sets.dedup();
        parallel::map_grained(sets.len(), GRAIN, |at| {
            let set = sets.get(at).copied().unwrap_or_default();
            let mut nodes: Vec<usize> = set
                .iter()
                .filter_map(|cell| lists.get(*cell as usize))
                .flatten()
                .copied()
                .filter(|node| self.has(*node) && reach_of(*node).len() > around)
                .collect();
            nodes.sort_unstable();
            nodes.dedup();
            // A larger reach is that of a far-reaching node of any of the set's cells.
            if nodes.iter().any(|node| reach_of(*node).len() > set.len() && within(set, reach_of(*node))) {
                return Vec::new();
            }
            nodes.retain(|node| within(reach_of(*node), set));
            nodes
        })
        .into_iter()
        .filter(|nodes| !nodes.is_empty())
        .collect()
    }

    /// The lower triangle of Tᵀ K T on the nodes `nodes` (rising, each once): [`packed::lower`]
    /// with the far-reaching ones taken with what follows them.
    pub(crate) fn lower(&self, matrix: &Blocked, nodes: &[usize]) -> Option<Vec<f64>> {
        let mut triangle = packed::lower(matrix, nodes)?;
        let mut set = |row: usize, column: usize, value: f64| {
            let (row, column) = if column > row { (column, row) } else { (row, column) };
            if let Some(slot) = triangle.get_mut(row * (row + 1) / 2 + column) {
                *slot = value;
            }
        };
        for (pf, far) in nodes.iter().enumerate() {
            let Some((group, j)) = self.of(*far) else { continue };
            for (pa, node) in nodes.iter().enumerate() {
                let mut value = group.w(*node, j);
                match self.of(*node) {
                    // Two far-reaching nodes: once, from the later one's x against the
                    // earlier one's K F̃ where the later one reaches and the earlier does not.
                    Some(_) if pa < pf => continue,
                    Some((other, ja)) => {
                        let mut q = 0;
                        for (p, inside) in other.inside.iter().enumerate() {
                            while group.rows.get(q).is_some_and(|row| row < inside) {
                                q += 1;
                            }
                            if group.rows.get(q) != Some(inside) {
                                continue;
                            }
                            let (x, w) = (other.x(p, ja), group.w.get(q * group.members.len() + j).unwrap_or(&[0.0; 9]));
                            for r in 0..3 {
                                for c in 0..3 {
                                    value[3 * r + c] -= x[r] * w[c] + x[3 + r] * w[3 + c] + x[6 + r] * w[6 + c];
                                }
                            }
                        }
                        for r in 0..3 {
                            for c in 0..3 {
                                // Of a node with itself, the lower triangle alone.
                                if pa > pf || c <= r {
                                    set(3 * pa + r, 3 * pf + c, value[3 * r + c]);
                                }
                            }
                        }
                    }
                    None => {
                        for r in 0..3 {
                            for c in 0..3 {
                                set(3 * pa + r, 3 * pf + c, value[3 * r + c]);
                            }
                        }
                    }
                }
            }
        }
        Some(triangle)
    }

    /// `out` = Tᵀ `r`: the far-reaching unknowns' entries less x against those within reach.
    pub(crate) fn inward(&self, r: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.extend_from_slice(r);
        let changed: Vec<Vec<[f64; 3]>> = parallel::map_grained(self.groups.len(), GRAIN, |g| {
            let Some(group) = self.groups.get(g) else { return Vec::new() };
            let m = group.members.len();
            let mut sums = vec![[0.0f64; 3]; m];
            for (p, node) in group.inside.iter().enumerate() {
                let Some(&[r0, r1, r2]) = r.get(3 * node..3 * node + 3).and_then(|r| r.first_chunk::<3>()) else { continue };
                for (sum, x) in sums.iter_mut().zip(group.x.get(p * m..(p + 1) * m).unwrap_or_default()) {
                    for c in 0..3 {
                        sum[c] += x[c] * r0 + x[3 + c] * r1 + x[6 + c] * r2;
                    }
                }
            }
            sums
        });
        for (group, sums) in self.groups.iter().zip(&changed) {
            for (member, sum) in group.members.iter().zip(sums) {
                if let Some(entries) = out.get_mut(3 * member..3 * member + 3) {
                    for (entry, sum) in entries.iter_mut().zip(sum) {
                        *entry -= sum;
                    }
                }
            }
        }
    }

    /// `z` = T `z`: the unknowns within every reach less x times the far-reaching ones'.
    pub(crate) fn outward(&self, z: &mut [f64]) {
        // The far-reaching unknowns as they come: one may lie within another's reach.
        let values: Vec<Vec<[f64; 3]>> = self
            .groups
            .iter()
            .map(|group| group.members.iter().map(|member| [0, 1, 2].map(|c| z.get(3 * member + c).copied().unwrap_or(0.0))).collect())
            .collect();
        for (group, values) in self.groups.iter().zip(&values) {
            let m = group.members.len();
            for (p, node) in group.inside.iter().enumerate() {
                let Some(entries) = z.get_mut(3 * node..3 * node + 3) else { continue };
                for (x, v) in group.x.get(p * m..(p + 1) * m).unwrap_or_default().iter().zip(values) {
                    for (d, entry) in entries.iter_mut().enumerate() {
                        *entry -= x[3 * d] * v[0] + x[3 * d + 1] * v[1] + x[3 * d + 2] * v[2];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use fieldcraft_solver::sparse::NodeGraph;

    use super::*;

    /// A chain of cells of three nodes, each sharing its ends with the cells beside it, and one
    /// node more (the last) that is in every cell from `from` to `to`: a function of a large
    /// cell over small ones. Every cell with a matrix of its own, symmetric and positive.
    fn system(cells: usize, from: usize, to: usize) -> (Blocked, Vec<Vec<usize>>, Vec<Vec<u32>>) {
        let far = 2 * cells + 1;
        let lists: Vec<Vec<usize>> = (0..cells)
            .map(|cell| {
                let mut list = vec![2 * cell, 2 * cell + 1, 2 * cell + 2];
                if (from..to).contains(&cell) {
                    list.push(far);
                }
                list
            })
            .collect();
        let graph = NodeGraph::new(far + 1, lists.iter());
        let mut matrix = Blocked::zero(&graph.adjacency).expect("small");
        for (cell, list) in lists.iter().enumerate() {
            let width = 3 * list.len();
            let mut data = vec![0.0; width * width];
            for i in 0..width {
                for j in 0..=i {
                    let value = ((cell * 131 + i * 17 + j * 5) as f64 * 0.37).sin() + if i == j { 12.0 } else { 0.0 };
                    data[i * width + j] = value;
                    data[j * width + i] = value;
                }
            }
            matrix.add(list, &data).unwrap();
        }
        let mut reach = vec![Vec::new(); far + 1];
        for (cell, list) in lists.iter().enumerate() {
            for node in list {
                reach[*node].push(cell as u32);
            }
        }
        (matrix, lists, reach)
    }

    /// Tᵀ K T on `nodes` from the matrix by rows and the columns of T, entry by entry.
    fn dense(matrix: &Blocked, extended: &Extended, nodes: &[usize]) -> Vec<Vec<f64>> {
        let rows = matrix.rows();
        let n = rows.n;
        let column = |unknown: usize| {
            let mut t = vec![0.0; n];
            t[unknown] = 1.0;
            extended.outward(&mut t);
            t
        };
        let unknowns: Vec<usize> = nodes.iter().flat_map(|node| 3 * node..3 * node + 3).collect();
        let columns: Vec<Vec<f64>> = unknowns.iter().map(|unknown| column(*unknown)).collect();
        columns
            .iter()
            .map(|a| {
                let mut ka = vec![0.0; n];
                rows.mul(a, &mut ka);
                columns.iter().map(|b| ka.iter().zip(b).map(|(x, y)| x * y).sum()).collect()
            })
            .collect()
    }

    #[test]
    fn a_far_reaching_node_is_orthogonal_to_what_lies_within_its_reach() {
        // Twelve cells; the far node reaches cells 2 to 8, more than the two a chain's node
        // has around it.
        let (matrix, lists, reach) = system(12, 2, 9);
        let far = 25;
        let extended = Extended::new(&matrix, &lists, &reach, &[], 2, &[]);
        assert!(extended.any() && extended.has(far) && !extended.has(6));
        let (group, _) = extended.of(far).unwrap();
        // Within its reach: the nodes of its cells but the two at the ends, which reach on.
        assert_eq!(group.inside, (5..=17).collect::<Vec<_>>());
        assert_eq!(group.rows, [4, 18, 25]);
        // One block, of the far node alone.
        assert_eq!(extended.blocks(&lists, &reach, 2), [vec![far]]);

        // T' K T: zero between the far node and the nodes within its reach, the matrix's own
        // between other nodes, and what `lower` gives everywhere.
        let nodes = [3, 4, 5, 11, 18, 19, far];
        let full = dense(&matrix, &extended, &nodes);
        let lower = extended.lower(&matrix, &nodes).unwrap();
        let scale = full.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        for row in 0..3 * nodes.len() {
            for column in 0..=row {
                let (a, b) = (lower[row * (row + 1) / 2 + column], full[row][column]);
                assert!((a - b).abs() < 1e-12 * scale, "({row}, {column}): {a} against {b}");
                let within_reach = |unknown: usize| (5..=17).contains(&nodes[unknown / 3]);
                if (row / 3 == 6 && within_reach(column)) || (column / 3 == 6 && within_reach(row)) {
                    assert!(b.abs() < 1e-12 * scale, "({row}, {column}): {b} is not zero");
                }
            }
        }
        let plain = packed::lower(&matrix, &nodes[..5]).unwrap();
        assert_eq!(extended.lower(&matrix, &nodes[..5]).unwrap(), plain, "no far node: the matrix's own");

        // Inward is the transpose of outward.
        let n = 3 * 26;
        let (a, b): (Vec<f64>, Vec<f64>) = ((0..n).map(|i| (i as f64 * 0.7).sin()).collect(), (0..n).map(|i| (i as f64 * 0.3).cos()).collect());
        let mut ta = a.clone();
        extended.outward(&mut ta);
        let mut ttb = Vec::new();
        extended.inward(&b, &mut ttb);
        let (left, right): (f64, f64) = (ta.iter().zip(&b).map(|(x, y)| x * y).sum(), a.iter().zip(&ttb).map(|(x, y)| x * y).sum());
        assert!((left - right).abs() < 1e-12 * left.abs().max(1.0), "{left} against {right}");
        // The same on one thread.
        let mut one = Vec::new();
        parallel::with_threads(1, || Extended::new(&matrix, &lists, &reach, &[], 2, &[]).inward(&b, &mut one));
        assert!(one.iter().zip(&ttb).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn two_far_reaching_nodes_meet_by_what_follows_both() {
        // Two far nodes of overlapping reach: the last node of the chain over cells 1 to 6,
        // and another over cells 4 to 9, put in by hand.
        let (_, mut lists, _) = system(11, 1, 7);
        let second = 2 * 11 + 2;
        for list in lists.iter_mut().skip(4).take(6) {
            list.push(second);
        }
        let graph = NodeGraph::new(second + 1, lists.iter());
        let mut matrix = Blocked::zero(&graph.adjacency).expect("small");
        for (cell, list) in lists.iter().enumerate() {
            let width = 3 * list.len();
            let mut data = vec![0.0; width * width];
            for i in 0..width {
                for j in 0..=i {
                    let value = ((cell * 71 + i * 13 + j * 7) as f64 * 0.41).sin() + if i == j { 14.0 } else { 0.0 };
                    data[i * width + j] = value;
                    data[j * width + i] = value;
                }
            }
            matrix.add(list, &data).unwrap();
        }
        let mut reach = vec![Vec::new(); second + 1];
        for (cell, list) in lists.iter().enumerate() {
            for node in list {
                reach[*node].push(cell as u32);
            }
        }
        let extended = Extended::new(&matrix, &lists, &reach, &[], 2, &[]);
        let first = 2 * 11 + 1;
        assert!(extended.has(first) && extended.has(second));
        // Neither reach holds the other: a block each.
        assert_eq!(extended.blocks(&lists, &reach, 2).len(), 2);
        let nodes = [2, 8, 9, 14, first, second];
        let full = dense(&matrix, &extended, &nodes);
        let lower = extended.lower(&matrix, &nodes).unwrap();
        let scale = full.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        for row in 0..3 * nodes.len() {
            for column in 0..=row {
                let (a, b) = (lower[row * (row + 1) / 2 + column], full[row][column]);
                assert!((a - b).abs() < 1e-12 * scale, "({row}, {column}): {a} against {b}");
            }
        }
        // The two are coupled through what follows both.
        assert!(full[12][15].abs() > 1e-6 * scale);
    }

    #[test]
    fn the_functions_of_a_leading_cell_are_taken_with_those_of_the_cells_that_follow() {
        // Two chains of three cells, with no node in common: cells 0 to 2 on nodes 0 to 6,
        // cells 3 to 5 on nodes 7 to 13. A connection couples cell 2 with cells 3 and 4, and
        // cell 2 leads.
        let chain = |first: usize| (0..3).map(move |cell| vec![first + 2 * cell, first + 2 * cell + 1, first + 2 * cell + 2]);
        let mut lists: Vec<Vec<usize>> = chain(0).chain(chain(7)).collect();
        let cells = lists.len();
        let pairs = [(2u32, 4u32), (2, 3), (2, 4)];
        for follower in [3, 4] {
            let both = [lists[2].clone(), lists[follower].clone()].concat();
            lists.push(both);
        }
        let graph = NodeGraph::new(14, lists.iter());
        let mut matrix = Blocked::zero(&graph.adjacency).expect("small");
        for (cell, list) in lists.iter().enumerate() {
            let width = 3 * list.len();
            let mut data = vec![0.0; width * width];
            for i in 0..width {
                for j in 0..=i {
                    let value = ((cell * 53 + i * 19 + j * 3) as f64 * 0.29).sin() + if i == j { 20.0 } else { 0.0 };
                    data[i * width + j] = value;
                    data[j * width + i] = value;
                }
            }
            matrix.add(list, &data).unwrap();
        }
        let mut reach = vec![Vec::new(); 14];
        for (cell, list) in lists.iter().take(cells).enumerate() {
            for node in list {
                reach[*node].push(cell as u32);
            }
        }
        let extended = Extended::new(&matrix, &lists, &reach, &[], 2, &pairs);
        // The nodes of cell 2 lead, as one group; no node of the other chain does.
        assert!((4..=6).all(|node| extended.has(node)) && (0..4).chain(7..14).all(|node| !extended.has(node)));
        let (group, place) = extended.of(5).unwrap();
        assert_eq!((group.members.as_slice(), place), ([4, 5, 6].as_slice(), 1));
        // What follows them: the nodes of cells 3 and 4 but node 11, which cell 5 has as well.
        assert_eq!(group.inside, [7, 8, 9, 10]);
        assert_eq!(group.rows, [2, 3, 4, 5, 6, 11]);
        // No node reaches far: no block of far-reaching nodes.
        assert!(extended.blocks(&lists, &reach, 2).is_empty());
        // T' K T: zero between the leading nodes and what follows them, and what `lower` gives.
        let nodes = [3, 4, 5, 6, 8, 9, 11];
        let full = dense(&matrix, &extended, &nodes);
        let lower = extended.lower(&matrix, &nodes).unwrap();
        let scale = full.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        for row in 0..3 * nodes.len() {
            for column in 0..=row {
                let (a, b) = (lower[row * (row + 1) / 2 + column], full[row][column]);
                assert!((a - b).abs() < 1e-12 * scale, "({row}, {column}): {a} against {b}");
                let (leads, follows) =
                    (|unknown: usize| (4..=6).contains(&nodes[unknown / 3]), |unknown: usize| (7..=10).contains(&nodes[unknown / 3]));
                if (leads(row) && follows(column)) || (leads(column) && follows(row)) {
                    assert!(b.abs() < 1e-12 * scale, "({row}, {column}): {b} is not zero");
                }
            }
        }
        // The other way round, cells 3 and 4 lead and cell 2 follows: its nodes but node 4,
        // which cell 1 has as well.
        let other = Extended::new(&matrix, &lists, &reach, &[], 2, &[(3, 2), (4, 2)]);
        assert_eq!(other.of(8).map(|(group, _)| group.inside.clone()), Some(vec![5, 6]));
    }

    #[test]
    fn a_linear_function_over_many_is_taken_with_the_linear_ones() {
        // A chain of cells, and one node more that is in all of them: more nodes within its
        // reach than are all taken.
        let cells = MOST_OF_ALL / 2 + 100;
        let far = 2 * cells + 1;
        let lists: Vec<Vec<usize>> = (0..cells).map(|cell| vec![2 * cell, 2 * cell + 1, 2 * cell + 2, far]).collect();
        let mut reach = vec![Vec::new(); far + 1];
        for (cell, list) in lists.iter().enumerate() {
            for node in list {
                reach[*node].push(cell as u32);
            }
        }
        let set: Vec<u32> = (0..cells as u32).collect();
        // Every second node linear, and the far one.
        let mut linear: Vec<bool> = (0..=far).map(|node| node % 2 == 0).collect();
        linear[far] = true;
        let all = inside(&lists, &reach, &[], &set, &[], &[far]);
        assert_eq!(all, (0..far).collect::<Vec<_>>(), "not known to be linear: with all of them");
        assert!(all.len() > MOST_OF_ALL);
        let few = inside(&lists, &reach, &linear, &set, &[], &[far]);
        assert_eq!(few, (0..far).step_by(2).collect::<Vec<_>>());
        // With another that is not linear: all of them again.
        assert_eq!(inside(&lists, &reach, &linear, &set, &[], &[1, far]).len(), all.len());
        // More than may follow at all: none.
        let many = MOST_INSIDE / 2 + 1;
        let lists: Vec<Vec<usize>> = (0..many).map(|cell| vec![2 * cell, 2 * cell + 1, 2 * cell + 2, 2 * many + 1]).collect();
        let mut reach = vec![Vec::new(); 2 * many + 2];
        for (cell, list) in lists.iter().enumerate() {
            for node in list {
                reach[*node].push(cell as u32);
            }
        }
        let set: Vec<u32> = (0..many as u32).collect();
        assert!(inside(&lists, &reach, &[], &set, &[], &[2 * many + 1]).is_empty());
    }

    #[test]
    fn nothing_far_reaching_changes_nothing() {
        let (matrix, lists, reach) = system(6, 0, 0);
        // Node 13 is in no cell: not in the system. No node reaches more than two cells.
        let extended = Extended::new(&matrix, &lists, &reach, &[], 2, &[]);
        assert!(!extended.any() && extended.blocks(&lists, &reach, 2).is_empty());
        let r: Vec<f64> = (0..3 * 14).map(|i| i as f64).collect();
        let (mut inward, mut outward) = (Vec::new(), r.clone());
        extended.inward(&r, &mut inward);
        extended.outward(&mut outward);
        assert!(inward == r && outward == r);
        let none = Extended::none(14);
        assert!(!none.any() && !none.has(3) && none.lower(&matrix, &[1, 2]).unwrap() == packed::lower(&matrix, &[1, 2]).unwrap());
        // A reach with nothing within it: left as it is.
        let (matrix, lists, reach) = system(4, 0, 4);
        assert!(Extended::new(&matrix, &lists, &reach, &[], 5, &[]).groups.is_empty());
        let all = Extended::new(&matrix, &lists, &reach, &[], 2, &[]);
        assert!(all.has(9), "every other node lies within its reach");
    }
}
