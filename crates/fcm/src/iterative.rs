//! The unknowns by iteration: for systems whose factorization would take far more work than
//! the system has entries, or more memory than there is.
//!
//! A factorization of the stiffness of a compact part takes work that grows with the square of
//! the unknowns and memory that grows faster than they do ([`fieldcraft_solver::cholesky`]). An
//! iteration takes work in proportion to the entries of the matrix, each time round, and its
//! memory is the matrix's; what it needs is a count of rounds that does not grow with the grid.
//!
//! # The method
//! The system is S, the stiffness with the unknowns every cell has to itself taken out
//! ([`crate::condense`]). It is solved by conjugate gradients (M. R. Hestenes, E. Stiefel,
//! *Methods of conjugate gradients for solving linear systems*, J. Res. Natl. Bur. Stand.
//! 49(6):409–436, 1952), preconditioned by an additive Schwarz method of two levels
//! (A. Toselli, O. Widlund, *Domain Decomposition Methods: Algorithms and Theory*, Springer
//! 2005, ch. 2 and 3): the approximate inverse is the sum of exact inverses on overlapping
//! pieces of the unknowns,
//!
//! M⁻¹ = R₀ᵀ S₀⁻¹ R₀ + Σ w² Rᵀ (R S Rᵀ)⁻¹ R, S₀ = R₀ S R₀ᵀ.
//!
//! **The coarse level** (index 0) is the linear shape functions of every level of the grid: the
//! trilinear finite elements of the refined grid, a thirtieth of the unknowns at degree 3. The
//! basis is hierarchic, so they are a subset of the unknowns and their matrix a part of S; it
//! is factored ([`Factor`]). They carry what is smooth over many cells from one end of the part
//! to the other in one step, which is what keeps the count of rounds from growing with the
//! number of cells.
//!
//! **The local level** is one block for every vertex of the grid: the functions that reach no
//! cell but the eight around it, which are the vertex's own linear function and those of the
//! edges and faces that meet there. Each block's matrix is factored densely (additive Schwarz
//! methods for shape functions of high degree: L. F. Pavarino, *Additive Schwarz methods for
//! the p-version finite element method*, Numer. Math. 66:493–516, 1993). An unknown is in the
//! blocks of several vertices and their answers for it are summed; a block's answer is
//! weighted by w², the mean over its unknowns of one over the blocks each is in (one weight
//! for a block: its answer is scaled, not bent, which a block that holds a cut cell does not
//! stand).
//!
//! Blocks of one cell each, all the functions that reach it, take three times the rounds on a
//! compact part and ten times on a thin one: where cells hold little material, a function they
//! share is nearly a sum of functions of the cells around it on the material there is, which
//! is what makes the stiffness of a cut grid ill-conditioned (F. de Prenter, C. V. Verhoosel,
//! G. J. van Zwieten, E. H. van Brummelen, *Condition number analysis and preconditioning of
//! the finite cell method*, Comput. Methods Appl. Mech. Engrg. 316:297–327, 2017), and a block
//! must hold the function with all of them to solve that exactly. Measured at degree 3 on the
//! oracle's bracket, 8 mm thick with holes through it, on cells of 2.2, 4.4 and 8 mm: 124, 268
//! and 390 rounds with a block for every cell, 39, 39 and 36 with one for every vertex; on a
//! solid block 105 and 31. The blocks around the vertices take three quarters of the memory of
//! those of the cells.
//!
//! **Cells of two sizes side by side.** A function of a cell beside a divided one reaches all
//! the cells that one is divided into, and a block of everything within its reach has
//! thousands of unknowns. So the blocks around vertices are those of functions that reach
//! eight cells at most ([`AROUND`]), and a farther-reaching function is taken together with
//! what follows it ([`crate::extended`]): that leaves it orthogonal to the functions within
//! its reach, and it is in a block with the other such functions around its vertex. Measured
//! on the bracket above at degree 3: 40 rounds with its cells divided once at its fillet and
//! holes, 63 divided twice, 49 to 66 divided three times, and 42 at degree 4 divided once.
//! With the far-reaching functions in blocks of the single cells they reach instead, it was
//! 117 rounds divided once, 233 twice, and 174 at degree 4 (J. N. Jomo, F. de Prenter,
//! M. Elhaddad, D. D'Angella, C. V. Verhoosel, S. Kollmannsberger, J. S. Kirschke, V. Nübel,
//! E. H. van Brummelen, E. Rank, *Robust and parallel scalable iterative solutions for
//! large-scale finite cell analyses*, Finite Elem. Anal. Des. 163:14–30, 2019, iterate on such
//! grids with blocks of cells).
//!
//! **Connected bodies.** Two bodies that a connection holds together are coupled stiffly
//! across it. Every two cells it couples are a block, with the functions of both, and the
//! functions of the cells that lead across it are taken with those of the cells that follow
//! ([`crate::extended`]): 53 rounds for three bonded blocks and 51 where they slide, for 85
//! and 78 with the blocks of two cells alone.
//!
//! # What it is not good at
//! **Walls much thinner than the cells.** A body of sheet on cells many times its thickness
//! takes more rounds, and more the thinner it is: 225 for a plate a tenth of its cells thick on
//! the trunk space, 111 with the tensor product. The coarse level is what lacks: its linear
//! functions are too stiff in the bending of a wall inside a cell. Such a body's factorization
//! is cheap, a wall being all surface, and the choice between the two counts it so
//! ([`crate::assembly`]).
//!
//! **A material that hardly changes its volume** (Poisson's ratio near one half), whose
//! stiffness against that one kind of strain dwarfs the rest. Such a system, and any other the
//! iteration does not settle in [`MOST`] rounds, is left to the factorization
//! ([`crate::assembly`]); one too large for that is iterated for up to [`MOST_ALONE`] rounds.
//!
//! # Determinism
//! Every sum runs in a fixed order: the products of the matrix as [`Blocked`] has them; the
//! blocks one by one into the result in their order; the inner products from the first
//! unknown to the last. The result is the same on any number of threads.

use fieldcraft_parallel as parallel;
use fieldcraft_solver::Limits;
use fieldcraft_solver::cholesky::{Factor, UpperColumns};
use fieldcraft_solver::sparse::Csr;

use crate::blocked::Blocked;
use crate::extended::Extended;
use crate::packed;
use crate::{Error, Result};

/// The residual at which the iteration is done, relative to the load: the unknowns are then
/// right to about as many digits, which is more than the shape functions resolve of the part.
pub(crate) const SETTLED: f64 = 1e-9; // unit-factors: a ratio of two forces

/// How far above the tolerance the true residual may end, when the recurrence's is below it:
/// the sums of a stiff system (a thin part on coarse cells) round at about this many times the
/// tolerance, and no further round brings the residual under it.
const ROUNDING: f64 = 10.0;

/// Rounds after which a system is left to the factorization: about ten times what a system
/// the two levels suit takes at the tolerance (30 to 40).
pub(crate) const MOST: usize = 400;

/// Rounds for a system the factorization cannot take, its factor or its rows above their
/// limits: five times as many.
pub(crate) const MOST_ALONE: usize = 5 * MOST;

/// Rounds between two looks at whether the caller has cancelled.
const LOOK: usize = 8;

/// The cells around a vertex of a grid: what a function may reach to be the middle of a block
/// ([`patches`]).
const AROUND: usize = 8;

/// Sets of cells a task of [`patches`] takes.
const GRAIN: usize = 64;

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

/// The cells each node that is in the system reaches, rising: `lists` has the nodes of every
/// cell, the first `cells` of them, and `own` says which nodes were taken out of the system
/// (they reach none).
pub(crate) fn reach(lists: &[Vec<usize>], cells: usize, own: &[bool]) -> Vec<Vec<u32>> {
    let mut reach: Vec<Vec<u32>> = vec![Vec::new(); own.len()];
    for (cell, list) in lists.iter().take(cells).enumerate() {
        let Ok(cell) = u32::try_from(cell) else { break };
        for node in list.iter().filter(|node| own.get(**node) == Some(&false)) {
            if let Some(reach) = reach.get_mut(*node)
                && reach.last() != Some(&cell)
            {
                reach.push(cell);
            }
        }
    }
    reach
}

/// The far-reaching nodes of `matrix`, each with what follows it ([`Extended`]): those that
/// reach more cells than a vertex of a grid has around it, and those that cells of another
/// body follow across a connection (`pairs`: the cell that leads and the one that follows).
/// `linear` says which nodes carry a linear shape function.
pub(crate) fn extended(matrix: &Blocked, lists: &[Vec<usize>], reach: &[Vec<u32>], linear: &[bool], pairs: &[(u32, u32)]) -> Extended {
    Extended::new(matrix, lists, reach, linear, AROUND, pairs)
}

/// The blocks of the local level: the nodes of each, rising. `lists` has the nodes of every
/// cell, the first `cells` of them, and then of every two cells a connection couples; `own`
/// says which nodes were taken out of the system, and are in no block; `reach` has the cells
/// of the others ([`reach`]) and `extended` the far-reaching ones that are taken with what
/// follows them. Every node of a list that is in the system is in one block at least.
pub(crate) fn patches(lists: &[Vec<usize>], cells: usize, own: &[bool], reach: &[Vec<u32>], extended: &Extended) -> Vec<Vec<usize>> {
    let kept = |node: &usize| own.get(*node) == Some(&false);
    let tidy = |mut nodes: Vec<usize>| {
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    };
    let reach_of = |node: usize| reach.get(node).map(Vec::as_slice).unwrap_or_default();
    // The sets of cells that are all a node reaches, of no more than a vertex has around it:
    // each once.
    let near = |node: usize| (2..=AROUND).contains(&reach_of(node).len());
    let mut sets: Vec<&[u32]> = (0..own.len()).filter(|node| near(*node)).map(reach_of).collect();
    sets.sort_unstable();
    sets.dedup();
    let within = |small: &[u32], large: &[u32]| small.iter().all(|cell| large.binary_search(cell).is_ok());
    // A set that is not part of a larger one has a block: the nodes that reach no cell
    // outside it. A larger set is that of a node of any of the set's cells: of the first.
    let around: Vec<Vec<usize>> = parallel::map_grained(sets.len(), GRAIN, |at| {
        let set = sets.get(at).copied().unwrap_or_default();
        let first = set.first().and_then(|cell| lists.get(*cell as usize)).map(Vec::as_slice).unwrap_or_default();
        if first.iter().any(|node| near(*node) && reach_of(*node).len() > set.len() && within(set, reach_of(*node))) {
            return Vec::new();
        }
        let mut nodes = tidy(set.iter().filter_map(|cell| lists.get(*cell as usize)).flatten().copied().collect());
        nodes.retain(|node| kept(node) && within(reach_of(*node), set));
        nodes
    });
    let mut blocks: Vec<Vec<usize>> = around.into_iter().filter(|nodes| !nodes.is_empty()).collect();
    // The far-reaching nodes that are taken with what follows them: with each other, around
    // the vertices of the larger cells.
    blocks.extend(extended.blocks(lists, reach, AROUND));
    // A cell with a node no such block holds (a far-reaching one that is taken as it is; a
    // cell that shares nothing) is a block itself.
    let mut held = vec![false; own.len()];
    for node in blocks.iter().flatten() {
        if let Some(flag) = held.get_mut(*node) {
            *flag = true;
        }
    }
    for list in lists.iter().take(cells) {
        if list.iter().any(|node| kept(node) && held.get(*node) == Some(&false)) {
            blocks.push(tidy(list.iter().copied().filter(|node| kept(node)).collect()));
        }
    }
    // And every two cells a connection couples.
    blocks.extend(
        lists.iter().skip(cells).map(|list| tidy(list.iter().copied().filter(|node| kept(node)).collect())).filter(|nodes| !nodes.is_empty()),
    );
    blocks
}

/// One block of the local level: its unknowns in rising order, and the Cholesky factor of
/// their matrix ([`packed`]).
struct Block {
    unknowns: Vec<usize>,
    factor: Vec<f64>,
}

impl Block {
    /// The block of the nodes `nodes` (rising, each once; three unknowns each) of `matrix`,
    /// the far-reaching ones taken with what follows them. None when there is no memory for
    /// its factor.
    fn new(matrix: &Blocked, extended: &Extended, nodes: &[usize]) -> Option<Block> {
        let unknowns: Vec<usize> = nodes.iter().flat_map(|node| 3 * node..3 * node + 3).collect();
        let n = unknowns.len();
        let mut factor = extended.lower(matrix, nodes)?;
        if !packed::cholesky(&mut factor, n) {
            // A block that is not positive to working precision (a sliver of material on its
            // fictitious stiffness): its diagonal alone, which is.
            for (i, unknown) in unknowns.iter().enumerate() {
                let diagonal = matrix.diagonal(*unknown).unwrap_or(1.0);
                let row = i * (i + 1) / 2;
                if let Some(entries) = factor.get_mut(row..row + i + 1) {
                    entries.fill(0.0);
                }
                if let Some(slot) = factor.get_mut(row + i) {
                    *slot = if diagonal > 0.0 && diagonal.is_finite() { diagonal.sqrt() } else { 1.0 };
                }
            }
        }
        Some(Block { unknowns, factor })
    }

    /// `out` = the inverse of the block's matrix times the block's part of `r`.
    fn solve(&self, r: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.extend(self.unknowns.iter().map(|unknown| r.get(*unknown).copied().unwrap_or(0.0)));
        packed::forward(&self.factor, out);
        packed::backward(&self.factor, out);
    }
}

/// The sum of the products of two lists, from the first to the last.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve `matrix` u = `load` on the nodes that are not `own` to a cell (three unknowns each;
/// the others' rows and columns are left out, and `load` is zero there) by the iteration of
/// the module's head, to the residual `settled` of `scale`, the size of the whole load, in at
/// most `most` rounds. `linear` says which nodes carry a linear shape function, `blocks` lists
/// the nodes of every block of the local level ([`patches`]), and `extended` has the
/// far-reaching nodes the blocks take with what follows them. `Ok(None)` when the rounds
/// did not settle it, or a step broke down: the system is then the factorization's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    matrix: &Blocked,
    own: &[bool],
    load: &[f64],
    scale: f64,
    linear: &[bool],
    blocks: &[Vec<usize>],
    extended: &Extended,
    limits: &Limits,
    settled: f64,
    most: usize,
    cancelled: &dyn Fn() -> bool,
) -> Result<Option<Outcome>> {
    let n = 3 * matrix.nodes();
    if load.len() != n || linear.len() != matrix.nodes() {
        return Err(Error::Internal("the iteration's system and its load differ in size".into()));
    }
    if !(scale > 0.0) || !scale.is_finite() || load.iter().all(|entry| *entry == 0.0) {
        // No load: nothing moves.
        return Ok(scale.is_finite().then(|| Outcome { u: vec![0.0; n], iterations: 0, residual: 0.0, coarse_entries: 0 }));
    }
    let taken = |node: usize| own.get(node).copied().unwrap_or(false);

    // The coarse level: the linear functions' part of the matrix, factored.
    let mut place = vec![usize::MAX; matrix.nodes()];
    let mut coarse = Vec::new();
    for (node, _) in linear.iter().enumerate().filter(|(node, linear)| **linear && !taken(*node)) {
        if let Some(slot) = place.get_mut(node) {
            *slot = coarse.len() / 3;
        }
        coarse.extend(3 * node..3 * node + 3);
    }
    let (mut ptr, mut idx, mut val) = (vec![0usize], Vec::new(), Vec::new());
    for unknown in &coarse {
        let (others, blocks) = matrix.row(unknown / 3);
        for (other, block) in others.iter().zip(blocks) {
            if let Some(at) = place.get(*other as usize).copied().filter(|at| *at != usize::MAX) {
                idx.extend(3 * at..3 * at + 3);
                val.extend_from_slice(block.get(3 * (unknown % 3)..3 * (unknown % 3) + 3).unwrap_or_default());
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

    // The local level, and each block's weight: the mean over its unknowns of one over the
    // blocks each is in.
    let blocks: Option<Vec<Block>> =
        parallel::map(blocks.len(), |block| Block::new(matrix, extended, blocks.get(block).map(Vec::as_slice).unwrap_or_default()))
            .into_iter()
            .collect();
    let Some(blocks) = blocks else {
        return Err(Error::Limit("no memory for the blocks the iteration solves the cells in (use a larger cell size or a lower degree)".into()));
    };
    let mut shared = vec![0.0f64; n];
    for unknown in blocks.iter().flat_map(|block| &block.unknowns) {
        if let Some(count) = shared.get_mut(*unknown) {
            *count += 1.0;
        }
    }
    let weights: Vec<f64> = blocks
        .iter()
        .map(|block| {
            let sum: f64 = block.unknowns.iter().map(|unknown| shared.get(*unknown).map_or(1.0, |count| 1.0 / count.max(1.0))).sum();
            sum / block.unknowns.len().max(1) as f64
        })
        .collect();
    drop(shared);
    let mut locals: Vec<Vec<f64>> = vec![Vec::new(); blocks.len()];
    let mut held = Vec::with_capacity(coarse.len());
    let mut inward = Vec::new();
    // z = M⁻¹ r.
    let mut apply = |r: &[f64], z: &mut [f64]| -> Result<()> {
        z.fill(0.0);
        // The blocks, on the far-reaching unknowns as they are taken with what follows them.
        let seen: &[f64] = if extended.any() {
            extended.inward(r, &mut inward);
            &inward
        } else {
            r
        };
        parallel::for_each_mut(&mut locals, |block, out| {
            if let Some(found) = blocks.get(block) {
                found.solve(seen, out);
            }
        });
        for ((block, local), weight) in blocks.iter().zip(&locals).zip(&weights) {
            for (unknown, value) in block.unknowns.iter().zip(local) {
                if let Some(slot) = z.get_mut(*unknown) {
                    *slot += weight * value;
                }
            }
        }
        if extended.any() {
            extended.outward(z);
        }
        // The coarse level, on the linear functions as they are.
        if let Some(factor) = &coarse_factor {
            held.clear();
            held.extend(coarse.iter().map(|unknown| r.get(*unknown).copied().unwrap_or(0.0)));
            factor.solve(&mut held)?;
            for (unknown, value) in coarse.iter().zip(&held) {
                if let Some(slot) = z.get_mut(*unknown) {
                    *slot += *value;
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
        matrix.mul_kept(own, &p, &mut ap);
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
            matrix.mul_kept(own, &u, &mut ap);
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

    fn size(load: &[f64]) -> f64 {
        dot(load, load).sqrt()
    }

    #[test]
    fn the_iteration_solves_what_the_factorization_does() {
        let nodes = 60;
        let (rows, load) = chain(nodes);
        let matrix = Blocked::from_rows(&rows);
        // Every fourth node is of the coarse level; the blocks are runs of five nodes that
        // overlap by one.
        let linear: Vec<bool> = (0..nodes).map(|node| node % 4 == 0).collect();
        let blocks: Vec<Vec<usize>> = (0..nodes).step_by(4).map(|first| (first..(first + 5).min(nodes)).collect()).collect();
        let run = |matrix: &Blocked, load: &[f64], linear: &[bool], most: usize, cancelled: bool| {
            solve(matrix, &[], load, size(load), linear, &blocks, &Extended::none(matrix.nodes()), &Limits::default(), SETTLED, most, &|| cancelled)
        };
        let solved = run(&matrix, &load, &linear, MOST, false).unwrap().expect("settled");
        let mut direct = load.clone();
        Factor::new(&UpperColumns::from_symmetric(&rows), usize::MAX).unwrap().solve(&mut direct).unwrap();
        let largest = direct.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (a, b) in solved.u.iter().zip(&direct) {
            assert!((a - b).abs() < 1e-7 * largest, "{a} against {b}");
        }
        assert!(solved.iterations > 0 && solved.iterations < 60 && solved.residual <= SETTLED, "{} rounds, {}", solved.iterations, solved.residual);
        assert!(solved.coarse_entries > 0);
        // The same to the bit on one thread and on many.
        let one = parallel::with_threads(1, || run(&matrix, &load, &linear, MOST, false)).unwrap().unwrap();
        assert!(one.u.iter().zip(&solved.u).all(|(a, b)| a.to_bits() == b.to_bits()) && one.iterations == solved.iterations);

        // No load moves nothing; too few rounds, and a matrix that is not positive, are the
        // factorization's; a caller that cancels ends it.
        let still = run(&matrix, &vec![0.0; 3 * nodes], &linear, MOST, false).unwrap().unwrap();
        assert!(still.iterations == 0 && still.u.iter().all(|x| *x == 0.0));
        assert!(run(&matrix, &load, &linear, 2, false).unwrap().is_none());
        let mut indefinite = rows.clone();
        for value in &mut indefinite.val {
            *value = -*value;
        }
        assert!(run(&Blocked::from_rows(&indefinite), &load, &linear, MOST, false).unwrap().is_none());
        assert!(run(&matrix, &load, &linear, MOST, true).is_err());
        assert!(run(&matrix, &load[1..], &linear, MOST, false).is_err());
        // Without a coarse level the blocks alone still solve it.
        let alone = run(&matrix, &load, &vec![false; nodes], MOST, false).unwrap().expect("settled");
        assert!(alone.coarse_entries == 0 && alone.residual <= SETTLED);
        for (a, b) in alone.u.iter().zip(&direct) {
            assert!((a - b).abs() < 1e-7 * largest, "{a} against {b}");
        }
    }

    #[test]
    fn nodes_taken_out_are_left_alone() {
        // Every third node taken out, its rows and columns with it: the others solve the
        // system of their own rows, and the nodes taken out stay at zero.
        let nodes = 30;
        let (rows, mut load) = chain(nodes);
        let matrix = Blocked::from_rows(&rows);
        let own: Vec<bool> = (0..nodes).map(|node| node % 3 == 1).collect();
        for (node, _) in own.iter().enumerate().filter(|(_, own)| **own) {
            load[3 * node..3 * node + 3].fill(0.0);
        }
        let linear: Vec<bool> = (0..nodes).map(|node| node % 6 == 0).collect();
        let blocks: Vec<Vec<usize>> =
            (0..nodes).step_by(3).map(|first| (first..(first + 4).min(nodes)).filter(|node| !own[*node]).collect()).collect();
        let solved = solve(&matrix, &own, &load, size(&load), &linear, &blocks, &Extended::none(nodes), &Limits::default(), SETTLED, MOST, &|| false)
            .unwrap()
            .expect("settled");
        let kept = matrix.rows_kept(&own);
        let mut direct: Vec<f64> = load.as_chunks::<3>().0.iter().zip(&own).filter(|(_, own)| !**own).flat_map(|(entries, _)| *entries).collect();
        Factor::new(&UpperColumns::from_symmetric(&kept), usize::MAX).unwrap().solve(&mut direct).unwrap();
        let found: Vec<f64> = solved.u.as_chunks::<3>().0.iter().zip(&own).filter(|(_, own)| !**own).flat_map(|(entries, _)| *entries).collect();
        let largest = direct.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (a, b) in found.iter().zip(&direct) {
            assert!((a - b).abs() < 1e-7 * largest, "{a} against {b}");
        }
        assert!(solved.u.as_chunks::<3>().0.iter().zip(&own).all(|(entries, own)| !own || entries == &[0.0; 3]));
    }

    /// The nodes of the cells of a grid of `nx` by `ny` squares: a node for every vertex, then
    /// one for every edge, then one for every square (its own). Returns the lists and the
    /// number of vertices and edges.
    fn squares(nx: usize, ny: usize) -> (Vec<Vec<usize>>, usize, usize) {
        let vertex = |i: usize, j: usize| i + (nx + 1) * j;
        let vertices = (nx + 1) * (ny + 1);
        // Edges along x, then along y.
        let along_x = |i: usize, j: usize| vertices + i + nx * j;
        let along_y = |i: usize, j: usize| vertices + nx * (ny + 1) + i + (nx + 1) * j;
        let edges = nx * (ny + 1) + (nx + 1) * ny;
        let mut lists = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                lists.push(vec![
                    vertex(i, j),
                    vertex(i + 1, j),
                    vertex(i, j + 1),
                    vertex(i + 1, j + 1),
                    along_x(i, j),
                    along_x(i, j + 1),
                    along_y(i, j),
                    along_y(i + 1, j),
                    vertices + edges + i + nx * j,
                ]);
            }
        }
        (lists, vertices, edges)
    }

    #[test]
    fn the_blocks_are_those_around_the_vertices() {
        let (lists, vertices, edges) = squares(3, 3);
        let nodes = vertices + edges + 9;
        let mut count = vec![0; nodes];
        for node in lists.iter().flatten() {
            count[*node] += 1;
        }
        // What one cell alone has is taken out: the squares' own nodes and the outer edges.
        // (The four corners are in one cell too; they are kept here, to be found in a block.)
        let own: Vec<bool> = (0..nodes).map(|node| count[node] == 1 && node >= vertices).collect();
        let cells = reach(&lists, lists.len(), &own);
        let blocks = patches(&lists, lists.len(), &own, &cells, &Extended::none(nodes));
        // One block for each of the four inner vertices: the vertex, the four edges that meet
        // there, and the vertices at the rim that reach none but its cells.
        assert_eq!(blocks.len(), 4);
        for block in &blocks {
            assert!(block.windows(2).all(|pair| pair[0] < pair[1]));
            assert_eq!(block.iter().filter(|node| (vertices..vertices + edges).contains(*node)).count(), 4);
            assert_eq!(block.iter().filter(|node| **node < vertices && count[**node] == 4).count(), 1);
        }
        let mut held = vec![false; nodes];
        for node in blocks.iter().flatten() {
            held[*node] = true;
        }
        assert!((0..nodes).all(|node| held[node] != own[node]), "every node that is left, and no other");

        // A node that reaches more cells than a vertex has around it is in no block around a
        // vertex: every cell it reaches is a block. Two coupled cells are one more.
        let mut far = lists.clone();
        let wide = nodes;
        for list in &mut far {
            list.push(wide);
        }
        far.push(vec![0, 1, 40]);
        let mut own_far = own.clone();
        own_far.push(false);
        let blocks = patches(&far, 9, &own_far, &reach(&far, 9, &own_far), &Extended::none(nodes + 1));
        assert_eq!(blocks.len(), 4 + 9 + 1);
        assert!(blocks[4..13].iter().all(|block| block.contains(&wide)) && blocks[..4].iter().all(|block| !block.contains(&wide)));
        assert_eq!(blocks[13], [0, 1]);
        // A cell alone, and no lists at all.
        let (alone, taken) = ([vec![2, 0, 1, 1]], [false, false, true]);
        assert_eq!(patches(&alone, 1, &taken, &reach(&alone, 1, &taken), &Extended::none(3)), [vec![0, 1]]);
        assert!(patches(&[], 0, &[], &[], &Extended::none(0)).is_empty());
    }

    #[test]
    fn a_block_is_solved_by_its_own_factor() {
        let (rows, _) = chain(6);
        let block = Block::new(&Blocked::from_rows(&rows), &Extended::none(6), &[1, 2, 4]).expect("small");
        assert_eq!(block.unknowns, [3, 4, 5, 6, 7, 8, 12, 13, 14]);
        // K_block x = r for a known x.
        let x: Vec<f64> = (0..9).map(|i| 0.5 + i as f64).collect();
        let mut r = vec![0.0; rows.n];
        for unknown in &block.unknowns {
            let (columns, values) = rows.row(*unknown);
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
    }
}
