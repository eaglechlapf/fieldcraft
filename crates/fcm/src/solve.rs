//! The linear static step on the grid: from a face-tagged surface, a material, supports and
//! loads to the displacement and stress anywhere in the part.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_solver::Limits;
use fieldcraft_solver::cholesky::{FactorError, PIVOT_TOLERANCE};
use fieldcraft_solver::ordering::element_dissection;

use crate::cell::{self, Face, Lame, Shapes, Surface};
use crate::gauss::gauss_legendre;
use crate::grid::{self, Grid, MAX_DEPTH, MAX_REFINE, Piece, QuadPoint, Triangles, cross, dot, sub};
use crate::inside::Solid;
use crate::islands::{self, CutLeaf, Held};
use crate::legendre::MAX_DEGREE;
use crate::moments::{Moments, Scratch};
use crate::refine::{Node, Tree};
use crate::space::{Kind, NONE, Space, local_modes};
use crate::{Error, Result};

/// Cells of the grid, at most (most of them outside the part), and as many again of the
/// refined levels.
const MAX_CELLS: usize = 4_000_000;
/// Entries of the sparse stiffness, at most (8 bytes each for the values and the column
/// indices: 2.9 GB).
pub(crate) const MAX_MATRIX_ENTRIES: usize = 180_000_000;
/// Quadrature points of one cut cell, at most.
const MAX_CELL_POINTS: usize = 4_000_000;
/// Bytes of cell matrices in memory together: the cells are integrated in blocks of as many as
/// fit (a cell's stiffness and its three support matrices), at least one and at most
/// [`MOST_BLOCK`]. The block depends on the degree only, never on the threads, and the cells
/// are assembled in order whatever it is.
const BLOCK_BYTES: usize = 1 << 30;
const MOST_BLOCK: usize = 256;
/// Cells classified per parallel task.
const GRAIN: usize = 4096;
/// Cells that dividing the ends of gaps may always add ([`Options::gaps`]); beyond these, as
/// many as the cells that hold material. A gap that closes to nothing (two faces that touch
/// along a line, a sphere on a plate) has no width to stop at, and a part of many slits has
/// many roots: the ends of gaps are then resolved as far as this pays for, and
/// [`Stats::holding`] says what is left.
const GAP_CELLS: usize = 2048;
/// A point this close to a grid plane, as a fraction of the cell size, may be evaluated in the
/// cell on either side.
const NEAR: f64 = 1e-6; // unit-factors: relative to the cell size
/// A cut cell that holds less than this fraction of its volume in material is a sliver: its
/// stress is read with its neighbours' ([`Solution::stress`], where the measurement is).
const SLIVER: f64 = 0.1;
/// A polynomial whose weight in the stress at a point is below this fraction of the largest is
/// left out of the mean ([`Solution::stress`]): it could move the result by no more than this
/// fraction of the difference between it and the others.
const LEAST_WEIGHT: f64 = 1e-3; // unit-factors: a ratio of weights
/// A cell whose surface turns by the limit ([`Options::TURN`]) to within this fraction resolves
/// it: whether a fillet of exactly two cells to the radius is divided is not left to rounding.
const AT_LIMIT: f64 = 1e-6; // unit-factors: relative to the limit

/// A linear elastic, isotropic material, in SI units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub young: f64,
    pub poisson: f64,
    /// Mass per volume, for gravity.
    pub density: f64,
}

impl Material {
    fn lame(&self) -> Result<Lame> {
        let (e, nu) = (self.young, self.poisson);
        if !(e > 0.0 && e.is_finite() && nu > -1.0 && nu < 0.5) {
            return Err(Error::Invalid(format!("isotropic constants E = {e}, ν = {nu} are not a stable material")));
        }
        if !(self.density >= 0.0 && self.density.is_finite()) {
            return Err(Error::Invalid(format!("density {} is not a mass per volume", self.density)));
        }
        Ok(Lame { lambda: e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), mu: e / (2.0 * (1.0 + nu)) })
    }
}

/// The material of the cell volume outside the part: the material itself with the fraction of
/// its stiffness.
fn fictitious_material(lame: &Lame, fraction: f64) -> Lame {
    Lame { lambda: fraction * lame.lambda, mu: fraction * lame.mu }
}

/// The fraction of the fictitious stiffness at which the shear stiffness of the fictitious volume
/// keeps the margin of [`Options::FICTITIOUS`] over the pivot tolerance for this material
/// ([`Options::fictitious`]): `FICTITIOUS · (λ + 2μ) / 2μ`.
fn margin_fraction(lame: &Lame) -> f64 {
    Options::FICTITIOUS * (lame.lambda + 2.0 * lame.mu).max(2.0 * lame.mu) / (2.0 * lame.mu)
}

/// Prescribed displacement components of a face, in the global axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Support {
    /// The face tag of the surface's triangles.
    pub face: u32,
    /// The components that are prescribed; the others are free.
    pub fixed: [bool; 3],
    /// The prescribed values (those of free components are ignored).
    pub value: [f64; 3],
}

/// A load, in SI units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Load {
    /// A pressure on a face, pushing into the part.
    Pressure { face: u32, pressure: f64 },
    /// A force per area on a face, in the global axes.
    Traction { face: u32, traction: [f64; 3] },
    /// A total force on a face, spread evenly over its area.
    Force { face: u32, force: [f64; 3] },
    /// A uniform acceleration of the whole part (it loads the part with density × acceleration).
    Gravity { acceleration: [f64; 3] },
}

/// The shape functions every cell carries ([`Options::basis`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Basis {
    /// The full tensor product of the one-dimensional functions of degree p: (p + 1)³ modes a
    /// cell.
    #[default]
    Tensor,
    /// The trunk space of Szabó and Babuška: the modes of the tensor product whose degrees sum
    /// to at most p (the two linear functions of an axis not counted). It holds every polynomial
    /// of total degree p with far fewer modes: 8 + 12 (p − 1) a cell up to degree 3, plus
    /// 3 (p − 2)(p − 3) from degree 4 and (p − 3)(p − 4)(p − 5)/6 from degree 6 on (74 instead
    /// of 216 at degree 5). Degree for degree it is less accurate than the tensor product. On
    /// cells that are the same everywhere (`tests/benchmarks.rs`: NAFEMS LE1, a plate with a
    /// hole) it reaches the tensor product's accuracy about two degrees higher, with half to
    /// three quarters of the unknowns and half the factor's entries, in no less time: there it
    /// saves memory, not time.
    ///
    /// On cells that are divided where the part needs it, it saves both: smaller cells make up
    /// for what it lacks. At a hole with the cells divided twice (`tests/hole.rs`,
    /// `where_the_refined_grid_sits`) the stress is within ± 1.1 % at degree 3 with a third of
    /// the tensor product's unknowns, which has ± 0.7 %, and within ± 0.5 % at degree 4 with
    /// a little over half of them. On a sheet metal tray hung by one wall (`tests/tray.rs`),
    /// with two rounds of [`Options::adapt`], the sag is 1.2 % low with 143 000 unknowns where
    /// the tensor product is 0.8 % low with 322 000 and takes twice as long.
    Trunk,
}

/// How the part is discretised.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// The size of the grid's cubic cells.
    pub cell_size: f64,
    /// The polynomial degree of the shape functions, 1..=[`MAX_DEGREE`].
    pub degree: usize,
    /// The depth of the octree under a cut cell where the surface in it is curved,
    /// 0..=[`MAX_DEPTH`]: for the cells the quadrature integrates, which are few unless
    /// [`Options::exact`] is off.
    pub depth: usize,
    /// The stiffness of the cell volume outside the part, as a fraction of the material's.
    ///
    /// **Nearly incompressible materials.** The factorization measures a pivot against the
    /// diagonal entry of its equation, which the material's longitudinal stiffness λ + 2μ sets,
    /// while the smallest pivot of an unknown that lives in the fictitious volume is set by the
    /// fraction of the shear stiffness 2μ. As Poisson's ratio approaches one half their ratio
    /// (λ + 2μ) / 2μ = (1 − ν) / (1 − 2ν) grows without bound (2500 at ν = 0.4999), and a
    /// fraction that is safe for steel leaves the system singular to working precision. When
    /// that happens the step is solved again with a hundred times the fraction, at most up to
    /// `FICTITIOUS · (λ + 2μ) / 2μ`, where the fictitious shear stiffness has the margin over the
    /// pivot tolerance that [`Options::FICTITIOUS`] gives a compressible material; a part that
    /// is not held stays singular at any fraction. [`Stats::fictitious`] is the fraction the
    /// solution was computed with. A larger fraction costs accuracy where a cell holds little
    /// material, and the pressure of a nearly incompressible material shows it first: for a
    /// block of ν = 0.4999 in tension whose worst cell holds 1.5 % material, the largest stress
    /// error is 0.3 % of the stress at the fraction 1e-8, 1 % at 2.5e-7 and 8 % at 2.5e-5
    /// (`tests/patch.rs`).
    pub fictitious: f64,
    /// Where the grid starts: this many cells before the lowest corner of the part's bounding
    /// box, each component in `[0, 1)`.
    pub offset: [f64; 3],
    /// The shape functions of a cell.
    pub basis: Basis,
    /// The levels of local refinement, 0..=[`MAX_REFINE`]; 0 is the grid alone.
    ///
    /// A cell is divided into eight of half its size where the part's surface in it is curved
    /// more tightly than the cell resolves, and the cells around it with it, level by level up
    /// to this number (the multi-level hp method of the crate's documentation: the cells of all
    /// levels carry one conforming basis, with no constraints at the hanging faces). A coarse
    /// grid then resolves a small fillet or hole without fine cells everywhere.
    ///
    /// **Which cells.** The measure is the turning of the surface across one cell: the cell's
    /// size times the largest curvature of the pieces of the surface it holds
    /// ([`grid::curvatures`]), which is the angle the surface's normal turns by over the length
    /// of the cell, or the cell's size over the radius of a fillet or a hole. A cut cell across
    /// which the surface turns more than [`Options::TURN`] (half of that below degree 4, and
    /// below degree 6 in the trunk space) is divided. A sharp edge is no curvature and divides
    /// nothing, and neither does a flat face, so `refine` costs nothing where it is not needed
    /// and stops by itself once the cells resolve every face, whatever the number of levels
    /// allowed. The cells that share a face with such a cell are divided with it: the finer
    /// functions vanish on the boundary of the cells that are divided, so the cell that holds
    /// the surface would otherwise be held there, right at the surface wherever the surface
    /// meets one of its faces. Measured on Kirsch's plate (`tests/hole.rs`,
    /// `where_the_refined_grid_sits`: cells of the hole's radius, degree 3 divided twice and
    /// degree 4 once, six grid positions), the stress at the hole is within ± 0.75 % and
    /// ± 0.6 % with those neighbours, as on the uniform grid of the finest cells, and within
    /// ± 5.6 % and ± 4.3 % without them; dividing the cells that share only an edge or a
    /// vertex as well changes the fourth digit and costs a tenth more unknowns.
    ///
    /// The octree of a divided cut cell is as much shallower as the cell is smaller
    /// ([`Options::depth`] less its level), so its smallest boxes are those of the grid's cell:
    /// in the same measurement the full depth under every divided cell changes the stress at
    /// the hole by 0.04 % at most, with up to seven times the quadrature points.
    pub refine: usize,
    /// Whether material that one cell holds in separate pieces has unknowns of its own in each
    /// piece ([`crate::islands`]): the two flanges of a sheet metal corner, the two jaws of a
    /// slit clamp, the two sides of any gap narrower than a cell. Off, a cell is one polynomial
    /// whatever it holds, and such a gap is solved as if it were welded shut wherever a cell
    /// reaches across it: on the fork of `tests/gap.rs` the prong that is not pulled takes a
    /// tenth of the load, and the tray of `tests/tray.rs`, hung by one wall, carries its load
    /// through its four open corners and deflects an eightieth of what it does.
    ///
    /// This costs nothing where no cell holds separate pieces of material, and changes nothing
    /// there: not the unknowns, not their numbering, not a bit of the result. A cell that cuts
    /// through a hole can hold the material on its two sides as two pieces; that is told apart
    /// too, which adds a few unknowns and moves the stress at the hole in its last digits. A
    /// piece of material that is connected to nothing else leaves the system singular; the step
    /// is then solved with the pieces joined, and [`Stats::joined`] says so.
    pub separate: bool,
    /// The levels of refinement at the root of a gap, 0..=[`MAX_REFINE`]; 0 for none.
    ///
    /// Where a gap ends inside a cell (a slit, the relief at a sheet metal corner), the cell
    /// holds one piece of material and its polynomial joins the two sides of the gap for as far
    /// as the cell reaches along it: nothing tells them apart there ([`Options::separate`]
    /// does where a cell holds the two sides and not the root). So a cell that holds one piece
    /// of material is divided when one of its eight parts would hold separate pieces
    /// ([`crate::islands`]) that are less than a quarter of the cell apart ([`Options::GAP`]),
    /// up to this many levels: the stretch along which the sides are joined shrinks to a few
    /// widths of the gap, and nothing else is divided for it. The cells this may add are bounded
    /// (as many again as hold material, 2048 at least), and when the system is too large with
    /// them the step is solved with a level less, down to none; [`Stats::holding`] counts the
    /// cells that are left holding the sides of a gap together.
    pub gaps: usize,
    /// Rounds of refinement that follow the solution, 0 for none.
    ///
    /// [`Options::refine`] divides cells where the surface is curved, whether or not anything
    /// happens there: every bend of a sheet metal part and every hole, when one bend carries
    /// the load. A round of this divides the cells the *solution* is least resolved in, and
    /// solves again. How well a cell resolves the solution is read from the solution itself:
    /// the strain energy of the part of it that only the functions of the highest degree carry
    /// in that cell (the hierarchic indicator of the p-version: B. Szabó, I. Babuška, *Finite
    /// Element Analysis*, Wiley 1991; a field the lower degrees represent has none). The cells
    /// that carry [`Options::SHARE`] of the sum, largest first, are divided (W. Dörfler, *A
    /// convergent adaptive algorithm for Poisson's equation*, SIAM J. Numer. Anal.
    /// 33(3):1106–1124, 1996), each round at most one level deeper than the last, so a stress
    /// that is singular (a sharp re-entrant edge, the end of a support) takes a round's share
    /// and no more. A round whose system would be too large is left out, and the solution
    /// before it is the result ([`Stats::left_out`]). [`Stats::rounds`] counts the rounds, and
    /// [`Stats::settled`] is how much the last one still changed the strain energy.
    ///
    /// The indicator ranks the cells of one solution; it is not a measure of the error. Summed
    /// over the part it is no fraction of the strain energy: in a cell that holds a thin slab
    /// of material the functions of the highest degree and the lower ones carry large parts
    /// that cancel. And it follows the energy, not a stress peak: at a hole of the radius of a
    /// cell in a plate under tension (`tests/hole.rs`) two rounds leave the stress at the hole
    /// within ± 7 % over six positions of the grid where two levels of [`Options::refine`]
    /// leave it within ± 0.7 % in the same time. The two work together: `refine` for what the
    /// surface shows, the rounds for what only the solution does.
    pub adapt: usize,
    /// The rounds of [`Options::adapt`] end once one has moved the solution by no more than
    /// this, in the energy norm and relative to the solution: √|ΔU / U| with the strain
    /// energies of the two solutions ([`Solution::change_from`]). 0 runs every round asked for.
    pub settle: f64,
    /// Whether a round of [`Options::adapt`] takes the integrals of the cut cells it leaves as
    /// they were from the round before, in place of integrating them again. The result is the
    /// same to the bit either way; integrating a cut cell is the most work a cell takes, and a
    /// round leaves most cells as they were.
    pub reuse: bool,
    /// Whether the cut cells are integrated exactly, from the triangles of the surface in them
    /// ([`crate::exact`]: the divergence theorem, at a cost that grows with the number of
    /// triangles in a cell), where that decides a cell with certainty; the others, and all of
    /// them without it, are integrated by the quadrature on an octree of [`Options::depth`]
    /// levels ([`crate::grid`]). [`Stats::exact_cells`] counts them.
    pub exact: bool,
    /// How the unknowns are solved for.
    pub solver: Solver,
}

/// How the unknowns of a step are solved for ([`Options::solver`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Solver {
    /// By the factorization, unless the system is large and the factorization would take far
    /// more work than an iteration, or more memory than is allowed: then by iteration, and by
    /// the factorization after all when the iteration does not settle ([`crate::iterative`]).
    #[default]
    Auto,
    /// By the sparse Cholesky factorization: the same result to the last bit on every machine.
    Direct,
    /// By iteration whatever the system's size (and by the factorization when it does not
    /// settle): the result is right to the iteration's tolerance, not to rounding.
    Iterative,
}

impl Options {
    /// The share of the indicator's sum whose cells a round of [`Options::adapt`] divides. Any
    /// share below one converges (Dörfler); a small one divides few cells a round and needs
    /// many rounds, each a whole solve, and a large one divides cells that did not need it.
    /// Measured on the sheet metal tray of `tests/tray.rs`, hung by one wall (the sag against
    /// 2.105 mm, from the grid's own cells): two rounds at 0.5 leave it about 11 % low, at 0.8
    /// 9 %, at 0.9 3 % with 163 000 unknowns, at 0.95 2 % with a quarter more unknowns.
    pub const SHARE: f64 = 0.9;
    /// The distance between two rounds' solutions below which a caller may take the result as
    /// settled ([`Options::settle`]): 5 % in the energy norm, the permissible error of the
    /// examples of O. C. Zienkiewicz, J. Z. Zhu, *A simple error estimator and adaptive
    /// procedure for practical engineering analysis*, Int. J. Numer. Meth. Engng
    /// 24(2):337–357, 1987. In strain energy that is a change of 0.25 %.
    pub const SETTLED: f64 = 0.05;
    /// A cell that holds the root of a gap is divided while it is more than this many widths
    /// of the gap across ([`Options::gaps`]).
    pub const GAP: f64 = 2.5;
    pub const DEGREE: usize = 3;
    /// The turning of the surface across one cell, in radians, that a cell of degree 4 or more
    /// resolves: half a radian, two cells to the radius. This is a measurement, not a choice:
    /// on Kirsch's plate (`tests/hole.rs`, `where_the_grid_sits`) the stress at the hole over
    /// six positions of a uniform grid is within ± 0.7 % with two cells to the radius at degree
    /// 4 and within ± 6.5 % with one, and at degree 3 within ± 0.8 % with four cells to the
    /// radius and ± 4 % with two. So a cell of degree 3 or less resolves half of this, a
    /// quarter of a radian, and so does the trunk space up to degree 5, which reaches the
    /// tensor product's accuracy about two degrees higher ([`Basis::Trunk`]). The facets of a
    /// tessellated face turn a little more than the face itself, so a cell at exactly the limit
    /// is divided.
    pub const TURN: f64 = 0.5;
    /// Flat faces are integrated exactly without an octree, so it refines only where the
    /// surface is curved ([`crate::grid`]), and there the depth sets the accuracy of the cell
    /// matrices: with too shallow an octree the stress at a curved face moves erratically as
    /// the degree is raised. Measured on a hole of the radius of one cell (`tests/hole.rs`,
    /// `depth_of_the_octree`, degrees 3 to 5), the stress at the hole changes by up to 6 % from
    /// depth 1 to 2, by less than 0.1 % from depth 2 to 3 and by less than 0.02 % from 3 to 4;
    /// each level about doubles the points of the curved cells.
    pub const DEPTH: usize = 3;
    /// The stiffness of a cut cell's volume outside the part keeps the unknowns of a cell that
    /// holds little material from being undetermined. The error it adds is of its own order, so
    /// it is as small as the factorization allows: two decades above the pivot the sparse
    /// Cholesky still accepts ([`PIVOT_TOLERANCE`]).
    pub const FICTITIOUS: f64 = 100.0 * PIVOT_TOLERANCE;

    /// The defaults with this cell size.
    pub fn new(cell_size: f64) -> Options {
        Options {
            cell_size,
            degree: Options::DEGREE,
            depth: Options::DEPTH,
            fictitious: Options::FICTITIOUS,
            offset: [0.0; 3],
            basis: Basis::Tensor,
            refine: 0,
            separate: true,
            gaps: 0,
            adapt: 0,
            settle: 0.0,
            reuse: true,
            exact: true,
            solver: Solver::Auto,
        }
    }

    /// The turning of the surface across a cell above which the cell is divided
    /// ([`Options::refine`], [`Options::TURN`]).
    fn turn(&self) -> f64 {
        let tensor_degree = match self.basis {
            Basis::Tensor => self.degree,
            Basis::Trunk => self.degree.saturating_sub(2),
        };
        if tensor_degree >= 4 { Options::TURN } else { 0.5 * Options::TURN }
    }

    fn check(&self) -> Result<()> {
        if self.refine > MAX_REFINE || self.gaps > MAX_REFINE || self.adapt > MAX_REFINE {
            return Err(Error::Invalid(format!("{} levels of refinement are above {MAX_REFINE}", self.refine.max(self.gaps).max(self.adapt))));
        }
        if !(self.settle >= 0.0 && self.settle.is_finite()) {
            return Err(Error::Invalid(format!("the change that ends the rounds of refinement, {}, is not a number from 0 on", self.settle)));
        }
        if !(1..=MAX_DEGREE).contains(&self.degree) {
            return Err(Error::Invalid(format!("degree {} is not in 1..={MAX_DEGREE}", self.degree)));
        }
        if self.depth > MAX_DEPTH {
            return Err(Error::Invalid(format!("octree depth {} is above {MAX_DEPTH}", self.depth)));
        }
        if !(self.fictitious > 0.0 && self.fictitious <= 1.0) {
            return Err(Error::Invalid(format!("fictitious stiffness {} is not in (0, 1]", self.fictitious)));
        }
        Ok(())
    }
}

/// The force the supports of a face exert on the part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reaction {
    pub face: u32,
    pub force: [f64; 3],
}

/// What a solution was computed with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Cells of the grid along each axis.
    pub cells: [usize; 3],
    /// Cells wholly inside the part (with refinement, of any level and not divided: [`Stats::levels`]).
    pub inside_cells: usize,
    /// Cells the surface cuts (with refinement, of any level and not divided).
    pub cut_cells: usize,
    /// Unknowns of the system.
    pub unknowns: usize,
    /// Quadrature points of the cut cells.
    pub quadrature_points: usize,
    /// The cut cells that were integrated exactly ([`Options::exact`]); the others were left to
    /// the quadrature on the octree.
    pub exact_cells: usize,
    /// Entries of the stiffness, as the factorization counts them.
    pub stiffness_entries: usize,
    /// Entries of the Cholesky factor.
    pub factor_entries: usize,
    /// The volume of the part as the quadrature sees it.
    pub volume: f64,
    /// The volume the surface encloses: the difference to `volume` is the quadrature's error.
    pub surface_volume: f64,
    /// The smallest fraction of a cut cell that holds material, of the cell's own volume (the
    /// conditioning of the system degrades as it falls).
    pub smallest_fraction: f64,
    /// The largest Nitsche parameter of a supported cell, a stiffness per area.
    pub largest_nitsche: f64,
    /// The fictitious stiffness the solution was computed with: [`Options::fictitious`], or a
    /// larger fraction when the system was singular to working precision at that one.
    pub fictitious: f64,
    /// The cells that hold material and are not divided, by level of refinement
    /// ([`Options::refine`]): `levels[0]` are cells of the grid, `levels[1]` cells of half
    /// their size, and so on. Their sum is `inside_cells + cut_cells`.
    pub levels: [usize; MAX_REFINE + 1],
    /// The cells that hold material in separate pieces, each of which has unknowns of its own
    /// there ([`Options::separate`]): the two sides of a gap narrower than the cell.
    pub split_cells: usize,
    /// The unknowns that telling such pieces apart added.
    pub separated: usize,
    /// The cells that still hold the two sides of a gap together where it ends, over more than
    /// twice [`Options::GAP`] widths of the gap: [`Options::gaps`] had no level left for them,
    /// or the cells it may add were used up.
    pub holding: usize,
    /// Whether the pieces were joined after all: told apart, the system was singular (a piece
    /// that nothing holds), and the step was solved as with [`Options::separate`] off.
    pub joined: bool,
    /// The rounds of refinement that followed the solution ([`Options::adapt`]).
    pub rounds: usize,
    /// The change of the strain energy in the last of those rounds, (U − U_before) / U: how
    /// far the solution still moved. 0 without a round.
    pub settled: f64,
    /// Whether a round of [`Options::adapt`] was left out: with its cells the system would have
    /// been too large, or could not be solved.
    pub left_out: bool,
    /// The rounds of the iteration that solved the system ([`Options::solver`]); 0 when the
    /// factorization did. [`Stats::factor_entries`] is then the factor of the coarse level.
    pub iterations: usize,
    /// What the iteration left when it ended, measured in energy against the whole
    /// (`crate::iterative`); 0 for the factorization.
    pub residual: f64,
}

/// How far two solutions of the same step are apart ([`Solution::change_from`]): an indicator
/// of convergence, not a bound of the error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Change {
    /// The change of the strain energy, (U − U_coarser) / U.
    pub strain_energy: f64,
    /// The change of the largest von Mises stress over the points, (peak − peak_coarser) / peak.
    pub peak_von_mises: f64,
    /// The largest difference of the von Mises stress at any one of the points, as a fraction of
    /// the peak.
    pub von_mises: f64,
    /// The points both solutions have a stress at.
    pub points: usize,
}

/// A solved step.
#[derive(Debug, Clone)]
pub struct Solution {
    grid: Grid,
    shapes: Shapes,
    lame: Lame,
    tree: Tree,
    space: Space,
    /// The cells that hold several islands, by node: the pieces of surface each owns and the
    /// island of each piece ([`islands::island_at`]).
    split: BTreeMap<u32, (Vec<Piece>, Vec<u16>)>,
    /// Where functions were told apart ([`Options::separate`]): for each element, itself and
    /// the elements a few links of material from it, in increasing order ([`Space::around`]).
    /// Empty where none were.
    near: Vec<Vec<u32>>,
    /// The fraction of each element's cell that holds material.
    fractions: Vec<f64>,
    /// Where the material of each element's cell is, in the cell's natural coordinates: its
    /// middle and its half extent along each axis ([`Solution::stress`]).
    material: Vec<([f64; 3], [f64; 3])>,
    /// Three displacement coefficients per mode.
    u: Vec<f64>,
    pub stats: Stats,
    /// For each element, the strain energy of what only the functions of the highest degree
    /// carry in its cell ([`Options::adapt`]).
    surplus: Vec<f64>,
    /// The strain energy of the part, ½ ∫ σ : ε dV.
    pub strain_energy: f64,
    /// The reaction of every supported face, by face tag.
    pub reactions: Vec<Reaction>,
}

/// Where a point is among a body's cells.
#[derive(Clone, Copy)]
struct Place<'a> {
    grid: &'a Grid,
    tree: &'a Tree,
    space: &'a Space,
    /// The cells that hold several islands ([`Solution`]).
    split: &'a BTreeMap<u32, (Vec<Piece>, Vec<u16>)>,
}

impl Place<'_> {
    /// The leaf under `node` that holds the point at `t` cells of the grid from its origin and
    /// holds material: its element, the point's natural coordinates in it and its size. A point
    /// within [`NEAR`] of the plane between two children may be in either.
    fn descend(&self, node: usize, t: [f64; 3], island: bool) -> Option<(usize, [f64; 3], f64)> {
        let cell = self.tree.nodes.get(node)?;
        let level = usize::from(cell.level);
        // In cells of this level from the cell's lowest corner.
        let steps = (1u64 << level.min(MAX_REFINE)) as f64;
        let within = [0, 1, 2].map(|a| t[a] * steps - cell.ijk[a] as f64);
        if cell.is_leaf() {
            let first = self.space.element_of.get(node).copied().filter(|element| *element != NONE && cell.material())? as usize;
            // In a cell of several islands, the one the point is in, when that is asked for:
            // the first otherwise.
            let island = u32::try_from(node).ok().filter(|_| island).and_then(|node| self.split.get(&node)).map_or(0, |(pieces, of_piece)| {
                islands::island_at(pieces, of_piece, [0, 1, 2].map(|a| self.grid.origin[a] + t[a] * self.grid.spacing))
            });
            return Some((first + island, within.map(|s| 2.0 * s - 1.0), self.grid.step(level)));
        }
        let mut halves = [[None; 2]; 3];
        for a in 0..3 {
            // In cells of the children's level from the plane between them.
            let from_middle = 2.0 * within[a] - 1.0;
            let upper = usize::from(from_middle >= 0.0);
            halves[a] = [Some(upper), (from_middle.abs() < NEAR).then_some(1 - upper)];
        }
        for k in halves[2].into_iter().flatten() {
            for j in halves[1].into_iter().flatten() {
                for i in halves[0].into_iter().flatten() {
                    if let Some(found) = self.descend(cell.children as usize + i + 2 * j + 4 * k, t, island) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }

    /// The element that holds `x`, its natural coordinates there and the size of its cell; with
    /// `island` off, the first element of the cell, whichever of its islands the point is in.
    fn locate_in(&self, x: [f64; 3], island: bool) -> Option<(usize, [f64; 3], f64)> {
        let mut t = [0.0; 3];
        let mut candidates = [[None; 2]; 3];
        for a in 0..3 {
            t[a] = (x[a] - self.grid.origin[a]) / self.grid.spacing;
            if !t[a].is_finite() {
                return None;
            }
            let count = self.grid.cells[a] as f64;
            let valid = |i: f64| (i >= 0.0 && i < count).then_some(i as usize);
            let i = t[a].floor();
            let other = if t[a] - i < NEAR {
                valid(i - 1.0)
            } else if i + 1.0 - t[a] < NEAR {
                valid(i + 1.0)
            } else {
                None
            };
            candidates[a] = [valid(i), other];
        }
        for k in candidates[2].into_iter().flatten() {
            for j in candidates[1].into_iter().flatten() {
                for i in candidates[0].into_iter().flatten() {
                    let Some(&root) = self.tree.root.get(self.grid.index([i, j, k])) else { continue };
                    if root == NONE {
                        continue;
                    }
                    if let Some(found) = self.descend(root as usize, t, island) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }
}

/// The share of the cells that hold material that the material fills, below which a body is
/// taken to be one of walls much thinner than its cells: where the rounds of an iteration grow
/// without the larger coarse level ([`crate::iterative`]: 41 rounds at 0.36, 52 at 0.27, 74 at
/// 0.18, for 31 or 32 with it).
pub(crate) const THIN: f64 = 0.3;

/// How a body is tied to others ([`crate::assembly`]).
pub(crate) struct Ties<'a> {
    /// The faces that are held to another body where they touch it.
    pub faces: &'a BTreeSet<u32>,
    /// Where a piece of such a face may touch the other body, and how it holds it there
    /// ([`cell::Touching`]).
    pub touching: &'a cell::Touching<'a>,
    /// The curved surfaces such faces lie on, for the normals of those that slide.
    pub curved: &'a BTreeMap<u32, cell::Curved>,
    /// Whether another body's tied face holds this one: it needs no support of its own then.
    pub held: bool,
}

/// A point where a tied face touches the other body, in an element's cell: the face, the point,
/// its weight (an area), the values of the element's functions there and the held traction of
/// each of their unknowns on the face, `tractions[3 n i + 3 f + k]` component i of function f's
/// component k, of n functions.
pub(crate) struct Tied<'a> {
    pub tag: u32,
    pub x: [f64; 3],
    pub weight: f64,
    pub values: &'a [f64],
    pub tractions: &'a [f64],
    /// Where the face slides on the other body: the unit normal it holds along. `None` where
    /// the two are bonded.
    pub along: Option<[f64; 3]>,
}

/// What a body hands over when its cells are ready, before their matrices are added up: enough
/// to add them into a system of several bodies ([`crate::assembly`]).
pub(crate) struct View<'a> {
    /// The body's cells.
    pub grid: Grid,
    pub tree: &'a Tree,
    /// The body's functions (three unknowns each), numbered in its own order of elimination.
    pub count: usize,
    /// Where a function sits.
    pub position: &'a dyn Fn(usize) -> [f64; 3],
    /// The functions of every element.
    pub elements: Vec<&'a [usize]>,
    /// Hand every element's stiffness and load, on its functions, to the caller, in order.
    #[allow(clippy::type_complexity)]
    pub assemble: &'a dyn Fn(&mut dyn FnMut(usize, &[f64], &[f64]) -> Result<()>) -> Result<()>,
    /// Hand every point where a tied face touches another body in an element's cell to the
    /// caller. Elements are asked on several threads at once.
    #[allow(clippy::type_complexity)]
    pub tied: &'a (dyn Fn(usize, &mut dyn FnMut(&Tied) -> Result<()>) -> Result<()> + Sync),
    /// The element whose cell holds the point, and the values of its functions there; a point
    /// no cell holds is read in the cell of the second point, one of the body's surface next
    /// to it.
    #[allow(clippy::type_complexity)]
    pub at: &'a (dyn Fn([f64; 3], [f64; 3]) -> Option<(usize, Vec<f64>)> + Sync),
    /// The Nitsche parameter of an element's cell, once its matrices were handed over.
    pub nitsche: &'a dyn Fn(usize) -> f64,
    /// Which functions are linear shape functions ([`crate::space::Space::linear`]).
    pub linear: &'a [bool],
    /// How the body asks for its unknowns to be solved, and its Poisson's ratio, which decides
    /// whether an iteration suits it.
    pub solver: Solver,
    pub poisson: f64,
    /// Which functions are the quadratic ones of an edge ([`crate::space::Space::quadratic`]).
    pub quadratic: &'a [bool],
    /// Whether the body is one of walls much thinner than its cells ([`THIN`]): its coarse
    /// level in an iteration has the quadratic functions of the edges too.
    pub thin: bool,
    /// The levels of refinement asked for, and the stiffness and factor entries of the solve
    /// before this one, for what a refusal says and when.
    pub refine: usize,
    pub before: Option<(usize, usize)>,
}

/// A body's share of a solved system: three displacement coefficients per function, and the
/// entries of the system's stiffness and of its factor.
pub(crate) struct Solved {
    pub u: Vec<f64>,
    pub stiffness_entries: usize,
    pub factor_entries: usize,
    /// The rounds of the iteration and what it left; zero when the system was factored.
    pub iterations: usize,
    pub residual: f64,
}

/// Where a body's equations go: into a system of its own, or one of several bodies.
pub(crate) type Join<'a> = &'a dyn Fn(&View) -> Result<Solved>;

/// What the cells of a body add up to while their matrices are handed over.
struct Tally {
    stats: Stats,
    nitsche: Vec<f64>,
    fractions: Vec<f64>,
    material: Vec<([f64; 3], [f64; 3])>,
}

/// The cells to divide next ([`Options::adapt`]) in each of the solutions of one step (one
/// body's, or those of the bodies of an assembly, which are ranked together: the indicator is
/// an energy, the same measure in every body): the leaves whose elements carry the share
/// `share` of the indicator's sum, largest first, each by its level and its place, with the
/// cells above it (a cell is divided only when the cells above it are).
pub(crate) fn least_resolved(solutions: &[&Solution], share: f64) -> Vec<BTreeSet<(u8, [u32; 3])>> {
    let surplus = |body: usize, element: usize| solutions.get(body).and_then(|solution| solution.surplus.get(element)).copied().unwrap_or(f64::NAN);
    let mut order: Vec<(usize, usize)> =
        solutions.iter().enumerate().flat_map(|(body, solution)| (0..solution.surplus.len()).map(move |element| (body, element))).collect();
    // Largest first, the lower body and element first among equals.
    order.sort_by(|a, b| surplus(b.0, b.1).total_cmp(&surplus(a.0, a.1)).then(a.cmp(b)));
    let total: f64 = solutions.iter().map(|solution| solution.surplus.iter().filter(|s| s.is_finite()).sum::<f64>()).sum();
    let mut marked = vec![BTreeSet::new(); solutions.len()];
    let mut sum = 0.0;
    for (body, element) in order {
        let own = surplus(body, element);
        if !(own.is_finite() && own > 0.0) {
            continue;
        }
        if !(sum < share * total) {
            break;
        }
        sum += own;
        let Some(solution) = solutions.get(body) else { continue };
        let Some(cell) = solution.space.elements.get(element).and_then(|node| solution.tree.nodes.get(*node as usize)) else { continue };
        if usize::from(cell.level) >= MAX_REFINE {
            continue;
        }
        if let Some(cells) = marked.get_mut(body) {
            for up in 0..=cell.level {
                cells.insert((cell.level - up, cell.ijk.map(|i| i >> up)));
            }
        }
    }
    marked
}

impl Solution {
    fn place(&self) -> Place<'_> {
        Place { grid: &self.grid, tree: &self.tree, space: &self.space, split: &self.split }
    }

    /// The element that holds `x`, its natural coordinates there and the size of its cell.
    fn locate(&self, x: [f64; 3]) -> Option<(usize, [f64; 3], f64)> {
        self.place().locate_in(x, true)
    }

    /// [`Solution::locate`]; with `island` off, the first element of the cell that holds `x`,
    /// whichever of the cell's islands the point is in: enough for what the cell alone says.
    fn locate_in(&self, x: [f64; 3], island: bool) -> Option<(usize, [f64; 3], f64)> {
        self.place().locate_in(x, island)
    }

    /// The displacement and its gradient ∂uᵢ/∂xⱼ at `x`: the displacement of the cell that holds
    /// the point, and the gradient of that cell's polynomial or, in a cell that holds little
    /// material, the weighted mean of the gradients of the polynomials around the point
    /// ([`Solution::stress`]).
    fn field(&self, x: [f64; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
        let (element, xi, size) = self.locate(x)?;
        let (u, own) = cell::field(&self.shapes, xi, size, &self.space.local(element, &self.u));
        let readers = self.readers(element, x);
        let least = readers.iter().map(|reader| reader.estimate).fold(f64::INFINITY, f64::min);
        let (mut sum, mut total, mut count) = ([[0.0; 3]; 3], 0.0, 0);
        for reader in &readers {
            // The weight of an estimate with this error among independent ones, over the largest.
            let weight = (least / reader.estimate) * (least / reader.estimate);
            if !(weight >= LEAST_WEIGHT) {
                continue;
            }
            let gradient = if reader.element == element {
                own
            } else {
                cell::field(&self.shapes, reader.xi, reader.size, &self.space.local(reader.element, &self.u)).1
            };
            for i in 0..3 {
                for j in 0..3 {
                    sum[i][j] += weight * gradient[i][j];
                }
            }
            total += weight;
            count += 1;
        }
        // One estimate is itself, to the last bit.
        Some((u, if count > 1 { sum.map(|row| row.map(|value| value / total)) } else { own }))
    }

    /// The polynomials that give the gradient at `x`, a point of `element`, when the element is a
    /// sliver ([`SLIVER`]): its own first, then those of the leaves around it, each with its
    /// estimated error ([`Solution::stress`]). None for any other element, whose own is used.
    fn readers(&self, element: usize, x: [f64; 3]) -> Vec<Reader> {
        let mut readers = Vec::new();
        if self.fractions.get(element).copied().unwrap_or(1.0) >= SLIVER {
            return readers;
        }
        let Some((node, cell)) = self.space.elements.get(element).and_then(|node| Some((*node as usize, self.tree.nodes.get(*node as usize)?)))
        else {
            return readers;
        };
        let t = [0, 1, 2].map(|a| (x[a] - self.grid.origin[a]) / self.grid.spacing);
        let reader = |node: usize, island: usize| -> Option<Reader> {
            let other = self.tree.nodes.get(node)?;
            let element = self.space.element(node, island)?;
            let fraction = self.fractions.get(element).copied().filter(|fraction| *fraction > 0.0)?;
            let (middle, half) = self.material.get(element).copied()?;
            let steps = (1u64 << other.level) as f64;
            let xi = [0, 1, 2].map(|a| 2.0 * (t[a] * steps - other.ijk[a] as f64) - 1.0);
            // How far a polynomial bounded on the cell's material can grow to the point, over
            // the root of the fraction of the cell that holds that material.
            let growth: f64 = (0..3).map(|a| chebyshev(self.shapes.p, ((xi[a] - middle[a]).abs() / half[a]).max(1.0))).product();
            Some(Reader { estimate: growth / fraction.sqrt(), element, xi, size: self.grid.step(usize::from(other.level)) })
        };
        readers.extend(reader(node, self.space.island(element)));
        // Where pieces of material were told apart, only the elements this one's material goes
        // on into, across a few faces at most, know its stress: the other side of a gap does not.
        let joined = self.near.get(element);
        let (level, at) = (usize::from(cell.level), cell.ijk.map(i64::from));
        for around in 0..27i64 {
            let beside = [at[0] + around % 3 - 1, at[1] + (around / 3) % 3 - 1, at[2] + around / 9 - 1];
            let Some(found) = self.tree.leaf_near(level, beside, t) else { continue };
            // One island where nothing was told apart; `Space::element` ends the others.
            for island in 0..usize::from(u16::MAX) {
                let Some(other) = self.space.element(found, island) else { break };
                let told_apart = joined.is_some_and(|joined| u32::try_from(other).is_ok_and(|other| joined.binary_search(&other).is_err()));
                if !told_apart
                    && let Some(found) = reader(found, island)
                    && readers.iter().all(|reader| reader.element != found.element)
                {
                    readers.push(found);
                }
                if joined.is_none() {
                    break;
                }
            }
        }
        readers
    }

    /// The fraction of its volume that the cell that holds a point of the part holds in material
    /// (with refinement, the cell that is not divided): 1 for a cell wholly inside the part.
    /// The polynomial of a cell that holds little is the least accurate ([`Solution::stress`]).
    pub fn material_fraction(&self, x: [f64; 3]) -> Option<f64> {
        let (element, _, _) = self.locate(x)?;
        self.fractions.get(element).copied()
    }

    /// The size of the cell that holds a point of the part (with refinement, the cell that is
    /// not divided). The solution is one polynomial over that length, so its stress is read
    /// well at points this far apart over the degree.
    pub fn cell_size(&self, x: [f64; 3]) -> Option<f64> {
        self.locate_in(x, false).map(|(_, _, size)| size)
    }

    /// The displacement at a point of the part; `None` outside the cells that hold material
    /// (with refinement, the cells that are not divided). In the part of a cut cell outside the
    /// material it is the cell's polynomial continued.
    pub fn displacement(&self, x: [f64; 3]) -> Option<[f64; 3]> {
        self.field(x).map(|(u, _)| u)
    }

    /// The stress at a point of the part, in the order xx, yy, zz, xy, xz, yz.
    ///
    /// **In a cell that holds little material.** The stress is the derivative of a cell's
    /// polynomial, and the polynomial of a cut cell is fitted to the material the cell holds. A
    /// cell that holds the fraction η of its volume takes its share of the solution's error on
    /// that much material, so its stress is off by about 1/√η times what a full cell's is (the
    /// same energy of the error on η of the volume), and it has to meet its neighbours'
    /// polynomials on faces that lie almost wholly outside the part. The stress in such a cell
    /// is noisy: on the bracket of `tests/bracket.rs` it is up to a quarter off in a cell that
    /// is one per cent full, between neighbours that are within a few per cent.
    ///
    /// The polynomials of the cells around know the stress there too. Continued beyond the
    /// material it is fitted to, a polynomial of degree p that is within a bound on an interval
    /// is within Tₚ(d) times the bound at d half lengths of the interval from its middle, Tₚ
    /// being Chebyshev's polynomial, which grows fastest of all (T. J. Rivlin, *The Chebyshev
    /// Polynomials*, Wiley 1974). So in a cell that holds less than a tenth of its volume in
    /// material the stress at a point is not read from the cell's polynomial alone: every cell
    /// around the point gives an estimate, with an error of about Π Tₚ(dₐ) / √η of its own (dₐ
    /// along each axis, from the middle of the cell's material in half extents of it, at least
    /// 1), and the stress is the mean of the estimates, each weighted by one over the square of
    /// that error, as independent estimates are. The displacement is the cell's own, and so is
    /// the stress in every cell that holds more. Nothing of this enters the solution: it is how
    /// the stress is read from it.
    ///
    /// The tenth is measured. With the limit at 0, 0.02, 0.05, 0.1, 0.2, 0.4 and 1:
    /// - on the bracket (8 mm cells divided twice at degree 3, the fillet read every half
    ///   millimetre at a chordal tolerance of 0.0877 mm, six grid positions) the largest stress
    ///   on the fillet is 17 % above the reference up to 0.05, read in a cell 7 % full, and
    ///   6.7 % above it from 0.1 on, in a cell half full; divided once at degree 4 it is 7.6 %
    ///   above at 0 and 3.6 % from 0.02 on;
    /// - on the tube of `tests/lame.rs` (the largest stress error over 25 points, twenty grids
    ///   and degrees) the error is the same or lower up to 0.05 but for one grid (0.64 % to
    ///   0.79 %), rises on one more at 0.1 (0.14 % to 0.25 %), and from 0.2 on the coarsest
    ///   grids, whose cells are 11 % full and more, are read far worse (0.65 % to 7.7 %);
    /// - a block in tension cut to slivers a millionth of a cell thick (`tests/patch.rs`) has
    ///   its stress to 2e-8 at any limit above that, and to 1e-2 without.
    ///
    /// So cells up to 7 % full were seen to be noisy, and cells of 11 % read better alone.
    /// Where the cells resolve the stress, a sliver's own polynomial is as good as its
    /// neighbours' and the mean costs a little: on Kirsch's quarter plate with divided cells
    /// (`tests/benchmarks.rs`) the stresses at the two ends of the hole, which lie in slivers
    /// on the planes of symmetry, move from within 0.6 % of the exact values to within 1.7 %.
    pub fn stress(&self, x: [f64; 3]) -> Option<[f64; 6]> {
        self.at(x).map(|(_, stress)| stress)
    }

    /// The displacement and the stress at a point of the part ([`Solution::displacement`],
    /// [`Solution::stress`]), read together: half the work of asking for each.
    pub fn at(&self, x: [f64; 3]) -> Option<([f64; 3], [f64; 6])> {
        let (u, g) = self.field(x)?;
        let (lambda, mu) = (self.lame.lambda, self.lame.mu);
        let dilatation = lambda * (g[0][0] + g[1][1] + g[2][2]);
        let stress = [
            dilatation + 2.0 * mu * g[0][0],
            dilatation + 2.0 * mu * g[1][1],
            dilatation + 2.0 * mu * g[2][2],
            mu * (g[0][1] + g[1][0]),
            mu * (g[0][2] + g[2][0]),
            mu * (g[1][2] + g[2][1]),
        ];
        Some((u, stress))
    }

    /// The von Mises stress at a point of the part.
    pub fn von_mises(&self, x: [f64; 3]) -> Option<f64> {
        self.stress(x).map(|s| fieldcraft_fem::von_mises(&s))
    }

    /// How far this solution is from a coarser one of the same step (the same part, material,
    /// supports and loads; usually one degree lower on the same grid): the relative change of
    /// the strain energy and of the von Mises stress at `points` (the points of interest, for
    /// example the vertices of the part's surface).
    ///
    /// This is the p-extension's own check (B. Szabó, I. Babuška, *Finite Element Analysis*,
    /// Wiley 1991: a sequence of solutions of rising degree on one mesh). For Galerkin solutions
    /// in nested spaces under prescribed loads the energy of their difference is twice the
    /// difference of their strain energies, so the square root of [`Change::strain_energy`] is
    /// the distance between the two solutions in the energy norm, relative to the solution.
    /// Here that holds approximately (the supports are imposed weakly and the cells' volume
    /// outside the part has a small stiffness). The numbers tell how much the last refinement
    /// still moved the result. They are an indicator, not a bound: a small change is necessary
    /// for a converged result and does not prove one, and a stress at a sharp re-entrant edge
    /// never settles. A ratio whose denominator is zero is zero when nothing changed and
    /// infinite otherwise.
    pub fn change_from(&self, coarser: &Solution, points: &[[f64; 3]]) -> Change {
        Solution::change_of(&[(self, coarser, points)])
    }

    /// [`Solution::change_from`] of several bodies solved together: each body's solution, its
    /// coarser one and its points. The strain energies are those of all the bodies, and the
    /// peak is the largest von Mises stress in any of them.
    pub fn change_of(bodies: &[(&Solution, &Solution, &[[f64; 3]])]) -> Change {
        let ratio = |change: f64, of: f64| {
            if of != 0.0 {
                change / of
            } else if change == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        };
        let (mut peak, mut peak_coarser, mut apart, mut count) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        let (mut energy, mut energy_coarser) = (0.0, 0.0);
        for (finer, coarser, points) in bodies {
            energy += finer.strain_energy;
            energy_coarser += coarser.strain_energy;
            for x in *points {
                let (Some(fine), Some(coarse)) = (finer.von_mises(*x), coarser.von_mises(*x)) else { continue };
                if !(fine.is_finite() && coarse.is_finite()) {
                    continue;
                }
                peak = peak.max(fine);
                peak_coarser = peak_coarser.max(coarse);
                apart = apart.max((fine - coarse).abs());
                count += 1;
            }
        }
        Change {
            strain_energy: ratio(energy - energy_coarser, energy),
            peak_von_mises: ratio(peak - peak_coarser, peak),
            von_mises: ratio(apart, peak),
            points: count,
        }
    }
}

/// What each face carries, and the body force per volume.
fn conditions(
    supports: &[Support],
    loads: &[Load],
    areas: &BTreeMap<u32, f64>,
    density: f64,
    ties: Option<&Ties>,
) -> Result<(BTreeMap<u32, Face>, [f64; 3])> {
    let area = |face: u32| areas.get(&face).copied().filter(|a| *a > 0.0).ok_or_else(|| Error::Invalid(format!("the surface has no face {face}")));
    let finite = |what: &str, values: &[f64]| {
        if values.iter().all(|x| x.is_finite()) { Ok(()) } else { Err(Error::Invalid(format!("{what} is not finite"))) }
    };
    let mut faces: BTreeMap<u32, Face> = BTreeMap::new();
    let mut body = [0.0; 3];
    for support in supports {
        area(support.face)?;
        finite("a prescribed displacement", &support.value)?;
        if !support.fixed.iter().any(|f| *f) {
            return Err(Error::Invalid(format!("the support of face {} fixes no component", support.face)));
        }
        let face = faces.entry(support.face).or_default();
        for i in 0..3 {
            if !support.fixed[i] {
                continue;
            }
            if face.fixed[i] {
                return Err(Error::Invalid(format!("component {} of face {} is prescribed twice", i + 1, support.face)));
            }
            face.fixed[i] = true;
            face.value[i] = support.value[i];
        }
    }
    for load in loads {
        match *load {
            Load::Pressure { face, pressure } => {
                area(face)?;
                finite("a pressure", &[pressure])?;
                faces.entry(face).or_default().pressure += pressure;
            }
            Load::Traction { face, traction } => {
                area(face)?;
                finite("a traction", &traction)?;
                let entry = faces.entry(face).or_default();
                for i in 0..3 {
                    entry.traction[i] += traction[i];
                }
            }
            Load::Force { face, force } => {
                let area = area(face)?;
                finite("a force", &force)?;
                let entry = faces.entry(face).or_default();
                for i in 0..3 {
                    entry.traction[i] += force[i] / area;
                }
            }
            Load::Gravity { acceleration } => {
                finite("an acceleration", &acceleration)?;
                for i in 0..3 {
                    body[i] += density * acceleration[i];
                }
            }
        }
    }
    // A face tied to another body is held to it where the two touch.
    for tag in ties.into_iter().flat_map(|ties| ties.faces) {
        area(*tag)?;
        let face = faces.entry(*tag).or_default();
        if face.supported() {
            return Err(Error::Invalid(format!("face {tag} is supported and connected to another body: one or the other")));
        }
        face.fixed = [true; 3];
        face.tied = true;
    }
    if !faces.values().any(Face::supported) && !ties.is_some_and(|ties| ties.held) {
        return Err(Error::Invalid("the part has no support: prescribe the displacement of at least one face".into()));
    }
    Ok((faces, body))
}

/// Faces between an element and one that reads its stress with it ([`Solution::stress`]): the
/// cells around a cell, which share a face, an edge or a vertex with it, are three faces away
/// at most.
const NEAR_LINKS: usize = 3;

/// A cell's polynomial as an estimate of the gradient at a point ([`Solution::stress`]).
struct Reader {
    /// How large its error is estimated to be, against a cell wholly inside the part at a
    /// point of its own.
    estimate: f64,
    element: usize,
    /// The point in the cell's natural coordinates: outside [−1, 1] for a neighbour's.
    xi: [f64; 3],
    /// The size of the cell.
    size: f64,
}

/// Chebyshev's polynomial Tₚ at `x` ≥ 1, by its recurrence Tₖ₊₁ = 2x Tₖ − Tₖ₋₁.
fn chebyshev(p: usize, x: f64) -> f64 {
    let (mut before, mut value) = (1.0, x);
    for _ in 1..p {
        (before, value) = (value, 2.0 * x * value - before);
    }
    value
}

/// What makes a model smaller, for the message of a limit.
pub(crate) fn smaller(refine: usize) -> &'static str {
    if refine > 0 { "use a larger cell size, a lower degree or fewer levels of refinement" } else { "use a larger cell size or a lower degree" }
}

pub(crate) fn factor_error(e: FactorError, refine: usize) -> Error {
    match e {
        FactorError::NotPositiveDefinite { column } | FactorError::Singular { column } => Error::Singular(format!(
            "the stiffness is singular to working precision (equation {column}): the supports leave a rigid motion of the part free, or a cell holds so little material that the fictitious stiffness no longer determines its unknowns"
        )),
        FactorError::TooLarge { entries, limit } => {
            Error::Limit(format!("the factorization needs {entries} entries; the limit is {limit} ({})", smaller(refine)))
        }
        FactorError::OutOfMemory { entries } => Error::Limit(format!("no memory for a factorization of {entries} entries")),
        FactorError::Internal(message) => Error::Internal(message),
        FactorError::Cancelled => Error::Invalid("cancelled".into()),
    }
}

/// What an element contributes.
struct Integrated {
    /// Its stiffness and load on its functions; `None` for a cell wholly inside the part with
    /// its own modes, which has the matrices of its level ([`Whole`]).
    stiffness: Option<Vec<f64>>,
    load: Option<Vec<f64>>,
    cut: Option<Cut>,
}

/// What an element of a cut cell leaves behind.
struct Cut {
    nitsche: f64,
    volume: f64,
    points: usize,
}

/// The stiffness and the body load of the cells of one level that are wholly inside the part,
/// on their own modes.
struct Whole {
    stiffness: Vec<f64>,
    load: Vec<f64>,
}

/// Which cells a test divides, in place of the surface's curvature: asked for every cell that
/// holds material with its level, its lowest corner and its size.
type Marker<'a> = &'a (dyn Fn(usize, [f64; 3], f64) -> bool + Sync);

/// Solve one linear static step of the part bounded by `surface` (closed, outward-oriented, in
/// SI units, its triangles tagged with their faces).
///
/// Fails with [`Error::Invalid`] on unusable input, [`Error::Limit`] when the grid, the system or
/// its factorization would be too large, and [`Error::Singular`] when the supports do not hold
/// the part.
pub fn solve(
    surface: &BoundarySurface,
    material: &Material,
    supports: &[Support],
    loads: &[Load],
    options: &Options,
    limits: &Limits,
) -> Result<Solution> {
    let mut next = Round::default();
    let alone = alone(limits);
    let (mut solution, used) = solve_marked(surface, material, supports, loads, options, limits, None, &next, &alone)?;
    // The rounds that follow the solution ([`Options::adapt`]): each divides the cells the last
    // solution is least resolved in and solves again, as the first solve was solved in the end
    // (with the levels where gaps end that fitted, with the pieces joined if they had to be).
    for round in 0..options.adapt.min(MAX_REFINE) {
        let cells = least_resolved(&[&solution], Options::SHARE).pop().unwrap_or_default();
        if !next.ask(cells, &solution) {
            break;
        }
        match solve_raising(surface, material, supports, loads, &used, limits, None, &next, &alone) {
            Ok(mut finer) => {
                finer.stats.rounds = round + 1;
                finer.stats.joined = solution.stats.joined;
                finer.stats.settled =
                    if finer.strain_energy != 0.0 { (finer.strain_energy - solution.strain_energy) / finer.strain_energy } else { 0.0 };
                let settled = finer.stats.settled.abs().sqrt() <= options.settle;
                solution = finer;
                if settled {
                    break;
                }
            }
            // Too large with this round's cells, or not to be solved with them: the solution
            // before it stands.
            Err(Error::Limit(_) | Error::Singular(_)) => {
                solution.stats.left_out = true;
                break;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(solution)
}

/// The system of one body alone ([`Join`]).
fn alone(limits: &Limits) -> impl Fn(&View) -> Result<Solved> + '_ {
    |view: &View| {
        crate::assembly::solve_system(&[view], &|_, _, _| None, limits)?.pop().ok_or_else(|| Error::Internal("the system has no solution".into()))
    }
}

/// A surface as far as what is kept of it tells it from another: its points and triangles,
/// counted, and its first and last point.
type SurfaceKey = (usize, usize, [u64; 3], [u64; 3]);

/// What a solve works out of a body's surface alone: the same on every grid.
struct OfSurface {
    of: SurfaceKey,
    solid: Solid,
    feature: Vec<[bool; 3]>,
}

/// What a solve works out of a body's surface on a grid: the same in every round on it.
struct OnGrid {
    of: SurfaceKey,
    grid: Grid,
    finest: usize,
    /// The triangles' corners on the planes of the finest cells, their faces, their unit
    /// normals and the area of each face.
    triangles: Vec<[[f64; 3]; 3]>,
    tags: Vec<u32>,
    normals: Vec<[f64; 3]>,
    areas: BTreeMap<u32, f64>,
    /// The surface in the grid's cells, and what each cell is.
    sorted: grid::Sorted,
    kinds: Vec<Kind>,
}

/// The cut cells of a solve, by level and place, each with its integrals
/// ([`Options::reuse`]), and what they were integrated of: the surface, the grid and its finest
/// cells, the degree and the depth of the quadrature, and whether the pieces of material were
/// told apart ([`Options::separate`]) and integrated exactly ([`Options::exact`]).
#[derive(Default)]
struct Cuts {
    of: Option<CutsOf>,
    cells: BTreeMap<(u8, [u32; 3]), Arc<CutLeaf>>,
}

type CutsOf = (SurfaceKey, Grid, [usize; 3], [bool; 2]);

/// What the solves of one body keep for each other: the rounds of one solve
/// ([`Options::adapt`]), and a count of the unknowns and the solve that follows it
/// ([`crate::assembly::Kept`]). Nothing in it changes a result: a solve takes from it what it
/// would work out again, the same.
#[derive(Default)]
pub(crate) struct Memory {
    of_surface: Mutex<Option<Arc<OfSurface>>>,
    on_grid: Mutex<Option<Arc<OnGrid>>>,
    cuts: Mutex<Cuts>,
}

fn in_use<T>(_: T) -> Error {
    Error::Internal("what a body's solves keep is in use".into())
}

impl Memory {
    /// What the surface alone gives: kept, or worked out and kept.
    fn of_surface(&self, surface: &BoundarySurface) -> Result<Arc<OfSurface>> {
        let bits = |point: Option<&[f64; 3]>| point.map_or([0; 3], |point| point.map(f64::to_bits));
        let of = (surface.points().len(), surface.triangles().len(), bits(surface.points().first()), bits(surface.points().last()));
        let mut kept = self.of_surface.lock().map_err(in_use)?;
        if let Some(found) = kept.as_ref().filter(|found| found.of == of) {
            return Ok(Arc::clone(found));
        }
        let found = Arc::new(OfSurface { of, solid: Solid::new(surface)?, feature: grid::feature_flags(surface) });
        *kept = Some(Arc::clone(&found));
        Ok(found)
    }

    /// What the surface gives on `grid`, with the finest cells there may be `finest` levels
    /// down: kept, or worked out and kept.
    fn on_grid(&self, surface: &BoundarySurface, of_surface: &OfSurface, grid: &Grid, finest: usize) -> Result<Arc<OnGrid>> {
        let mut kept = self.on_grid.lock().map_err(in_use)?;
        if let Some(found) = kept.as_ref().filter(|found| found.of == of_surface.of && found.grid == *grid && found.finest == finest) {
            return Ok(Arc::clone(found));
        }
        let vertices = surface.points();
        // On the planes of the finest cells there may be: a face of the part that lies on one
        // lies on it exactly.
        let triangles = fieldcraft_parallel::map_grained(surface.triangles().len(), GRAIN, |t| -> Result<[[f64; 3]; 3]> {
            let corner = |v: u32| {
                vertices
                    .get(v as usize)
                    .map(|x| grid.snap_to(*x, finest))
                    .ok_or_else(|| Error::Internal("a surface triangle names a missing vertex".into()))
            };
            let t = surface.triangles().get(t).ok_or_else(|| Error::Internal("a surface triangle is missing".into()))?;
            Ok([corner(t[0])?, corner(t[1])?, corner(t[2])?])
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        let tags: Vec<u32> = (0..triangles.len()).map(|t| surface.tri_face().get(t).copied().unwrap_or(0)).collect();
        let mut normals = Vec::with_capacity(triangles.len());
        let mut areas: BTreeMap<u32, f64> = BTreeMap::new();
        for (t, tag) in triangles.iter().zip(&tags) {
            let normal = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            let length = dot(normal, normal).sqrt();
            normals.push(if length > 0.0 { [normal[0] / length, normal[1] / length, normal[2] / length] } else { [0.0; 3] });
            *areas.entry(*tag).or_insert(0.0) += 0.5 * length;
        }
        // The cells: cut where they own a piece of the surface, else by their centre.
        let sorted = grid::sort_surface(grid, &triangles);
        let kinds = fieldcraft_parallel::map_grained(grid.len(), GRAIN, |c| -> Result<Kind> {
            if sorted.pieces.get(c).is_some_and(|pieces| !pieces.is_empty()) {
                return Ok(Kind::Cut);
            }
            let ijk = grid.coords(c);
            let (a, b) = (grid.lo(ijk), grid.hi(ijk));
            Ok(if of_surface.solid.contains([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])])? { Kind::Inside } else { Kind::Outside })
        })
        .into_iter()
        .collect::<Result<Vec<Kind>>>()?;
        let found = Arc::new(OnGrid { of: of_surface.of, grid: *grid, finest, triangles, tags, normals, areas, sorted, kinds });
        *kept = Some(Arc::clone(&found));
        Ok(found)
    }
}

/// What a solve of [`solve`] takes from the solves before it ([`Options::adapt`]).
#[derive(Default)]
pub(crate) struct Round {
    /// The cells to divide besides those the surface's curvature names, by level and place.
    asked: BTreeSet<(u8, [u32; 3])>,
    /// The entries of the stiffness and of its factor in the solve before, when there was one.
    before: Option<(usize, usize)>,
    /// What the body's solves keep for each other.
    memory: Arc<Memory>,
    /// Whether the unknowns are only counted: they are not put in an order then.
    counting: bool,
}

impl Round {
    /// The first solve of a body whose solves keep `memory`; to count its unknowns only, or
    /// to solve for them.
    pub(crate) fn first(memory: Arc<Memory>, counting: bool) -> Round {
        Round { memory, counting, ..Round::default() }
    }

    /// Ask the next solve to divide `cells` as well, after `solution`, the solve before it;
    /// whether any of them was not asked for already.
    pub(crate) fn ask(&mut self, cells: BTreeSet<(u8, [u32; 3])>, solution: &Solution) -> bool {
        let before = self.asked.len();
        self.asked.extend(cells);
        self.before = Some((solution.stats.stiffness_entries, solution.stats.factor_entries));
        self.asked.len() > before
    }
}

/// One solve of [`solve`] with exactly these options: dividing the cells `marker` names (up to
/// [`Options::refine`] levels) when there is one, and the cells `round` asks for besides. The fictitious stiffness is as small as the factorization allows
/// ([`Options::fictitious`]): the fraction asked for, then a hundred times more at a time up to
/// the one that keeps the margin for this material.
#[allow(clippy::too_many_arguments)]
fn solve_raising(
    surface: &BoundarySurface,
    material: &Material,
    supports: &[Support],
    loads: &[Load],
    options: &Options,
    limits: &Limits,
    marker: Option<Marker>,
    round: &Round,
    join: Join,
) -> Result<Solution> {
    options.check()?;
    let most = most_fictitious(material)?;
    let mut fictitious = options.fictitious;
    // Each step multiplies the fraction, so the loop ends.
    loop {
        match solve_at(surface, material, supports, loads, options, limits, fictitious, marker, round, None, join) {
            Err(Error::Singular(_)) if fictitious < most => fictitious = (fictitious * FICTITIOUS_STEP).min(most),
            result => return result,
        }
    }
}

/// The first solve of [`solve`], with what it falls back on when the step cannot be solved as
/// asked, and the options it was solved with in the end.
#[allow(clippy::too_many_arguments)]
fn solve_marked(
    surface: &BoundarySurface,
    material: &Material,
    supports: &[Support],
    loads: &[Load],
    options: &Options,
    limits: &Limits,
    marker: Option<Marker>,
    round: &Round,
    join: Join,
) -> Result<(Solution, Options)> {
    let attempt = |options: &Options| solve_raising(surface, material, supports, loads, options, limits, marker, round, join);
    // The cells divided where gaps end may make the system too large ([`Options::gaps`]): then
    // with a level less of them, down to none.
    let mut fewer = *options;
    let first = loop {
        match attempt(&fewer) {
            Err(Error::Limit(_)) if fewer.separate && fewer.gaps > 0 => fewer.gaps -= 1,
            result => break result,
        }
    };
    match first {
        // Told apart, a piece of material may be held by nothing ([`Options::separate`]): the
        // step with the pieces joined, when that one can be solved.
        Err(Error::Singular(message)) if options.separate => {
            let joined = Options { separate: false, ..*options };
            match attempt(&joined) {
                Ok(mut solution) => {
                    solution.stats.joined = true;
                    Ok((solution, joined))
                }
                Err(Error::Singular(_)) => Err(Error::Singular(message)),
                Err(other) => Err(other),
            }
        }
        result => result.map(|solution| (solution, fewer)),
    }
}

/// The largest fictitious stiffness a material leaves room for ([`Options::fictitious`]), as a
/// fraction of its own.
pub(crate) fn most_fictitious(material: &Material) -> Result<f64> {
    Ok(margin_fraction(&material.lame()?).min(1.0))
}

/// One body of several ([`crate::assembly`]): one solve of it with exactly these options and
/// this fictitious stiffness, dividing the cells `round` asks for besides, its faces tied as
/// `ties` say, its equations handed to `join`. The rounds of [`Options::adapt`] are the
/// caller's to run, over all the bodies together.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_tied(
    surface: &BoundarySurface,
    material: &Material,
    supports: &[Support],
    loads: &[Load],
    options: &Options,
    limits: &Limits,
    fictitious: f64,
    round: &Round,
    ties: &Ties,
    join: Join,
) -> Result<Solution> {
    options.check()?;
    solve_at(surface, material, supports, loads, options, limits, fictitious, None, round, Some(ties), join)
}

/// How much the fictitious stiffness is raised at a time when the system is singular to working
/// precision: the two decades of margin [`Options::FICTITIOUS`] has over the pivot tolerance.
pub(crate) const FICTITIOUS_STEP: f64 = 100.0;

/// Divide the cells of `tree` that are too coarse, level by level up to `levels`
/// ([`Options::refine`]): the cut cells across which the surface turns by more than `turn` and
/// the cells around them, or the cells `marker` names; and, to any level, the cells `asked` for
/// by level and place ([`Options::adapt`]), without the cells around them. `touching` and
/// `pieces` hold the surface in every cell of the tree ([`grid::Sorted`]) and get those of the
/// new cells.
#[allow(clippy::too_many_arguments)]
fn divide_cells(
    grid: &Grid,
    tree: &mut Tree,
    surface: &Triangles,
    solid: &Solid,
    touching: &mut Vec<Vec<u32>>,
    pieces: &mut Vec<Vec<Piece>>,
    levels: usize,
    turn: f64,
    marker: Option<Marker>,
    asked: &BTreeSet<(u8, [u32; 3])>,
) -> Result<()> {
    let curvature = if marker.is_none() { grid::curvatures(surface) } else { Vec::new() };
    let base = tree.nodes.len();
    // The cells of the level being looked at: the ones the level before made.
    let mut first = 0;
    let deepest_asked = asked.iter().map(|(level, _)| usize::from(*level) + 1).max().unwrap_or(0);
    for level in 0..levels.max(deepest_asked).min(MAX_REFINE) {
        let end = tree.nodes.len();
        let size = grid.step(level);
        let leaf = |node: usize| tree.nodes.get(node).filter(|cell| cell.is_leaf() && cell.material());
        let marked: Vec<bool> = (first..end)
            .map(|node| {
                let Some(cell) = leaf(node) else { return false };
                match marker {
                    Some(marker) => marker(level, grid.fine_lo(level, cell.position()), size),
                    None => {
                        let curved = |piece: &Piece| curvature.get(piece.triangle as usize).copied().unwrap_or(0.0);
                        level < levels
                            && cell.kind == Kind::Cut
                            && size * pieces.get(node).map_or(0.0, |owned| owned.iter().map(curved).fold(0.0, f64::max)) > turn * (1.0 + AT_LIMIT)
                    }
                }
            })
            .collect();
        let is_marked = |node: usize| node >= first && marked.get(node - first).copied().unwrap_or(false);
        let divided: Vec<usize> = (first..end)
            .filter(|&node| {
                let Some(cell) = leaf(node) else { return false };
                if is_marked(node) || asked.contains(&(cell.level, cell.ijk)) {
                    return true;
                }
                if marker.is_some() {
                    return false;
                }
                // The cells across the faces of a marked one are divided with it.
                let at = cell.ijk.map(i64::from);
                (0..3).any(|axis| {
                    [-1, 1].into_iter().any(|step| {
                        let mut beside = at;
                        beside[axis] += step;
                        tree.find(level, beside).is_some_and(is_marked)
                    })
                })
            })
            .collect();
        if divided.is_empty() {
            break;
        }
        if end - base + 8 * divided.len() > MAX_CELLS {
            return Err(Error::Limit(format!(
                "refinement would make more than {MAX_CELLS} cells (use fewer levels of refinement or a larger cell size)"
            )));
        }
        divide_nodes(grid, tree, surface, solid, touching, pieces, &divided)?;
        first = end;
    }
    Ok(())
}

/// Divide the leaves `divided` of `tree` into eight each, sorting the surface each holds
/// (`touching`, `pieces`) into the new cells.
fn divide_nodes(
    grid: &Grid,
    tree: &mut Tree,
    surface: &Triangles,
    solid: &Solid,
    touching: &mut Vec<Vec<u32>>,
    pieces: &mut Vec<Vec<Piece>>,
    divided: &[usize],
) -> Result<()> {
    let children = fieldcraft_parallel::map(divided.len(), |i| -> Result<Vec<(Kind, Vec<u32>, Vec<Piece>)>> {
        let broken = || Error::Internal("a cell to divide is missing".into());
        let node = *divided.get(i).ok_or_else(broken)?;
        let cell = tree.nodes.get(node).ok_or_else(broken)?;
        let reach = touching.get(node).ok_or_else(broken)?;
        let (level, at) = (usize::from(cell.level), cell.position());
        let finer = grid.step(level + 1);
        (0..8usize)
            .map(|child| {
                let at = [2 * at[0] + (child & 1), 2 * at[1] + ((child >> 1) & 1), 2 * at[2] + ((child >> 2) & 1)];
                let (lo, hi) = (grid.fine_lo(level + 1, at), grid.fine_hi(level + 1, at));
                let (reach, owned) = grid::sort_into_box(surface.corners, reach, lo, hi, finer);
                // A cell that owns a piece is cut; a part of a cell wholly inside is inside;
                // any other lies on one side of the surface, which its centre decides.
                let kind = if !owned.is_empty() {
                    Kind::Cut
                } else if cell.kind == Kind::Inside || solid.contains([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])])? {
                    Kind::Inside
                } else {
                    Kind::Outside
                };
                Ok((kind, reach, owned))
            })
            .collect()
    });
    for (node, children) in divided.iter().zip(children) {
        let children = children?;
        let mut kinds = [Kind::Outside; 8];
        for (kind, child) in kinds.iter_mut().zip(&children) {
            *kind = child.0;
        }
        if children.len() != kinds.len() {
            return Err(Error::Internal("a divided cell has not eight children".into()));
        }
        tree.divide(*node, kinds)?;
        for (_, reach, owned) in children {
            touching.push(reach);
            pieces.push(owned);
        }
        // The surface in a divided cell is its children's.
        if let Some(list) = touching.get_mut(*node) {
            *list = Vec::new();
        }
        if let Some(list) = pieces.get_mut(*node) {
            *list = Vec::new();
        }
    }
    Ok(())
}

/// [`solve`] with the fictitious stiffness `fictitious` in place of the options'.
#[allow(clippy::too_many_arguments)]
fn solve_at(
    surface: &BoundarySurface,
    material: &Material,
    supports: &[Support],
    loads: &[Load],
    options: &Options,
    limits: &Limits,
    fictitious: f64,
    marker: Option<Marker>,
    round: &Round,
    ties: Option<&Ties>,
    join: Join,
) -> Result<Solution> {
    let asked = &round.asked;
    let lame = material.lame()?;
    options.check()?;
    let p = options.degree;
    let shapes =
        Shapes::new(p, local_modes(p, options.basis == Basis::Trunk)).ok_or_else(|| Error::Internal("the cell's modes lack a vertex mode".into()))?;
    let m = shapes.len();
    let n = 3 * m;
    let block_size = (BLOCK_BYTES / (4 * 8 * n * n)).clamp(1, MOST_BLOCK);

    // The surface on the grid.
    let vertices = surface.points();
    if surface.triangles().is_empty() {
        return Err(Error::Invalid("the surface has no triangles".into()));
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for v in vertices {
        for a in 0..3 {
            lo[a] = lo[a].min(v[a]);
            hi[a] = hi[a].max(v[a]);
        }
    }
    let grid = Grid::over(lo, hi, options.cell_size, options.offset, MAX_CELLS)?;
    // The finest cells there can be: divided for the surface's curvature, then by the rounds
    // that follow the solution (each a level at most), then where gaps end. The same in every
    // round, so a cell is the same cell in each.
    let finest = (options.refine + options.adapt + if options.separate { options.gaps } else { 0 }).min(MAX_REFINE);
    // What the surface gives alone, and on this grid: the same in every round, and kept.
    let of_surface = round.memory.of_surface(surface)?;
    let placed = round.memory.on_grid(surface, &of_surface, &grid, finest)?;
    let (triangles, tags, normals, solid) = (&placed.triangles[..], &placed.tags[..], &placed.normals[..], &of_surface.solid);
    let (faces, body) = conditions(supports, loads, &placed.areas, material.density, ties)?;
    let on_surface =
        Surface { normals, tags, faces: &faces, touching: ties.map(|ties| ties.touching), curved: ties.map_or(&cell::NO_CURVES, |ties| ties.curved) };
    let on_grid = Triangles { corners: triangles, normals, across: surface.neighbors(), feature: &of_surface.feature };

    // The cells that hold material, divided where they are too coarse, each with the surface
    // in it.
    let mut tree = Tree::new(&grid, &placed.kinds);
    let (mut touching, mut pieces): (Vec<Vec<u32>>, Vec<Vec<Piece>>) = (Vec::with_capacity(tree.nodes.len()), Vec::with_capacity(tree.nodes.len()));
    for (cell, root) in tree.root.iter().enumerate() {
        if *root != NONE {
            touching.push(placed.sorted.touching.get(cell).cloned().unwrap_or_default());
            pieces.push(placed.sorted.pieces.get(cell).cloned().unwrap_or_default());
        }
    }
    if options.refine > 0 || !asked.is_empty() {
        divide_cells(&grid, &mut tree, &on_grid, solid, &mut touching, &mut pieces, options.refine, options.turn(), marker, asked)?;
    }

    // The cut cells: their quadrature, and the separate pieces of material each holds. Where
    // a gap between such pieces ends, the cells are divided and the new ones integrated
    // ([`Options::gaps`]), round by round.
    let rule = gauss_legendre(p + 1);
    let surface_rule = gauss_legendre(2 * p + 1);
    let (second_rule, third_rule) = (gauss_legendre(2 * p + 1), gauss_legendre(3 * p + 2));
    let mut cuts: Vec<Option<Arc<CutLeaf>>> = Vec::new();
    // The cut cells of the solve before ([`Options::reuse`]), when they were integrated as
    // this one's are, on this grid.
    let cuts_of: CutsOf = (of_surface.of, grid, [finest, p, options.depth], [options.separate, options.exact]);
    let before = std::mem::take(&mut *round.memory.cuts.lock().map_err(in_use)?);
    let reused = (options.reuse && before.of == Some(cuts_of)).then_some(&before.cells);
    let mut links: Vec<[u32; 4]> = Vec::new();
    // What the ends of gaps may cost: as many cells again as hold material, or [`GAP_CELLS`]
    // where that is more.
    let allowed = tree.nodes.iter().filter(|cell| cell.is_leaf() && cell.material()).count().max(GAP_CELLS);
    let (mut added, mut holding) = (0usize, 0usize);
    // A round divides the cells that hold a gap's sides together, the largest first, so cells
    // that are already small beside larger ones take a round more for every level between
    // them: as many rounds as the levels asked for and the levels there are.
    let rounds = if options.gaps == 0 { 0 } else { (options.gaps + tree.deepest()).min(2 * MAX_REFINE) };
    for round in 0..=rounds {
        cuts.resize(tree.nodes.len(), None);
        let mut todo = Vec::new();
        for (node, cell) in tree.nodes.iter().enumerate() {
            let Some(slot) = cuts.get_mut(node).filter(|slot| slot.is_none() && cell.is_leaf() && cell.kind == Kind::Cut) else { continue };
            match reused.and_then(|cells| cells.get(&(cell.level, cell.ijk))) {
                Some(leaf) => *slot = Some(Arc::clone(leaf)),
                None => todo.push(node),
            }
        }
        let integrated = fieldcraft_parallel::map(todo.len(), |index| -> Result<CutLeaf> {
            let broken = || Error::Internal("a cut cell is missing".into());
            let node = *todo.get(index).ok_or_else(broken)?;
            let cell = tree.nodes.get(node).ok_or_else(broken)?;
            let level = usize::from(cell.level);
            let rules = [&rule[..], &second_rule[..], &third_rule[..]];
            islands::integrate(
                &shapes.tables,
                grid.fine_lo(level, cell.position()),
                grid.fine_hi(level, cell.position()),
                grid.step(level),
                touching.get(node).map(Vec::as_slice).unwrap_or_default(),
                pieces.get(node).map(Vec::as_slice).unwrap_or_default(),
                &on_grid,
                solid,
                rules,
                // The octree is as much shallower as the cell is smaller.
                options.depth.saturating_sub(level),
                MAX_CELL_POINTS,
                options.separate,
                options.exact,
            )
        });
        for (node, leaf) in todo.iter().zip(integrated) {
            if let Some(slot) = cuts.get_mut(*node) {
                *slot = Some(Arc::new(leaf?));
            }
        }
        if !options.separate {
            break;
        }
        let held = |node: usize| -> Option<Held> {
            let cell = tree.nodes.get(node)?;
            let level = usize::from(cell.level);
            Some(Held {
                lo: grid.fine_lo(level, cell.position()),
                hi: grid.fine_hi(level, cell.position()),
                pieces: pieces.get(node).map(Vec::as_slice).unwrap_or_default(),
                of_piece: cuts.get(node).and_then(|slot| slot.as_deref()).map(|leaf| leaf.of_piece.as_slice()).unwrap_or_default(),
            })
        };
        links = islands::links(&tree, solid, &held)?;
        // A gap wider than half a cell of the grid is none that a cell is divided for: the
        // largest cell is more than [`Options::GAP`] widths across only of a narrower one.
        const _: () = assert!(Options::GAP >= 2.0);
        let sides = islands::sides(&tree, &links, &held, 0.5 * grid.spacing);
        let islands_of = |node: u32| cuts.get(node as usize).and_then(|slot| slot.as_deref()).map_or(1, CutLeaf::islands);
        let divided = crate::space::holding(&tree, &islands_of, &links, &sides, Options::GAP, &|level| grid.step(level));
        if divided.is_empty() {
            break;
        }
        // Each divided cell is seven cells more at most. Past the levels allowed, or past what
        // the ends of gaps may cost, the cells that are left hold their gaps' sides together:
        // the ones that do so over more than twice the widths asked for are counted.
        let more = 7 * divided.len();
        if round == rounds || added + more > allowed || tree.nodes.len() + 8 * divided.len() > 2 * MAX_CELLS {
            holding = crate::space::holding(&tree, &islands_of, &links, &sides, 2.0 * Options::GAP, &|level| grid.step(level)).len();
            break;
        }
        added += more;
        divide_nodes(&grid, &mut tree, &on_grid, solid, &mut touching, &mut pieces, &divided)?;
        for node in divided {
            if let Some(slot) = cuts.get_mut(node) {
                *slot = None;
            }
        }
    }
    cuts.resize(tree.nodes.len(), None);
    drop(before);
    // For the solve after this one.
    if options.reuse {
        let cells = (tree.nodes.iter().zip(&cuts)).filter_map(|(cell, leaf)| Some(((cell.level, cell.ijk), Arc::clone(leaf.as_ref()?)))).collect();
        *round.memory.cuts.lock().map_err(in_use)? = Cuts { of: Some(cuts_of), cells };
    }
    let cut_of = |node: usize| cuts.get(node).and_then(|slot| slot.as_deref());
    // The pieces of surface an island of a cell owns: all of the cell's where it has one island.
    let pieces_of = |node: usize, island: usize| -> Cow<[Piece]> {
        let all = pieces.get(node).map(Vec::as_slice).unwrap_or_default();
        match cut_of(node).map(|leaf| leaf.of_piece.as_slice()).filter(|of_piece| !of_piece.is_empty()) {
            Some(of_piece) => Cow::Owned(all.iter().zip(of_piece).filter(|(_, of)| usize::from(**of) == island).map(|(piece, _)| *piece).collect()),
            None => Cow::Borrowed(all),
        }
    };

    // The unknowns, numbered in a fill-reducing order: every island of a cell an element, and
    // the functions told apart that material does not join.
    let mut space = Space::with_islands(
        &tree,
        &shapes.modes,
        p,
        &|node| cut_of(node as usize).map_or(1, CutLeaf::islands),
        options.separate.then_some(&links[..]),
    );
    let separated = space.told_apart;
    let elements = space.elements.len();
    if elements == 0 {
        return Err(Error::Invalid("no cell holds material: the surface must be closed and oriented outward".into()));
    }
    let unknowns = 3 * space.count;
    if unknowns > limits.max_dofs {
        return Err(Error::Limit(format!("{unknowns} unknowns; the limit is {} ({})", limits.max_dofs, smaller(options.refine))));
    }
    // A first refusal before the cells are ordered and integrated, on a count that is not
    // below the blocks of the stiffness (nine entries each): the entries are counted, and held
    // to the limit of the way they are solved, once the system is put together
    // ([`crate::assembly`]).
    let most = if options.solver == Solver::Direct { MAX_MATRIX_ENTRIES } else { MAX_MATRIX_ENTRIES.max(limits.max_factor_entries) };
    let blocks =
        (0..elements).fold(0usize, |sum, element| sum.saturating_add(space.functions(element).len().saturating_mul(space.functions(element).len())));
    if blocks > most {
        return Err(Error::Limit(format!(
            "the stiffness would have {} entries; the limit is {most} ({})",
            blocks.saturating_mul(9),
            smaller(options.refine)
        )));
    }
    // Nested dissection by elements: the functions of a cell all couple, so a separator is
    // the functions two groups of cells share, not a layer of functions through the cells.
    // Not for a count of the unknowns, which is what it is in any order.
    if !round.counting {
        let order = {
            let lists: Vec<&[usize]> = (0..elements).map(|element| space.functions(element)).collect();
            let centres: Vec<[f64; 3]> = lists
                .iter()
                .map(|list| {
                    let mut centre = [0.0; 3];
                    for at in list.iter().filter_map(|mode| space.xyz.get(*mode)) {
                        for a in 0..3 {
                            centre[a] += at[a] / list.len() as f64;
                        }
                    }
                    centre
                })
                .collect();
            element_dissection(space.count, &lists, &centres)
        };
        space.renumber(&order)?;
    }
    // A cell the surface does not cut: every one of a level has the same matrices.
    let mut whole = Vec::with_capacity(m);
    for (z, wz) in &rule {
        for (y, wy) in &rule {
            for (x, wx) in &rule {
                whole.push(QuadPoint { xi: [*x, *y, *z], weight: wx * wy * wz, inside: true });
            }
        }
    }
    let body_load =
        |shape_integrals: &[f64]| -> Vec<f64> { (0..n).map(|row| body[row % 3] * shape_integrals.get(row / 3).copied().unwrap_or(0.0)).collect() };
    let outside_material = fictitious_material(&lame, fictitious);
    let wholes: Vec<Whole> = {
        let mut scratch = Scratch::default();
        let reference = cell::integrate(&shapes, &whole);
        (0..=tree.deepest())
            .map(|level| {
                let size = grid.step(level);
                Whole {
                    stiffness: cell::stiffness(&shapes, size, &reference.inside, &lame, &mut scratch),
                    load: body_load(&cell::body(&shapes, size, &reference.inside, &mut scratch)),
                }
            })
            .collect()
    };

    let missing = || Error::Internal("an element is missing".into());
    let cell_of = |element: usize| -> Result<(usize, &Node, &Whole)> {
        let node = space.elements.get(element).copied().ok_or_else(missing)? as usize;
        let cell = tree.nodes.get(node).ok_or_else(missing)?;
        Ok((node, cell, wholes.get(usize::from(cell.level)).ok_or_else(missing)?))
    };
    let integrate = |element: usize| -> Result<Integrated> {
        let (node, cell, whole) = cell_of(element)?;
        let map = space.map(element);
        match cell.kind {
            Kind::Inside => {
                return Ok(Integrated {
                    stiffness: map.map(|map| map.matrix(m, &whole.stiffness)),
                    load: map.map(|map| map.vector(m, &whole.load)),
                    cut: None,
                });
            }
            // No element holds no material ([`Space`]).
            Kind::Outside => return Err(missing()),
            Kind::Cut => {}
        }
        let level = usize::from(cell.level);
        let size = grid.step(level);
        let corner = grid.fine_lo(level, cell.position());
        let island = space.island(element);
        let leaf = cut_of(node).ok_or_else(missing)?;
        let inside = leaf.inside.get(island).ok_or_else(missing)?;
        let pieces = pieces_of(node, island);
        let mut scratch = Scratch::default();
        let mut stiffness = cell::stiffness(&shapes, size, inside, &lame, &mut scratch);
        // To an island the rest of the cell is outside it: the other islands too.
        let rest;
        let outside = if leaf.inside.len() > 1 {
            let mut sum = leaf.outside.clone();
            leaf.inside.iter().enumerate().filter(|(other, _)| *other != island).for_each(|(_, other)| sum.absorb(other));
            rest = sum;
            &rest
        } else {
            &leaf.outside
        };
        if !outside.is_zero() {
            let outside = cell::stiffness(&shapes, size, outside, &outside_material, &mut scratch);
            for (total, fictitious) in stiffness.iter_mut().zip(&outside) {
                *total += fictitious;
            }
        }
        let mut load = body_load(&cell::body(&shapes, size, inside, &mut scratch));
        let boundary = cell::boundary(&shapes, corner, size, &pieces, &on_surface, &lame, &surface_rule, &mut scratch);
        for (total, surface_load) in load.iter_mut().zip(&boundary.load) {
            *total += surface_load;
        }
        let mut nitsche = 0.0;
        if !boundary.traction.is_empty() {
            // On the cell's own modes: the functions that reach it span no more
            // ([`crate::refine`]).
            nitsche = cell::nitsche_parameter(&shapes, &stiffness, &boundary.traction).ok_or_else(|| {
                Error::Singular(format!(
                    "cell ({}, {}, {}) of level {level} holds so little material that its stiffness is singular to working precision (move the grid with the offset, or raise the fictitious stiffness)",
                    cell.ijk[0], cell.ijk[1], cell.ijk[2]
                ))
            })?;
            for ((total, consistency), penalty) in stiffness.iter_mut().zip(&boundary.consistency).zip(&boundary.penalty) {
                *total += consistency + nitsche * penalty;
            }
            for ((total, consistency), penalty) in load.iter_mut().zip(&boundary.consistency_load).zip(&boundary.penalty_load) {
                *total += consistency + nitsche * penalty;
            }
        }
        if let Some(map) = map {
            stiffness = map.matrix(m, &stiffness);
            load = map.vector(m, &load);
        }
        // The points of a cell are counted with its first island.
        let cut = Cut { nitsche, volume: 0.125 * size * size * size * inside.total(), points: if island == 0 { leaf.points } else { 0 } };
        Ok(Integrated { stiffness: Some(stiffness), load: Some(load), cut: Some(cut) })
    };

    // What the cells add up to as their matrices are handed over, elements in order.
    let stats = Stats {
        cells: grid.cells,
        inside_cells: 0,
        cut_cells: 0,
        unknowns,
        quadrature_points: 0,
        exact_cells: 0,
        stiffness_entries: 0,
        factor_entries: 0,
        volume: 0.0,
        surface_volume: surface.volume(),
        smallest_fraction: 1.0,
        largest_nitsche: 0.0,
        fictitious,
        levels: [0; MAX_REFINE + 1],
        split_cells: cuts.iter().flatten().filter(|leaf| leaf.islands() > 1).count(),
        separated: 3 * separated,
        holding,
        joined: false,
        rounds: 0,
        settled: 0.0,
        left_out: false,
        iterations: 0,
        residual: 0.0,
    };
    let tally =
        RefCell::new(Tally { stats, nitsche: vec![0.0; elements], fractions: vec![1.0; elements], material: vec![([0.0; 3], [1.0; 3]); elements] });
    let in_use = || Error::Internal("the cells' sums are in use".into());
    let assemble = |sink: &mut dyn FnMut(usize, &[f64], &[f64]) -> Result<()>| -> Result<()> {
        let mut tally = tally.try_borrow_mut().map_err(|_| in_use())?;
        let Tally { stats, nitsche, fractions, material } = &mut *tally;
        for block in fieldcraft_parallel::blocks(elements, block_size) {
            let first = block.start;
            let cells = fieldcraft_parallel::map(block.len(), |i| integrate(first + i));
            for (i, integrated) in cells.into_iter().enumerate() {
                let integrated = integrated?;
                let element = first + i;
                let (node, cell, whole) = cell_of(element)?;
                let island = space.island(element);
                let (cell_stiffness, cell_load) =
                    (integrated.stiffness.as_ref().unwrap_or(&whole.stiffness), integrated.load.as_ref().unwrap_or(&whole.load));
                sink(element, cell_stiffness, cell_load)?;
                let size = grid.step(usize::from(cell.level));
                // A cell is counted with its first island.
                if island == 0
                    && let Some(count) = stats.levels.get_mut(usize::from(cell.level))
                {
                    *count += 1;
                }
                match integrated.cut {
                    Some(cut) => {
                        stats.cut_cells += usize::from(island == 0);
                        stats.quadrature_points += cut.points;
                        stats.exact_cells += usize::from(island == 0 && cut_of(node).is_some_and(|leaf| leaf.exact));
                        stats.volume += cut.volume;
                        stats.smallest_fraction = stats.smallest_fraction.min(cut.volume / (size * size * size));
                        if let Some(fraction) = fractions.get_mut(element) {
                            *fraction = cut.volume / (size * size * size);
                        }
                        // A uniform box with the material's middle and spread: its half extent is
                        // √3 standard deviations.
                        let spread = cut_of(node).and_then(|leaf| leaf.inside.get(island)).and_then(Moments::spread);
                        if let (Some(place), Some((mean, deviation))) = (material.get_mut(element), spread) {
                            *place = (mean, deviation.map(|sigma| 3f64.sqrt() * sigma));
                        }
                        stats.largest_nitsche = stats.largest_nitsche.max(cut.nitsche);
                        if let Some(parameter) = nitsche.get_mut(element) {
                            *parameter = cut.nitsche;
                        }
                    }
                    None => {
                        stats.inside_cells += 1;
                        stats.volume += size * size * size;
                    }
                }
            }
        }
        Ok(())
    };

    // The cells of several islands, for reading the solution: which island a point is in.
    let mut split = BTreeMap::new();
    for (node, leaf) in cuts.iter().enumerate() {
        if let (Some(leaf), Some(owned), Ok(node)) = (leaf.as_ref().filter(|leaf| !leaf.of_piece.is_empty()), pieces.get(node), u32::try_from(node)) {
            split.insert(node, (owned.clone(), leaf.of_piece.clone()));
        }
    }
    // The values of an element's functions from those of its cell's modes.
    let on_functions = |element: usize, values: &[f64]| -> Vec<f64> {
        match space.map(element) {
            Some(map) => {
                let mut wide = vec![0.0; 3 * m];
                for (slot, value) in wide.iter_mut().step_by(3).zip(values) {
                    *slot = *value;
                }
                map.vector(m, &wide).iter().step_by(3).copied().collect()
            }
            None => values.to_vec(),
        }
    };
    let place = Place { grid: &grid, tree: &tree, space: &space, split: &split };
    let at = |x: [f64; 3], y: [f64; 3]| -> Option<(usize, Vec<f64>)> {
        let (element, xi, size) = place.locate_in(x, true).or_else(|| {
            let (element, xi, size) = place.locate_in(y, true)?;
            Some((element, [0, 1, 2].map(|a| xi[a] + 2.0 * (x[a] - y[a]) / size), size))
        })?;
        let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
        cell::eval(&shapes, xi, 2.0 / size, &mut values, &mut gradients);
        Some((element, on_functions(element, &values)))
    };
    let pieces_of_element = |element: usize| pieces_of(space.elements.get(element).map_or(usize::MAX, |node| *node as usize), space.island(element));
    let tied = |element: usize, visit: &mut dyn FnMut(&Tied) -> Result<()>| -> Result<()> {
        if ties.is_none() {
            return Ok(());
        }
        let owned = pieces_of_element(element);
        if !on_surface.ties(&owned) {
            return Ok(());
        }
        let (_, cell, _) = cell_of(element)?;
        let level = usize::from(cell.level);
        let (corner, size) = (grid.fine_lo(level, cell.position()), grid.step(level));
        let functions = space.functions(element).len();
        let mut failed = None;
        cell::tied_points(&shapes, corner, size, &owned, &on_surface, &lame, &surface_rule, |tag, x, weight, values, tractions, along| {
            if failed.is_some() {
                return;
            }
            // On the element's functions: Bᵀ of the cell's modes' values and tractions.
            let values = on_functions(element, values);
            let mut all = Vec::with_capacity(9 * functions);
            for i in 0..3 {
                let of_modes: Vec<f64> = tractions.iter().flat_map(|mode| mode[i]).collect();
                match space.map(element) {
                    Some(map) => all.extend(map.vector(m, &of_modes)),
                    None => all.extend(of_modes),
                }
            }
            if let Err(e) = visit(&Tied { tag, x, weight, values: &values, tractions: &all, along }) {
                failed = Some(e);
            }
        });
        failed.map_or(Ok(()), Err)
    };
    let position = |function: usize| {
        let at = space.xyz.get(function).copied().unwrap_or_default();
        [0, 1, 2].map(|a| grid.origin[a] + 0.5 * grid.spacing * at[a])
    };
    let nitsche_of = |element: usize| tally.try_borrow().ok().and_then(|tally| tally.nitsche.get(element).copied()).unwrap_or(0.0);
    // How full of material the cells that hold any are, taken together.
    let thin = {
        let (mut material, mut cells) = (0.0, 0.0);
        for element in 0..elements {
            let Ok((node, cell, _)) = cell_of(element) else { continue };
            let size = grid.step(usize::from(cell.level));
            let whole = size * size * size;
            let island = space.island(element);
            if island == 0 {
                cells += whole;
            }
            material += cut_of(node).and_then(|leaf| leaf.inside.get(island)).map_or(whole, |inside| 0.125 * whole * inside.total());
        }
        material < THIN * cells
    };
    let view = View {
        grid,
        tree: &tree,
        count: space.count,
        position: &position,
        elements: (0..elements).map(|element| space.functions(element)).collect(),
        assemble: &assemble,
        tied: &tied,
        at: &at,
        nitsche: &nitsche_of,
        linear: &space.linear,
        solver: options.solver,
        poisson: material.poisson,
        quadratic: &space.quadratic,
        thin,
        refine: options.refine,
        before: round.before,
    };

    // The solution: of this body's system alone, or of the system of several bodies.
    let solved = join(&view)?;
    drop(view);
    let Tally { mut stats, nitsche, fractions, material } = tally.into_inner();
    stats.stiffness_entries = solved.stiffness_entries;
    stats.factor_entries = solved.factor_entries;
    stats.iterations = solved.iterations;
    stats.residual = solved.residual;
    let u = solved.u;
    if u.len() != unknowns || u.iter().any(|x| !x.is_finite()) {
        return Err(Error::Singular("the solution is not finite: the stiffness is singular to working precision".into()));
    }

    // The strain energy of the part: ½ uᵀ K u element by element, in the modes of its cell, with
    // the stiffness of the part alone.
    let mut strain_energy = 0.0;
    // The modes of the highest degree along an axis.
    let top: Vec<bool> = shapes.modes.iter().map(|mode| mode.iter().any(|index| usize::from(*index) == p)).collect();
    let mut surplus = Vec::with_capacity(elements);
    for block in fieldcraft_parallel::blocks(elements, block_size) {
        let first = block.start;
        let energies = fieldcraft_parallel::map(block.len(), |i| {
            let element = first + i;
            let Ok((node, cell, whole)) = cell_of(element) else { return (0.0, 0.0) };
            let local = space.local(element, &u);
            let own;
            let k = match cut_of(node).and_then(|leaf| leaf.inside.get(space.island(element))) {
                Some(moments) => {
                    own = cell::stiffness(&shapes, grid.step(usize::from(cell.level)), moments, &lame, &mut Scratch::default());
                    &own
                }
                None => &whole.stiffness,
            };
            let mut energy = 0.0;
            for (row, x) in k.chunks_exact(n).zip(&local) {
                energy += x * row.iter().zip(&local).map(|(a, b)| a * b).sum::<f64>();
            }
            // And of what the functions of the highest degree alone carry ([`Options::adapt`]).
            let highest = |unknown: usize| top.get(unknown / 3).copied().unwrap_or(false);
            let mut surplus = 0.0;
            for (r, (row, x)) in k.chunks_exact(n).zip(&local).enumerate() {
                if highest(r) {
                    surplus += x * row.iter().zip(&local).enumerate().filter(|(c, _)| highest(*c)).map(|(_, (a, b))| a * b).sum::<f64>();
                }
            }
            (0.5 * energy, 0.5 * surplus)
        });
        for (energy, most) in energies {
            strain_energy += energy;
            surplus.push(most);
        }
    }

    // The reactions of the supported faces.
    let pieces_of = pieces_of_element;
    let supported: Vec<usize> = (0..elements).filter(|element| on_surface.supports(&pieces_of(*element))).collect();
    let mut reactions: BTreeMap<u32, [f64; 3]> = BTreeMap::new();
    for block in fieldcraft_parallel::blocks(supported.len(), block_size) {
        let first = block.start;
        let cells = fieldcraft_parallel::map(block.len(), |i| {
            let mut own: BTreeMap<u32, [f64; 3]> = BTreeMap::new();
            let Some(&element) = supported.get(first + i) else { return own };
            let Ok((_, cell, _)) = cell_of(element) else { return own };
            let level = usize::from(cell.level);
            let local = space.local(element, &u);
            let parameter = nitsche.get(element).copied().unwrap_or(0.0);
            let corner = grid.fine_lo(level, cell.position());
            cell::reaction(&shapes, corner, grid.step(level), &pieces_of(element), &on_surface, &lame, &surface_rule, parameter, &local, &mut own);
            own
        });
        // Element by element, in order.
        for own in cells {
            for (face, force) in own {
                let total = reactions.entry(face).or_insert([0.0; 3]);
                for i in 0..3 {
                    total[i] += force[i];
                }
            }
        }
    }
    let reactions = reactions.into_iter().map(|(face, force)| Reaction { face, force }).collect();

    drop(cuts);
    // The positions of the modes were for the ordering only.
    space.xyz = Vec::new();
    let near = if separated > 0 { (0..elements).map(|element| space.around(element, NEAR_LINKS).unwrap_or_default()).collect() } else { Vec::new() };
    Ok(Solution { grid, shapes, lame, tree, space, split, near, fractions, material, u, stats, surplus, strain_energy, reactions })
}

#[cfg(test)]
mod tests {
    //! Patch tests with cells divided as a test asks, whatever the surface: the constant stress
    //! of a block in tension and the linear stress of a bar hanging under its own weight
    //! (S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed., McGraw-Hill 1970). Both
    //! fields are polynomials of the cells' degree, so a conforming, linearly independent basis
    //! gives them exactly however the cells are divided.

    use super::*;
    use fieldcraft_mesh::surface::SurfaceOptions;

    /// A box from the origin to `size`, the face at the lower end of axis a tagged 2a and the
    /// one at its upper end 2a + 1.
    fn block(size: [f64; 3]) -> BoundarySurface {
        let (mut positions, mut triangles, mut tags) = (Vec::new(), Vec::new(), Vec::new());
        for axis in 0..3 {
            let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
            for side in 0..2 {
                let corner = |tb: f64, tc: f64| {
                    let mut p = [0.0; 3];
                    p[axis] = side as f64 * size[axis];
                    p[b] = tb * size[b];
                    p[c] = tc * size[c];
                    p
                };
                let base = positions.len() as u32;
                positions.extend([corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0), corner(0.0, 1.0)]);
                // Counter-clockwise seen from outside.
                triangles.extend(if side == 1 {
                    [[base, base + 1, base + 2], [base, base + 2, base + 3]]
                } else {
                    [[base, base + 2, base + 1], [base, base + 3, base + 2]]
                });
                tags.extend([(2 * axis + side) as u32; 2]);
            }
        }
        BoundarySurface::new(&positions, &triangles, &tags, &SurfaceOptions::default()).unwrap()
    }

    const STEEL: Material = Material { young: 210e9, poisson: 0.3, density: 7850.0 };
    const SIZE: [f64; 3] = [0.75, 0.5, 0.25];
    const STRESS: f64 = 100e6;

    /// About a third of the cells of every level, by where they are: divided cells next to
    /// undivided ones of the same and of other levels.
    fn scattered(level: usize, lo: [f64; 3], size: f64) -> bool {
        let at = lo.map(|x| (x / size).floor() as i64);
        (at[0] + 2 * at[1] + 3 * at[2] + level as i64).rem_euclid(3) == 0
    }

    /// The block in tension along x with sliding supports on its three lower faces: the largest
    /// errors of the displacement (over the elongation) and the stress (over σ₀), and the
    /// solution.
    fn tension(options: &Options, marker: Marker) -> (f64, f64, Solution) {
        let supports = [
            Support { face: 0, fixed: [true, false, false], value: [0.0; 3] },
            Support { face: 2, fixed: [false, true, false], value: [0.0; 3] },
            Support { face: 4, fixed: [false, false, true], value: [0.0; 3] },
        ];
        let loads = [Load::Traction { face: 1, traction: [STRESS, 0.0, 0.0] }];
        let solution = solve_marked(
            &block(SIZE),
            &STEEL,
            &supports,
            &loads,
            options,
            &Limits::default(),
            Some(marker),
            &Round::default(),
            &alone(&Limits::default()),
        )
        .unwrap()
        .0;
        let elongation = STRESS * SIZE[0] / STEEL.young;
        let (mut displacement, mut stress) = (0.0f64, 0.0f64);
        for fx in [0.0, 0.07, 0.31, 0.5, 0.88, 1.0] {
            for fy in [0.0, 0.23, 0.6, 1.0] {
                for fz in [0.0, 0.41, 1.0] {
                    let x = [fx * SIZE[0], fy * SIZE[1], fz * SIZE[2]];
                    let exact =
                        [STRESS * x[0] / STEEL.young, -STEEL.poisson * STRESS * x[1] / STEEL.young, -STEEL.poisson * STRESS * x[2] / STEEL.young];
                    let (u, s) = (solution.displacement(x).unwrap(), solution.stress(x).unwrap());
                    for i in 0..3 {
                        displacement = displacement.max((u[i] - exact[i]).abs() / elongation);
                    }
                    stress = stress.max((s[0] - STRESS).abs() / STRESS);
                    for component in &s[1..] {
                        stress = stress.max(component.abs() / STRESS);
                    }
                }
            }
        }
        (displacement, stress, solution)
    }

    fn check_tension(name: &str, options: &Options, marker: Marker, tolerance: f64) -> Solution {
        let (displacement, stress, solution) = tension(options, marker);
        let stats = &solution.stats;
        let volume = SIZE[0] * SIZE[1] * SIZE[2];
        println!(
            "{name}: leaves by level {:?}, {} unknowns, volume error {:.1e}, smallest fraction {:.1e}, displacement error {displacement:.1e}, stress error {stress:.1e}",
            &stats.levels[..=options.refine],
            stats.unknowns,
            (stats.volume - volume).abs() / volume,
            stats.smallest_fraction
        );
        assert!(stats.levels[1..].iter().sum::<usize>() > 0, "{name}: no cell is divided");
        assert_eq!(stats.levels.iter().sum::<usize>(), stats.inside_cells + stats.cut_cells);
        assert!((stats.volume - volume).abs() < 1e-12 * volume, "{name}: volume {}", stats.volume);
        assert!(displacement < tolerance && stress < tolerance, "{name}: displacement error {displacement:.3e}, stress error {stress:.3e}");
        let force = STRESS * SIZE[1] * SIZE[2];
        let reaction = |face: u32| solution.reactions.iter().find(|r| r.face == face).unwrap().force;
        assert!((reaction(0)[0] + force).abs() < 1e-6 * force, "{name}: reaction {:?}", reaction(0));
        assert!(reaction(2)[1].abs() < tolerance * force && reaction(4)[2].abs() < tolerance * force);
        let energy = 0.5 * STRESS * STRESS / STEEL.young * volume;
        assert!((solution.strain_energy - energy).abs() < 2.0 * tolerance * energy, "{name}: energy {} vs {energy}", solution.strain_energy);
        solution
    }

    /// Every face on a plane of the grid, so nothing is fictitious: the constant stress is found
    /// to rounding with one cell divided, with cells of three levels scattered through the
    /// block, in the tensor product and in the trunk space.
    #[test]
    fn constant_stress_with_divided_cells_is_exact() {
        let first: Marker = &|level, lo, _| level == 0 && lo == [0.0; 3];
        for degree in [1, 2, 3] {
            let options = Options { degree, refine: 1, ..Options::new(0.25) };
            let solution = check_tension(&format!("one cell divided, degree {degree}"), &options, first, 1e-9);
            assert_eq!(solution.stats.levels[..2], [5, 8]);
            // The unknowns by hand. The grid's 3 × 2 × 1 cells have (3p + 1)(2p + 1)(p + 1)
            // functions; those of the divided cell that no other cell shares are handed down
            // to its children: all but the ones on its two faces to its neighbours, p² (p + 1).
            // The children, a grid of 2 × 2 × 2, have (2p + 1)³ less the ones on those two
            // faces, where the finer functions vanish: 2p · 2p · (2p + 1).
            let p = degree;
            let coarse = (3 * p + 1) * (2 * p + 1) * (p + 1) - p * p * (p + 1);
            assert_eq!(solution.stats.unknowns, 3 * (coarse + 2 * p * 2 * p * (2 * p + 1)), "degree {degree}");
        }
        for (degree, basis, refine) in
            [(2, Basis::Tensor, 2), (3, Basis::Tensor, 2), (3, Basis::Trunk, 2), (4, Basis::Trunk, 2), (2, Basis::Tensor, 3)]
        {
            let options = Options { degree, basis, refine, ..Options::new(0.25) };
            let solution = check_tension(&format!("scattered, degree {degree}, {basis:?}, {refine} levels"), &options, &scattered, 1e-9);
            assert!(solution.stats.levels[refine] > 0, "cells of level {refine}: {:?}", solution.stats.levels);
            assert!((solution.stats.smallest_fraction - 1.0).abs() < 1e-12);
        }
    }

    /// The faces anywhere in their cells: cut cells are divided too, and what is left is the
    /// fictitious stiffness, as without refinement (`tests/patch.rs`).
    #[test]
    fn constant_stress_with_divided_cut_cells() {
        for (degree, basis) in [(2, Basis::Tensor), (3, Basis::Tensor), (3, Basis::Trunk)] {
            let options = Options { degree, basis, refine: 2, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) };
            let solution = check_tension(&format!("cut cells, degree {degree}, {basis:?}"), &options, &scattered, 5e-3);
            assert!(solution.stats.levels[2] > 0 && solution.stats.smallest_fraction < 0.1);
        }
    }

    /// A bar hanging from its upper face under its own weight, with Poisson's ratio zero: the
    /// stress σ_zz = ρ g z grows linearly from the free lower end and the displacement
    /// u_z = ρ g (z² − L²) / 2E is quadratic, which degree 2 holds.
    fn hanging(options: &Options, marker: Marker, tolerance: f64) {
        let material = Material { young: 70e9, poisson: 0.0, density: 2700.0 };
        let (size, gravity) = ([0.5, 0.25, 1.0], 9.81);
        let supports = [Support { face: 5, fixed: [true; 3], value: [0.0; 3] }];
        let loads = [Load::Gravity { acceleration: [0.0, 0.0, -gravity] }];
        let solution = solve_marked(
            &block(size),
            &material,
            &supports,
            &loads,
            options,
            &Limits::default(),
            Some(marker),
            &Round::default(),
            &alone(&Limits::default()),
        )
        .unwrap()
        .0;
        let weight = material.density * gravity;
        let scale = weight * size[2] * size[2] / (2.0 * material.young);
        let (mut displacement, mut stress) = (0.0f64, 0.0f64);
        for fx in [0.0, 0.3, 0.5, 1.0] {
            for fy in [0.0, 0.45, 1.0] {
                for fz in [0.0, 0.07, 0.25, 0.5, 0.62, 0.875, 1.0] {
                    let x = [fx * size[0], fy * size[1], fz * size[2]];
                    let exact = weight * (x[2] * x[2] - size[2] * size[2]) / (2.0 * material.young);
                    let (u, s) = (solution.displacement(x).unwrap(), solution.stress(x).unwrap());
                    displacement = displacement.max((u[2] - exact).abs() / scale).max(u[0].abs() / scale).max(u[1].abs() / scale);
                    stress = stress.max((s[2] - weight * x[2]).abs() / (weight * size[2]));
                    for component in [s[0], s[1], s[3], s[4], s[5]] {
                        stress = stress.max(component.abs() / (weight * size[2]));
                    }
                }
            }
        }
        let stats = &solution.stats;
        let volume = size[0] * size[1] * size[2];
        println!(
            "hanging bar, degree {}, {:?}: leaves by level {:?}, {} unknowns, smallest fraction {:.1e}, displacement error {displacement:.1e}, stress error {stress:.1e}",
            options.degree,
            options.basis,
            &stats.levels[..=options.refine],
            stats.unknowns,
            stats.smallest_fraction
        );
        assert!(stats.levels[options.refine] > 0, "{:?}", stats.levels);
        assert!(displacement < tolerance && stress < tolerance, "displacement error {displacement:.3e}, stress error {stress:.3e}");
        // The support carries the weight, and the energy is ½ ∫ σ² / E.
        let reaction = solution.reactions.iter().find(|r| r.face == 5).unwrap().force;
        assert!((reaction[2] - weight * volume).abs() < 1e-6 * weight * volume, "{reaction:?}");
        assert!(reaction[0].abs() + reaction[1].abs() < tolerance.max(1e-6) * weight * volume, "{reaction:?}");
        let energy = weight * weight * volume * size[2] * size[2] / (6.0 * material.young);
        assert!((solution.strain_energy - energy).abs() < 2.0 * tolerance * energy, "energy {} vs {energy}", solution.strain_energy);
    }

    #[test]
    fn linear_stress_with_divided_cells() {
        for (degree, basis) in [(2, Basis::Tensor), (3, Basis::Tensor), (2, Basis::Trunk), (4, Basis::Trunk)] {
            // Faces on planes of the grid: exact to rounding.
            hanging(&Options { degree, basis, refine: 2, ..Options::new(0.25) }, &scattered, 1e-9);
        }
        // Cut cells divided: to the fictitious stiffness.
        hanging(&Options { degree: 2, refine: 2, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) }, &scattered, 5e-3);
        hanging(&Options { degree: 3, refine: 2, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) }, &scattered, 5e-3);
    }

    /// The stress in a cell that holds a sliver of material is the mean of what the cells around
    /// the point say, each by its estimated error ([`Solution::stress`]): in a block in tension
    /// whose first cells hold a millionth of their volume, it is the stress to the fictitious
    /// stiffness of the cells beside them and no longer to that of the slivers.
    #[test]
    fn the_stress_in_a_sliver_is_read_from_the_cells_around_it() {
        let options = Options { degree: 2, offset: [1.0 - 1e-6, 0.0, 0.0], ..Options::new(0.25) };
        let (_, _, solution) = tension(&options, &scattered);
        assert_eq!(solution.stats.levels[0], solution.stats.cut_cells);
        for x in [[1e-7, 0.2, 0.1], [0.0, 0.0, 0.0], [2e-7, 0.5, 0.25]] {
            let fraction = solution.material_fraction(x).unwrap();
            assert!((fraction - 1e-6).abs() < 1e-9, "{x:?}: {fraction}");
            let stress = solution.stress(x).unwrap();
            assert!((stress[0] - STRESS).abs() < 1e-6 * STRESS && stress[1..].iter().all(|s| s.abs() < 1e-6 * STRESS), "{x:?}: {stress:?}");
        }
        // The cells beside them are full, and a point outside the cells has no fraction.
        assert!((solution.material_fraction([0.1, 0.2, 0.1]).unwrap() - 1.0).abs() < 1e-12);
        assert!(solution.material_fraction([5.0, 0.2, 0.1]).is_none());
        assert_eq!((solution.cell_size([0.1, 0.2, 0.1]), solution.cell_size([5.0, 0.2, 0.1])), (Some(0.25), None));
        // Chebyshev's polynomials: T₁ = x, T₃ = 4x³ − 3x.
        assert_eq!((chebyshev(1, 1.5), chebyshev(3, 1.0), chebyshev(3, 2.0)), (1.5, 1.0, 26.0));
    }

    /// A point on the plane between a divided cell and its neighbour, or between two cells of a
    /// divided one, is found in a cell that holds material, and levels above the limit are
    /// refused.
    #[test]
    fn points_between_cells_of_different_levels_are_located() {
        let options = Options { degree: 2, refine: 2, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) };
        let (_, _, solution) = tension(&options, &scattered);
        for x in [[0.0; 3], SIZE, [0.19, 0.13, 0.12], [0.065, 0.005, 0.245], [0.75, 0.25, 0.0]] {
            assert!(solution.displacement(x).is_some(), "{x:?}");
        }
        assert!(solution.displacement([5.0, 0.2, 0.1]).is_none() && solution.displacement([f64::NAN; 3]).is_none());
        let too_deep = Options { refine: MAX_REFINE + 1, ..options };
        assert!(matches!(solve(&block(SIZE), &STEEL, &[], &[], &too_deep, &Limits::default()), Err(Error::Invalid(_))));
    }
}
