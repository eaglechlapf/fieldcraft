//! The linear static step on the grid: from a face-tagged surface, a material, supports and
//! loads to the displacement and stress anywhere in the part.

use std::collections::BTreeMap;

use fieldcraft_fem::{DofLayout, DofSet};
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_solver::Limits;
use fieldcraft_solver::cholesky::{Factor, FactorError, PIVOT_TOLERANCE, UpperColumns};
use fieldcraft_solver::ordering::fill_reducing_order;
use fieldcraft_solver::sparse::{NodeGraph, add_block};

use crate::cell::{self, Face, Lame, Shapes, Surface};
use crate::gauss::gauss_legendre;
use crate::grid::{self, Grid, MAX_DEPTH, MAX_REFINE, Piece, QuadPoint, Triangles, cross, dot, sub};
use crate::inside::Solid;
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
const MAX_MATRIX_ENTRIES: usize = 180_000_000;
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
/// A point this close to a grid plane, as a fraction of the cell size, may be evaluated in the
/// cell on either side.
const NEAR: f64 = 1e-6; // unit-factors: relative to the cell size
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
    /// the benchmarks of `tests/benchmarks.rs` (NAFEMS LE1, a plate with a hole) it reaches the
    /// tensor product's accuracy about two degrees higher, with half to three quarters of the
    /// unknowns and half the factor's entries, in no less time: it saves memory, not time.
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
    /// 0..=[`MAX_DEPTH`].
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
}

impl Options {
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
        if self.refine > MAX_REFINE {
            return Err(Error::Invalid(format!("{} levels of refinement are above {MAX_REFINE}", self.refine)));
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
    /// Three displacement coefficients per mode.
    u: Vec<f64>,
    pub stats: Stats,
    /// The strain energy of the part, ½ ∫ σ : ε dV.
    pub strain_energy: f64,
    /// The reaction of every supported face, by face tag.
    pub reactions: Vec<Reaction>,
}

impl Solution {
    /// The leaf under `node` that holds the point at `t` cells of the grid from its origin and
    /// holds material: its element, the point's natural coordinates in it and its size. A point
    /// within [`NEAR`] of the plane between two children may be in either.
    fn descend(&self, node: usize, t: [f64; 3]) -> Option<(usize, [f64; 3], f64)> {
        let cell = self.tree.nodes.get(node)?;
        let level = usize::from(cell.level);
        // In cells of this level from the cell's lowest corner.
        let steps = (1u64 << level.min(MAX_REFINE)) as f64;
        let within = [0, 1, 2].map(|a| t[a] * steps - cell.ijk[a] as f64);
        if cell.is_leaf() {
            let element = self.space.element_of.get(node).copied().filter(|element| *element != NONE && cell.material())?;
            return Some((element as usize, within.map(|s| 2.0 * s - 1.0), self.grid.step(level)));
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
                    if let Some(found) = self.descend(cell.children as usize + i + 2 * j + 4 * k, t) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }

    /// The element that holds `x`, its natural coordinates there and the size of its cell.
    fn locate(&self, x: [f64; 3]) -> Option<(usize, [f64; 3], f64)> {
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
                    if let Some(found) = self.descend(root as usize, t) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }

    /// The displacement and its gradient ∂uᵢ/∂xⱼ at `x`.
    fn field(&self, x: [f64; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
        let (element, xi, size) = self.locate(x)?;
        let local = self.space.local(element, &self.u);
        Some(cell::field(&self.shapes, xi, size, &local))
    }

    /// The displacement at a point of the part; `None` outside the cells that hold material
    /// (with refinement, the cells that are not divided). In the part of a cut cell outside the
    /// material it is the cell's polynomial continued.
    pub fn displacement(&self, x: [f64; 3]) -> Option<[f64; 3]> {
        self.field(x).map(|(u, _)| u)
    }

    /// The stress at a point of the part, in the order xx, yy, zz, xy, xz, yz.
    pub fn stress(&self, x: [f64; 3]) -> Option<[f64; 6]> {
        let (_, g) = self.field(x)?;
        let (lambda, mu) = (self.lame.lambda, self.lame.mu);
        let dilatation = lambda * (g[0][0] + g[1][1] + g[2][2]);
        Some([
            dilatation + 2.0 * mu * g[0][0],
            dilatation + 2.0 * mu * g[1][1],
            dilatation + 2.0 * mu * g[2][2],
            mu * (g[0][1] + g[1][0]),
            mu * (g[0][2] + g[2][0]),
            mu * (g[1][2] + g[2][1]),
        ])
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
        for x in points {
            let (Some(fine), Some(coarse)) = (self.von_mises(*x), coarser.von_mises(*x)) else { continue };
            if !(fine.is_finite() && coarse.is_finite()) {
                continue;
            }
            peak = peak.max(fine);
            peak_coarser = peak_coarser.max(coarse);
            apart = apart.max((fine - coarse).abs());
            count += 1;
        }
        Change {
            strain_energy: ratio(self.strain_energy - coarser.strain_energy, self.strain_energy),
            peak_von_mises: ratio(peak - peak_coarser, peak),
            von_mises: ratio(apart, peak),
            points: count,
        }
    }
}

/// What each face carries, and the body force per volume.
fn conditions(supports: &[Support], loads: &[Load], areas: &BTreeMap<u32, f64>, density: f64) -> Result<(BTreeMap<u32, Face>, [f64; 3])> {
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
    if !faces.values().any(Face::supported) {
        return Err(Error::Invalid("the part has no support: prescribe the displacement of at least one face".into()));
    }
    Ok((faces, body))
}

fn factor_error(e: FactorError) -> Error {
    match e {
        FactorError::NotPositiveDefinite { column } | FactorError::Singular { column } => Error::Singular(format!(
            "the stiffness is singular to working precision (equation {column}): the supports leave a rigid motion of the part free, or a cell holds so little material that the fictitious stiffness no longer determines its unknowns"
        )),
        FactorError::TooLarge { entries, limit } => {
            Error::Limit(format!("the factorization needs {entries} entries; the limit is {limit} (use a larger cell size or a lower degree)"))
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

/// What the quadrature of a cut cell leaves behind.
struct Cut {
    /// The moments of the part in the cell: its stiffness alone, with no fictitious stiffness
    /// and no support terms, is computed from them again for the strain energy.
    inside: Moments,
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
    solve_marked(surface, material, supports, loads, options, limits, None)
}

/// [`solve`], dividing the cells `marker` names (up to [`Options::refine`] levels) when there is
/// one.
fn solve_marked(
    surface: &BoundarySurface,
    material: &Material,
    supports: &[Support],
    loads: &[Load],
    options: &Options,
    limits: &Limits,
    marker: Option<Marker>,
) -> Result<Solution> {
    let lame = material.lame()?;
    options.check()?;
    // The fictitious stiffness is as small as the factorization allows ([`Options::fictitious`]):
    // the fraction asked for, then a hundred times more at a time up to the one that keeps the
    // margin for this material. Each step multiplies the fraction, so the loop ends.
    let most = margin_fraction(&lame).min(1.0);
    let mut fictitious = options.fictitious;
    loop {
        match solve_at(surface, material, supports, loads, options, limits, fictitious, marker) {
            Err(Error::Singular(_)) if fictitious < most => fictitious = (fictitious * FICTITIOUS_STEP).min(most),
            result => return result,
        }
    }
}

/// How much the fictitious stiffness is raised at a time when the system is singular to working
/// precision: the two decades of margin [`Options::FICTITIOUS`] has over the pivot tolerance.
const FICTITIOUS_STEP: f64 = 100.0;

/// Divide the cells of `tree` that are too coarse, level by level up to `levels`
/// ([`Options::refine`]): the cut cells across which the surface turns by more than `turn` and
/// the cells around them, or the cells `marker` names. `touching` and `pieces` hold the surface
/// in every cell of the tree ([`grid::Sorted`]) and get those of the new cells.
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
) -> Result<()> {
    let curvature = if marker.is_none() { grid::curvatures(surface) } else { Vec::new() };
    let base = tree.nodes.len();
    // The cells of the level being looked at: the ones the level before made.
    let mut first = 0;
    for level in 0..levels.min(MAX_REFINE) {
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
                        cell.kind == Kind::Cut
                            && size * pieces.get(node).map_or(0.0, |owned| owned.iter().map(curved).fold(0.0, f64::max)) > turn * (1.0 + AT_LIMIT)
                    }
                }
            })
            .collect();
        let is_marked = |node: usize| node >= first && marked.get(node - first).copied().unwrap_or(false);
        let divided: Vec<usize> = (first..end)
            .filter(|&node| {
                let Some(cell) = leaf(node) else { return false };
                if is_marked(node) || marker.is_some() {
                    return is_marked(node);
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
        let finer = grid.step(level + 1);
        let children = fieldcraft_parallel::map(divided.len(), |i| -> Result<Vec<(Kind, Vec<u32>, Vec<Piece>)>> {
            let broken = || Error::Internal("a cell to divide is missing".into());
            let node = *divided.get(i).ok_or_else(broken)?;
            let cell = tree.nodes.get(node).ok_or_else(broken)?;
            let reach = touching.get(node).ok_or_else(broken)?;
            let at = cell.position();
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
        for (node, children) in divided.into_iter().zip(children) {
            let children = children?;
            let mut kinds = [Kind::Outside; 8];
            for (kind, child) in kinds.iter_mut().zip(&children) {
                *kind = child.0;
            }
            if children.len() != kinds.len() {
                return Err(Error::Internal("a divided cell has not eight children".into()));
            }
            tree.divide(node, kinds)?;
            for (_, reach, owned) in children {
                touching.push(reach);
                pieces.push(owned);
            }
            // The surface in a divided cell is its children's.
            if let Some(list) = touching.get_mut(node) {
                *list = Vec::new();
            }
            if let Some(list) = pieces.get_mut(node) {
                *list = Vec::new();
            }
        }
        first = end;
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
) -> Result<Solution> {
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
    let mut triangles = Vec::with_capacity(surface.triangles().len());
    for t in surface.triangles() {
        // On the planes of the finest cells there may be: a face of the part that lies on one
        // lies on it exactly.
        let corner = |v: u32| {
            vertices
                .get(v as usize)
                .map(|x| grid.snap_to(*x, options.refine))
                .ok_or_else(|| Error::Internal("a surface triangle names a missing vertex".into()))
        };
        triangles.push([corner(t[0])?, corner(t[1])?, corner(t[2])?]);
    }
    let tags: Vec<u32> = (0..triangles.len()).map(|t| surface.tri_face().get(t).copied().unwrap_or(0)).collect();
    let mut normals = Vec::with_capacity(triangles.len());
    let mut areas: BTreeMap<u32, f64> = BTreeMap::new();
    for (t, tag) in triangles.iter().zip(&tags) {
        let normal = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        let length = dot(normal, normal).sqrt();
        normals.push(if length > 0.0 { [normal[0] / length, normal[1] / length, normal[2] / length] } else { [0.0; 3] });
        *areas.entry(*tag).or_insert(0.0) += 0.5 * length;
    }
    let (faces, body) = conditions(supports, loads, &areas, material.density)?;
    let on_surface = Surface { normals: &normals, tags: &tags, faces: &faces };
    let feature = grid::feature_flags(surface);
    let on_grid = Triangles { corners: &triangles, normals: &normals, across: surface.neighbors(), feature: &feature };

    // The cells: cut where they own a piece of the surface, else by their centre.
    let solid = Solid::new(surface)?;
    let mut sorted = grid::sort_surface(&grid, &triangles);
    let kinds = fieldcraft_parallel::map_grained(grid.len(), GRAIN, |c| -> Result<Kind> {
        if sorted.pieces.get(c).is_some_and(|pieces| !pieces.is_empty()) {
            return Ok(Kind::Cut);
        }
        let ijk = grid.coords(c);
        let (a, b) = (grid.lo(ijk), grid.hi(ijk));
        Ok(if solid.contains([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])])? { Kind::Inside } else { Kind::Outside })
    })
    .into_iter()
    .collect::<Result<Vec<Kind>>>()?;

    // The cells that hold material, divided where they are too coarse, each with the surface
    // in it.
    let mut tree = Tree::new(&grid, &kinds);
    let (mut touching, mut pieces): (Vec<Vec<u32>>, Vec<Vec<Piece>>) = (Vec::with_capacity(tree.nodes.len()), Vec::with_capacity(tree.nodes.len()));
    for (cell, root) in tree.root.iter().enumerate() {
        if *root != NONE {
            touching.push(sorted.touching.get_mut(cell).map(std::mem::take).unwrap_or_default());
            pieces.push(sorted.pieces.get_mut(cell).map(std::mem::take).unwrap_or_default());
        }
    }
    drop(sorted);
    drop(kinds);
    if options.refine > 0 {
        divide_cells(&grid, &mut tree, &on_grid, &solid, &mut touching, &mut pieces, options.refine, options.turn(), marker)?;
    }

    // The unknowns, numbered in a fill-reducing order.
    let mut space = Space::new(&tree, &shapes.modes, p);
    let elements = space.elements.len();
    if elements == 0 {
        return Err(Error::Invalid("no cell holds material: the surface must be closed and oriented outward".into()));
    }
    let unknowns = 3 * space.count;
    if unknowns > limits.max_dofs {
        return Err(Error::Limit(format!("{unknowns} unknowns; the limit is {} (use a larger cell size or a lower degree)", limits.max_dofs)));
    }
    let too_large = |entries: usize| {
        Error::Limit(format!(
            "the stiffness would have {entries} entries; the limit is {MAX_MATRIX_ENTRIES} (use a larger cell size or a lower degree)"
        ))
    };
    let blocks =
        (0..elements).fold(0usize, |sum, element| sum.saturating_add(space.functions(element).len().saturating_mul(space.functions(element).len())));
    if blocks > MAX_MATRIX_ENTRIES {
        return Err(too_large(blocks.saturating_mul(9)));
    }
    let order = {
        let graph = NodeGraph::new(space.count, (0..elements).map(|element| space.functions(element)));
        fill_reducing_order(&space.xyz, &graph.adjacency)
    };
    space.renumber(&order)?;
    let layout = DofLayout::uniform(0..space.count as u32, DofSet::TRANSLATIONS);
    if layout.dofs() != unknowns {
        return Err(Error::Internal("the layout does not hold three unknowns per mode".into()));
    }
    let mut stiffness = {
        let graph = NodeGraph::new(space.count, (0..elements).map(|element| space.functions(element)));
        let entries = graph.entries(&layout);
        if entries > MAX_MATRIX_ENTRIES {
            return Err(too_large(entries));
        }
        graph.zero_matrix(&layout)
    };
    let mut load = vec![0.0; unknowns];

    // A cell the surface does not cut: every one of a level has the same matrices.
    let rule = gauss_legendre(p + 1);
    let surface_rule = gauss_legendre(2 * p + 1);
    let (second_rule, third_rule) = (gauss_legendre(2 * p + 1), gauss_legendre(3 * p + 2));
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
        let touching = touching.get(node).map(Vec::as_slice).unwrap_or_default();
        let pieces = pieces.get(node).map(Vec::as_slice).unwrap_or_default();
        let mut volume = cell::Volume::new(&shapes);
        let rules = [&rule[..], &second_rule[..], &third_rule[..]];
        // The octree is as much shallower as the cell is smaller.
        let depth = options.depth.saturating_sub(level);
        let points = grid::visit_cut_quadrature(corner, size, touching, &on_grid, &solid, rules, depth, MAX_CELL_POINTS, |point| volume.add(&point))?;
        let mut scratch = Scratch::default();
        let mut stiffness = cell::stiffness(&shapes, size, &volume.inside, &lame, &mut scratch);
        if !volume.outside.is_zero() {
            let outside = cell::stiffness(&shapes, size, &volume.outside, &outside_material, &mut scratch);
            for (total, fictitious) in stiffness.iter_mut().zip(&outside) {
                *total += fictitious;
            }
        }
        let mut load = body_load(&cell::body(&shapes, size, &volume.inside, &mut scratch));
        let boundary = cell::boundary(&shapes, corner, size, pieces, &on_surface, &lame, &surface_rule, &mut scratch);
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
        let cut = Cut { nitsche, volume: 0.125 * size * size * size * volume.inside.total(), points, inside: volume.inside };
        Ok(Integrated { stiffness: Some(stiffness), load: Some(load), cut: Some(cut) })
    };

    // Assembly, elements in order.
    let mut stats = Stats {
        cells: grid.cells,
        inside_cells: 0,
        cut_cells: 0,
        unknowns,
        quadrature_points: 0,
        factor_entries: 0,
        volume: 0.0,
        surface_volume: surface.volume(),
        smallest_fraction: 1.0,
        largest_nitsche: 0.0,
        fictitious,
        levels: [0; MAX_REFINE + 1],
    };
    let mut nitsche = vec![0.0; elements];
    let mut inside: Vec<Option<Moments>> = vec![None; elements];
    for block in fieldcraft_parallel::blocks(elements, block_size) {
        let first = block.start;
        let cells = fieldcraft_parallel::map(block.len(), |i| integrate(first + i));
        for (i, integrated) in cells.into_iter().enumerate() {
            let integrated = integrated?;
            let element = first + i;
            let (_, cell, whole) = cell_of(element)?;
            let nodes = space.functions(element);
            let (cell_stiffness, cell_load) =
                (integrated.stiffness.as_ref().unwrap_or(&whole.stiffness), integrated.load.as_ref().unwrap_or(&whole.load));
            add_block(&mut stiffness, &layout, nodes, DofSet::TRANSLATIONS, cell_stiffness)?;
            for (a, mode) in nodes.iter().enumerate() {
                for i in 0..3 {
                    if let (Some(total), Some(value)) = (load.get_mut(3 * mode + i), cell_load.get(3 * a + i)) {
                        *total += value;
                    }
                }
            }
            let size = grid.step(usize::from(cell.level));
            if let Some(count) = stats.levels.get_mut(usize::from(cell.level)) {
                *count += 1;
            }
            match integrated.cut {
                Some(cut) => {
                    stats.cut_cells += 1;
                    stats.quadrature_points += cut.points;
                    stats.volume += cut.volume;
                    stats.smallest_fraction = stats.smallest_fraction.min(cut.volume / (size * size * size));
                    stats.largest_nitsche = stats.largest_nitsche.max(cut.nitsche);
                    if let Some(parameter) = nitsche.get_mut(element) {
                        *parameter = cut.nitsche;
                    }
                    if let Some(moments) = inside.get_mut(element) {
                        *moments = Some(cut.inside);
                    }
                }
                None => {
                    stats.inside_cells += 1;
                    stats.volume += size * size * size;
                }
            }
        }
    }

    // The solution.
    stiffness.mirror_lower();
    let factor = Factor::new(&UpperColumns::from_symmetric(&stiffness), limits.max_factor_entries).map_err(factor_error)?;
    stats.factor_entries = factor.entries();
    let mut u = load;
    factor.solve(&mut u)?;
    drop(factor);
    drop(stiffness);
    if u.iter().any(|x| !x.is_finite()) {
        return Err(Error::Singular("the solution is not finite: the stiffness is singular to working precision".into()));
    }

    // The strain energy of the part: ½ uᵀ K u element by element, in the modes of its cell, with
    // the stiffness of the part alone.
    let mut strain_energy = 0.0;
    for block in fieldcraft_parallel::blocks(elements, block_size) {
        let first = block.start;
        let energies = fieldcraft_parallel::map(block.len(), |i| {
            let element = first + i;
            let Ok((_, cell, whole)) = cell_of(element) else { return 0.0 };
            let local = space.local(element, &u);
            let own;
            let k = match inside.get(element).and_then(Option::as_ref) {
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
            0.5 * energy
        });
        for energy in energies {
            strain_energy += energy;
        }
    }
    drop(inside);

    // The reactions of the supported faces.
    let pieces_of = |element: usize| space.elements.get(element).and_then(|node| pieces.get(*node as usize)).map(Vec::as_slice).unwrap_or_default();
    let supported: Vec<usize> = (0..elements).filter(|element| on_surface.supports(pieces_of(*element))).collect();
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
            cell::reaction(&shapes, corner, grid.step(level), pieces_of(element), &on_surface, &lame, &surface_rule, parameter, &local, &mut own);
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

    // The positions of the modes were for the ordering only.
    space.xyz = Vec::new();
    Ok(Solution { grid, shapes, lame, tree, space, u, stats, strain_energy, reactions })
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
        let solution = solve_marked(&block(SIZE), &STEEL, &supports, &loads, options, &Limits::default(), Some(marker)).unwrap();
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
        let solution = solve_marked(&block(size), &material, &supports, &loads, options, &Limits::default(), Some(marker)).unwrap();
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
