//! The linear static step on the grid: from a face-tagged surface, a material, supports and
//! loads to the displacement and stress anywhere in the part.

use std::collections::BTreeMap;

use fieldcraft_fem::{DofLayout, DofSet};
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_solver::Limits;
use fieldcraft_solver::cholesky::{Factor, FactorError, PIVOT_TOLERANCE, UpperColumns};
use fieldcraft_solver::ordering::fill_reducing_order;
use fieldcraft_solver::sparse::{NodeGraph, add_block};

use crate::cell::{self, Face, Lame, Surface};
use crate::gauss::gauss_legendre;
use crate::grid::{self, Grid, MAX_DEPTH, QuadPoint, cross, dot, sub};
use crate::inside::Solid;
use crate::legendre::MAX_DEGREE;
use crate::space::{Kind, NONE, Space, modes};
use crate::{Error, Result};

/// Cells of the grid, at most (most of them outside the part).
const MAX_CELLS: usize = 4_000_000;
/// Entries of the sparse stiffness, at most (8 bytes each for the values, the column indices
/// and the stiffness of the part alone).
const MAX_MATRIX_ENTRIES: usize = 120_000_000;
/// Quadrature points of one cut cell, at most.
const MAX_CELL_POINTS: usize = 4_000_000;
/// Cells integrated at a time: their matrices are in memory together.
const BLOCK: usize = 16;
/// Cells classified per parallel task.
const GRAIN: usize = 4096;
/// A point this close to a grid plane, as a fraction of the cell size, may be evaluated in the
/// cell on either side.
const NEAR: f64 = 1e-6; // unit-factors: relative to the cell size

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

/// How the part is discretised.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    /// The size of the grid's cubic cells.
    pub cell_size: f64,
    /// The polynomial degree of the shape functions, 1..=[`MAX_DEGREE`].
    pub degree: usize,
    /// The depth of the octree under a cut cell, 0..=[`MAX_DEPTH`].
    pub depth: usize,
    /// The stiffness of the cell volume outside the part, as a fraction of the material's.
    pub fictitious: f64,
    /// Where the grid starts: this many cells before the lowest corner of the part's bounding
    /// box, each component in `[0, 1)`.
    pub offset: [f64; 3],
}

impl Options {
    pub const DEGREE: usize = 3;
    /// Flat faces are integrated exactly at any depth; a deeper octree shortens the lists of
    /// triangles a leaf has to look at and costs more points.
    pub const DEPTH: usize = 1;
    /// The stiffness of a cut cell's volume outside the part keeps the unknowns of a cell that
    /// holds little material from being undetermined. The error it adds is of its own order, so
    /// it is as small as the factorization allows: two decades above the pivot the sparse
    /// Cholesky still accepts ([`PIVOT_TOLERANCE`]).
    pub const FICTITIOUS: f64 = 100.0 * PIVOT_TOLERANCE;

    /// The defaults with this cell size.
    pub fn new(cell_size: f64) -> Options {
        Options { cell_size, degree: Options::DEGREE, depth: Options::DEPTH, fictitious: Options::FICTITIOUS, offset: [0.0; 3] }
    }

    fn check(&self) -> Result<()> {
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
    /// Cells wholly inside the part.
    pub inside_cells: usize,
    /// Cells the surface cuts.
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
    /// The smallest fraction of a cut cell that holds material (the conditioning of the system
    /// degrades as it falls).
    pub smallest_fraction: f64,
    /// The largest Nitsche parameter of a supported cell, a stiffness per area.
    pub largest_nitsche: f64,
}

/// A solved step.
#[derive(Debug, Clone)]
pub struct Solution {
    grid: Grid,
    degree: usize,
    lame: Lame,
    modes: usize,
    slot: Vec<u32>,
    cell_modes: Vec<usize>,
    /// Three displacement coefficients per mode.
    u: Vec<f64>,
    pub stats: Stats,
    /// The strain energy of the part, ½ ∫ σ : ε dV.
    pub strain_energy: f64,
    /// The reaction of every supported face, by face tag.
    pub reactions: Vec<Reaction>,
}

impl Solution {
    /// The active cell that holds `x` and its natural coordinates there.
    fn locate(&self, x: [f64; 3]) -> Option<(usize, [f64; 3])> {
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
                    let Some(&slot) = self.slot.get(self.grid.index([i, j, k])) else { continue };
                    if slot != NONE {
                        return Some((slot as usize, [2.0 * (t[0] - i as f64) - 1.0, 2.0 * (t[1] - j as f64) - 1.0, 2.0 * (t[2] - k as f64) - 1.0]));
                    }
                }
            }
        }
        None
    }

    /// The displacement and its gradient ∂uᵢ/∂xⱼ at `x`.
    fn field(&self, x: [f64; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
        let (slot, xi) = self.locate(x)?;
        let cell = self.cell_modes.get(slot * self.modes..(slot + 1) * self.modes)?;
        let local = local_unknowns(cell, &self.u);
        Some(cell::field(self.degree, xi, self.grid.spacing, &local))
    }

    /// The displacement at a point of the part; `None` outside the cells that hold material. In
    /// the part of a cut cell outside the material it is the cell's polynomial continued.
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
}

/// The unknowns of a cell's modes, three per mode.
fn local_unknowns(cell: &[usize], u: &[f64]) -> Vec<f64> {
    cell.iter().flat_map(|mode| (0..3).map(move |i| u.get(3 * mode + i).copied().unwrap_or(0.0))).collect()
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

/// What a cut cell contributes.
struct CutCell {
    stiffness: Vec<f64>,
    /// The stiffness of the part alone: no fictitious stiffness, no support terms.
    physical: Vec<f64>,
    load: Vec<f64>,
    nitsche: f64,
    volume: f64,
    points: usize,
}

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
    let lame = material.lame()?;
    options.check()?;
    let p = options.degree;
    let m = modes(p);
    let n = 3 * m;

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
    let h = grid.spacing;
    let mut triangles = Vec::with_capacity(surface.triangles().len());
    for t in surface.triangles() {
        let corner = |v: u32| {
            vertices.get(v as usize).map(|x| grid.snap(*x)).ok_or_else(|| Error::Internal("a surface triangle names a missing vertex".into()))
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
    let on_grid = grid::Triangles { corners: &triangles, normals: &normals, across: surface.neighbors(), feature: &feature };

    // The cells: cut where they own a piece of the surface, else by their centre.
    let solid = Solid::new(surface)?;
    let sorted = grid::sort_surface(&grid, &triangles);
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

    // The unknowns, numbered in a fill-reducing order.
    let mut space = Space::new(&grid, &kinds, p);
    if space.active.is_empty() {
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
    if space.active.len().saturating_mul(m * m) > MAX_MATRIX_ENTRIES {
        return Err(too_large(space.active.len().saturating_mul(n * n)));
    }
    let order = {
        let graph = NodeGraph::new(space.count, space.cell_modes.chunks(m));
        fill_reducing_order(&space.xyz, &graph.adjacency)
    };
    space.renumber(&order)?;
    let layout = DofLayout::uniform(0..space.count as u32, DofSet::TRANSLATIONS);
    if layout.dofs() != unknowns {
        return Err(Error::Internal("the layout does not hold three unknowns per mode".into()));
    }
    let mut stiffness = {
        let graph = NodeGraph::new(space.count, space.cell_modes.chunks(m));
        let entries = graph.entries(&layout);
        if entries > MAX_MATRIX_ENTRIES {
            return Err(too_large(entries));
        }
        graph.zero_matrix(&layout)
    };
    // The stiffness of the part alone, on the same pattern: its values take the matrix's place
    // while a cell is added to them.
    let mut physical = vec![0.0; stiffness.val.len()];
    let mut load = vec![0.0; unknowns];

    // A cell wholly inside the part: every one has this matrix.
    let rule = gauss_legendre(p + 1);
    let surface_rule = gauss_legendre(2 * p + 1);
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
    let reference = cell::integrate(p, h, &whole);
    let reference_stiffness = cell::stiffness(m, &reference.inside, &lame);
    let reference_load = body_load(&reference.body);

    let cut_cell = |slot: usize| -> Result<Option<CutCell>> {
        let Some(&index) = space.active.get(slot) else { return Err(Error::Internal("an active cell is missing".into())) };
        if kinds.get(index) != Some(&Kind::Cut) {
            return Ok(None);
        }
        let ijk = grid.coords(index);
        let corner = grid.lo(ijk);
        let touching = sorted.touching.get(index).map(Vec::as_slice).unwrap_or_default();
        let pieces = sorted.pieces.get(index).map(Vec::as_slice).unwrap_or_default();
        let points = grid::cut_quadrature(corner, h, touching, &on_grid, &solid, &rule, options.depth, MAX_CELL_POINTS)?;
        let volume = cell::integrate(p, h, &points);
        let physical = cell::stiffness(m, &volume.inside, &lame);
        let outside = cell::stiffness(m, &volume.outside, &lame);
        let mut stiffness: Vec<f64> = physical.iter().zip(&outside).map(|(inside, outside)| inside + options.fictitious * outside).collect();
        let mut load = body_load(&volume.body);
        let boundary = cell::boundary(p, corner, h, pieces, &on_surface, &lame, &surface_rule);
        for (total, surface_load) in load.iter_mut().zip(&boundary.load) {
            *total += surface_load;
        }
        let mut nitsche = 0.0;
        if !boundary.traction.is_empty() {
            nitsche = cell::nitsche_parameter(p, &stiffness, &boundary.traction).ok_or_else(|| {
                Error::Singular(format!(
                    "cell ({}, {}, {}) holds so little material that its stiffness is singular to working precision (move the grid with the offset, or raise the fictitious stiffness)",
                    ijk[0], ijk[1], ijk[2]
                ))
            })?;
            for ((total, consistency), penalty) in stiffness.iter_mut().zip(&boundary.consistency).zip(&boundary.penalty) {
                *total += consistency + nitsche * penalty;
            }
            for ((total, consistency), penalty) in load.iter_mut().zip(&boundary.consistency_load).zip(&boundary.penalty_load) {
                *total += consistency + nitsche * penalty;
            }
        }
        Ok(Some(CutCell { stiffness, physical, load, nitsche, volume: volume.volume, points: points.len() }))
    };

    // Assembly, cells in order.
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
    };
    let mut nitsche = vec![0.0; space.active.len()];
    for block in fieldcraft_parallel::blocks(space.active.len(), BLOCK) {
        let first = block.start;
        let cells = fieldcraft_parallel::map(block.len(), |i| cut_cell(first + i));
        for (i, cut) in cells.into_iter().enumerate() {
            let cut = cut?;
            let slot = first + i;
            let nodes = space.cell(slot);
            let (cell_stiffness, cell_physical, cell_load) = match &cut {
                Some(cut) => (&cut.stiffness, &cut.physical, &cut.load),
                None => (&reference_stiffness, &reference_stiffness, &reference_load),
            };
            add_block(&mut stiffness, &layout, nodes, DofSet::TRANSLATIONS, cell_stiffness)?;
            std::mem::swap(&mut stiffness.val, &mut physical);
            let added = add_block(&mut stiffness, &layout, nodes, DofSet::TRANSLATIONS, cell_physical);
            std::mem::swap(&mut stiffness.val, &mut physical);
            added?;
            for (a, mode) in nodes.iter().enumerate() {
                for i in 0..3 {
                    if let (Some(total), Some(value)) = (load.get_mut(3 * mode + i), cell_load.get(3 * a + i)) {
                        *total += value;
                    }
                }
            }
            match &cut {
                Some(cut) => {
                    stats.cut_cells += 1;
                    stats.quadrature_points += cut.points;
                    stats.volume += cut.volume;
                    stats.smallest_fraction = stats.smallest_fraction.min(cut.volume / (h * h * h));
                    stats.largest_nitsche = stats.largest_nitsche.max(cut.nitsche);
                    if let Some(parameter) = nitsche.get_mut(slot) {
                        *parameter = cut.nitsche;
                    }
                }
                None => {
                    stats.inside_cells += 1;
                    stats.volume += h * h * h;
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
    if u.iter().any(|x| !x.is_finite()) {
        return Err(Error::Singular("the solution is not finite: the stiffness is singular to working precision".into()));
    }

    // The strain energy of the part: ½ uᵀ K u with the stiffness of the part alone.
    std::mem::swap(&mut stiffness.val, &mut physical);
    drop(physical);
    stiffness.mirror_lower();
    let mut forces = vec![0.0; unknowns];
    stiffness.mul(&u, &mut forces);
    drop(stiffness);
    let strain_energy = 0.5 * u.iter().zip(&forces).map(|(x, y)| x * y).sum::<f64>();

    // The reactions of the supported faces.
    let mut reactions: BTreeMap<u32, [f64; 3]> = BTreeMap::new();
    for (slot, &index) in space.active.iter().enumerate() {
        let pieces = sorted.pieces.get(index).map(Vec::as_slice).unwrap_or_default();
        if !on_surface.supports(pieces) {
            continue;
        }
        let local = local_unknowns(space.cell(slot), &u);
        let parameter = nitsche.get(slot).copied().unwrap_or(0.0);
        cell::reaction(p, grid.lo(grid.coords(index)), h, pieces, &on_surface, &lame, &surface_rule, parameter, &local, &mut reactions);
    }
    let reactions = reactions.into_iter().map(|(face, force)| Reaction { face, force }).collect();

    Ok(Solution { grid, degree: p, lame, modes: m, slot: space.slot, cell_modes: space.cell_modes, u, stats, strain_energy, reactions })
}
