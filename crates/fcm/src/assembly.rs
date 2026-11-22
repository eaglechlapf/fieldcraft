//! Several bodies in one solve, bonded where they touch.
//!
//! Every body is solved as one body is ([`crate::solve`]): on a grid of its own, with its own
//! cell size, material, supports and loads, its cells cut by its own surface. Two bodies share
//! no unknown, however close they are, so nothing joins them by accident; what joins them is a
//! [`Bond`], which names faces of the two that may touch.
//!
//! # A bond
//! Where the faces touch, the displacements of the two bodies are made equal weakly, by
//! Nitsche's method for an interface (A. Hansbo, P. Hansbo, *An unfitted finite element method,
//! based on Nitsche's method, for elliptic interface problems*, Comput. Methods Appl. Mech.
//! Engrg. 191(47–48):5537–5552, 2002), with the traction taken from one of the two bodies alone,
//! the limit of the weighted average of C. Annavarapu, M. Hautefeuille, J. E. Dolbow, *A robust
//! Nitsche's formulation for interface problems*, Comput. Methods Appl. Mech. Engrg.
//! 225–228:44–54, 2012. With A that body, B the other, Γ where they touch, t(u) = σ_A(u) n the
//! traction of A's displacement on Γ and [u] = u_A − u_B the jump:
//!
//! a(u, v) += − ∫ t(u_A) · [v] dΓ − ∫ t(v_A) · [u] dΓ + β ∫ [u] · [v] dΓ.
//!
//! The traction is continuous across Γ for the exact solution and the jump vanishes, so the
//! terms are consistent for any β, whatever the two materials. On A alone they are those of a
//! support that fixes every component of Γ ([`crate::cell`]), so A's cells get them, and their
//! parameter β (twice the largest eigenvalue of A's cell problem, computed, not chosen), as they
//! get a support's; the bond adds the blocks that couple the two bodies,
//!
//! K_AB = ∫ t(v_A) · u_B − β ∫ v_A · u_B, K_BB = β ∫ u_B · v_B,
//!
//! which are what a prescribed displacement's load would be if B's displacement were known.
//! A is the body whose named faces have the smaller area: the face that is more likely to be
//! covered by the other.
//!
//! # Where two faces touch
//! Γ is not drawn; it is found. The pieces of A's named faces are integrated by their quadrature
//! points, and a point counts when it lies within the bond's tolerance of a triangle of B's
//! named faces. So faces that overlap in part are bonded over the part, to the fineness of the
//! points (the pieces are not cut along the other face's edge), and faces that cover each other
//! are bonded exactly. B's displacement at the point is that of B's cell there; a point that
//! rounding or the facets of a curved face put just outside B's cells reads the cell of the
//! nearest point of B's face, whose polynomial reaches it.
//!
//! The two bodies' cells do not match, and B's functions are polynomials only cell by cell. So
//! a piece of A's face (a triangle in one cell of A) is first cut along B's cells
//! ([`leaf_parts`]), and each part is integrated by a rule that is exact for the products of
//! the two bodies' functions on it: the bond then passes the patch test on any two grids, as a
//! consistent method must (a constant stress goes through it to rounding).
//!
//! # One system
//! The bodies' unknowns are ordered together, by nested dissection of all their elements and of
//! the pairs of cells a bond couples ([`fieldcraft_solver::ordering::element_dissection`]), and
//! factored as one matrix. A body that only a bond holds needs no support of its own.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use fieldcraft_fem::{DofLayout, DofSet};
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_solver::Limits;
use fieldcraft_solver::cholesky::{Factor, UpperColumns};
use fieldcraft_solver::ordering::element_dissection;
use fieldcraft_solver::sparse::{NodeGraph, add_block};

use crate::grid::{self, Grid, cross, dot, sub};
use crate::islands::closest_point;
use crate::refine::Tree;
use crate::solve::{self, Load, MAX_MATRIX_ENTRIES, Material, Options, Solution, Solved, Support, Tied, Ties, View};
use crate::space::NONE;
use crate::{Error, Result};

/// Bodies of one solve, at most (their solves are nested).
pub const MAX_PARTS: usize = 64;
/// Buckets along an axis of the box of a bond's faces, at most.
const MOST_BUCKETS: usize = 48;
/// Triangle references over all buckets of a bond's faces, at most.
const MOST_REFERENCES: usize = 50_000_000;

/// One body of an assembly: what [`crate::solve`] takes for one body.
#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    pub surface: &'a BoundarySurface,
    pub material: Material,
    pub supports: &'a [Support],
    pub loads: &'a [Load],
    /// The grid of this body ([`Options::adapt`] is not run for a body of an assembly yet).
    pub options: Options,
}

/// Two bodies bonded where faces of theirs touch.
#[derive(Debug, Clone, PartialEq)]
pub struct Bond {
    /// A body (its index among the parts) and the faces of it (their tags) that may touch the
    /// other's.
    pub first: (usize, Vec<u32>),
    pub second: (usize, Vec<u32>),
    /// How far apart a point of the one's faces and the other's may be and still touch, a
    /// length: above the rounding of flat faces that lie on each other, and above the chordal
    /// tolerance of the triangulations where curved faces do.
    pub tolerance: f64,
}

/// The triangles of the faces of one body that a bond names, sorted into buckets.
struct Reach {
    triangles: Vec<[[f64; 3]; 3]>,
    lo: [f64; 3],
    step: [f64; 3],
    count: [usize; 3],
    buckets: Vec<Vec<u32>>,
    tolerance: f64,
}

impl Reach {
    fn new(surface: &BoundarySurface, faces: &BTreeSet<u32>, tolerance: f64) -> Result<Reach> {
        let points = surface.points();
        let mut triangles = Vec::new();
        for (triangle, tag) in surface.triangles().iter().zip(surface.tri_face()) {
            if !faces.contains(tag) {
                continue;
            }
            let corner = |v: u32| points.get(v as usize).copied().ok_or_else(|| Error::Internal("a surface triangle names a missing vertex".into()));
            triangles.push([corner(triangle[0])?, corner(triangle[1])?, corner(triangle[2])?]);
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for corner in triangles.iter().flatten() {
            for a in 0..3 {
                lo[a] = lo[a].min(corner[a] - tolerance);
                hi[a] = hi[a].max(corner[a] + tolerance);
            }
        }
        let mut reach = Reach { triangles, lo, step: [1.0; 3], count: [1; 3], buckets: Vec::new(), tolerance };
        if reach.triangles.is_empty() {
            return Ok(reach);
        }
        // Fewer buckets until the triangles' references fit: each halving leaves an eighth.
        let mut most = MOST_BUCKETS;
        loop {
            for a in 0..3 {
                let extent = hi[a] - lo[a];
                reach.count[a] = if extent > 0.0 && extent.is_finite() { most } else { 1 };
                reach.step[a] = if reach.count[a] > 1 { extent / most as f64 } else { 1.0 };
            }
            let range = |reach: &Reach, triangle: &[[f64; 3]; 3], a: usize| {
                let (least, largest) =
                    (triangle[0][a].min(triangle[1][a]).min(triangle[2][a]), triangle[0][a].max(triangle[1][a]).max(triangle[2][a]));
                (reach.bucket(a, least - tolerance), reach.bucket(a, largest + tolerance))
            };
            let references = reach.triangles.iter().fold(0usize, |sum, triangle| {
                let spans = [0, 1, 2].map(|a| range(&reach, triangle, a));
                sum.saturating_add(spans.iter().map(|(from, to)| to + 1 - from).product::<usize>())
            });
            if references > MOST_REFERENCES && most > 1 {
                most /= 2;
                continue;
            }
            let mut buckets = vec![Vec::new(); reach.count[0] * reach.count[1] * reach.count[2]];
            for (t, triangle) in reach.triangles.iter().enumerate() {
                let spans = [0, 1, 2].map(|a| range(&reach, triangle, a));
                for k in spans[2].0..=spans[2].1 {
                    for j in spans[1].0..=spans[1].1 {
                        for i in spans[0].0..=spans[0].1 {
                            if let Some(bucket) = buckets.get_mut(i + reach.count[0] * (j + reach.count[1] * k)) {
                                bucket.push(t as u32);
                            }
                        }
                    }
                }
            }
            reach.buckets = buckets;
            return Ok(reach);
        }
    }

    /// The bucket along axis `a` of a coordinate, clamped into the box.
    fn bucket(&self, a: usize, x: f64) -> usize {
        let t = (x - self.lo[a]) / self.step[a];
        if !(t > 0.0) { 0 } else { (t as usize).min(self.count[a] - 1) }
    }

    /// The point of the faces nearest to `x`, when it is within the tolerance, and the square of
    /// its distance.
    fn nearest(&self, x: [f64; 3]) -> Option<([f64; 3], f64)> {
        if self.buckets.is_empty() || (0..3).any(|a| !(x[a] >= self.lo[a] && x[a] <= self.lo[a] + self.step[a] * self.count[a] as f64)) {
            return None;
        }
        let bucket = self.buckets.get(self.bucket(0, x[0]) + self.count[0] * (self.bucket(1, x[1]) + self.count[1] * self.bucket(2, x[2])))?;
        let mut best: Option<([f64; 3], f64)> = None;
        for t in bucket {
            let Some(triangle) = self.triangles.get(*t as usize) else { continue };
            let y = closest_point(x, triangle);
            let apart = sub(x, y);
            let apart = dot(apart, apart);
            if apart <= self.tolerance * self.tolerance && best.is_none_or(|(_, least)| apart < least) {
                best = Some((y, apart));
            }
        }
        best
    }
}

/// A bond as the solve reads it: the body whose traction it takes, the faces of that body, and
/// the other body with its faces.
struct Coupling {
    flux: usize,
    faces: BTreeSet<u32>,
    other: usize,
    reach: Reach,
}

/// What the nested solves of the bodies share.
struct Shared<'a> {
    parts: &'a [Part<'a>],
    couplings: &'a [Coupling],
    limits: &'a Limits,
    /// Of each body: the faces a bond takes the traction of, whether a bond holds it, and the
    /// fictitious stiffness of this attempt.
    tied: &'a [BTreeSet<u32>],
    held: &'a [bool],
    fictitious: &'a [f64],
    /// The cells of each body, once its solve has them: a copy the other bodies' pieces are
    /// cut along.
    frames: &'a [OnceLock<(Grid, Tree)>],
    solved: RefCell<Vec<Option<Solved>>>,
    solutions: RefCell<Vec<Option<Solution>>>,
}

/// The body a point of a tied face of `body` touches, and the nearest point of its faces.
fn partner(couplings: &[Coupling], body: usize, tag: u32, x: [f64; 3]) -> Option<(usize, [f64; 3])> {
    let mut best: Option<(usize, [f64; 3], f64)> = None;
    for coupling in couplings.iter().filter(|coupling| coupling.flux == body && coupling.faces.contains(&tag)) {
        if let Some((y, apart)) = coupling.reach.nearest(x)
            && best.is_none_or(|(_, _, least)| apart < least)
        {
            best = Some((coupling.other, y, apart));
        }
    }
    best.map(|(other, y, _)| (other, y))
}

/// The views of the bodies whose solves are under way, the last first.
struct Chain<'a> {
    view: &'a View<'a>,
    up: Option<&'a Chain<'a>>,
}

fn broken(what: &str) -> Error {
    Error::Internal(format!("assembly: {what}"))
}

/// Solve body `index` and, inside its solve, the bodies after it; the last one's solve solves
/// the system of all.
fn nest(shared: &Shared, index: usize, up: Option<&Chain>) -> Result<()> {
    let part = shared.parts.get(index).ok_or_else(|| broken("a body is missing"))?;
    let (couplings, frames) = (shared.couplings, shared.frames);
    let touching = |tag: u32, piece: &[[f64; 3]; 3], visit: &mut dyn FnMut(&[[f64; 3]; 3], &dyn Fn([f64; 3]) -> bool)| {
        // The bodies this face is bonded to, each once.
        let others: BTreeSet<usize> =
            couplings.iter().filter(|coupling| coupling.flux == index && coupling.faces.contains(&tag)).map(|coupling| coupling.other).collect();
        for other in others {
            let touches = |x: [f64; 3]| partner(couplings, index, tag, x).is_some_and(|(body, _)| body == other);
            match frames.get(other).and_then(OnceLock::get) {
                Some((grid, tree)) => leaf_parts(grid, tree, piece, &mut |part| visit(part, &touches)),
                None => visit(piece, &touches),
            }
        }
    };
    let ties = Ties {
        faces: shared.tied.get(index).ok_or_else(|| broken("a body's ties are missing"))?,
        touching: &touching,
        held: shared.held.get(index).copied().unwrap_or(false),
    };
    let join = |view: &View| -> Result<Solved> {
        let link = Chain { view, up };
        if let Some(frame) = frames.get(index) {
            // Set once: a body's solve reaches this once an attempt.
            let _ = frame.set((view.grid, view.tree.clone()));
        }
        if index + 1 < shared.parts.len() {
            nest(shared, index + 1, Some(&link))?;
        } else {
            let mut views = Vec::with_capacity(shared.parts.len());
            let mut at = Some(&link);
            while let Some(link) = at {
                views.push(link.view);
                at = link.up;
            }
            views.reverse();
            let solved = solve_system(&views, &|body, tag, x| partner(couplings, body, tag, x), shared.limits)?;
            let mut kept = shared.solved.try_borrow_mut().map_err(|_| broken("the solutions are in use"))?;
            *kept = solved.into_iter().map(Some).collect();
        }
        let mut kept = shared.solved.try_borrow_mut().map_err(|_| broken("the solutions are in use"))?;
        kept.get_mut(index).and_then(Option::take).ok_or_else(|| broken("a body has no solution"))
    };
    let fictitious = shared.fictitious.get(index).copied().unwrap_or(part.options.fictitious);
    let solution =
        solve::solve_tied(part.surface, &part.material, part.supports, part.loads, &part.options, shared.limits, fictitious, &ties, &join)?;
    let mut solutions = shared.solutions.try_borrow_mut().map_err(|_| broken("the solutions are in use"))?;
    if let Some(slot) = solutions.get_mut(index) {
        *slot = Some(solution);
    }
    Ok(())
}

/// Solve several bodies as one step, bonded where the faces a [`Bond`] names touch; the
/// solutions come in the bodies' order. A body needs a support of its own or a bond that holds
/// it. Refused like a single body's solve: a system above the limits, a body or a group of
/// bodies that nothing holds.
pub fn solve_assembly(parts: &[Part], bonds: &[Bond], limits: &Limits) -> Result<Vec<Solution>> {
    if parts.is_empty() || parts.len() > MAX_PARTS {
        return Err(Error::Invalid(format!("an assembly has 1 to {MAX_PARTS} bodies; this one has {}", parts.len())));
    }
    // The area of faces of a body, and whether it has them.
    let area = |body: usize, faces: &[u32]| -> Result<f64> {
        let part = parts.get(body).ok_or_else(|| Error::Invalid(format!("a bond names body {body}; there are {}", parts.len())))?;
        let points = part.surface.points();
        let mut areas: BTreeMap<u32, f64> = faces.iter().map(|face| (*face, 0.0)).collect();
        for (triangle, tag) in part.surface.triangles().iter().zip(part.surface.tri_face()) {
            let (Some(total), [Some(a), Some(b), Some(c)]) = (areas.get_mut(tag), triangle.map(|v| points.get(v as usize))) else { continue };
            let doubled = cross(sub(*b, *a), sub(*c, *a));
            *total += 0.5 * dot(doubled, doubled).sqrt();
        }
        if let Some((face, _)) = areas.iter().find(|(_, area)| !(**area > 0.0)) {
            return Err(Error::Invalid(format!("a bond names face {face} of body {body}, which its surface does not have")));
        }
        if faces.is_empty() {
            return Err(Error::Invalid(format!("a bond names no face of body {body}")));
        }
        Ok(areas.values().sum())
    };
    let mut couplings = Vec::with_capacity(bonds.len());
    let mut tied = vec![BTreeSet::new(); parts.len()];
    let mut held = vec![false; parts.len()];
    for bond in bonds {
        if !(bond.tolerance > 0.0 && bond.tolerance.is_finite()) {
            return Err(Error::Invalid(format!("a bond's tolerance {} is not a positive length", bond.tolerance)));
        }
        if bond.first.0 == bond.second.0 {
            return Err(Error::Invalid(format!("a bond ties body {} to itself", bond.first.0)));
        }
        let (first, second) = (area(bond.first.0, &bond.first.1)?, area(bond.second.0, &bond.second.1)?);
        // The traction is taken from the body with the smaller faces.
        let (flux, other) = if second < first { (&bond.second, &bond.first) } else { (&bond.first, &bond.second) };
        let faces: BTreeSet<u32> = flux.1.iter().copied().collect();
        let others: BTreeSet<u32> = other.1.iter().copied().collect();
        let surface = parts.get(other.0).ok_or_else(|| broken("a body is missing"))?.surface;
        if let Some(own) = tied.get_mut(flux.0) {
            own.extend(&faces);
        }
        if let Some(own) = held.get_mut(other.0) {
            *own = true;
        }
        couplings.push(Coupling { flux: flux.0, faces, other: other.0, reach: Reach::new(surface, &others, bond.tolerance)? });
    }
    // As small a fictitious stiffness as the factorization allows, as for one body: a hundred
    // times more at a time for every body, up to what its material leaves. Each step raises
    // one at least, so the loop ends.
    let most: Vec<f64> = parts.iter().map(|part| solve::most_fictitious(&part.material)).collect::<Result<_>>()?;
    let mut fictitious: Vec<f64> = parts.iter().map(|part| part.options.fictitious).collect();
    loop {
        let frames: Vec<OnceLock<(Grid, Tree)>> = (0..parts.len()).map(|_| OnceLock::new()).collect();
        let shared = Shared {
            parts,
            couplings: &couplings,
            limits,
            tied: &tied,
            held: &held,
            fictitious: &fictitious,
            frames: &frames,
            solved: RefCell::new(Vec::new()),
            solutions: RefCell::new((0..parts.len()).map(|_| None).collect()),
        };
        match nest(&shared, 0, None) {
            Err(Error::Singular(_)) if fictitious.iter().zip(&most).any(|(now, most)| now < most) => {
                for (now, most) in fictitious.iter_mut().zip(&most) {
                    *now = (*now * solve::FICTITIOUS_STEP).min(*most).max(*now);
                }
            }
            Err(e) => return Err(e),
            Ok(()) => {
                return shared.solutions.into_inner().into_iter().map(|solution| solution.ok_or_else(|| broken("a body has no solution"))).collect();
            }
        }
    }
}

/// The parts of a triangle in the cells of a body that hold material and are not divided, each
/// a triangle in one cell: what is left of it outside them is in no part.
fn leaf_parts(grid: &Grid, tree: &Tree, triangle: &[[f64; 3]; 3], visit: &mut dyn FnMut(&[[f64; 3]; 3])) {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for corner in triangle {
        for a in 0..3 {
            lo[a] = lo[a].min(corner[a]);
            hi[a] = hi[a].max(corner[a]);
        }
    }
    // The cells of the grid the triangle's box reaches.
    let mut range = [(0usize, 0usize); 3];
    for a in 0..3 {
        let count = grid.cells[a];
        let (from, to) = (((lo[a] - grid.origin[a]) / grid.spacing).floor(), ((hi[a] - grid.origin[a]) / grid.spacing).floor());
        if count == 0 || !(from.is_finite() && to.is_finite()) || to < 0.0 || from >= count as f64 {
            return;
        }
        range[a] = (from.max(0.0) as usize, (to as usize).min(count - 1));
    }
    let mut stack: Vec<u32> = Vec::new();
    for k in range[2].0..=range[2].1 {
        for j in range[1].0..=range[1].1 {
            for i in range[0].0..=range[0].1 {
                let Some(&root) = tree.root.get(grid.index([i, j, k])) else { continue };
                if root == NONE {
                    continue;
                }
                stack.push(root);
                // Every cell under the root at most once: the stack empties.
                while let Some(node) = stack.pop() {
                    let Some(cell) = tree.nodes.get(node as usize) else { continue };
                    let level = usize::from(cell.level);
                    let (from, to) = (grid.fine_lo(level, cell.position()), grid.fine_hi(level, cell.position()));
                    if (0..3).any(|a| from[a] > hi[a] || to[a] < lo[a]) {
                        continue;
                    }
                    if !cell.is_leaf() {
                        stack.extend((0..8).map(|child| cell.children + child));
                        continue;
                    }
                    if !cell.material() {
                        continue;
                    }
                    let polygon: Vec<[f64; 3]> = grid::clip(triangle, from, to).into_iter().map(|(corner, _)| corner).collect();
                    for pair in polygon.get(1..).unwrap_or_default().windows(2) {
                        let part = [polygon[0], pair[0], pair[1]];
                        let doubled = cross(sub(part[1], part[0]), sub(part[2], part[0]));
                        if dot(doubled, doubled) > 0.0 {
                            visit(&part);
                        }
                    }
                }
            }
        }
    }
}

/// What a bond adds between the functions of two elements, without the parameter β: the
/// integrals over the points where the first's tied face touches the second.
struct Pair {
    /// ∫ t_l(v) N_b of the first's unknowns (rows) and the second's (columns).
    flux: Vec<f64>,
    /// ∫ N_a N_b, functions of the first by functions of the second.
    mass: Vec<f64>,
    /// ∫ N_b N_b′ of the second's functions.
    own: Vec<f64>,
}

/// One matrix of the bodies whose `views` are ready: ordered, added up, factored and solved,
/// with the blocks of the bonds; `partner` is the body a point of a tied face touches and the
/// nearest point of that body's faces. The solutions come in the views' order.
pub(crate) fn solve_system(
    views: &[&View],
    partner: &dyn Fn(usize, u32, [f64; 3]) -> Option<(usize, [f64; 3])>,
    limits: &Limits,
) -> Result<Vec<Solved>> {
    let mut offsets = Vec::with_capacity(views.len());
    let mut total = 0usize;
    for view in views {
        offsets.push(total);
        total = total.saturating_add(view.count);
    }
    if 3 * total > limits.max_dofs {
        return Err(Error::Limit(format!("{} unknowns; the limit is {} (use a larger cell size or a lower degree)", 3 * total, limits.max_dofs)));
    }

    // The bonds: the pairs of elements they couple.
    let mut pairs: BTreeMap<(usize, usize, usize, usize), Pair> = BTreeMap::new();
    for (first, view) in views.iter().enumerate() {
        (view.tied)(&mut |point: &Tied| -> Result<()> {
            let (second, y) = partner(first, point.tag, point.x).ok_or_else(|| broken("a tied point touches no body"))?;
            let other = views.get(second).ok_or_else(|| broken("a bond's body is missing"))?;
            let (element, values) = (other.at)(point.x, y).ok_or_else(|| {
                Error::Invalid(format!(
                    "body {first} touches body {second} at ({:.6e}, {:.6e}, {:.6e}), where the second has no cell: its faces named by the bond do not bound its material there",
                    point.x[0], point.x[1], point.x[2]
                ))
            })?;
            let (na, nb) = (point.values.len(), values.len());
            if point.tractions.len() != 9 * na {
                return Err(broken("a tied point's tractions"));
            }
            let pair = pairs.entry((first, point.element, second, element)).or_insert_with(|| Pair {
                flux: vec![0.0; 9 * na * nb],
                mass: vec![0.0; na * nb],
                own: vec![0.0; nb * nb],
            });
            if pair.mass.len() != na * nb || pair.own.len() != nb * nb {
                return Err(broken("an element's functions changed"));
            }
            let w = point.weight;
            for l in 0..3 {
                let Some(traction) = point.tractions.get(l * 3 * na..(l + 1) * 3 * na) else { continue };
                for (row, t) in traction.iter().enumerate() {
                    let scaled = w * t;
                    if scaled == 0.0 {
                        continue;
                    }
                    if let Some(out) = pair.flux.get_mut(row * 3 * nb..(row + 1) * 3 * nb) {
                        for (slot, value) in out.iter_mut().skip(l).step_by(3).zip(&values) {
                            *slot += scaled * value;
                        }
                    }
                }
            }
            for (row, na_value) in pair.mass.chunks_exact_mut(nb.max(1)).zip(point.values) {
                for (slot, value) in row.iter_mut().zip(&values) {
                    *slot += w * na_value * value;
                }
            }
            for (row, first_value) in pair.own.chunks_exact_mut(nb.max(1)).zip(&values) {
                for (slot, value) in row.iter_mut().zip(&values) {
                    *slot += w * first_value * value;
                }
            }
            Ok(())
        })?;
    }

    // The order of elimination: one body alone is in its own already.
    let global = |body: usize, function: usize| offsets.get(body).map_or(function, |offset| offset + function);
    let place: Vec<usize> = if views.len() == 1 && pairs.is_empty() {
        (0..total).collect()
    } else {
        let mut lists: Vec<Vec<usize>> = Vec::new();
        for (body, view) in views.iter().enumerate() {
            lists.extend(view.elements.iter().map(|functions| functions.iter().map(|function| global(body, *function)).collect()));
        }
        for (first, element, second, other) in pairs.keys() {
            let functions = |body: usize, element: usize| views.get(body).and_then(|view| view.elements.get(element)).copied().unwrap_or_default();
            lists.push(
                functions(*first, *element)
                    .iter()
                    .map(|f| global(*first, *f))
                    .chain(functions(*second, *other).iter().map(|f| global(*second, *f)))
                    .collect(),
            );
        }
        let mut positions = vec![[0.0; 3]; total];
        for (body, view) in views.iter().enumerate() {
            for function in 0..view.count {
                if let Some(slot) = positions.get_mut(global(body, function)) {
                    *slot = (view.position)(function);
                }
            }
        }
        let centres: Vec<[f64; 3]> = lists
            .iter()
            .map(|list| {
                let mut centre = [0.0; 3];
                for at in list.iter().filter_map(|node| positions.get(*node)) {
                    for a in 0..3 {
                        centre[a] += at[a] / list.len() as f64;
                    }
                }
                centre
            })
            .collect();
        let slices: Vec<&[usize]> = lists.iter().map(Vec::as_slice).collect();
        let order = element_dissection(total, &slices, &centres);
        let mut place = vec![usize::MAX; total];
        for (k, node) in order.iter().enumerate() {
            if let Some(slot) = place.get_mut(*node) {
                *slot = k;
            }
        }
        if place.contains(&usize::MAX) {
            return Err(broken("the order is not a permutation"));
        }
        place
    };
    let placed = |body: usize, element: usize| -> Vec<usize> {
        let functions = views.get(body).and_then(|view| view.elements.get(element)).copied().unwrap_or_default();
        functions.iter().filter_map(|function| place.get(global(body, *function)).copied()).collect()
    };

    // The matrix.
    let refine = views.first().map_or(0, |view| view.refine);
    let too_large = |entries: usize| {
        Error::Limit(format!("the stiffness would have {entries} entries; the limit is {MAX_MATRIX_ENTRIES} ({})", solve::smaller(refine)))
    };
    let layout = DofLayout::uniform(0..total as u32, DofSet::TRANSLATIONS);
    if layout.dofs() != 3 * total {
        return Err(broken("the layout does not hold three unknowns per function"));
    }
    let (mut stiffness, stiffness_entries) = {
        let bodies = views.iter().enumerate().flat_map(|(body, view)| (0..view.elements.len()).map(move |element| (body, element)));
        let coupled = pairs.keys().map(|(first, element, second, other)| {
            let mut both = placed(*first, *element);
            both.extend(placed(*second, *other));
            both
        });
        let graph = NodeGraph::new(total, bodies.map(|(body, element)| placed(body, element)).chain(coupled));
        let entries = graph.entries(&layout);
        if entries > MAX_MATRIX_ENTRIES {
            return Err(too_large(entries));
        }
        // A round of [`Options::adapt`] is refused here, before its cells are integrated, when
        // its factor cannot fit: the larger system of the same part is taken to have no fewer
        // entries of the factor for each of the stiffness than the solve before it had (the
        // fill of a nested dissection grows with the system).
        if let [view] = views
            && let Some((before, factor)) = view.before.filter(|(before, _)| *before > 0 && entries >= *before)
        {
            let least = entries as f64 * factor as f64 / before as f64;
            if least > limits.max_factor_entries as f64 {
                return Err(Error::Limit(format!(
                    "the factorization would need {least:.0} entries or more; the limit is {} ({})",
                    limits.max_factor_entries,
                    solve::smaller(refine)
                )));
            }
        }
        (graph.zero_matrix(&layout), entries)
    };
    let mut load = vec![0.0; 3 * total];
    for (body, view) in views.iter().enumerate() {
        (view.assemble)(&mut |element: usize, cell_stiffness: &[f64], cell_load: &[f64]| -> Result<()> {
            let nodes = placed(body, element);
            add_block(&mut stiffness, &layout, &nodes, DofSet::TRANSLATIONS, cell_stiffness)?;
            for (a, node) in nodes.iter().enumerate() {
                for i in 0..3 {
                    if let (Some(sum), Some(value)) = (load.get_mut(3 * node + i), cell_load.get(3 * a + i)) {
                        *sum += value;
                    }
                }
            }
            Ok(())
        })?;
    }
    // The bonds' blocks, with the parameter of the cell whose traction they take.
    for ((first, element, second, other), pair) in &pairs {
        let beta = views.get(*first).map_or(0.0, |view| (view.nitsche)(*element));
        if !(beta > 0.0 && beta.is_finite()) {
            return Err(broken("a tied cell has no parameter"));
        }
        let (mut nodes, others) = (placed(*first, *element), placed(*second, *other));
        let (na, nb) = (nodes.len(), others.len());
        nodes.extend(others);
        let n = 3 * (na + nb);
        let mut block = vec![0.0; n * n];
        for row in 0..3 * na {
            for (fb, mass) in pair.mass.get((row / 3) * nb..(row / 3 + 1) * nb).unwrap_or_default().iter().enumerate() {
                for l in 0..3 {
                    let flux = pair.flux.get(row * 3 * nb + 3 * fb + l).copied().unwrap_or(0.0);
                    let value = flux - if l == row % 3 { beta * mass } else { 0.0 };
                    let column = 3 * (na + fb) + l;
                    if let Some(slot) = block.get_mut(row * n + column) {
                        *slot = value;
                    }
                    if let Some(slot) = block.get_mut(column * n + row) {
                        *slot = value;
                    }
                }
            }
        }
        for (fb, row) in pair.own.chunks_exact(nb.max(1)).enumerate() {
            for (other, mass) in row.iter().enumerate() {
                for l in 0..3 {
                    if let Some(slot) = block.get_mut((3 * (na + fb) + l) * n + 3 * (na + other) + l) {
                        *slot = beta * mass;
                    }
                }
            }
        }
        add_block(&mut stiffness, &layout, &nodes, DofSet::TRANSLATIONS, &block)?;
    }
    drop(pairs);

    // The solution. The factorization reads the matrix's lower triangle, and nothing reads the
    // matrix after it.
    let factor = Factor::new(&UpperColumns::from_symmetric(&stiffness), limits.max_factor_entries).map_err(|e| solve::factor_error(e, refine))?;
    let factor_entries = factor.entries();
    let mut u = load;
    factor.solve(&mut u)?;
    drop(factor);
    drop(stiffness);
    if u.iter().any(|x| !x.is_finite()) {
        return Err(Error::Singular("the solution is not finite: the stiffness is singular to working precision".into()));
    }
    Ok(views
        .iter()
        .enumerate()
        .map(|(body, view)| {
            let mut own = vec![0.0; 3 * view.count];
            for (function, out) in own.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                if let Some(values) = place.get(global(body, function)).and_then(|node| u.get(3 * node..3 * node + 3)) {
                    out.copy_from_slice(values);
                }
            }
            Solved { u: own, stiffness_entries, factor_entries }
        })
        .collect())
}
