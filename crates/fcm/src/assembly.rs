//! Several bodies in one solve, joined where they touch.
//!
//! Every body is solved as one body is ([`crate::solve`]): on a grid of its own, with its own
//! cell size, material, supports and loads, its cells cut by its own surface. Two bodies share
//! no unknown, however close they are, so nothing joins them by accident; what joins them is a
//! [`Bond`], which names faces of the two that may touch and says how the two hold each other
//! there: bonded, or sliding ([`Grip`]).
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
//! # A sliding connection
//! Two bodies that keep touching and slide along each other without friction share only the
//! component of the displacement across the faces, and pull on each other only across them.
//! With n the normal of A's face, tₙ(u) = n · σ_A(u) n and [u]ₙ = [u] · n, the terms are those
//! of a bond with the traction and the jump replaced by their parts along n:
//!
//! a(u, v) += − ∫ tₙ(u_A) [v]ₙ dΓ − ∫ tₙ(v_A) [u]ₙ dΓ + β ∫ [u]ₙ [v]ₙ dΓ,
//!
//! the frictionless limit of the sliding interface of C. Annavarapu, M. Hautefeuille,
//! J. E. Dolbow, *A Nitsche stabilized finite element method for frictional sliding on embedded
//! interfaces. Part I: Single interface*, Comput. Methods Appl. Mech. Engrg. 268:417–436, 2014,
//! for two bodies that do not part: the solve is linear, so a sliding connection pulls as well
//! as it pushes (it is not contact). The tangential traction of the exact solution vanishes on
//! Γ, so the terms are consistent like a bond's.
//!
//! **The normal of a curved face.** The triangles of a round face are flat, and held across
//! every one of them a pin could not turn in its hole: the facets would lock it like a spline.
//! A face named in [`Part::curved`] with the surface it lies on (a cylinder, a sphere, a cone,
//! a torus) takes its normal from that surface at every point, and the pin turns and slides.
//! Other curved faces keep their triangles' normals, which is right for a face swept along a
//! line (it slides along the line) and stiffer than the surface for any other.
//!
//! # Bodies left free
//! A sliding connection holds less than a bond, and bodies it leaves free to move have no
//! static solution. Before anything is solved, [`free_motions`] finds the rigid motions that no
//! support and no connection resists (the null space of the bodies' restraints on their six
//! motions each), and the solve is refused with them.
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

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use fieldcraft_fem::{DofLayout, DofSet};
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_solver::Limits;
use fieldcraft_solver::cholesky::{Factor, FactorError, UpperColumns, predicted_of_nodes};
use fieldcraft_solver::ordering::element_dissection;
use fieldcraft_solver::sparse::{Csr, NodeGraph, add_block};

use crate::blocked::Blocked;
use crate::cell::{Curved, Grip};
use crate::condense::condense;
use crate::grid::{self, Grid, cross, dot, sub};
use crate::islands::closest_point;
use crate::iterative;
use crate::refine::Tree;
use crate::solve::{self, Load, MAX_MATRIX_ENTRIES, Material, Options, Round, Solution, Solved, Solver, Support, Tied, Ties, View};
use crate::space::NONE;
use crate::{Error, Result};

/// Unknowns from which a system may be solved by iteration ([`Solver::Auto`]): below, the
/// factorization takes under a second, and its result is exact.
const ITERATIVE_FROM: usize = 50_000;
/// What an iteration costs, in multiply–adds of a factorization for every entry of the
/// stiffness, by how many times cells are divided (not, once, twice, more): its set-up (the
/// cells' own unknowns, the blocks' factors, what follows the far-reaching functions) and its
/// rounds. Measured at degree 3 on 24 threads on the oracle's bracket, as the factorization's
/// work for an entry times the iteration's time over the factorization's: 560 on cells of
/// 2.2 mm (38 rounds), 620 on cells of 4.4 mm divided once (40 rounds), 1000 on cells of 8 mm
/// divided twice (63 rounds), and with cells divided three times 1150 on cells of 16 mm (49
/// rounds) and 2450 on cells of 8 mm (66 rounds).
const ITERATION: [f64; 4] = [600.0, 600.0, 1000.0, 2400.0];
/// How many times that where bodies are connected (47 to 85 rounds measured on three blocks
/// bonded and sliding, for 40 of one body), and where a body has the trunk basis, which is what
/// a caller gives one that is thin on its cells (225 rounds on a plate a tenth of its cells
/// thick).
const CONNECTED: f64 = 2.0;
const THIN: f64 = 3.0;
/// Poisson's ratio above which a body's system is left to the factorization: towards one half
/// the rounds of the iteration grow (32 at 0.3, 47 at 0.49, 69 at 0.499 and 166 at 0.4999 on
/// one block).
const NEARLY_INCOMPRESSIBLE: f64 = 0.499;

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
    /// The grid of this body. The rounds of [`Options::adapt`] are run over all the bodies
    /// together ([`solve_assembly`]).
    pub options: Options,
    /// The faces that lie on a cylinder, a sphere, a cone or a torus, each with its surface: a
    /// face that slides on another body takes its normal from it ([`Curved`]). Faces not listed
    /// have the normals of their triangles, which is exact for flat ones.
    pub curved: &'a [(u32, Curved)],
}

/// Two bodies joined where faces of theirs touch.
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
    /// How the two hold each other where they touch.
    pub grip: Grip,
}

/// The triangles of the faces of one body that a bond names (or of all its faces), sorted into
/// buckets.
struct Reach {
    triangles: Vec<[[f64; 3]; 3]>,
    /// The face of each.
    tags: Vec<u32>,
    lo: [f64; 3],
    step: [f64; 3],
    count: [usize; 3],
    buckets: Vec<Vec<u32>>,
    tolerance: f64,
}

impl Reach {
    fn new(surface: &BoundarySurface, faces: Option<&BTreeSet<u32>>, tolerance: f64) -> Result<Reach> {
        let points = surface.points();
        let (mut triangles, mut tags) = (Vec::new(), Vec::new());
        for (triangle, tag) in surface.triangles().iter().zip(surface.tri_face()) {
            if faces.is_some_and(|faces| !faces.contains(tag)) {
                continue;
            }
            let corner = |v: u32| points.get(v as usize).copied().ok_or_else(|| Error::Internal("a surface triangle names a missing vertex".into()));
            triangles.push([corner(triangle[0])?, corner(triangle[1])?, corner(triangle[2])?]);
            tags.push(*tag);
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for corner in triangles.iter().flatten() {
            for a in 0..3 {
                lo[a] = lo[a].min(corner[a] - tolerance);
                hi[a] = hi[a].max(corner[a] + tolerance);
            }
        }
        let mut reach = Reach { triangles, tags, lo, step: [1.0; 3], count: [1; 3], buckets: Vec::new(), tolerance };
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
        self.nearest_triangle(x).map(|(y, apart, _)| (y, apart))
    }

    /// [`Reach::nearest`], with the triangle the point is on.
    fn nearest_triangle(&self, x: [f64; 3]) -> Option<([f64; 3], f64, usize)> {
        if self.buckets.is_empty() || (0..3).any(|a| !(x[a] >= self.lo[a] && x[a] <= self.lo[a] + self.step[a] * self.count[a] as f64)) {
            return None;
        }
        let bucket = self.buckets.get(self.bucket(0, x[0]) + self.count[0] * (self.bucket(1, x[1]) + self.count[1] * self.bucket(2, x[2])))?;
        let mut best: Option<([f64; 3], f64, usize)> = None;
        for t in bucket {
            let Some(triangle) = self.triangles.get(*t as usize) else { continue };
            let y = closest_point(x, triangle);
            let apart = sub(x, y);
            let apart = dot(apart, apart);
            if apart <= self.tolerance * self.tolerance && best.is_none_or(|(_, least, _)| apart < least) {
                best = Some((y, apart, *t as usize));
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
    grip: Grip,
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
    /// Of each body: the curved surfaces its faces lie on.
    curved: &'a [BTreeMap<u32, Curved>],
    /// Of each body: what its solve takes from the rounds before ([`Options::adapt`]).
    rounds: &'a [Round],
    /// The cells of each body, once its solve has them: a copy the other bodies' pieces are
    /// cut along.
    frames: &'a [OnceLock<(Grid, Tree)>],
    /// Set when the unknowns are only counted: the system is not solved, and the count is left
    /// here.
    count: Option<&'a Cell<Option<usize>>>,
    solved: RefCell<Vec<Option<Solved>>>,
    solutions: RefCell<Vec<Option<Solution>>>,
}

/// The connection through which a point of a tied face of `body` touches another body (its
/// place among the couplings), and the nearest point of that body's faces.
fn partner(couplings: &[Coupling], body: usize, tag: u32, x: [f64; 3]) -> Option<(usize, [f64; 3])> {
    let mut best: Option<(usize, [f64; 3], f64)> = None;
    for (at, coupling) in couplings.iter().enumerate().filter(|(_, coupling)| coupling.flux == body && coupling.faces.contains(&tag)) {
        if let Some((y, apart)) = coupling.reach.nearest(x)
            && best.is_none_or(|(_, _, least)| apart < least)
        {
            best = Some((at, y, apart));
        }
    }
    best.map(|(at, y, _)| (at, y))
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
    let touching = |tag: u32, piece: &[[f64; 3]; 3], visit: &mut dyn FnMut(&[[f64; 3]; 3], &dyn Fn([f64; 3]) -> bool, Grip)| {
        // The bodies this face is tied to and how, each once.
        let ways: BTreeSet<(usize, Grip)> = couplings
            .iter()
            .filter(|coupling| coupling.flux == index && coupling.faces.contains(&tag))
            .map(|coupling| (coupling.other, coupling.grip))
            .collect();
        for (other, grip) in ways {
            let touches = |x: [f64; 3]| {
                partner(couplings, index, tag, x)
                    .and_then(|(found, _)| couplings.get(found))
                    .is_some_and(|coupling| coupling.other == other && coupling.grip == grip)
            };
            match frames.get(other).and_then(OnceLock::get) {
                Some((grid, tree)) => leaf_parts(grid, tree, piece, &mut |part| visit(part, &touches, grip)),
                None => visit(piece, &touches, grip),
            }
        }
    };
    let ties = Ties {
        faces: shared.tied.get(index).ok_or_else(|| broken("a body's ties are missing"))?,
        touching: &touching,
        curved: shared.curved.get(index).ok_or_else(|| broken("a body's curved faces are missing"))?,
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
            if let Some(count) = shared.count {
                // Counting: nothing is integrated or solved, and this error ends the bodies'
                // solves; the caller reads the count.
                count.set(Some(views.iter().fold(0usize, |sum, view| sum.saturating_add(3 * view.count))));
                return Err(broken("counted"));
            }
            let touched = |body: usize, tag: u32, x: [f64; 3]| {
                partner(couplings, body, tag, x).and_then(|(found, y)| couplings.get(found).map(|coupling| (coupling.other, y)))
            };
            let solved = solve_system(&views, &touched, shared.limits)?;
            let mut kept = shared.solved.try_borrow_mut().map_err(|_| broken("the solutions are in use"))?;
            *kept = solved.into_iter().map(Some).collect();
        }
        let mut kept = shared.solved.try_borrow_mut().map_err(|_| broken("the solutions are in use"))?;
        kept.get_mut(index).and_then(Option::take).ok_or_else(|| broken("a body has no solution"))
    };
    let fictitious = shared.fictitious.get(index).copied().unwrap_or(part.options.fictitious);
    let round = shared.rounds.get(index).ok_or_else(|| broken("a body's round is missing"))?;
    let solution =
        solve::solve_tied(part.surface, &part.material, part.supports, part.loads, &part.options, shared.limits, fictitious, round, &ties, &join)?;
    let mut solutions = shared.solutions.try_borrow_mut().map_err(|_| broken("the solutions are in use"))?;
    if let Some(slot) = solutions.get_mut(index) {
        *slot = Some(solution);
    }
    Ok(())
}

/// The bonds as the solves read them, and of each body the faces a bond takes the traction of,
/// whether a bond holds it, and the curved surfaces its faces lie on.
struct Bonded {
    couplings: Vec<Coupling>,
    tied: Vec<BTreeSet<u32>>,
    held: Vec<bool>,
    curved: Vec<BTreeMap<u32, Curved>>,
}

/// The bonds of an assembly, checked.
fn couplings(parts: &[Part], bonds: &[Bond]) -> Result<Bonded> {
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
        couplings.push(Coupling { flux: flux.0, faces, other: other.0, reach: Reach::new(surface, Some(&others), bond.tolerance)?, grip: bond.grip });
    }
    let curved = parts.iter().map(|part| part.curved.iter().copied().collect()).collect();
    Ok(Bonded { couplings, tied, held, curved })
}

/// A motion nothing resists: bodies that move as rigid pieces against no support and no
/// connection ([`free_motions`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Motion {
    /// The bodies that move, each with the displacement of the middle of its box and its turn
    /// about that point: a vector along the axis, as long as the displacement the turn gives a
    /// point half the box's diagonal away. All are scaled together, the largest number to 1.
    pub bodies: Vec<(usize, [f64; 3], [f64; 3])>,
}

/// The least share of the strongest restraint that still holds a motion ([`free_motions`]): a
/// connection over a patch a hundred-thousandth of a body's size across still holds the body
/// against turning about it, with (1e-5)², and the rounding of the decomposition, about 1e-13
/// for the most bodies there can be (N. J. Higham, *Analysis of the Cholesky decomposition of a
/// semi-definite matrix*, in Reliable Numerical Computation, Oxford University Press,
/// 1990, pp. 161–185), does not.
const HELD: f64 = 1e-10;
/// A body moves in a motion when its share of it is above this part of the largest.
const MOVES: f64 = 1e-6; // unit-factors: a ratio of two displacements

/// The rigid motions of the bodies that nothing resists: no support and no bond, and no sliding
/// connection across its faces. With any, the bodies have no static solution; an assembly that
/// can be solved has none.
///
/// Every body has six rigid motions, and every restraint resists some of them: a support those
/// that move its face in a fixed component, a bond those that move the two bodies apart where
/// they touch, a sliding connection those that move them apart across the faces. The energy of
/// each restraint under the motions, ∫ |P (r_A − r_B)|² dΓ over its area, summed over all of
/// them is a positive semi-definite matrix of six rows a body, and the motions it gives no
/// energy are free. They are found by its Cholesky decomposition with complete pivoting, which
/// reveals its rank (Higham 1990, as at [`HELD`]): one motion for every pivot that is left
/// below [`HELD`] of the largest.
pub fn free_motions(parts: &[Part], bonds: &[Bond]) -> Result<Vec<Motion>> {
    let Bonded { couplings, curved, .. } = couplings(parts, bonds)?;
    motions(parts, &couplings, &curved)
}

/// [`free_motions`] of the bonds as the solves read them.
fn motions(parts: &[Part], couplings: &[Coupling], curved: &[BTreeMap<u32, Curved>]) -> Result<Vec<Motion>> {
    let n = 6 * parts.len();
    // Each body's motions: three displacements, and three turns about the middle of its box,
    // scaled by half its diagonal so that all six move its points by about as much.
    let frames: Vec<([f64; 3], f64)> = parts
        .iter()
        .map(|part| {
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for point in part.surface.points() {
                for a in 0..3 {
                    lo[a] = lo[a].min(point[a]);
                    hi[a] = hi[a].max(point[a]);
                }
            }
            let reach = 0.5 * dot(sub(hi, lo), sub(hi, lo)).sqrt();
            ([0, 1, 2].map(|a| 0.5 * (lo[a] + hi[a])), if reach > 0.0 && reach.is_finite() { reach } else { 1.0 })
        })
        .collect();
    let moved = |body: usize, x: [f64; 3]| -> [[f64; 3]; 6] {
        let (centre, reach) = frames.get(body).copied().unwrap_or(([0.0; 3], 1.0));
        let arm = sub(x, centre).map(|c| c / reach);
        let mut motions = [[0.0; 3]; 6];
        for axis in 0..3 {
            motions[axis][axis] = 1.0;
            let mut along = [0.0; 3];
            along[axis] = 1.0;
            motions[3 + axis] = cross(along, arm);
        }
        motions
    };
    let sampled = |body: usize, faces: &dyn Fn(u32) -> bool, visit: &mut dyn FnMut(u32, [f64; 3], [f64; 3], f64)| {
        if let Some(part) = parts.get(body) {
            side_points(part.surface, faces, visit);
        }
    };
    let mut gram = vec![0.0; n * n];
    for (body, part) in parts.iter().enumerate() {
        for support in part.supports {
            let (mut energy, mut area) = ([[0.0; 6]; 6], 0.0);
            sampled(body, &|tag| tag == support.face, &mut |_, x, _, weight| {
                let motions = moved(body, x);
                for i in 0..6 {
                    for j in 0..6 {
                        energy[i][j] += weight * (0..3).filter(|k| support.fixed[*k]).map(|k| motions[i][k] * motions[j][k]).sum::<f64>();
                    }
                }
                area += weight;
            });
            if !(area > 0.0) {
                continue;
            }
            for i in 0..6 {
                for j in 0..6 {
                    if let Some(slot) = gram.get_mut((6 * body + i) * n + 6 * body + j) {
                        *slot += energy[i][j] / area;
                    }
                }
            }
        }
    }
    for coupling in couplings {
        let (first, second) = (coupling.flux, coupling.other);
        let surfaces = curved.get(first);
        let (mut energy, mut area) = (vec![[0.0; 12]; 12], 0.0);
        sampled(first, &|tag| coupling.faces.contains(&tag), &mut |tag, x, facet, weight| {
            if coupling.reach.nearest(x).is_none() {
                return;
            }
            let (own, other) = (moved(first, x), moved(second, x));
            // The two bodies' motions apart: the first's, and the second's the other way.
            let apart: [[f64; 3]; 12] = std::array::from_fn(|i| if i < 6 { own[i] } else { other[i - 6].map(|c| -c) });
            let normal = match coupling.grip {
                Grip::Bonded => None,
                Grip::Sliding => Some(surfaces.and_then(|surfaces| surfaces.get(&tag)).map_or(facet, |surface| surface.normal(x, facet))),
            };
            for i in 0..12 {
                for j in 0..12 {
                    energy[i][j] += weight
                        * match normal {
                            None => dot(apart[i], apart[j]),
                            Some(normal) => dot(apart[i], normal) * dot(apart[j], normal),
                        };
                }
            }
            area += weight;
        });
        if !(area > 0.0) {
            continue;
        }
        let row = |i: usize| if i < 6 { 6 * first + i } else { 6 * second + i - 6 };
        for i in 0..12 {
            for j in 0..12 {
                if let Some(slot) = gram.get_mut(row(i) * n + row(j)) {
                    *slot += energy[i][j] / area;
                }
            }
        }
    }

    // The decomposition with complete pivoting: the largest of the diagonal left is the next
    // pivot, and when it is below the share of the first that holds, the rest is free.
    let at = |i: usize, j: usize| i * n + j;
    let mut order: Vec<usize> = (0..n).collect();
    let largest = (0..n).map(|i| gram.get(at(i, i)).copied().unwrap_or(0.0)).fold(0.0, f64::max);
    let mut rank = 0;
    while rank < n {
        let (mut pivot, mut value) = (rank, f64::NEG_INFINITY);
        for i in rank..n {
            let diagonal = gram.get(at(i, i)).copied().unwrap_or(0.0);
            if diagonal > value {
                (pivot, value) = (i, diagonal);
            }
        }
        if !(value > HELD * largest && value.is_finite()) {
            break;
        }
        if pivot != rank {
            order.swap(pivot, rank);
            for k in 0..n {
                gram.swap(at(pivot, k), at(rank, k));
            }
            for k in 0..n {
                gram.swap(at(k, pivot), at(k, rank));
            }
        }
        let root = value.sqrt();
        for i in rank + 1..n {
            if let Some(slot) = gram.get_mut(at(i, rank)) {
                *slot /= root;
            }
        }
        for i in rank + 1..n {
            let left = gram.get(at(i, rank)).copied().unwrap_or(0.0);
            if left == 0.0 {
                continue;
            }
            for j in rank + 1..n {
                let other = gram.get(at(j, rank)).copied().unwrap_or(0.0);
                if let Some(slot) = gram.get_mut(at(i, j)) {
                    *slot -= left * other;
                }
            }
        }
        if let Some(slot) = gram.get_mut(at(rank, rank)) {
            *slot = root;
        }
        rank += 1;
    }
    // One free motion for every column left: itself, with what the pivoted ones must add to it
    // to give no energy, −L₁₁⁻ᵀ of its row of the factor.
    let mut free = Vec::with_capacity(n - rank);
    for column in rank..n {
        let mut held: Vec<f64> = (0..rank).map(|k| -gram.get(at(column, k)).copied().unwrap_or(0.0)).collect();
        for k in (0..rank).rev() {
            let mut value = held.get(k).copied().unwrap_or(0.0);
            for later in k + 1..rank {
                value -= gram.get(at(later, k)).copied().unwrap_or(0.0) * held.get(later).copied().unwrap_or(0.0);
            }
            let pivot = gram.get(at(k, k)).copied().unwrap_or(1.0);
            if let Some(slot) = held.get_mut(k) {
                *slot = value / pivot;
            }
        }
        let mut motion = vec![0.0; n];
        for (k, value) in held.iter().enumerate() {
            if let Some(slot) = order.get(k).and_then(|mode| motion.get_mut(*mode)) {
                *slot = *value;
            }
        }
        if let Some(slot) = order.get(column).and_then(|mode| motion.get_mut(*mode)) {
            *slot = 1.0;
        }
        let most = motion.iter().fold(0.0f64, |most, value| most.max(value.abs()));
        if !(most > 0.0 && most.is_finite()) {
            return Err(broken("a free motion is not finite"));
        }
        let bodies = motion
            .as_chunks::<6>()
            .0
            .iter()
            .enumerate()
            .filter(|(_, own)| own.iter().any(|value| value.abs() > MOVES * most))
            .map(|(body, own)| (body, [own[0], own[1], own[2]].map(|value| value / most), [own[3], own[4], own[5]].map(|value| value / most)))
            .collect();
        free.push(Motion { bodies });
    }
    Ok(free)
}

/// Visit the triangles of the faces of a surface that `faces` keeps, each by the middles of its
/// sides with a third of its area (exact for what is quadratic over a triangle): the face, the
/// point, the triangle's unit normal and the weight.
fn side_points(surface: &BoundarySurface, faces: &dyn Fn(u32) -> bool, visit: &mut dyn FnMut(u32, [f64; 3], [f64; 3], f64)) {
    let points = surface.points();
    for (triangle, tag) in surface.triangles().iter().zip(surface.tri_face()) {
        if !faces(*tag) {
            continue;
        }
        let [Some(a), Some(b), Some(c)] = triangle.map(|v| points.get(v as usize).copied()) else { continue };
        let doubled = cross(sub(b, a), sub(c, a));
        let length = dot(doubled, doubled).sqrt();
        if !(length > 0.0 && length.is_finite()) {
            continue;
        }
        let normal = doubled.map(|c| c / length);
        for (p, q) in [(a, b), (b, c), (c, a)] {
            visit(*tag, [0, 1, 2].map(|i| 0.5 * (p[i] + q[i])), normal, length / 6.0);
        }
    }
}

/// The area over which the faces of each bond touch, as the solve takes it: of the faces whose
/// traction the bond takes (the smaller ones), what lies within the bond's tolerance of the
/// other body's. Zero for faces that do not touch: such a bond joins nothing.
pub fn bond_areas(parts: &[Part], bonds: &[Bond]) -> Result<Vec<f64>> {
    let Bonded { couplings, .. } = couplings(parts, bonds)?;
    Ok(couplings
        .iter()
        .map(|coupling| {
            let mut area = 0.0;
            if let Some(part) = parts.get(coupling.flux) {
                side_points(part.surface, &|tag| coupling.faces.contains(&tag), &mut |_, x, _, weight| {
                    if coupling.reach.nearest(x).is_some() {
                        area += weight;
                    }
                });
            }
            area
        })
        .collect())
}

/// What a solve of bodies that can still move is refused with.
fn refuse_free(motions: &[Motion]) -> Result<()> {
    let Some(motion) = motions.first() else { return Ok(()) };
    let bodies: Vec<String> = motion.bodies.iter().map(|(body, _, _)| body.to_string()).collect();
    Err(Error::Invalid(format!(
        "nothing holds body {} against {} of its rigid motions: a body needs supports and connections that leave it no way to slide or turn",
        bodies.join(", "),
        motions.len()
    )))
}

/// The unknowns [`solve_assembly`] would solve for: the bodies' grids are built and their cells
/// sorted, and nothing is integrated or solved. For choosing cell sizes that fit a budget
/// before the work is done.
pub fn assembly_unknowns(parts: &[Part], bonds: &[Bond], limits: &Limits) -> Result<usize> {
    let Bonded { couplings, tied, held, curved } = couplings(parts, bonds)?;
    let fictitious: Vec<f64> = parts.iter().map(|part| part.options.fictitious).collect();
    let frames: Vec<OnceLock<(Grid, Tree)>> = (0..parts.len()).map(|_| OnceLock::new()).collect();
    let rounds: Vec<Round> = parts.iter().map(|_| Round::default()).collect();
    let count = Cell::new(None);
    let shared = Shared {
        parts,
        couplings: &couplings,
        limits,
        tied: &tied,
        held: &held,
        fictitious: &fictitious,
        curved: &curved,
        rounds: &rounds,
        frames: &frames,
        count: Some(&count),
        solved: RefCell::new(Vec::new()),
        solutions: RefCell::new(Vec::new()),
    };
    match (nest(&shared, 0, None), count.get()) {
        (_, Some(unknowns)) => Ok(unknowns),
        (Err(e), None) => Err(e),
        (Ok(()), None) => Err(broken("the unknowns were not counted")),
    }
}

/// Solve several bodies as one step, joined where the faces a [`Bond`] names touch; the
/// solutions come in the bodies' order. A body needs supports of its own or connections that
/// hold it against every rigid motion ([`free_motions`]). Refused like a single body's solve: a
/// system above the limits, a body or a group of bodies that can still move.
///
/// **Rounds that follow the solution** ([`Options::adapt`]) are run as for one body, over all
/// the bodies together: as many as the body that asks for the most, each a whole solve. The
/// cells are ranked across the bodies by the one indicator, so a round divides cells where the
/// assembly is least resolved, in whichever body that is, and none in a body that is resolved.
/// The rounds end once one moves the strain energy of all the bodies by no more than the
/// smallest [`Options::settle`] among them; each solution's [`crate::Stats::rounds`],
/// [`crate::Stats::settled`] and [`crate::Stats::left_out`] are the assembly's.
pub fn solve_assembly(parts: &[Part], bonds: &[Bond], limits: &Limits) -> Result<Vec<Solution>> {
    let Bonded { couplings, tied, held, curved } = couplings(parts, bonds)?;
    refuse_free(&motions(parts, &couplings, &curved)?)?;
    let most: Vec<f64> = parts.iter().map(|part| solve::most_fictitious(&part.material)).collect::<Result<_>>()?;
    // One solve of all the bodies, each dividing the cells its round asks for.
    let attempt = |rounds: &[Round]| -> Result<Vec<Solution>> {
        // As small a fictitious stiffness as the factorization allows, as for one body: a
        // hundred times more at a time for every body, up to what its material leaves. Each
        // step raises one at least, so the loop ends.
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
                curved: &curved,
                rounds,
                frames: &frames,
                count: None,
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
                    return shared
                        .solutions
                        .into_inner()
                        .into_iter()
                        .map(|solution| solution.ok_or_else(|| broken("a body has no solution")))
                        .collect();
                }
            }
        }
    };
    let mut rounds: Vec<Round> = parts.iter().map(|_| Round::default()).collect();
    let mut solutions = attempt(&rounds)?;
    let asked = parts.iter().map(|part| part.options.adapt).max().unwrap_or(0).min(grid::MAX_REFINE);
    let settle = parts.iter().map(|part| part.options.settle).fold(f64::INFINITY, f64::min);
    let energy = |solutions: &[Solution]| solutions.iter().map(|solution| solution.strain_energy).sum::<f64>();
    for round in 0..asked {
        let marked = solve::least_resolved(&solutions.iter().collect::<Vec<_>>(), Options::SHARE);
        let mut more = false;
        for ((own, cells), solution) in rounds.iter_mut().zip(marked).zip(&solutions) {
            more |= own.ask(cells, solution);
        }
        if !more {
            break;
        }
        match attempt(&rounds) {
            Ok(mut finer) => {
                let (now, before) = (energy(&finer), energy(&solutions));
                let settled = if now != 0.0 { (now - before) / now } else { 0.0 };
                for solution in &mut finer {
                    solution.stats.rounds = round + 1;
                    solution.stats.settled = settled;
                }
                solutions = finer;
                if settled.abs().sqrt() <= settle {
                    break;
                }
            }
            // Too large with this round's cells, or not to be solved with them: the solutions
            // before it stand.
            Err(Error::Limit(_) | Error::Singular(_)) => {
                for solution in &mut solutions {
                    solution.stats.left_out = true;
                }
                break;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(solutions)
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

/// Faces of two bodies that touch, as [`find_bonds`] found them.
#[derive(Debug, Clone, PartialEq)]
pub struct Touch {
    pub bond: Bond,
    /// The area over which they touch, as the triangles of the two surfaces sample it.
    pub area: f64,
    /// Every face of the first body with a face of the second that it touches, and the area
    /// the two share.
    pub pairs: Vec<(u32, u32, f64)>,
}

/// A face touches another over at least this share of the smaller of the two: less is an edge
/// or a corner that rests on a face, which is no bond.
const LEAST_SHARE: f64 = 0.01;

/// The faces of every two bodies that touch: lie within `tolerance` of each other, facing each
/// other, over more than an edge. One [`Bond`] for each two bodies that have such faces, with
/// all of them. `tolerance` is a length above the chordal tolerance of the triangulations.
///
/// A triangle touches where sample points of it (its middle and a point towards each corner)
/// have a triangle of the other body within the tolerance that faces the other way; each body's
/// triangles are sampled against the other's, so a small face on a large one is found from the
/// small one's side.
pub fn find_bonds(surfaces: &[&BoundarySurface], tolerance: f64) -> Result<Vec<Touch>> {
    if !(tolerance > 0.0 && tolerance.is_finite()) {
        return Err(Error::Invalid(format!("the tolerance {tolerance} is not a positive length")));
    }
    if surfaces.len() > MAX_PARTS {
        return Err(Error::Invalid(format!("an assembly has at most {MAX_PARTS} bodies; this one has {}", surfaces.len())));
    }
    let reaches: Vec<Reach> = surfaces.iter().map(|surface| Reach::new(surface, None, tolerance)).collect::<Result<_>>()?;
    let normal = |triangle: &[[f64; 3]; 3]| cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]));
    // The area of each face of a body.
    let areas = |reach: &Reach| {
        let mut areas: BTreeMap<u32, f64> = BTreeMap::new();
        for (triangle, tag) in reach.triangles.iter().zip(&reach.tags) {
            let doubled = normal(triangle);
            *areas.entry(*tag).or_insert(0.0) += 0.5 * dot(doubled, doubled).sqrt();
        }
        areas
    };
    // The area of each face of a body against each face of another, where triangles of the
    // first touch the second.
    let touching = |from: &Reach, onto: &Reach| {
        let mut found: BTreeMap<(u32, u32), f64> = BTreeMap::new();
        let end = [0, 1, 2].map(|a| onto.lo[a] + onto.step[a] * onto.count[a] as f64);
        for (triangle, tag) in from.triangles.iter().zip(&from.tags) {
            // Outside the other body's box nothing touches.
            if (0..3).any(|a| triangle.iter().all(|corner| corner[a] < onto.lo[a]) || triangle.iter().all(|corner| corner[a] > end[a])) {
                continue;
            }
            let doubled = normal(triangle);
            let area = 0.5 * dot(doubled, doubled).sqrt();
            if !(area > 0.0) {
                continue;
            }
            for weights in [[1.0 / 3.0; 3], [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]] {
                let x = [0, 1, 2].map(|a| weights[0] * triangle[0][a] + weights[1] * triangle[1][a] + weights[2] * triangle[2][a]);
                let Some((_, _, other)) = onto.nearest_triangle(x) else { continue };
                let (Some(facing), Some(other_tag)) = (onto.triangles.get(other).map(normal), onto.tags.get(other)) else { continue };
                if dot(doubled, facing) < 0.0 {
                    *found.entry((*tag, *other_tag)).or_insert(0.0) += 0.25 * area;
                }
            }
        }
        found
    };
    let mut touches = Vec::new();
    for (first, reach) in reaches.iter().enumerate() {
        let own = areas(reach);
        for (second, other) in reaches.iter().enumerate().skip(first + 1) {
            let theirs = areas(other);
            // Either body's estimate of every pair of faces: the larger of the two.
            let mut pairs = touching(reach, other);
            for ((b, a), area) in touching(other, reach) {
                let seen = pairs.entry((a, b)).or_insert(0.0);
                *seen = seen.max(area);
            }
            let (mut faces, mut others, mut total, mut joined) = (BTreeSet::new(), BTreeSet::new(), 0.0, Vec::new());
            for ((a, b), area) in pairs {
                // Two faces touch over no more than the smaller of them.
                let smaller = own.get(&a).copied().unwrap_or(0.0).min(theirs.get(&b).copied().unwrap_or(0.0));
                let area = area.min(smaller);
                if area > LEAST_SHARE * smaller && area > 0.0 {
                    faces.insert(a);
                    others.insert(b);
                    total += area;
                    joined.push((a, b, area));
                }
            }
            if !faces.is_empty() {
                let bond = Bond {
                    first: (first, faces.into_iter().collect()),
                    second: (second, others.into_iter().collect()),
                    tolerance,
                    grip: Grip::Bonded,
                };
                touches.push(Touch { bond, area: total, pairs: joined });
            }
        }
    }
    Ok(touches)
}

/// What a bond adds between the functions of two elements, without the parameter β: the
/// integrals over the points where the first's tied face touches the second and holds it one
/// way, bonded or sliding.
struct Pair {
    /// ∫ t_l(v) N_b: for each of the first's unknowns, each component l of the held traction,
    /// the second's functions.
    flux: Vec<f64>,
    /// ∫ N_a N_b, functions of the first by functions of the second: one number where the two
    /// are bonded, and where they slide the six of ∫ N_a N_b n_k n_l ([`SYMMETRIC`]).
    mass: Vec<f64>,
    /// ∫ N_b N_b′ of the second's functions, likewise.
    own: Vec<f64>,
}

/// Where the product of components k and l of the normal is kept among the six of a pair of
/// functions.
const SYMMETRIC: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];

/// The stiffness as the way it is solved keeps it.
enum Stiffness {
    /// Rows of entries, for the factorization.
    Rows(Csr),
    /// Blocks of three by three, for the iteration.
    Blocks(Blocked),
}

impl Stiffness {
    /// Add the matrix `data` of the element of `nodes` (three unknowns each).
    fn add(&mut self, layout: &DofLayout, nodes: &[usize], data: &[f64]) -> Result<()> {
        match self {
            Stiffness::Rows(rows) => Ok(add_block(rows, layout, nodes, DofSet::TRANSLATIONS, data)?),
            Stiffness::Blocks(blocked) => blocked.add(nodes, data),
        }
    }
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
    let mut pairs: BTreeMap<(usize, usize, usize, usize, bool), Pair> = BTreeMap::new();
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
            let each = if point.along.is_some() { 6 } else { 1 };
            let pair = pairs.entry((first, point.element, second, element, point.along.is_some())).or_insert_with(|| Pair {
                flux: vec![0.0; 9 * na * nb],
                mass: vec![0.0; each * na * nb],
                own: vec![0.0; each * nb * nb],
            });
            if pair.mass.len() != each * na * nb || pair.own.len() != each * nb * nb {
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
                    if let Some(out) = pair.flux.get_mut((3 * row + l) * nb..(3 * row + l + 1) * nb) {
                        for (slot, value) in out.iter_mut().zip(&values) {
                            *slot += scaled * value;
                        }
                    }
                }
            }
            match point.along {
                None => {
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
                }
                Some(n) => {
                    let across = [w * n[0] * n[0], w * n[0] * n[1], w * n[0] * n[2], w * n[1] * n[1], w * n[1] * n[2], w * n[2] * n[2]];
                    for (row, na_value) in pair.mass.chunks_exact_mut(6 * nb.max(1)).zip(point.values) {
                        for (six, value) in row.as_chunks_mut::<6>().0.iter_mut().zip(&values) {
                            for (slot, weight) in six.iter_mut().zip(across) {
                                *slot += weight * na_value * value;
                            }
                        }
                    }
                    for (row, first_value) in pair.own.chunks_exact_mut(6 * nb.max(1)).zip(&values) {
                        for (six, value) in row.as_chunks_mut::<6>().0.iter_mut().zip(&values) {
                            for (slot, weight) in six.iter_mut().zip(across) {
                                *slot += weight * first_value * value;
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
    }

    // The pairs of elements the bonds couple, each once however it is held.
    let coupled: BTreeSet<(usize, usize, usize, usize)> =
        pairs.keys().map(|(first, element, second, other, _)| (*first, *element, *second, *other)).collect();
    // The order of elimination: one body alone is in its own already.
    let global = |body: usize, function: usize| offsets.get(body).map_or(function, |offset| offset + function);
    let place: Vec<usize> = if views.len() == 1 && pairs.is_empty() {
        (0..total).collect()
    } else {
        let mut lists: Vec<Vec<usize>> = Vec::new();
        for (body, view) in views.iter().enumerate() {
            lists.extend(view.elements.iter().map(|functions| functions.iter().map(|function| global(body, *function)).collect()));
        }
        for (first, element, second, other) in &coupled {
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
    let too_large = |entries: usize, limit: usize| {
        Error::Limit(format!("the stiffness would have {entries} entries; the limit is {limit} ({})", solve::smaller(refine)))
    };
    let layout = DofLayout::uniform(0..total as u32, DofSet::TRANSLATIONS);
    if layout.dofs() != 3 * total {
        return Err(broken("the layout does not hold three unknowns per function"));
    }
    // How the system is solved is chosen before it is added up, from its pattern: by the
    // factorization, or by iteration where the factorization would take far more work than
    // the system has entries, or more memory than is allowed. The iteration keeps the matrix
    // by blocks, in half the memory, and takes as much more of it.
    let (mut stiffness, stiffness_entries, needed) = {
        let bodies = views.iter().enumerate().flat_map(|(body, view)| (0..view.elements.len()).map(move |element| (body, element)));
        let joined = coupled.iter().map(|(first, element, second, other)| {
            let mut both = placed(*first, *element);
            both.extend(placed(*second, *other));
            both
        });
        let graph = NodeGraph::new(total, bodies.map(|(body, element)| placed(body, element)).chain(joined));
        let stiffness_entries = graph.entries(&layout);
        let asked = |solver: Solver| views.iter().any(|view| view.solver == solver);
        // What the factorization would cost is worked out where the choice hangs on it: not
        // for a system that is iterated in any case, as asked or because it is too large as
        // rows, since that takes as long as a few rounds.
        let iterated_anyway = !asked(Solver::Direct) && (asked(Solver::Iterative) || stiffness_entries > MAX_MATRIX_ENTRIES);
        let cost = (!iterated_anyway).then(|| predicted_of_nodes(&graph.adjacency));
        let iterate = match cost {
            None => true,
            Some(_) if asked(Solver::Direct) => false,
            Some((entries, work)) => {
                // A material that hardly changes its volume is not the iteration's
                // ([`iterative`]); what does not fit as a factor is, whatever it suits.
                let suits = views.iter().all(|view| view.poisson <= NEARLY_INCOMPRESSIBLE);
                let divided = views.iter().map(|view| view.tree.deepest()).max().unwrap_or(0);
                let mut iteration = ITERATION.get(divided).or(ITERATION.last()).copied().unwrap_or(f64::INFINITY);
                if !coupled.is_empty() {
                    iteration *= CONNECTED;
                }
                if views.iter().any(|view| view.trunk) {
                    iteration *= THIN;
                }
                entries > limits.max_factor_entries || (3 * total >= ITERATIVE_FROM && suits && work > iteration * stiffness_entries as f64)
            }
        };
        // The entries a factor needs, where they are known.
        let needed = cost.map(|(entries, _)| entries);
        // By blocks the matrix may have as many entries as the factor is allowed: that is the
        // memory the caller has given the solve.
        let limit = if iterate { MAX_MATRIX_ENTRIES.max(limits.max_factor_entries) } else { MAX_MATRIX_ENTRIES };
        if stiffness_entries > limit {
            return Err(too_large(stiffness_entries, limit));
        }
        // A round of [`Options::adapt`] is refused here, before its cells are integrated, when
        // its factor cannot fit: the larger system of the same part is taken to have no fewer
        // entries of the factor for each of the stiffness than the solve before it had (the
        // fill of a nested dissection grows with the system).
        // (Only where the factorization is the one way asked for: an iteration takes a
        // system whose factor does not fit.)
        if let Some((before, factor)) =
            views.iter().find_map(|view| view.before).filter(|(before, _)| asked(Solver::Direct) && *before > 0 && stiffness_entries >= *before)
        {
            let least = stiffness_entries as f64 * factor as f64 / before as f64;
            if least > limits.max_factor_entries as f64 {
                return Err(Error::Limit(format!(
                    "the factorization would need {least:.0} entries or more; the limit is {} ({})",
                    limits.max_factor_entries,
                    solve::smaller(refine)
                )));
            }
        }
        let stiffness = if iterate {
            let blocked = Blocked::zero(&graph.adjacency);
            Stiffness::Blocks(
                blocked
                    .ok_or_else(|| Error::Limit(format!("no memory for a stiffness of {stiffness_entries} entries ({})", solve::smaller(refine))))?,
            )
        } else {
            Stiffness::Rows(graph.zero_matrix(&layout))
        };
        (stiffness, stiffness_entries, needed)
    };
    let mut load = vec![0.0; 3 * total];
    for (body, view) in views.iter().enumerate() {
        (view.assemble)(&mut |element: usize, cell_stiffness: &[f64], cell_load: &[f64]| -> Result<()> {
            let nodes = placed(body, element);
            stiffness.add(&layout, &nodes, cell_stiffness)?;
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
    for ((first, element, second, other, sliding), pair) in &pairs {
        let beta = views.get(*first).map_or(0.0, |view| (view.nitsche)(*element));
        if !(beta > 0.0 && beta.is_finite()) {
            return Err(broken("a tied cell has no parameter"));
        }
        let (mut nodes, others) = (placed(*first, *element), placed(*second, *other));
        let (na, nb) = (nodes.len(), others.len());
        nodes.extend(others);
        let n = 3 * (na + nb);
        let mut block = vec![0.0; n * n];
        if *sliding {
            // Across the face alone: β multiplies ∫ N N′ n_k n_l where a bond has ∫ N N′ δ_kl.
            for row in 0..3 * na {
                for fb in 0..nb {
                    for l in 0..3 {
                        let flux = pair.flux.get((3 * row + l) * nb + fb).copied().unwrap_or(0.0);
                        let mass = pair.mass.get(((row / 3) * nb + fb) * 6 + SYMMETRIC[row % 3][l]).copied().unwrap_or(0.0);
                        let (value, column) = (flux - beta * mass, 3 * (na + fb) + l);
                        if let Some(slot) = block.get_mut(row * n + column) {
                            *slot = value;
                        }
                        if let Some(slot) = block.get_mut(column * n + row) {
                            *slot = value;
                        }
                    }
                }
            }
            for fb in 0..nb {
                for other in 0..nb {
                    for k in 0..3 {
                        for l in 0..3 {
                            let mass = pair.own.get((fb * nb + other) * 6 + SYMMETRIC[k][l]).copied().unwrap_or(0.0);
                            if let Some(slot) = block.get_mut((3 * (na + fb) + k) * n + 3 * (na + other) + l) {
                                *slot = beta * mass;
                            }
                        }
                    }
                }
            }
            stiffness.add(&layout, &nodes, &block)?;
            continue;
        }
        for row in 0..3 * na {
            for (fb, mass) in pair.mass.get((row / 3) * nb..(row / 3 + 1) * nb).unwrap_or_default().iter().enumerate() {
                for l in 0..3 {
                    let flux = pair.flux.get((3 * row + l) * nb + fb).copied().unwrap_or(0.0);
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
        stiffness.add(&layout, &nodes, &block)?;
    }
    drop(pairs);

    let (u, factor_entries, iterations, residual) = match stiffness {
        Stiffness::Rows(rows) => {
            if let Some(entries) = needed.filter(|entries| *entries > limits.max_factor_entries) {
                return Err(solve::factor_error(FactorError::TooLarge { entries, limit: limits.max_factor_entries }, refine));
            }
            // The factorization reads the matrix's lower triangle.
            let upper = UpperColumns::from_symmetric(&rows);
            drop(rows);
            let factor = Factor::new(&upper, limits.max_factor_entries).map_err(|e| solve::factor_error(e, refine))?;
            drop(upper);
            let mut u = load;
            factor.solve(&mut u)?;
            (u, factor.entries(), 0, 0.0)
        }
        Stiffness::Blocks(mut blocked) => {
            let mut linear = vec![false; total];
            for (body, view) in views.iter().enumerate() {
                for (function, flag) in view.linear.iter().enumerate().take(view.count) {
                    if let Some(slot) = place.get(global(body, function)).and_then(|node| linear.get_mut(*node)) {
                        *slot = *flag;
                    }
                }
            }
            // The nodes of every cell, and of every two cells a connection couples: what a
            // cell has to itself is taken out of the system, and the blocks of the iteration
            // are made of the rest.
            let cells = views.iter().map(|view| view.elements.len()).sum::<usize>();
            let mut lists: Vec<Vec<usize>> = views
                .iter()
                .enumerate()
                .flat_map(|(body, view)| (0..view.elements.len()).map(move |element| (body, element)))
                .map(|(body, element)| placed(body, element))
                .collect();
            lists.extend(coupled.iter().map(|(first, element, second, other)| {
                let mut both = placed(*first, *element);
                both.extend(placed(*second, *other));
                both
            }));
            let condensed = condense(&mut blocked, &load, &lists)?;
            let reach = iterative::reach(&lists, cells, &condensed.own);
            // Across a connection the cells of the body that has fewer of them there lead, and
            // those of the other follow.
            let first_cell: Vec<usize> = views.iter().scan(0usize, |sum, view| Some(std::mem::replace(sum, *sum + view.elements.len()))).collect();
            let cell = |body: usize, element: usize| first_cell.get(body).and_then(|first| u32::try_from(first + element).ok());
            let mut sides: BTreeMap<(usize, usize), (BTreeSet<usize>, BTreeSet<usize>)> = BTreeMap::new();
            for (first, element, second, other) in &coupled {
                let entry = sides.entry((*first, *second)).or_default();
                entry.0.insert(*element);
                entry.1.insert(*other);
            }
            let leads: Vec<(u32, u32)> = coupled
                .iter()
                .filter_map(|(first, element, second, other)| {
                    let (a, b) = (cell(*first, *element)?, cell(*second, *other)?);
                    let first_leads = sides.get(&(*first, *second)).is_none_or(|(elements, others)| elements.len() <= others.len());
                    Some(if first_leads { (a, b) } else { (b, a) })
                })
                .collect();
            let extended = iterative::extended(&blocked, &lists, &reach, &linear, &leads);
            let blocks = iterative::patches(&lists, cells, &condensed.own, &reach, &extended);
            drop(lists);
            drop(reach);
            let scale = load.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
            let progress = fieldcraft_model::progress::current();
            let cancelled = || progress.as_ref().is_some_and(|p| p.is_cancelled());
            // With no factorization to fall back on, the iteration is given longer.
            let alone = needed.is_some_and(|entries| entries > limits.max_factor_entries) || blocked.entries() > MAX_MATRIX_ENTRIES;
            let most = if alone { iterative::MOST_ALONE } else { iterative::MOST };
            let iterated = iterative::solve(
                &blocked,
                &condensed.own,
                &condensed.load,
                scale,
                &linear,
                &blocks,
                &extended,
                limits,
                iterative::SETTLED,
                most,
                &cancelled,
            )?;
            drop(blocks);
            let (mut u, factor_entries, iterations, residual) = match iterated {
                Some(outcome) => (outcome.u, outcome.coarse_entries, outcome.iterations, outcome.residual),
                // What the iteration did not settle is the factorization's, if it can take it:
                // of what the cells' own unknowns left.
                None => {
                    let unsettled =
                        |why: String| Error::Limit(format!("the iteration did not settle the system, and {why} ({})", solve::smaller(refine)));
                    let too_much = |entries: usize| {
                        unsettled(format!("the factorization would need {entries} entries; the limit is {}", limits.max_factor_entries))
                    };
                    if let Some(entries) = needed.filter(|entries| *entries > limits.max_factor_entries) {
                        return Err(too_much(entries));
                    }
                    if blocked.entries() > MAX_MATRIX_ENTRIES {
                        return Err(unsettled(format!(
                            "the stiffness has {} entries; a factorization takes {MAX_MATRIX_ENTRIES} at most",
                            blocked.entries()
                        )));
                    }
                    let (u, factor_entries) = condensed.factored(&blocked, limits.max_factor_entries).map_err(|e| match e {
                        FactorError::TooLarge { entries, .. } => too_much(entries),
                        other => solve::factor_error(other, refine),
                    })?;
                    (u, factor_entries, 0, 0.0)
                }
            };
            condensed.recover(&blocked, &load, &mut u);
            (u, factor_entries, iterations, residual)
        }
    };
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
            Solved { u: own, stiffness_entries, factor_entries, iterations, residual }
        })
        .collect())
}
