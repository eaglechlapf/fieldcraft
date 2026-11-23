//! FieldCraft's finite cell solver: a linear static step solved on a background grid, with no
//! body-fitted mesh of the part.
//!
//! The finite cell method (J. Parvizian, A. Düster, E. Rank, *Finite cell method: h- and
//! p-extension for embedded domain problems in solid mechanics*, Comput. Mech. 41(1):121–133,
//! 2007; A. Düster, J. Parvizian, Z. Yang, E. Rank, *The finite cell method for three-dimensional
//! problems of solid mechanics*, Comput. Methods Appl. Mech. Engrg. 197(45–48):3768–3782, 2008)
//! embeds the part in a Cartesian grid of cells, puts higher-order shape functions on every cell
//! that holds material, and integrates the weak form over the part only. The grid needs to know
//! one thing about the geometry: whether a point is inside. Short edges, sliver faces and small
//! fillets, which stop a body-fitted mesher, are only places where the answer changes.
//!
//! The pipeline, for one closed face-tagged triangle surface:
//! 1. **Grid** ([`grid`]): a Cartesian grid over the surface's bounding box. Each surface
//!    triangle is cut into pieces, one cell each; a cell that owns a piece is *cut*, any other is
//!    inside or outside by its centre ([`inside`]: the winding number with exact predicates).
//! 2. **Refinement** ([`Options::refine`], off by default): a cut cell across which the surface
//!    turns more than its degree resolves (a fillet or a hole with fewer than two cells to the
//!    radius at degree 4, four at degree 3) is divided into eight, with the cells across its
//!    faces, level by level. The cells of all levels carry one conforming basis by the
//!    multi-level hp method: a divided cell stays and its children are laid over it, the finer
//!    functions vanish on the boundary of the cells that are divided, and a function a divided
//!    cell's children can represent is handed down to them, so there are no hanging nodes to
//!    constrain (N. Zander, T. Bog, S. Kollmannsberger, D. Schillinger, E. Rank, *Multi-level
//!    hp-adaptivity: high-order mesh adaptivity without the difficulties of constraining hanging
//!    nodes*, Comput. Mech. 55(3):499–517, 2015). Every cell that is not divided is integrated
//!    like a cell of the grid, in its own shape functions; the functions of the cells above it
//!    are combinations of those.
//!    The cells the solution is least resolved in can be divided as well, in rounds of solving
//!    and dividing ([`Options::adapt`], off by default): the indicator is the strain energy of
//!    what the functions of the highest degree alone carry in a cell (B. Szabó, I. Babuška,
//!    *Finite Element Analysis*, Wiley 1991), the cells that carry a fixed share of its sum are
//!    divided (W. Dörfler, *A convergent adaptive algorithm for Poisson's equation*, SIAM J.
//!    Numer. Anal. 33(3):1106–1124, 1996), and the rounds end when one no longer moves the
//!    solution ([`Options::settle`]).
//! 3. **Basis** ([`legendre`]): the tensor product of integrated Legendre polynomials of degree
//!    p on every cell, or its trunk space ([`Options::basis`]). Modes on a shared vertex, edge
//!    or face of the grid are one unknown.
//! 4. **Cut cells**: what a cell the surface cuts needs are the Legendre moments of its
//!    material, from which every cell matrix follows by sum factorisation. The
//!    surface is a triangulation, so they are integrals of polynomials over a polyhedron, and
//!    the divergence theorem gives them exactly from the triangles in the cell and one inside
//!    test ([`exact`], [`Options::exact`]). A cell in a position that does not decide with
//!    certainty (its highest corner on the surface) is integrated by quadrature instead
//!    ([`grid`]): one axis at a time, split where the surface and its edges cross, with Gauss
//!    rules that integrate a cell cut by flat faces exactly, on an octree of depth
//!    [`Options::depth`] where the surface is curved. The volume outside the part is weighted
//!    with [`Options::fictitious`].
//! 5. **Islands** ([`Options::separate`], [`islands`]): a cell that holds material in separate
//!    pieces (the two flanges at the open corner of a sheet metal part, the two sides of a slit)
//!    carries shape functions of its own for each piece, and two cells share the functions of
//!    the face between them only where material goes on through it. One polynomial a cell
//!    would join the pieces: the gap would be welded shut wherever a cell reaches across it.
//!    Where a gap ends, the cell at its root still holds the two sides together over its own
//!    length, so the cells whose functions do that are divided ([`Options::gaps`]).
//! 6. **Supports**: prescribed displacement components of a face, imposed weakly by Nitsche's
//!    method on the face's own triangles, with the stabilisation parameter of each cell from that
//!    cell's eigenvalue problem (no penalty number to choose).
//! 7. **Loads**: pressures, tractions and forces integrated over the face's triangles, gravity
//!    over the part's quadrature points.
//! 8. **Solution**: assembled into [`fieldcraft_solver`]'s sparse matrix and factored by its
//!    Cholesky, the unknowns ordered by its nested dissection by elements
//!    ([`fieldcraft_solver::ordering::element_dissection`]): the functions of a cell all
//!    couple, so the part is divided along the functions that two groups of cells share, which
//!    are far fewer than a layer of functions cut through the cells. Displacement and stress
//!    are evaluated anywhere in the part from the cell's polynomial.
//!
//! This is a first version: one body, linear elastic, isotropic, static. On thin sheet it is
//! stiff: cells of degree 3 a dozen sheet thicknesses across bend a plate 3 to 9 % too little
//! (`tests/tray.rs`), which finer cells or a higher degree lower slowly, and every cell carries
//! the full degree through the thickness, where a plate needs two. Its accuracy is
//! bounded by the tessellation of curved faces, by the fictitious stiffness, which shows in the
//! stress of a cell that holds very little material, and by the grid: on cells as large as a
//! hole's radius the stress at the hole is within a few per cent only from degree 5 on
//! (`tests/benchmarks.rs`), unless the cells are refined there ([`Options::refine`]). The
//! refinement follows the curvature of the surface and nothing else: it does not look at the
//! solution, so a stress that peaks at a sharp re-entrant edge or where a support ends is not
//! refined towards, and it divides a cell along all three axes, also where a plate or an
//! extrusion would need it along two. The stress is the derivative of the cell's polynomial and
//! is least accurate on the boundary of the part, most of all where a supported face ends. In a
//! cell that holds little material that polynomial is fitted to very little, and its stress is
//! noisy; in a cell that holds less than a tenth of its volume the stress at a point is
//! therefore read as the mean of the polynomials of the cells around the point, each weighted
//! by its estimated error ([`Solution::stress`]).
//! The solution itself is not changed by that: the unknowns of such a cell are still determined
//! by little material, which costs conditioning and not accuracy elsewhere. The part is known by
//! the triangles of its surface alone, and small cells resolve the creases between the facets of
//! a concave fillet as the notches they are: on the bracket of `tests/bracket.rs` cells of a
//! third of the fillet's radius read the peak 4 to 7 % high over facets of a chordal error of
//! 1/25 of a cell and within 0.7 % at 1/200, so the surface is to be triangulated far more
//! finely than the smallest cell. The system is
//! factored directly, so memory grows quickly with the degree: the caps in [`solve()`] refuse a
//! model that would not fit. [`Solution::change_from`] compares two solutions as an indicator of
//! convergence.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]
// Matrix kernels read as the textbook formulas with index loops.
#![allow(clippy::needless_range_loop)]

pub mod assembly;
mod blocked;
mod cell;
mod condense;
mod dense;
pub mod exact;
mod extended;
pub mod gauss;
pub mod grid;
pub mod inside;
pub mod islands;
mod iterative;
pub mod legendre;
mod moments;
mod packed;
mod refine;
mod solve;
mod space;

pub use assembly::{
    Bond, Kept, Motion, Part, Touch, assembly_unknowns, assembly_unknowns_keeping, bond_areas, distance_to_faces, facet_deviation, find_bonds,
    free_motions, meet_in_notch, solve_assembly, solve_assembly_keeping,
};
pub use cell::{Curved, Grip};
pub use solve::{Basis, Change, Load, Material, Options, Reaction, Solution, Solver, Stats, Support, solve};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("limit reached: {0}")]
    Limit(String),
    /// The stiffness is singular to working precision.
    #[error("{0}")]
    Singular(String),
    /// A broken internal invariant (a bug), reported instead of a panic.
    #[error("internal error: {0}")]
    Internal(String),
    #[error(transparent)]
    Surface(#[from] fieldcraft_mesh::Error),
    #[error(transparent)]
    Solver(#[from] fieldcraft_solver::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
