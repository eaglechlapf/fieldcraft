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
//! 2. **Basis** ([`legendre`]): the tensor product of integrated Legendre polynomials of degree
//!    p on every cell, or its trunk space ([`Options::basis`]). Modes on a shared vertex, edge
//!    or face of the grid are one unknown.
//! 3. **Cut cells** ([`grid`]): a cell the surface cuts is integrated one axis at a time, split
//!    where the surface and its edges cross, with Gauss rules that integrate a cell cut by flat
//!    faces exactly; where the surface is curved, an octree of depth [`Options::depth`] divides
//!    the cell first. Points outside the part are weighted with [`Options::fictitious`]. The
//!    points are not summed mode pair by mode pair: they are summed into the Legendre moments of
//!    the cell's material, from which every cell matrix follows by sum factorisation.
//! 4. **Supports**: prescribed displacement components of a face, imposed weakly by Nitsche's
//!    method on the face's own triangles, with the stabilisation parameter of each cell from that
//!    cell's eigenvalue problem (no penalty number to choose).
//! 5. **Loads**: pressures, tractions and forces integrated over the face's triangles, gravity
//!    over the part's quadrature points.
//! 6. **Solution**: assembled into [`fieldcraft_solver`]'s sparse matrix, ordered by its nested
//!    dissection and factored by its Cholesky. Displacement and stress are evaluated anywhere in
//!    the part from the cell's polynomial.
//!
//! This is a first version: one body, linear elastic, isotropic, static. Its accuracy is
//! bounded by the tessellation of curved faces, by the fictitious stiffness, which shows in the
//! stress of a cell that holds very little material, and by the grid, which is uniform: there is
//! no local refinement, so a feature needs cells of its own size everywhere (on cells as large
//! as a hole's radius the stress at the hole is within a few per cent only from degree 5 on,
//! `tests/benchmarks.rs`). The stress is the derivative of the cell's polynomial and is least
//! accurate on the boundary of the part, most of all where a supported face ends. The system is
//! factored directly, so memory grows quickly with the degree: the caps in [`solve()`] refuse a
//! model that would not fit. [`Solution::change_from`] compares two solutions as an indicator of
//! convergence.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]
// Matrix kernels read as the textbook formulas with index loops.
#![allow(clippy::needless_range_loop)]

mod cell;
mod dense;
pub mod gauss;
pub mod grid;
pub mod inside;
pub mod legendre;
mod moments;
mod solve;
mod space;

pub use solve::{Basis, Change, Load, Material, Options, Reaction, Solution, Stats, Support, solve};

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
