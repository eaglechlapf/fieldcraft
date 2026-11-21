//! Standard benchmarks on the grid: the published targets of problems the oracle also runs
//! (`tests/oracle`), each solved over several cell sizes and degrees, with the error to the
//! target printed. The shapes are generated in code as tessellated surfaces.
//!
//! - **Kirsch's plate with a hole** (G. Kirsch, Z. VDI 42:797–807, 1898; Timoshenko and Goodier,
//!   *Theory of Elasticity*, 3rd ed., 1970): a quarter of a 60 × 60 plate with a hole of radius 1
//!   in tension. The stress at the top of the hole is 3 times the remote stress, at its side −1
//!   times. Solved in plane strain (the in-plane stresses of this problem do not depend on the
//!   elastic constants), one cell thick.
//! - **NAFEMS LE1**, the elliptic membrane (NAFEMS, *The Standard NAFEMS Benchmarks*, TNSB Rev. 3,
//!   1990): σ_yy = 92.7 MPa at D (2, 0). Half the 0.1 m thickness, with the mid-plane a symmetry
//!   plane, as the oracle's case models it.
//! - **MacNeal–Harder straight and curved cantilevers** (R. H. MacNeal, R. L. Harder, *A proposed
//!   standard set of problems to test finite element accuracy*, Finite Elem. Anal. Des. 1:3–20,
//!   1985, Table 3): tip displacements under unit tip loads.
//! - **MacNeal–Harder thick-walled cylinder** (the same paper): the radial displacement of the
//!   bore as Poisson's ratio approaches one half, against Lamé's closed form. At ν = 0.4999 and
//!   degrees 4 and 5 the default fictitious stiffness leaves the system singular to working
//!   precision, and the solver raises it a hundred times.
//!
//! Kirsch's plate and LE1 are also solved with local refinement (`Options::refine`): a coarse
//! grid whose cells are divided at the hole, or towards the end of the ellipse where it is
//! curved most.

mod common;

use std::time::Instant;

use common::{AT_X0, AT_Y0, AT_Z0, AT_Z1, INNER, OUTER_A, OUTER_B, X_HI, X_LO, block, quarter_tube, strip};
use fieldcraft_fcm::{Basis, Load, Material, Options, Solution, Support, solve};
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_solver::Limits;

const OFF_GRID: [f64; 3] = [0.37, 0.21, 0.43];

fn slide(face: u32, axis: usize) -> Support {
    let mut fixed = [false; 3];
    fixed[axis] = true;
    Support { face, fixed, value: [0.0; 3] }
}

/// Solve, or print why not.
fn attempt(
    case: &str,
    surface: &BoundarySurface,
    material: &Material,
    supports: &[Support],
    loads: &[Load],
    options: &Options,
) -> Option<(Solution, String)> {
    let started = Instant::now();
    match solve(surface, material, supports, loads, options, &Limits::default()) {
        Ok(solution) => {
            // With refinement, the cells of each level.
            let levels = if options.refine > 0 { format!(", cells by level {:?}", &solution.stats.levels[..=options.refine]) } else { String::new() };
            let run = format!(
                "{case}: cell {}, degree {}{levels}: {} unknowns, {:.1} s",
                options.cell_size,
                options.degree,
                solution.stats.unknowns,
                started.elapsed().as_secs_f64()
            );
            Some((solution, run))
        }
        Err(e) => {
            println!("{case}: cell {}, degree {}: {e}", options.cell_size, options.degree);
            None
        }
    }
}

fn percent(value: f64, target: f64) -> String {
    format!("{value:.5e} ({:+.2} % of {target})", 100.0 * (value - target) / target)
}

/// A quarter turn in `segments` steps: (cos, sin), exact at both ends.
fn turn(i: usize, segments: usize) -> (f64, f64) {
    if i == 0 {
        (1.0, 0.0)
    } else if i == segments {
        (0.0, 1.0)
    } else {
        let t = std::f64::consts::FRAC_PI_2 * i as f64 / segments as f64;
        (t.cos(), t.sin())
    }
}

/// A quarter of a square plate of half width `half` with a hole of radius 1, pulled along x, for
/// each cell size, degree and number of levels of refinement.
fn kirsch(case: &str, half: f64, thin: bool, basis: Basis, cases: &[(f64, usize, usize)]) {
    const SEGMENTS: usize = 128;
    let material = Material { young: 1000.0, poisson: 0.3, density: 0.0 };
    for &(cell, degree, refine) in cases {
        // Plane strain, one cell thick with both faces on grid planes; or a thin plate with a free
        // face (plane stress), cut by every cell.
        let (thickness, at) = if thin { (0.05, 0.0) } else { (cell, 0.5 * cell) };
        let surface = strip(
            SEGMENTS,
            thickness,
            |i| {
                let (c, s) = turn(i, SEGMENTS);
                // The outer curve is the square's boundary along the same ray.
                let outer = match i.cmp(&(SEGMENTS / 2)) {
                    std::cmp::Ordering::Less => [half, half * s / c],
                    std::cmp::Ordering::Equal => [half, half],
                    std::cmp::Ordering::Greater => [half * c / s, half],
                };
                ([c, s], outer)
            },
            |i| if i < SEGMENTS / 2 { OUTER_A } else { OUTER_B },
        );
        let supports = [slide(AT_X0, 0), slide(AT_Y0, 1), slide(AT_Z0, 2), slide(AT_Z1, 2)];
        let supports = if thin { &supports[..3] } else { &supports[..] };
        let loads = [Load::Traction { face: OUTER_A, traction: [1.0, 0.0, 0.0] }];
        // One cell thick, its two faces on grid planes; the grid is off the part in x and y.
        let options = Options { degree, basis, refine, offset: [0.37, 0.21, if thin { 0.43 } else { 0.0 }], ..Options::new(cell) };
        if let Some((solution, run)) = attempt(case, &surface, &material, supports, &loads, &options) {
            let top = solution.stress([0.0, 1.0, at]).unwrap()[0];
            let side = solution.stress([1.0, 0.0, at]).unwrap()[1];
            println!("{run}; σ_xx at the top of the hole {}, σ_yy at its side {}", percent(top, 3.0), percent(side, -1.0));
        }
    }
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn kirsch_plate_with_a_hole() {
    kirsch("kirsch", 30.0, false, Basis::Tensor, &[(2.0, 3, 0), (2.0, 4, 0), (1.0, 2, 0), (1.0, 3, 0), (0.5, 2, 0), (1.0, 4, 0)]);
    // Cells as large as the hole's radius resolve it from degree 5 on, which fits the limits of
    // the system in the trunk space.
    kirsch("kirsch, trunk space", 30.0, false, Basis::Trunk, &[(1.0, 5, 0), (1.0, 6, 0), (1.0, 7, 0)]);
}

/// The same plate with local refinement (`Options::refine`): cells as large as the hole's
/// radius, and twice and four times as large, divided at the hole only. The uniform grid of
/// this plate cannot have two cells a radius at degree 4 (the system would not fit); divided
/// at the hole, cells of the radius reach Kirsch's values with a few thousand unknowns more
/// than they have alone.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn kirsch_plate_with_a_hole_refined() {
    let cases = [(1.0, 3, 0), (1.0, 3, 1), (1.0, 3, 2), (1.0, 4, 0), (1.0, 4, 1), (1.0, 4, 2), (2.0, 3, 3), (2.0, 4, 2), (4.0, 4, 3)];
    kirsch("kirsch, refined", 30.0, false, Basis::Tensor, &cases);
    kirsch("kirsch, refined, trunk space", 30.0, false, Basis::Trunk, &[(1.0, 5, 0), (1.0, 5, 2), (2.0, 6, 2)]);
    // A thin plate with a free face: every cell is cut, and the divided ones are too.
    kirsch("kirsch, half width 8, thin, refined", 8.0, true, Basis::Tensor, &[(1.0, 3, 0), (1.0, 3, 2), (1.0, 4, 0), (1.0, 4, 1), (0.5, 4, 0)]);
}

/// The same hole in a plate small enough for cells of half its radius at degree 4. The plate is
/// too narrow for Kirsch's values to hold exactly (they are for an infinite plate), so this
/// shows whether the stress at the hole settles as the grid and the degree are refined.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn kirsch_hole_in_a_small_plate_settles() {
    kirsch("kirsch, half width 8", 8.0, false, Basis::Tensor, &[(1.0, 3, 0), (1.0, 4, 0), (1.0, 5, 0), (0.5, 3, 0), (0.5, 4, 0), (0.25, 3, 0)]);
    kirsch("kirsch, half width 8, thin", 8.0, true, Basis::Tensor, &[(1.0, 3, 0), (1.0, 4, 0), (1.0, 5, 0), (0.5, 3, 0), (0.5, 4, 0)]);
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn nafems_le1_elliptic_membrane() {
    const SEGMENTS: usize = 96;
    let material = Material { young: 210_000.0, poisson: 0.3, density: 0.0 };
    let surface = strip(
        SEGMENTS,
        0.05,
        |i| {
            let (c, s) = turn(i, SEGMENTS);
            ([2.0 * c, s], [3.25 * c, 2.75 * s])
        },
        |_| OUTER_A,
    );
    let supports = [slide(AT_X0, 0), slide(AT_Y0, 1), slide(AT_Z0, 2)];
    // 10 MPa pulling outward on the outer ellipse.
    let loads = [Load::Pressure { face: OUTER_A, pressure: -10.0 }];
    for (cell, degree) in [(0.5, 3), (0.5, 4), (0.5, 5), (0.25, 3), (0.25, 4), (0.25, 5)] {
        let options = Options { degree, offset: OFF_GRID, ..Options::new(cell) };
        if let Some((solution, run)) = attempt("le1", &surface, &material, &supports, &loads, &options) {
            println!("{run}; σ_yy at D {} MPa", percent(solution.stress([2.0, 0.0, 0.0]).unwrap()[1], 92.7));
        }
    }
    // With local refinement: the inner ellipse has the radius of curvature 0.5 at D, as large
    // as a cell, and 4 at its other end, so the cells are divided towards D.
    for (cell, degree, refine) in [(0.5, 3, 1), (0.5, 3, 2), (0.5, 4, 1), (0.5, 4, 2), (0.25, 3, 1), (0.25, 4, 1), (1.0, 3, 2), (1.0, 4, 2)] {
        let options = Options { degree, refine, offset: OFF_GRID, ..Options::new(cell) };
        if let Some((solution, run)) = attempt("le1, refined", &surface, &material, &supports, &loads, &options) {
            println!("{run}; σ_yy at D {} MPa", percent(solution.stress([2.0, 0.0, 0.0]).unwrap()[1], 92.7));
        }
    }
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn macneal_harder_straight_cantilever() {
    let material = Material { young: 1.0e7, poisson: 0.3, density: 0.0 };
    let surface = block([0.0; 3], [6.0, 0.2, 0.1]);
    let supports = [Support { face: X_LO, fixed: [true; 3], value: [0.0; 3] }];
    for (cell, degree) in [(1.0, 2), (1.0, 3), (1.0, 4), (1.0, 5), (0.5, 3), (0.5, 4)] {
        let options = Options { degree, offset: OFF_GRID, ..Options::new(cell) };
        let mut line = String::new();
        for (name, axis, target) in [("extension", 0, 3.0e-5), ("in-plane", 1, 0.1081), ("out-of-plane", 2, 0.4321)] {
            let mut force = [0.0; 3];
            force[axis] = 1.0;
            if let Some((solution, run)) = attempt("straight", &surface, &material, &supports, &[Load::Force { face: X_HI, force }], &options) {
                if line.is_empty() {
                    line = run;
                }
                line += &format!("; {name} {}", percent(solution.displacement([6.0, 0.1, 0.05]).unwrap()[axis], target));
            }
        }
        println!("{line}");
    }
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn macneal_harder_curved_cantilever() {
    let material = Material { young: 1.0e7, poisson: 0.25, density: 0.0 };
    let surface = quarter_tube(4.12, 4.32, 0.1, 96);
    // Clamped where it meets the plane y = 0; the tip is in the plane x = 0.
    let supports = [Support { face: AT_Y0, fixed: [true; 3], value: [0.0; 3] }];
    for (cell, degree) in [(1.0, 3), (1.0, 4), (1.0, 5), (0.5, 3), (0.5, 4)] {
        let options = Options { degree, offset: OFF_GRID, ..Options::new(cell) };
        let mut line = String::new();
        for (name, axis, target) in [("in-plane", 1, 0.08734), ("out-of-plane", 2, 0.5022)] {
            let mut force = [0.0; 3];
            force[axis] = 1.0;
            if let Some((solution, run)) = attempt("curved", &surface, &material, &supports, &[Load::Force { face: AT_X0, force }], &options) {
                if line.is_empty() {
                    line = run;
                }
                line += &format!("; {name} {}", percent(solution.displacement([0.0, 4.22, 0.05]).unwrap()[axis], target));
            }
        }
        println!("{line}");
    }
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn macneal_harder_thick_cylinder_near_incompressible() {
    let (a, b) = (3.0, 9.0);
    let surface = quarter_tube(a, b, 1.0, 64);
    let supports = [slide(AT_X0, 0), slide(AT_Y0, 1), slide(AT_Z0, 2), slide(AT_Z1, 2)];
    let loads = [Load::Pressure { face: INNER, pressure: 1.0 }];
    for poisson in [0.49, 0.499, 0.4999] {
        let material = Material { young: 1000.0, poisson, density: 0.0 };
        // Just inside the facets of the bore, where Lamé's displacement is evaluated too.
        let r = 1.001 * a;
        let exact = (1.0 + poisson) / material.young * (a * a / (b * b - a * a)) * ((1.0 - 2.0 * poisson) * r + b * b / r);
        for degree in 2..=5 {
            let options = Options { degree, offset: OFF_GRID, ..Options::new(3.0) };
            // Where the default fictitious stiffness leaves the system singular to working
            // precision, the solver raises it (`Options::fictitious`) and says so in its stats.
            if let Some((solution, run)) = attempt(&format!("thick cylinder, ν = {poisson}"), &surface, &material, &supports, &loads, &options) {
                let q = std::f64::consts::FRAC_1_SQRT_2;
                let u = solution.displacement([r * q, r * q, 0.5]).unwrap();
                let radial = (u[0] + u[1]) * q;
                println!("{run}; fictitious stiffness {:.1e}; radial displacement of the bore {}", solution.stats.fictitious, percent(radial, exact));
                assert!(solution.stats.fictitious >= Options::FICTITIOUS);
                if degree >= 3 {
                    assert!((radial - exact).abs() < 5e-3 * exact, "ν = {poisson}, degree {degree}: {radial} against {exact}");
                }
            } else {
                panic!("ν = {poisson}, degree {degree} is not solved");
            }
        }
    }
}
