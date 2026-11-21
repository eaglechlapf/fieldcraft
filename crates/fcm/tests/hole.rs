//! Where the stress at a hole goes wrong: diagnostics for Kirsch's plate (`benchmarks.rs`).
//!
//! `quarter_field` prints the field of a quarter model against Kirsch's closed form on a
//! symmetry plane, just off it and round the hole; `whole_plate` solves the whole plate, where
//! no support touches the hole.
//!
//! What these showed:
//! - The values that went up and down from one degree to the next were the cut-cell quadrature
//!   of the curved face, which an octree of depth 1 left inexact (`depth_of_the_octree`;
//!   `fieldcraft_fcm::Options::DEPTH`).
//! - What is left is resolution, not the weakly imposed supports. The stress at a hole falls to
//!   a third within one radius; a cell as large as the radius carries that peak in one
//!   polynomial, and the value at the hole then depends on where the cell's faces fall, in the
//!   whole plate as much as in the quarter model. `where_the_grid_sits` measures the spread
//!   over six grid positions: ± 10 %, 6.5 % and 3 % at degrees 3, 4 and 5 with one cell a
//!   radius, ± 4 % and 0.7 % at degrees 3 and 4 with two, ± 0.8 % at degree 3 with four. A grid
//!   that is the same everywhere needs two cells a radius of the smallest stressed feature at
//!   degree 4, and four at degree 3.
//! - Refining only there does as well (`fieldcraft_fcm::Options::refine`,
//!   `where_the_refined_grid_sits`): cells of the hole's radius divided at the hole, twice at
//!   degree 3 and once at degree 4, have the spread of the uniform grid of the finest cells.

mod common;

use common::{AT_X0, AT_Y0, AT_Z0, AT_Z1, OUTER_A, OUTER_B, ring, strip};
use fieldcraft_fcm::{Basis, Load, Material, Options, Support, solve};
use fieldcraft_solver::Limits;

fn slide(face: u32, axis: usize) -> Support {
    let mut fixed = [false; 3];
    fixed[axis] = true;
    Support { face, fixed, value: [0.0; 3] }
}

fn hoop(s: [f64; 6], angle: f64) -> f64 {
    let (sin, cos) = angle.sin_cos();
    s[0] * sin * sin + s[1] * cos * cos - 2.0 * s[3] * sin * cos
}

/// The quarter plate of half width 30: the hoop stress round the hole and σ_xx above it against
/// Kirsch's closed form.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn quarter_field() {
    const N: usize = 128;
    let half = 30.0;
    let material = Material { young: 1000.0, poisson: 0.3, density: 0.0 };
    for (cell, degree) in [(1.0, 2), (1.0, 3)] {
        let surface = strip(
            N,
            cell,
            |i| {
                let t = std::f64::consts::FRAC_PI_2 * i as f64 / N as f64;
                let (c, s) = if i == 0 {
                    (1.0, 0.0)
                } else if i == N {
                    (0.0, 1.0)
                } else {
                    (t.cos(), t.sin())
                };
                let outer = if i < N / 2 {
                    [half, half * s / c]
                } else if i == N / 2 {
                    [half, half]
                } else {
                    [half * c / s, half]
                };
                ([c, s], outer)
            },
            |i| if i < N / 2 { OUTER_A } else { OUTER_B },
        );
        let supports = [slide(AT_X0, 0), slide(AT_Y0, 1), slide(AT_Z0, 2), slide(AT_Z1, 2)];
        let loads = [Load::Traction { face: OUTER_A, traction: [1.0, 0.0, 0.0] }];
        let options = Options { degree, offset: [0.37, 0.21, 0.0], ..Options::new(cell) };
        let s = solve(&surface, &material, &supports, &loads, &options, &Limits::default()).unwrap();
        let mut line = format!("quarter cell {cell} p{degree}: hoop at r=1.002 (exact):");
        for degrees in [0.5f64, 10.0, 30.0, 45.0, 60.0, 80.0, 89.5] {
            let a = degrees.to_radians();
            let r: f64 = 1.002;
            let exact = 0.5 * (1.0 + 1.0 / (r * r)) - 0.5 * (1.0 + 3.0 / r.powi(4)) * (2.0 * a).cos();
            line += &format!(" {degrees}°: {:.3} ({exact:.3})", hoop(s.stress([r * a.cos(), r * a.sin(), 0.5 * cell]).unwrap(), a));
        }
        line += "; σ_xx on x=0 at y (exact):";
        for y in [1.0f64, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0] {
            let exact = 0.5 * (2.0 + 1.0 / (y * y) + 3.0 / y.powi(4));
            line += &format!(" {y}: {:.3} ({exact:.3})", s.stress([0.0, y, 0.5 * cell]).unwrap()[0]);
        }
        line += "; σ_xx at x=0.05 off the plane, y:";
        for y in [1.0f64, 1.1, 1.25] {
            line += &format!(" {y}: {:.3}", s.stress([0.05, y, 0.5 * cell]).unwrap()[0]);
        }
        println!("{line}; nitsche {:.2e}, smallest fraction {:.1e}", s.stats.largest_nitsche, s.stats.smallest_fraction);
    }
}

/// Where the cut cells are integrated by quadrature ([`Options::exact`] off, as here, or a cell
/// whose highest corner lies on the surface, as most of this plate's would be: its faces are
/// in the grid's planes), the octree's depth is the accuracy of the matrices of a cell a curved
/// face cuts. A quarter plate of half width 8 on cells as large as the hole's radius: the
/// stress at the top of the hole settles with the depth (to 0.1 % from depth 2 to 3), and at
/// the default depth it falls steadily with the degree to the value a finer grid gives. With
/// an octree of depth 1 it went up and down from degree to degree instead.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn depth_of_the_octree() {
    const N: usize = 128;
    let half = 8.0;
    let material = Material { young: 1000.0, poisson: 0.3, density: 0.0 };
    let top = |cell: f64, degree: usize, depth: usize| -> f64 {
        let surface = strip(
            N,
            cell,
            |i| {
                let t = std::f64::consts::FRAC_PI_2 * i as f64 / N as f64;
                let (c, s) = if i == 0 {
                    (1.0, 0.0)
                } else if i == N {
                    (0.0, 1.0)
                } else {
                    (t.cos(), t.sin())
                };
                let outer = if i < N / 2 {
                    [half, half * s / c]
                } else if i == N / 2 {
                    [half, half]
                } else {
                    [half * c / s, half]
                };
                ([c, s], outer)
            },
            |i| if i < N / 2 { OUTER_A } else { OUTER_B },
        );
        let supports = [slide(AT_X0, 0), slide(AT_Y0, 1), slide(AT_Z0, 2), slide(AT_Z1, 2)];
        let loads = [Load::Traction { face: OUTER_A, traction: [1.0, 0.0, 0.0] }];
        let options = Options { degree, depth, exact: false, offset: [0.37, 0.21, 0.0], ..Options::new(cell) };
        solve(&surface, &material, &supports, &loads, &options, &Limits::default()).unwrap().stress([0.0, 1.0, 0.5 * cell]).unwrap()[0]
    };
    let mut settled = Vec::new();
    for degree in [3, 4, 5] {
        let by_depth: Vec<f64> = (1..=4).map(|depth| top(1.0, degree, depth)).collect();
        println!("cell 1, degree {degree}: σ_xx at the top of the hole at depths 1 to 4: {by_depth:.4?}");
        let deepest = by_depth[3];
        assert!((by_depth[1] - deepest).abs() < 2e-3 * deepest, "degree {degree}, depth 2: {by_depth:?}");
        assert!((by_depth[2] - deepest).abs() < 5e-4 * deepest, "degree {degree}, depth 3: {by_depth:?}");
        assert_eq!(Options::DEPTH, 3);
        settled.push(by_depth[2]);
    }
    let finer = top(0.5, 4, Options::DEPTH);
    println!("cell 0.5, degree 4: {finer:.4}");
    assert!(settled[0] > settled[1] && settled[1] > settled[2], "the stress falls with the degree: {settled:?}");
    assert!((settled[2] - finer).abs() < 0.01 * finer, "degree 5 on cells of the hole's radius: {} against {finer}", settled[2]);
}

/// A whole plate of half width 8 with the whole hole: no support touches the hole.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn whole_plate() {
    const N: usize = 512;
    let half = 8.0;
    let (left, bottom) = (8u32, 9u32);
    let material = Material { young: 1000.0, poisson: 0.3, density: 0.0 };
    for (cell, degree) in [(1.0, 3), (1.0, 4), (0.5, 3)] {
        let surface = ring(
            N,
            cell,
            |i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / N as f64;
                let (c, s) = (t.cos(), t.sin());
                let m = c.abs().max(s.abs());
                ([c, s], [half * (c / m), half * (s / m)])
            },
            |a, b| {
                if a[0] == half && b[0] == half {
                    OUTER_A
                } else if a[0] == -half && b[0] == -half {
                    left
                } else if a[1] == -half && b[1] == -half {
                    bottom
                } else {
                    OUTER_B
                }
            },
        );
        let supports = [slide(left, 0), slide(bottom, 1), slide(AT_Z0, 2), slide(AT_Z1, 2)];
        let loads = [Load::Traction { face: OUTER_A, traction: [1.0, 0.0, 0.0] }];
        let options = Options { degree, offset: [0.37, 0.21, 0.0], ..Options::new(cell) };
        match solve(&surface, &material, &supports, &loads, &options, &Limits::default()) {
            Ok(s) => {
                let z = 0.5 * cell;
                println!(
                    "whole plate cell {cell} p{degree}: {} unknowns; σ_xx at (0,1) {:.3}, at (0,-1) {:.3}; σ_yy at (1,0) {:.3}, at (-1,0) {:.3}",
                    s.stats.unknowns,
                    s.stress([0.0, 1.0, z]).unwrap()[0],
                    s.stress([0.0, -1.0, z]).unwrap()[0],
                    s.stress([1.0, 0.0, z]).unwrap()[1],
                    s.stress([-1.0, 0.0, z]).unwrap()[1]
                );
            }
            Err(e) => println!("whole plate cell {cell} p{degree}: {e}"),
        }
    }
}

/// Where the grid sits. The stress at a hole falls to a third within one radius, so a cell as
/// large as the radius has to carry the whole peak in one polynomial, and how well it does
/// depends on where its faces fall. The same whole plate (no support near the hole) on grids
/// that differ only in their start: the spread of σ_xx over the top and the bottom of the hole
/// is printed for each cell size and degree. It is the honest accuracy of a stress peak at that
/// resolution, and it narrows with the degree and faster with the cell size.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn where_the_grid_sits() {
    const N: usize = 512;
    let half = 8.0;
    let (left, bottom) = (8u32, 9u32);
    let material = Material { young: 1000.0, poisson: 0.3, density: 0.0 };
    let mut spreads = Vec::new();
    for (cell, degree) in [(1.0, 3), (1.0, 4), (1.0, 5), (0.5, 3), (0.5, 4), (0.25, 3)] {
        let surface = ring(
            N,
            cell,
            |i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / N as f64;
                let (c, s) = (t.cos(), t.sin());
                let m = c.abs().max(s.abs());
                ([c, s], [half * (c / m), half * (s / m)])
            },
            |a, b| {
                if a[0] == half && b[0] == half {
                    OUTER_A
                } else if a[0] == -half && b[0] == -half {
                    left
                } else if a[1] == -half && b[1] == -half {
                    bottom
                } else {
                    OUTER_B
                }
            },
        );
        let supports = [slide(left, 0), slide(bottom, 1), slide(AT_Z0, 2), slide(AT_Z1, 2)];
        let loads = [Load::Traction { face: OUTER_A, traction: [1.0, 0.0, 0.0] }];
        let (mut lowest, mut highest) = (f64::INFINITY, 0.0f64);
        for offset in [[0.0, 0.0, 0.0], [0.0, 0.21, 0.0], [0.37, 0.21, 0.0], [0.37, 0.9, 0.0], [0.9, 0.1, 0.0], [0.9, 0.5, 0.0]] {
            let options = Options { degree, offset, ..Options::new(cell) };
            let s = solve(&surface, &material, &supports, &loads, &options, &Limits::default()).unwrap();
            for y in [1.0, -1.0] {
                let stress = s.stress([0.0, y, 0.5 * cell]).unwrap()[0];
                (lowest, highest) = (lowest.min(stress), highest.max(stress));
            }
        }
        let spread = (highest - lowest) / (highest + lowest);
        println!(
            "cell {cell}, degree {degree}: σ_xx at the hole from {lowest:.3} to {highest:.3} over six grid positions (± {:.1} %)",
            100.0 * spread
        );
        spreads.push(spread);
    }
    // One cell a radius at degree 3 is not enough for a peak; four cells a radius are.
    assert!(spreads[0] > 0.05, "{spreads:?}");
    assert!(spreads[5] < 0.03, "{spreads:?}");
}

/// Where the grid sits, with local refinement (`Options::refine`): the whole plate of
/// `where_the_grid_sits` on cells as large as the hole's radius, divided once and twice at the
/// hole, against uniform grids. The spread of σ_xx at the top and the bottom of the hole over
/// the same six grid positions is printed with the unknowns and the time of a solve. A grid of
/// one cell a radius, divided at the hole only, reaches the spread of the uniform grid of the
/// same finest cells with a fraction of its unknowns. At degree 4 the second level changes
/// nothing: cells of half the radius resolve the hole, and no cell is divided again. The plate
/// is one cell of the grid thick, so the uniform grids have one layer of cells where the divided
/// ones have two or four: a part that is not a plate gains more.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn where_the_refined_grid_sits() {
    const N: usize = 512;
    let half = 8.0;
    let (left, bottom) = (8u32, 9u32);
    let material = Material { young: 1000.0, poisson: 0.3, density: 0.0 };
    let mut spreads = Vec::new();
    let cases = [
        (1.0, 3, 0, 0, Basis::Tensor),
        (1.0, 3, 1, 0, Basis::Tensor),
        (1.0, 3, 2, 0, Basis::Tensor),
        (0.5, 3, 0, 0, Basis::Tensor),
        (0.25, 3, 0, 0, Basis::Tensor),
        (1.0, 4, 0, 0, Basis::Tensor),
        (1.0, 4, 1, 0, Basis::Tensor),
        (1.0, 4, 2, 0, Basis::Tensor),
        (0.5, 4, 0, 0, Basis::Tensor),
        (2.0, 4, 2, 0, Basis::Tensor),
        (2.0, 3, 3, 0, Basis::Tensor),
        // Rounds that follow the solution in place of the levels at the hole.
        (1.0, 3, 0, 2, Basis::Tensor),
        (1.0, 3, 0, 3, Basis::Tensor),
        // The trunk space on the cells divided at the hole.
        (1.0, 3, 2, 0, Basis::Trunk),
        (1.0, 4, 2, 0, Basis::Trunk),
    ];
    for (cell, degree, refine, adapt, basis) in cases {
        // One cell of the grid thick, whatever the cells at the hole.
        let surface = ring(
            N,
            cell,
            |i| {
                let t = 2.0 * std::f64::consts::PI * i as f64 / N as f64;
                let (c, s) = (t.cos(), t.sin());
                let m = c.abs().max(s.abs());
                ([c, s], [half * (c / m), half * (s / m)])
            },
            |a, b| {
                if a[0] == half && b[0] == half {
                    OUTER_A
                } else if a[0] == -half && b[0] == -half {
                    left
                } else if a[1] == -half && b[1] == -half {
                    bottom
                } else {
                    OUTER_B
                }
            },
        );
        let supports = [slide(left, 0), slide(bottom, 1), slide(AT_Z0, 2), slide(AT_Z1, 2)];
        let loads = [Load::Traction { face: OUTER_A, traction: [1.0, 0.0, 0.0] }];
        let pull: f64 = (surface.triangles().iter().zip(surface.tri_face()))
            .filter(|(_, tag)| **tag == OUTER_A)
            .map(|(triangle, _)| {
                let [a, b, c] = triangle.map(|v| surface.points()[v as usize]);
                // The loaded face is in a plane x = const: half the cross product's x.
                0.5 * ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])).abs()
            })
            .sum();
        let (mut lowest, mut highest) = (f64::INFINITY, 0.0f64);
        let (mut fewest, mut most, mut seconds) = (usize::MAX, 0usize, 0.0f64);
        let mut levels = [0usize; 5];
        for offset in [[0.0, 0.0, 0.0], [0.0, 0.21, 0.0], [0.37, 0.21, 0.0], [0.37, 0.9, 0.0], [0.9, 0.1, 0.0], [0.9, 0.5, 0.0]] {
            let options = Options { degree, offset, refine, adapt, basis, ..Options::new(cell) };
            let started = std::time::Instant::now();
            let s = solve(&surface, &material, &supports, &loads, &options, &Limits::default()).unwrap();
            seconds += started.elapsed().as_secs_f64() / 6.0;
            (fewest, most) = (fewest.min(s.stats.unknowns), most.max(s.stats.unknowns));
            // The left face carries the pull on the loaded one (the right side less the segment
            // at a corner that rounding tags otherwise), and the quadrature of the divided cells
            // gives the plate's volume to the tessellation of the hole.
            let reaction = s.reactions.iter().find(|r| r.face == left).unwrap().force[0];
            assert!((reaction + pull).abs() < 1e-6 * pull, "cell {cell}, degree {degree}, {refine} levels: reaction {reaction} against {pull}");
            assert!((s.stats.volume - s.stats.surface_volume).abs() < 1e-4 * s.stats.surface_volume, "volume {}", s.stats.volume);
            assert_eq!(s.stats.levels.iter().sum::<usize>(), s.stats.inside_cells + s.stats.cut_cells);
            for (total, count) in levels.iter_mut().zip(&s.stats.levels) {
                *total = (*total).max(*count);
            }
            for y in [1.0, -1.0] {
                let stress = s.stress([0.0, y, 0.5 * cell]).unwrap()[0];
                (lowest, highest) = (lowest.min(stress), highest.max(stress));
            }
        }
        let spread = (highest - lowest) / (highest + lowest);
        println!(
            "cell {cell}, degree {degree} ({basis:?}), {refine} levels, {adapt} rounds: σ_xx at the hole from {lowest:.3} to {highest:.3} over six grid positions (± {:.1} %); {fewest} to {most} unknowns, at most {levels:?} cells by level, {seconds:.1} s a solve",
            100.0 * spread
        );
        spreads.push((spread, most));
    }
    // Divided twice at the hole, cells of the radius are as good as a uniform grid of four cells
    // a radius at degree 3 and of two at degree 4, with fewer unknowns.
    assert!(spreads[2].0 < 0.03 && spreads[2].1 < spreads[4].1 / 4, "{spreads:?}");
    assert!(spreads[7].0 < 0.02 && spreads[7].1 < spreads[8].1, "{spreads:?}");
    // Rounds that follow the solution ([`Options::adapt`]) follow the strain energy, of which a
    // stress peak holds little: two of them leave the stress at the hole within ± 7 % where two
    // levels at the hole leave it within ± 0.7 %, and a third comes to ± 1.6 %.
    assert!(spreads[11].0 > 2.0 * spreads[2].0 && spreads[12].0 < spreads[11].0 && spreads[12].0 < 0.03, "{spreads:?}");
    // On cells divided at the hole the trunk space does with a third of the unknowns at degree
    // 3 (± 1.1 % where the tensor product has ± 0.7 %) and with a little over half at degree 4
    // (± 0.5 %): what it lacks of the tensor product, the smaller cells make up for.
    assert!(spreads[13].0 < 0.02 && 3 * spreads[13].1 < spreads[2].1 + spreads[2].1 / 10, "{spreads:?}");
    assert!(spreads[14].0 < 0.01 && spreads[14].0 < spreads[2].0 && 5 * spreads[14].1 < 3 * spreads[2].1, "{spreads:?}");
}
