//! Where the stress at a hole goes wrong: diagnostics for Kirsch's plate (`benchmarks.rs`).
//!
//! In a quarter model, the top and the side of the hole lie on the symmetry planes, whose
//! supports are imposed weakly (Nitsche). The stress normal to such a plane, evaluated on the
//! plane, is the least accurate value there: `quarter_field` prints the field against Kirsch's
//! closed form on the plane, just off it and round the hole. `whole_plate` solves the whole
//! plate, where no support touches the hole, and the same stresses behave.
mod common;

use common::{AT_X0, AT_Y0, AT_Z0, AT_Z1, OUTER_A, OUTER_B, ring, strip};
use fieldcraft_fcm::{Load, Material, Options, Support, solve};
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
