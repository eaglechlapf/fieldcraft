//! Theory: a body under the same pressure all round has the stress −p in every direction and
//! shrinks evenly, u = −p (1 − 2ν) x / E, whatever its shape (S. P. Timoshenko, J. N. Goodier,
//! *Theory of Elasticity*, 3rd ed., McGraw-Hill 1970, uniform hydrostatic pressure).
//!
//! The part is a tetrahedron in the corner of the coordinate planes: sliding supports on the
//! three faces in those planes stand for the pressure there, and the pressure acts on the
//! inclined face, which cuts the cells at an angle to every axis. The displacement is linear, so
//! every degree represents it, and the quadrature gives the volume of the cut cells exactly. What
//! is left is the fictitious stiffness: the cell that holds least material holds 0.03 % of its
//! volume, and the stress there is the least accurate value (within 1 % of the pressure).

mod common;

use common::{INCLINED, X_LO, Y_LO, Z_LO, corner};
use fieldcraft_fcm::{Load, Material, Options, Support, solve};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210e9, poisson: 0.3, density: 7850.0 };
const REACH: [f64; 3] = [0.9, 0.6, 0.45];
const PRESSURE: f64 = 50e6;

#[test]
fn a_corner_under_pressure_shrinks_evenly() {
    let supports = [
        Support { face: X_LO, fixed: [true, false, false], value: [0.0; 3] },
        Support { face: Y_LO, fixed: [false, true, false], value: [0.0; 3] },
        Support { face: Z_LO, fixed: [false, false, true], value: [0.0; 3] },
    ];
    let loads = [Load::Pressure { face: INCLINED, pressure: PRESSURE }];
    let volume = REACH[0] * REACH[1] * REACH[2] / 6.0;
    let shrink = PRESSURE * (1.0 - 2.0 * STEEL.poisson) / STEEL.young;
    for (degree, depth, tolerance) in [(1, 2, 2e-2), (2, 2, 5e-3), (2, 4, 5e-3), (3, 3, 2e-2)] {
        let options = Options { degree, depth, offset: [0.17, 0.41, 0.29], ..Options::new(0.25) };
        let solution = solve(&corner(REACH), &STEEL, &supports, &loads, &options, &Limits::default()).unwrap();
        let (mut displacement, mut stress) = (0.0f64, 0.0f64);
        // Points of the tetrahedron by barycentric coordinates, its corners and faces included.
        for weights in [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.3, 0.3, 0.4],
            [0.5, 0.0, 0.2, 0.3],
            [0.1, 0.6, 0.2, 0.1],
        ] {
            let x = [weights[1] * REACH[0], weights[2] * REACH[1], weights[3] * REACH[2]];
            let u = solution.displacement(x).unwrap();
            let s = solution.stress(x).unwrap();
            for i in 0..3 {
                displacement = displacement.max((u[i] + shrink * x[i]).abs() / (shrink * REACH[0]));
                stress = stress.max((s[i] + PRESSURE).abs() / PRESSURE).max(s[3 + i].abs() / PRESSURE);
            }
        }
        let stats = &solution.stats;
        println!(
            "degree {degree}, depth {depth}: cells {:?} ({} inside, {} cut), {} unknowns, {} points, volume error {:.2e}, smallest fraction {:.2e}, displacement error {displacement:.2e}, stress error {stress:.2e}",
            stats.cells,
            stats.inside_cells,
            stats.cut_cells,
            stats.unknowns,
            stats.quadrature_points,
            (stats.volume - volume).abs() / volume,
            stats.smallest_fraction
        );
        assert!((stats.surface_volume - volume).abs() < 1e-12 * volume);
        assert!((stats.volume - volume).abs() < 1e-12 * volume);
        assert!(displacement < tolerance, "degree {degree}, depth {depth}: displacement error {displacement:.3e}");
        assert!(stress < tolerance, "degree {degree}, depth {depth}: stress error {stress:.3e}");
        // Each support carries the pressure on its own face: p times the face's area.
        for (face, axis) in [(X_LO, 0), (Y_LO, 1), (Z_LO, 2)] {
            let area = 0.5 * REACH[(axis + 1) % 3] * REACH[(axis + 2) % 3];
            let force = solution.reactions.iter().find(|r| r.face == face).unwrap().force[axis];
            assert!((force - PRESSURE * area).abs() < 1e-6 * PRESSURE * area, "face {face}: {force} vs {}", PRESSURE * area);
        }
    }
}
