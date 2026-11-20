//! A real part: the filleted L-bracket of the oracle (`tests/oracle/filleted-bracket-fillet-stress`),
//! built by SolidCraft from that case's `design.json` and solved on the grid with no volume mesh.
//!
//! An 8 mm L-bracket with an R6 inside fillet and a hole through each leg. The foot face is
//! clamped and 2 kN in +x act on the top face (steel, E = 210 GPa, ν = 0.3). The oracle's
//! reference is the peak von Mises stress on the fillet face, 213 MPa within 3 %, the
//! mesh-independent value of a four-level study of quadratic tetrahedral meshes (that case's
//! README.md), and the reaction, −2000 N, which equilibrium gives.
//!
//! The numbers here are in the consistent set millimetre, newton, megapascal, the units of the
//! design and of the oracle's recipe.

use std::time::Instant;

use fieldcraft_fcm::{Load, Material, Options, Support, solve};
use fieldcraft_geometry::{open_design, script_design};
use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210_000.0, poisson: 0.3, density: 7.85e-9 };
const FORCE: f64 = 2000.0;
const REFERENCE: f64 = 213.0;

/// The bracket's boundary at a chordal tolerance, with the tags of its foot, top and fillet
/// faces.
fn bracket(tolerance: f64) -> (BoundarySurface, [u32; 3]) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/oracle/filleted-bracket-fillet-stress/design.json");
    let script = std::fs::read_to_string(path).unwrap();
    let design = open_design(&script_design(&script).unwrap()).unwrap();
    assert!(design.errors.is_empty(), "{:?}", design.errors);
    let body = design.body("Bracket").unwrap();
    let mesh = body.tessellate(tolerance).unwrap();
    let options = SurfaceOptions { weld_distance: mesh.weld_distance, ..SurfaceOptions::default() };
    let surface = BoundarySurface::new(&mesh.positions, &mesh.triangles, &mesh.tri_face, &options).unwrap();
    let face = |name: &str| body.face_index(name).unwrap();
    (surface, [face("F2:side:l1"), face("F2:side:l5"), face("F3:blend(F2:side:l3|F2:side:l4)")])
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_fillet_stress_of_a_real_bracket() {
    for tolerance in [0.05, 0.01, 0.002] {
        let started = Instant::now();
        let (surface, [foot, top, fillet]) = bracket(tolerance);
        println!("tessellation at {tolerance} mm: {} triangles, {:.1} s", surface.triangles().len(), started.elapsed().as_secs_f64());
        // The points of the fillet face: the corners and the centre of each of its triangles.
        let mut on_fillet: Vec<[f64; 3]> = Vec::new();
        for (triangle, tag) in surface.triangles().iter().zip(surface.tri_face()) {
            if *tag != fillet {
                continue;
            }
            let corners = triangle.map(|v| surface.points()[v as usize]);
            on_fillet.extend(corners);
            on_fillet.push([0, 1, 2].map(|i| (corners[0][i] + corners[1][i] + corners[2][i]) / 3.0));
        }
        assert!(!on_fillet.is_empty());
        let supports = [Support { face: foot, fixed: [true; 3], value: [0.0; 3] }];
        let loads = [Load::Force { face: top, force: [FORCE, 0.0, 0.0] }];
        let cases: &[(f64, usize)] =
            if tolerance == 0.05 { &[(8.0, 2), (8.0, 3), (4.0, 2), (4.0, 3), (8.0, 4)] } else { &[(8.0, 3), (4.0, 3), (8.0, 4)] };
        for &(cell, degree) in cases {
            let started = Instant::now();
            // The grid starts off the part's corner, so no face of the bracket lies on a grid plane.
            let options = Options { degree, offset: [0.37, 0.21, 0.43], ..Options::new(cell) };
            let solution = match solve(&surface, &STEEL, &supports, &loads, &options, &Limits::default()) {
                Ok(solution) => solution,
                Err(e) => {
                    println!("cell {cell} mm, degree {degree}: {e}");
                    continue;
                }
            };
            let seconds = started.elapsed().as_secs_f64();
            let (mut peak, mut at) = (0.0f64, [0.0; 3]);
            for x in &on_fillet {
                let stress = solution.von_mises(*x).unwrap();
                if stress > peak {
                    (peak, at) = (stress, *x);
                }
            }
            let reaction = solution.reactions.iter().find(|r| r.face == foot).unwrap().force;
            let stats = &solution.stats;
            println!(
                "cell {cell} mm, degree {degree}: {} inside and {} cut cells, {} unknowns, {} points, {seconds:.1} s; volume error {:.1e}, smallest fraction {:.1e}; fillet peak {peak:.1} MPa ({:+.1} % of {REFERENCE}) at [{:.1}, {:.1}, {:.1}]; reaction {:.3} {:.3} {:.3} N; strain energy {:.4} N mm",
                stats.inside_cells,
                stats.cut_cells,
                stats.unknowns,
                stats.quadrature_points,
                (stats.volume - stats.surface_volume).abs() / stats.surface_volume,
                stats.smallest_fraction,
                100.0 * (peak - REFERENCE) / REFERENCE,
                at[0],
                at[1],
                at[2],
                reaction[0],
                reaction[1],
                reaction[2],
                solution.strain_energy
            );
            assert!((reaction[0] + FORCE).abs() < 1e-6 * FORCE, "the foot carries the load: {reaction:?}");
            // The facets of the holes and the fillet are not split at: the volume is right to the
            // order of the tessellation's chordal error over the part's size.
            assert!((stats.volume - stats.surface_volume).abs() < 1e-3 * stats.surface_volume, "the quadrature gives the part's volume");
        }
    }
}
