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
//!
//! What the peak of one solution is worth shows over several positions of the grid
//! (`the_fillet_stress_over_grid_positions`, the fillet read every half millimetre, a chordal
//! tolerance of 0.05 mm, degree 3): on 8 mm cells it lies anywhere from 184 to 243 MPa, on 4 mm
//! cells from 201 to 219. With local refinement (`Options::refine`) the 8 mm cells are divided
//! at the fillet and the holes: once, they are 4 mm there and the peak lies from 202 to 219 MPa
//! with two thirds of the unknowns of the 4 mm grid; twice, from 211 to 221; at degree 4 once,
//! from 210 to 219. The bracket is small against its fillet and holes, so more than half of its
//! cells are divided and the saving is modest.
//!
//! Read that densely, the peak is the largest stress anywhere on the fillet, and before the
//! stress of a cell that holds little material was read with its neighbours'
//! (`Solution::stress`) it was the noise of such cells: up to 248 MPa on the cells divided
//! twice, 17 % above the reference, in cells 0.3 to 7 % full.

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
        // Cell size, degree and levels of local refinement (`Options::refine`): 8 mm cells
        // divided at the R6 fillet and the holes against 4 mm cells everywhere.
        let cases: &[(f64, usize, usize)] = if tolerance == 0.05 {
            &[(8.0, 2, 0), (8.0, 3, 0), (4.0, 2, 0), (4.0, 3, 0), (8.0, 4, 0), (8.0, 3, 1), (8.0, 3, 2), (8.0, 4, 1)]
        } else {
            &[(8.0, 3, 0), (4.0, 3, 0), (8.0, 4, 0), (8.0, 3, 1), (8.0, 3, 2)]
        };
        for &(cell, degree, refine) in cases {
            let started = Instant::now();
            // The grid starts off the part's corner, so no face of the bracket lies on a grid plane.
            let options = Options { degree, refine, offset: [0.37, 0.21, 0.43], ..Options::new(cell) };
            let solution = match solve(&surface, &STEEL, &supports, &loads, &options, &Limits::default()) {
                Ok(solution) => solution,
                Err(e) => {
                    println!("cell {cell} mm, degree {degree}, {refine} levels: {e}");
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
            let levels = if refine > 0 { format!(", {refine} levels (cells by level {:?})", &stats.levels[..=refine]) } else { String::new() };
            println!(
                "cell {cell} mm, degree {degree}{levels}: {} inside and {} cut cells, {} unknowns, {} points, {seconds:.1} s; volume error {:.1e}, smallest fraction {:.1e}; fillet peak {peak:.1} MPa ({:+.1} % of {REFERENCE}) at [{:.1}, {:.1}, {:.1}]; reaction {:.3} {:.3} {:.3} N; strain energy {:.4} N mm",
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

/// Points all over a face: every triangle of the face divided evenly until no edge is longer
/// than `spacing`, its corners and edges included.
fn samples(surface: &BoundarySurface, face: u32, spacing: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for (triangle, tag) in surface.triangles().iter().zip(surface.tri_face()) {
        if *tag != face {
            continue;
        }
        let [a, b, c] = triangle.map(|v| surface.points()[v as usize]);
        let length = |p: [f64; 3], q: [f64; 3]| (0..3).map(|i| (p[i] - q[i]) * (p[i] - q[i])).sum::<f64>().sqrt();
        let steps = (length(a, b).max(length(b, c)).max(length(c, a)) / spacing).ceil().max(1.0) as usize;
        for i in 0..=steps {
            for j in 0..=steps - i {
                let (s, t) = (i as f64 / steps as f64, j as f64 / steps as f64);
                points.push([0, 1, 2].map(|k| a[k] + s * (b[k] - a[k]) + t * (c[k] - a[k])));
            }
        }
    }
    points
}

/// The spacing of the points the fillet peak is read at, in millimetres: far below the finest
/// cells, so the peak is the largest stress on the face and not the largest at the corners of
/// its facets.
const SAMPLING: f64 = 0.5;

/// Where the grid sits, on the bracket: the fillet peak over six positions of the grid, for
/// 4 mm cells everywhere and for 8 mm cells divided at the fillet and the holes
/// (`Options::refine`). One position tells little: the peak of a fillet of 6 mm on cells of 4 mm
/// moves by several per cent with the grid, as the stress at a hole does (`tests/hole.rs`).
///
/// The peak is the largest von Mises stress over points [`SAMPLING`] apart all over the fillet,
/// as an application reads it, at two chordal tolerances of the tessellation: 0.05 mm and
/// 0.0877 mm, a thousandth of the bracket's diagonal.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_fillet_stress_over_grid_positions() {
    for tolerance in [0.05, 0.0877] {
        let (surface, [foot, top, fillet]) = bracket(tolerance);
        let on_fillet = samples(&surface, fillet, SAMPLING);
        println!("tessellation at {tolerance} mm: {} triangles, the fillet read at {} points", surface.triangles().len(), on_fillet.len());
        let supports = [Support { face: foot, fixed: [true; 3], value: [0.0; 3] }];
        let loads = [Load::Force { face: top, force: [FORCE, 0.0, 0.0] }];
        let mut spreads = Vec::new();
        for (cell, degree, refine) in [(8.0, 3, 0), (4.0, 3, 0), (8.0, 3, 1), (8.0, 3, 2), (8.0, 4, 0), (8.0, 4, 1)] {
            let (mut lowest, mut highest, mut at) = (f64::INFINITY, 0.0f64, [0.0; 3]);
            let (mut fewest, mut most, mut seconds) = (usize::MAX, 0usize, 0.0f64);
            let (mut least, mut largest) = (f64::INFINITY, 0.0f64);
            let mut peaks = String::new();
            for offset in [[0.37, 0.21, 0.43], [0.0, 0.0, 0.0], [0.11, 0.63, 0.87], [0.71, 0.05, 0.29], [0.5, 0.5, 0.5], [0.93, 0.41, 0.17]] {
                let options = Options { degree, refine, offset, ..Options::new(cell) };
                let started = Instant::now();
                let solution = solve(&surface, &STEEL, &supports, &loads, &options, &Limits::default()).unwrap();
                seconds += started.elapsed().as_secs_f64() / 6.0;
                let (mut peak, mut peak_at) = (0.0f64, [0.0; 3]);
                for x in &on_fillet {
                    let stress = solution.von_mises(*x).unwrap();
                    if stress > peak {
                        (peak, peak_at) = (stress, *x);
                    }
                }
                peaks += &format!(" {peak:.1}");
                if peak > highest {
                    (highest, at) = (peak, peak_at);
                }
                lowest = lowest.min(peak);
                (fewest, most) = (fewest.min(solution.stats.unknowns), most.max(solution.stats.unknowns));
                (least, largest) = (least.min(solution.strain_energy), largest.max(solution.strain_energy));
                let reaction = solution.reactions.iter().find(|r| r.face == foot).unwrap().force;
                assert!((reaction[0] + FORCE).abs() < 1e-6 * FORCE, "the foot carries the load: {reaction:?}");
                let stats = &solution.stats;
                assert!((stats.volume - stats.surface_volume).abs() < 1e-3 * stats.surface_volume, "the quadrature gives the part's volume");
            }
            let spread = (highest - lowest) / (highest + lowest);
            println!(
                "cell {cell} mm, degree {degree}, {refine} levels: fillet peak from {lowest:.1} to {highest:.1} MPa over six grid positions (± {:.1} %, {:+.1} % to {:+.1} % of {REFERENCE}:{peaks}), the highest at [{:.2}, {:.2}, {:.2}]; strain energy {least:.3} to {largest:.3} N mm; {fewest} to {most} unknowns, {seconds:.1} s a solve",
                100.0 * spread,
                100.0 * (lowest - REFERENCE) / REFERENCE,
                100.0 * (highest - REFERENCE) / REFERENCE,
                at[0],
                at[1],
                at[2]
            );
            spreads.push((lowest / REFERENCE - 1.0, highest / REFERENCE - 1.0));
        }
        // Divided at the fillet, the 8 mm cells put the peak within a few per cent of the
        // reference wherever the grid sits: at degree 3 once and twice, and at degree 4 once.
        // Alone they are up to 15 % off.
        for (case, low, high) in [(2, -0.07, 0.05), (3, -0.03, 0.08), (5, -0.03, 0.05)] {
            let (lowest, highest) = spreads[case];
            assert!(lowest > low && highest < high, "tolerance {tolerance}, case {case}: {lowest:+.3} to {highest:+.3}");
        }
        assert!(spreads[0].0 < -0.1 || spreads[0].1 > 0.1, "tolerance {tolerance}: {:?}", spreads[0]);
    }
}
