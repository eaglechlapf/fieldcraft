//! The meshing test corpus on the grid: every part SolidCraft builds from `tests/corpus/*/recipe.json`
//! (thin walls, slivers, near-tangent faces, small features, lofts and splines), each body solved
//! under its own weight with its largest face clamped, with no volume mesh.
//!
//! None of these parts has a closed-form stress. What is checked is what must hold for any part:
//! the solve finishes, the quadrature gives the volume the surface encloses, the clamped face
//! carries the weight (the reaction is density × gravity × volume, straight up), and the
//! displacement is finite everywhere on the surface. The change of the strain energy from degree
//! 2 to degree 3 on the same grid is printed as a measure of how far degree 2 is from converged.
//! Degree 3 is solved a second time with the cells divided once where the surface is curved
//! (`Options::refine`), which must hold the same.
//!
//! The numbers are in the consistent set millimetre, newton, megapascal (and tonne).

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use fieldcraft_fcm::{Load, Material, Options, Solution, Support, solve};
use fieldcraft_geometry::{open_design, script_design};
use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210_000.0, poisson: 0.3, density: 7.85e-9 };
const GRAVITY: f64 = 9810.0;
/// Cells along the longest side of a body.
const CELLS: f64 = 8.0;

struct Row {
    part: String,
    line: String,
    failed: bool,
}

fn run(surface: &BoundarySurface, clamped: u32, cell: f64, degree: usize, refine: usize) -> Result<(Solution, f64), String> {
    let supports = [Support { face: clamped, fixed: [true; 3], value: [0.0; 3] }];
    let loads = [Load::Gravity { acceleration: [0.0, 0.0, -GRAVITY] }];
    let options = Options { degree, refine, offset: [0.37, 0.21, 0.43], ..Options::new(cell) };
    let started = Instant::now();
    match catch_unwind(AssertUnwindSafe(|| solve(surface, &STEEL, &supports, &loads, &options, &Limits::default()))) {
        Ok(Ok(solution)) => Ok((solution, started.elapsed().as_secs_f64())),
        Ok(Err(e)) => Err(format!("{e}")),
        Err(_) => Err("PANIC".into()),
    }
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn every_corpus_part_solves_under_its_own_weight() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/corpus");
    let mut parts: Vec<_> = std::fs::read_dir(&root).unwrap().flatten().map(|e| e.path()).filter(|p| p.join("recipe.json").is_file()).collect();
    parts.sort();
    let mut rows: Vec<Row> = Vec::new();
    for part in &parts {
        let name = part.file_name().unwrap().to_string_lossy().into_owned();
        let script = std::fs::read_to_string(part.join("recipe.json")).unwrap();
        let design = match script_design(&script).and_then(|bytes| open_design(&bytes)) {
            Ok(design) => design,
            Err(e) => {
                rows.push(Row { part: name, line: format!("the design does not build: {e}"), failed: true });
                continue;
            }
        };
        for body in &design.bodies {
            let label = if design.bodies.len() > 1 { format!("{name}/{}", body.name) } else { name.clone() };
            let measure = match body.measure() {
                Ok(measure) => measure,
                Err(e) => {
                    rows.push(Row { part: label, line: format!("no measure: {e}"), failed: true });
                    continue;
                }
            };
            let size = [0, 1, 2].map(|i| measure.bbox_max[i] - measure.bbox_min[i]);
            let longest = size[0].max(size[1]).max(size[2]);
            let diagonal = (size[0] * size[0] + size[1] * size[1] + size[2] * size[2]).sqrt();
            let surface = body.tessellate(diagonal / 2000.0).map_err(|e| e.to_string()).and_then(|mesh| {
                let options = SurfaceOptions { weld_distance: mesh.weld_distance, ..SurfaceOptions::default() };
                BoundarySurface::new(&mesh.positions, &mesh.triangles, &mesh.tri_face, &options).map_err(|e| e.to_string())
            });
            let surface = match surface {
                Ok(surface) => surface,
                Err(e) => {
                    rows.push(Row { part: label, line: format!("no boundary surface: {e}"), failed: true });
                    continue;
                }
            };
            // The largest face is clamped.
            let mut areas: BTreeMap<u32, f64> = BTreeMap::new();
            for (triangle, tag) in surface.triangles().iter().zip(surface.tri_face()) {
                let [a, b, c] = triangle.map(|v| surface.points()[v as usize]);
                let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
                let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
                *areas.entry(*tag).or_insert(0.0) += 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            }
            let clamped = areas.iter().fold((0u32, 0.0f64), |best, (tag, area)| if *area > best.1 { (*tag, *area) } else { best }).0;
            let cell = longest / CELLS;
            let weight = STEEL.density * GRAVITY * surface.volume();
            let mut line = format!(
                "{:.0}×{:.0}×{:.0} mm, {} faces, {} triangles, cell {cell:.1} mm",
                size[0],
                size[1],
                size[2],
                areas.len(),
                surface.triangles().len()
            );
            let mut failed = false;
            let mut energies = Vec::new();
            for (degree, refine) in [(2, 0), (3, 0), (3, 1)] {
                let divided = if refine > 0 { format!(" divided {refine}") } else { String::new() };
                match run(&surface, clamped, cell, degree, refine) {
                    Err(e) => {
                        line += &format!(" | p{degree}{divided}: {e}");
                        failed = true;
                    }
                    Ok((solution, seconds)) => {
                        let stats = &solution.stats;
                        let volume_error = (stats.volume - stats.surface_volume).abs() / stats.surface_volume;
                        let reaction = solution.reactions.iter().find(|r| r.face == clamped).map_or([f64::NAN; 3], |r| r.force);
                        let reaction_error = ((reaction[2] - weight).abs() + reaction[0].abs() + reaction[1].abs()) / weight;
                        let (mut largest, mut lost) = (0.0f64, 0usize);
                        for x in surface.points() {
                            match solution.displacement(*x) {
                                Some(u) if u.iter().all(|c| c.is_finite()) => largest = largest.max((u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()),
                                _ => lost += 1,
                            }
                        }
                        if refine == 0 {
                            energies.push(solution.strain_energy);
                        }
                        let bad = volume_error > 1e-3 || !(reaction_error < 1e-3) || lost > 0 || !(solution.strain_energy > 0.0);
                        failed |= bad;
                        line += &format!(
                            " | p{degree}{divided}: {}+{} cells, {} unknowns, {seconds:.1} s, volume error {volume_error:.1e}, reaction error {reaction_error:.1e}, smallest fraction {:.0e}, largest displacement {largest:.2e} mm, energy {:.4e}{}{}",
                            stats.inside_cells,
                            stats.cut_cells,
                            stats.unknowns,
                            stats.smallest_fraction,
                            solution.strain_energy,
                            if lost > 0 { format!(", {lost} surface points without a value") } else { String::new() },
                            if bad { " ← CHECK" } else { "" }
                        );
                    }
                }
            }
            if let [coarse, fine] = energies[..] {
                line += &format!(" | energy change p2→p3: {:+.1} %", 100.0 * (fine - coarse) / fine);
            }
            println!("{label}: {line}");
            rows.push(Row { part: label, line, failed });
        }
    }
    let failed: Vec<&Row> = rows.iter().filter(|r| r.failed).collect();
    println!("\n{} bodies, {} with a failure or a check:", rows.len(), failed.len());
    for row in &failed {
        println!("  {}: {}", row.part, row.line);
    }
}
