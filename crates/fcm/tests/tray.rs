//! A real sheet metal part: a tray of 2 mm steel, 400 × 300 mm with four walls 50 mm high bent
//! up from its floor, built by SolidCraft's sheet metal commands from the recipe below (a base
//! flange, four edge flanges with an inside bend radius of 2 mm, a Ø60 passage and six Ø8 holes
//! in the floor, two Ø6 holes in each long wall). Its four corners are open: the ends of two
//! neighbouring walls are 2 mm apart, as the bends leave them.
//!
//! A cell of a twentieth of the tray's diagonal is 25 mm, a dozen sheet thicknesses, and holds
//! both walls at a corner. Two load cases show what that does:
//! - **Bolted.** The four holes near the corners are fixed and a pressure of 1 kPa acts on the
//!   floor. The corners carry little, and joined or not the floor sags the same to 1 %.
//! - **Hung by one wall.** The outside of the back wall is fixed and 100 N pull the front wall
//!   down. Everything goes through the bend along the back wall; the side walls end free at the
//!   back corners. One polynomial a cell joins them to the back wall there, they carry the
//!   load as ribs, and the floor sags an eightieth of what it does.
//!
//! The reference is FieldCraft's own solver on a body-fitted mesh of the same part: quadratic
//! tetrahedra of 4 mm (157 000 elements, 920 000 unknowns), on which the floor's edge at the
//! passage sags 0.1532 mm bolted (0.1533 mm on tetrahedra of 2.5 mm) and 2.175 mm hung by the
//! wall. That is a cross-check between two solvers of ours, not a closed form: the tolerances
//! say how far the grid is from it, and the hung tray is a hard case for both (a plate of 2 mm
//! bent over 300 mm, with stress concentrated where the bend meets the corner reliefs).
//!
//! What is left of the error once the corners are open is the bend along the fixed wall, which
//! cells of 12.5 mm do not resolve. A section through it alone (the wall, the bend and a strip
//! of the floor, 10 mm wide; measured once and not kept as a test) converges to a deflection
//! that cells of 25 mm miss by 8 %, cells divided to 6 mm at the bend by 2 % and to 3 mm by
//! 0.7 %. Dividing the cells of every bend and hole that far is more than the solver holds;
//! dividing the ones the solution is least resolved in ([`Options::adapt`]) is not.
//!
//! The numbers are in the consistent set millimetre, newton, megapascal.

use std::collections::BTreeMap;
use std::time::Instant;

use fieldcraft_fcm::{Load, Material, Options, Support, solve};
use fieldcraft_geometry::{open_design, script_design};
use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210_000.0, poisson: 0.3, density: 7.85e-9 };
/// The tray's design: SolidCraft's own script form.
const RECIPE: &str = r#"{"commands": [
  {"command": "sheet.manage_rules", "params": {"name": "Steel2", "thickness": "2 mm", "k_factor": "0.44", "bend_radius": "2 mm", "active": true}},
  {"command": "sketch.create", "params": {"plane": "XY", "name": "BaseSketch"}},
  {"command": "sketch.draw.dimensioned", "params": {"command": "sketch.rectangle.center", "params": {"center": [0, 0], "corner": [200, 150]}, "dimensions": {"width": "400", "height": "300"}}},
  {"command": "sketch.finish", "params": {}},
  {"command": "sheet.flange", "params": {"sketch": "BaseSketch", "type": "base", "name": "Base", "body_name": "Tray"}},
  {"command": "sheet.flange", "params": {"type": "edge", "edges": [[0, 150, 2], [0, -150, 2], [200, 0, 2], [-200, 0, 2]], "height": 50, "angle": "90 deg", "name": "Walls", "body": "Tray"}},
  {"command": "solid.hole", "params": {"position": [0, 0, 2], "diameter": 60, "name": "Passage"}},
  {"command": "solid.hole", "params": {"position": [150, 100, 2], "diameter": 8, "name": "Mount1"}},
  {"command": "solid.hole", "params": {"position": [-150, 100, 2], "diameter": 8, "name": "Mount2"}},
  {"command": "solid.hole", "params": {"position": [150, -100, 2], "diameter": 8, "name": "Mount3"}},
  {"command": "solid.hole", "params": {"position": [-150, -100, 2], "diameter": 8, "name": "Mount4"}},
  {"command": "solid.hole", "params": {"position": [0, 100, 2], "diameter": 8, "name": "Mount5"}},
  {"command": "solid.hole", "params": {"position": [0, -100, 2], "diameter": 8, "name": "Mount6"}},
  {"command": "solid.hole", "params": {"position": [100, 150, 30], "diameter": 6, "depth": 2, "name": "Side1"}},
  {"command": "solid.hole", "params": {"position": [-100, 150, 30], "diameter": 6, "depth": 2, "name": "Side2"}},
  {"command": "solid.hole", "params": {"position": [100, -150, 30], "diameter": 6, "depth": 2, "name": "Side3"}},
  {"command": "solid.hole", "params": {"position": [-100, -150, 30], "diameter": 6, "depth": 2, "name": "Side4"}}
]}"#;
/// The volume of the tray, mm³: the flat pattern's bends and holes worked out by hand agree
/// with what SolidCraft measures.
const VOLUME: f64 = 361_623.99;
/// The sag of the floor's edge at the passage, at (30, 0, 2), on the body-fitted mesh (the
/// module's reference): bolted and hung by the back wall, mm.
const BOLTED: f64 = 0.1532;
const HUNG: f64 = 2.175;
/// A twentieth of the tray's diagonal: the cell an application takes when none is given.
const CELL: f64 = 25.12468905280223;

struct Tray {
    surface: BoundarySurface,
    /// The faces of the four bolt holes near the corners, the floor's upper face and the
    /// outsides of the back and the front wall.
    bolts: Vec<u32>,
    floor: u32,
    back: u32,
    front: u32,
}

fn tray(tolerance: f64) -> Tray {
    let design = open_design(&script_design(RECIPE).unwrap()).unwrap();
    assert!(design.errors.is_empty(), "{:?}", design.errors);
    let body = design.body("Tray").unwrap();
    let measured = body.measure().unwrap().volume;
    assert!((measured / VOLUME - 1.0).abs() < 1e-6, "volume {measured}");
    let mesh = body.tessellate(tolerance).unwrap();
    let options = SurfaceOptions { weld_distance: mesh.weld_distance, ..SurfaceOptions::default() };
    let surface = BoundarySurface::new(&mesh.positions, &mesh.triangles, &mesh.tri_face, &options).unwrap();
    // The faces by where they are: the box each face's triangles lie in.
    let mut boxes: BTreeMap<u32, ([f64; 3], [f64; 3])> = BTreeMap::new();
    for (triangle, tag) in surface.triangles().iter().zip(surface.tri_face()) {
        let (lo, hi) = boxes.entry(*tag).or_insert(([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]));
        for v in triangle {
            let p = surface.points()[*v as usize];
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
    }
    let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
    let flat = |axis: usize, at: f64, least: f64| -> u32 {
        let found: Vec<u32> = boxes
            .iter()
            .filter(|(_, (lo, hi))| near(lo[axis], at) && near(hi[axis], at) && hi[(axis + 1) % 3] - lo[(axis + 1) % 3] > least)
            .map(|(tag, _)| *tag)
            .collect();
        assert_eq!(found.len(), 1, "one face in the plane {at} across axis {axis}: {found:?}");
        found[0]
    };
    // The halves of the Ø8 holes at (±150, ±100): within 4 mm of their axes, through the floor.
    let bolts: Vec<u32> = boxes
        .iter()
        .filter(|(_, (lo, hi))| {
            let centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
            near(lo[2], 0.0)
                && near(hi[2], 2.0)
                && hi[0] - lo[0] < 8.1
                && hi[1] - lo[1] < 8.1
                && (centre[0].abs() - 150.0).abs() < 4.1
                && (centre[1].abs() - 100.0).abs() < 4.1
        })
        .map(|(tag, _)| *tag)
        .collect();
    assert_eq!(bolts.len(), 8, "{bolts:?}");
    Tray { floor: flat(2, 2.0, 300.0), back: flat(1, 150.0, 40.0), front: flat(1, -150.0, 40.0), bolts, surface }
}

/// The sag at the floor's edge at the passage, and the solution's own account.
fn sag(tray: &Tray, hung: bool, options: &Options) -> (f64, String) {
    let (supports, loads): (Vec<Support>, Vec<Load>) = if hung {
        (vec![Support { face: tray.back, fixed: [true; 3], value: [0.0; 3] }], vec![Load::Force { face: tray.front, force: [0.0, 0.0, -100.0] }])
    } else {
        (
            tray.bolts.iter().map(|face| Support { face: *face, fixed: [true; 3], value: [0.0; 3] }).collect(),
            vec![Load::Pressure { face: tray.floor, pressure: 0.001 }],
        )
    };
    let started = Instant::now();
    let solution = solve(&tray.surface, &STEEL, &supports, &loads, options, &Limits::default()).unwrap();
    let stats = &solution.stats;
    assert!(!stats.joined, "{stats:?}");
    let volume_error = (stats.volume - stats.surface_volume).abs() / stats.surface_volume;
    assert!(volume_error < 1e-4, "volume error {volume_error}");
    let total = solution.reactions.iter().fold([0.0; 3], |sum, r| [sum[0] + r.force[0], sum[1] + r.force[1], sum[2] + r.force[2]]);
    let account = format!(
        "{} unknowns, {} rounds (the last changed the strain energy by {:.2} %), cells by level {:?}, {} cells of several pieces ({} unknowns more), reaction {:.3?} N, {:.1} s",
        stats.unknowns,
        stats.rounds,
        100.0 * stats.settled,
        stats.levels,
        stats.split_cells,
        stats.separated,
        total,
        started.elapsed().as_secs_f64()
    );
    (-solution.displacement([30.0, 0.0, 2.0]).unwrap()[2], account)
}

#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_sheet_metal_tray_with_open_corners() {
    let tray = tray(0.05);
    println!("{} triangles", tray.surface.triangles().len());
    let grid = |separate: bool, gaps: usize| Options { refine: 1, separate, gaps, offset: [0.37, 0.21, 0.43], ..Options::new(CELL) };

    // Hung by its back wall. Joined, the corners carry the load.
    let (joined, account) = sag(&tray, true, &grid(false, 0));
    println!("hung, corners joined: {joined:.4} mm of {HUNG} ({account})");
    assert!(joined < 0.05 * HUNG, "{joined}");
    // Told apart, the walls are still held together where the gap between them ends.
    let (apart, account) = sag(&tray, true, &grid(true, 0));
    println!("hung, pieces told apart: {apart:.4} mm of {HUNG} ({account})");
    assert!(apart > 0.4 * HUNG && apart < 0.6 * HUNG, "{apart}");
    // With the cells divided where the gaps end, what is left is the plate's own bending on
    // cells of degree 3.
    let (divided, account) = sag(&tray, true, &grid(true, 4));
    println!("hung, cells divided where the gaps end: {divided:.4} mm of {HUNG} ({account})");
    assert!((divided / HUNG - 1.0).abs() < 0.15, "{divided}");

    // And with the cells divided where the solution is least resolved, which is the bend along
    // the wall that holds the tray: two rounds, neither of which settles it within 5 %.
    let rounds = Options { adapt: 2, settle: Options::SETTLED, ..grid(true, 4) };
    let (adapted, account) = sag(&tray, true, &rounds);
    println!("hung, two rounds that follow the solution: {adapted:.4} mm of {HUNG} ({account})");
    assert!((adapted / HUNG - 1.0).abs() < 0.05, "{adapted}");
    // From the grid's own cells, with none divided for the curvature of a bend or a hole, the
    // same two rounds do better than one level at every bend and hole, with fewer unknowns.
    let (followed, account) = sag(&tray, true, &Options { refine: 0, ..rounds });
    println!("hung, two rounds from the grid's own cells: {followed:.4} mm of {HUNG} ({account})");
    assert!((followed / HUNG - 1.0).abs() < 0.08 && (followed / HUNG - 1.0).abs() < (divided / HUNG - 1.0).abs(), "{followed}");

    // Bolted. The corners carry little.
    for (separate, gaps, tolerance) in [(false, 0, 0.02), (true, 4, 0.012)] {
        let (bolted, account) = sag(&tray, false, &grid(separate, gaps));
        println!("bolted, separate {separate}, {gaps} levels where gaps end: {bolted:.5} mm of {BOLTED} ({account})");
        assert!((bolted / BOLTED - 1.0).abs() < tolerance, "{bolted}");
    }
    let (bolted, account) = sag(&tray, false, &rounds);
    println!("bolted, two rounds that follow the solution: {bolted:.5} mm of {BOLTED} ({account})");
    assert!((bolted / BOLTED - 1.0).abs() < 0.005, "{bolted}");
}
