//! Theory: Lamé's thick-walled cylinder under internal pressure, in plane strain.
//!
//! A tube with inner radius a and outer radius b under the internal pressure p has
//! σ_r = A (1 − b²/r²), σ_θ = A (1 + b²/r²) with A = p a² / (b² − a²), and in plane strain the
//! radial displacement u_r = (1 + ν)/E · A ((1 − 2ν) r + b²/r) (S. P. Timoshenko, J. N. Goodier,
//! *Theory of Elasticity*, 3rd ed., McGraw-Hill 1970, the thick-walled cylinder).
//!
//! The model is a quarter of the tube with sliding supports on its two cuts and on both ends.
//! Its curved faces cut the cells at every angle, so this is the case the quadrature of cut cells
//! and the Nitsche supports on them have to get right. The surface is a tessellation, 32 flat
//! strips to the quarter circle, whose own error (a sagitta of 3e-4 of the radius) is below the
//! tolerances here.

mod common;

use common::{AT_X0, AT_Y0, AT_Z0, AT_Z1, INNER, quarter_tube};
use fieldcraft_fcm::{Load, Material, Options, Solution, Support, solve};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210e9, poisson: 0.3, density: 7850.0 };
const INNER_RADIUS: f64 = 0.1;
const OUTER_RADIUS: f64 = 0.2;
const LENGTH: f64 = 0.05;
const PRESSURE: f64 = 100e6;

fn tube(options: &Options) -> Solution {
    tube_within(options, &Limits::default()).unwrap()
}

fn tube_within(options: &Options, limits: &Limits) -> Result<Solution, fieldcraft_fcm::Error> {
    let supports = [
        Support { face: AT_X0, fixed: [true, false, false], value: [0.0; 3] },
        Support { face: AT_Y0, fixed: [false, true, false], value: [0.0; 3] },
        Support { face: AT_Z0, fixed: [false, false, true], value: [0.0; 3] },
        Support { face: AT_Z1, fixed: [false, false, true], value: [0.0; 3] },
    ];
    let loads = [Load::Pressure { face: INNER, pressure: PRESSURE }];
    solve(&quarter_tube(INNER_RADIUS, OUTER_RADIUS, LENGTH, 32), &STEEL, &supports, &loads, options, limits)
}

/// The largest errors of the radial displacement (relative to its value at the bore) and of the
/// hoop and radial stress (relative to the hoop stress at the bore) over points in the wall.
fn errors(solution: &Solution) -> (f64, f64) {
    let (a, b) = (INNER_RADIUS, OUTER_RADIUS);
    let scale = PRESSURE * a * a / (b * b - a * a);
    let radial = |r: f64| (1.0 + STEEL.poisson) / STEEL.young * scale * ((1.0 - 2.0 * STEEL.poisson) * r + b * b / r);
    let hoop_at_bore = scale * (1.0 + b * b / (a * a));
    let (mut displacement, mut stress) = (0.0f64, 0.0f64);
    for fraction in [0.02, 0.25, 0.5, 0.75, 0.98] {
        let r = a + fraction * (b - a);
        for degrees in [5.0f64, 30.0, 45.0, 70.0, 85.0] {
            let (sin, cos) = degrees.to_radians().sin_cos();
            let x = [r * cos, r * sin, 0.4 * LENGTH];
            let u = solution.displacement(x).unwrap();
            let s = solution.stress(x).unwrap();
            displacement = displacement.max((u[0] * cos + u[1] * sin - radial(r)).abs() / radial(a));
            // The stress in the polar axes: σ_r = n·σ·n, σ_θ = t·σ·t.
            let sigma_r = s[0] * cos * cos + s[1] * sin * sin + 2.0 * s[3] * sin * cos;
            let sigma_t = s[0] * sin * sin + s[1] * cos * cos - 2.0 * s[3] * sin * cos;
            stress = stress.max((sigma_r - scale * (1.0 - b * b / (r * r))).abs() / hoop_at_bore);
            stress = stress.max((sigma_t - scale * (1.0 + b * b / (r * r))).abs() / hoop_at_bore);
        }
    }
    (displacement, stress)
}

fn report(name: &str, solution: &Solution) -> (f64, f64) {
    let volume = std::f64::consts::FRAC_PI_4 * (OUTER_RADIUS * OUTER_RADIUS - INNER_RADIUS * INNER_RADIUS) * LENGTH;
    let (displacement, stress) = errors(solution);
    let stats = &solution.stats;
    // The pressure on the bore pushes the quarter outward: p a L along each of x and y, which
    // the two cuts take.
    let force = PRESSURE * INNER_RADIUS * LENGTH;
    let reaction = |face: u32, i: usize| solution.reactions.iter().find(|r| r.face == face).unwrap().force[i];
    println!(
        "{name}: cells {:?} ({} inside, {} cut), {} unknowns, {} points, volume error {:.2e}, smallest fraction {:.2e}, displacement error {displacement:.2e}, stress error {stress:.2e}, reactions {:.4} {:.4} of p a L",
        stats.cells,
        stats.inside_cells,
        stats.cut_cells,
        stats.unknowns,
        stats.quadrature_points,
        (stats.volume - volume).abs() / volume,
        stats.smallest_fraction,
        -reaction(AT_X0, 0) / force,
        -reaction(AT_Y0, 1) / force
    );
    (displacement, stress)
}

/// Four cells of degree 2 across the quarter: a coarse check that stays in the trunk gate.
#[test]
fn a_coarse_grid_is_within_a_few_per_cent() {
    let options = Options { degree: 2, depth: 1, ..Options::new(0.1) };
    let solution = tube(&options);
    let (displacement, stress) = report("degree 2", &solution);
    assert!(displacement < 0.05, "displacement error {displacement:.3e}");
    assert!(stress < 0.25, "stress error {stress:.3e}");
}

/// The change between two degrees on the same cells is the convergence indicator: it falls as
/// the degree rises, and its energy part tracks the error to the closed form.
#[test]
fn the_change_between_degrees_falls_as_the_solution_converges() {
    let solve_at = |degree: usize| tube(&Options { degree, ..Options::new(0.1) });
    let solutions = [solve_at(1), solve_at(2), solve_at(3), solve_at(4)];
    // Points through the wall, off the supports.
    let mut points = Vec::new();
    for fraction in [0.02, 0.25, 0.5, 0.75, 0.98] {
        for degrees in [5.0f64, 30.0, 45.0, 70.0, 85.0] {
            let r = INNER_RADIUS + fraction * (OUTER_RADIUS - INNER_RADIUS);
            points.push([r * degrees.to_radians().cos(), r * degrees.to_radians().sin(), 0.4 * LENGTH]);
        }
    }
    let same = solutions[2].change_from(&solutions[2], &points);
    assert_eq!((same.strain_energy, same.peak_von_mises, same.von_mises, same.points), (0.0, 0.0, 0.0, points.len()));
    // A point outside the part counts for nothing.
    assert_eq!(solutions[2].change_from(&solutions[1], &[[10.0, 10.0, 10.0]]).points, 0);
    let mut previous = (f64::INFINITY, f64::INFINITY);
    for pair in solutions.windows(2) {
        let change = pair[1].change_from(&pair[0], &points);
        let (_, stress) = errors(&pair[1]);
        println!(
            "change to the next degree: strain energy {:+.2e}, peak von Mises {:+.2e}, von Mises anywhere {:.2e}; stress error to Lamé {stress:.2e}",
            change.strain_energy, change.peak_von_mises, change.von_mises
        );
        assert_eq!(change.points, points.len());
        assert!(change.strain_energy.abs() < previous.0 && change.von_mises < previous.1, "{change:?} after {previous:?}");
        previous = (change.strain_energy.abs(), change.von_mises);
    }
    assert!(previous.0 < 5e-3 && previous.1 < 0.05, "degree 3 to 4: {previous:?}");
}

/// Raising the degree on the same four cells: the error falls by about a decade from degree 1
/// to 2 and from 2 to 3, to a few tenths of a per cent at degree 4.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_error_falls_with_the_degree() {
    let mut previous = f64::INFINITY;
    for degree in 1..=4 {
        let options = Options { degree, depth: 1, ..Options::new(0.1) };
        let (displacement, stress) = report(&format!("degree {degree}"), &tube(&options));
        assert!(displacement < previous, "degree {degree}: displacement error {displacement:.3e} after {previous:.3e}");
        previous = displacement;
        if degree == 4 {
            assert!(displacement < 0.005, "displacement error {displacement:.3e}");
            assert!(stress < 0.02, "stress error {stress:.3e}");
        }
    }
}

/// The trunk space on the same four cells: degree for degree it has fewer unknowns and a larger
/// error than the tensor product, and reaches the tensor product's accuracy at degree 4 with
/// two more degrees and fewer unknowns.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_trunk_space_converges_with_fewer_unknowns() {
    let tensor = tube(&Options { degree: 4, depth: 1, ..Options::new(0.1) });
    let (tensor_displacement, tensor_stress) = report("tensor product, degree 4", &tensor);
    let mut previous = f64::INFINITY;
    for degree in 2..=6 {
        let solution = tube(&Options { degree, depth: 1, basis: fieldcraft_fcm::Basis::Trunk, ..Options::new(0.1) });
        let (displacement, stress) = report(&format!("trunk, degree {degree}"), &solution);
        assert!(displacement < previous, "degree {degree}: displacement error {displacement:.3e} after {previous:.3e}");
        previous = displacement;
        if degree == 6 {
            assert!(solution.stats.unknowns < tensor.stats.unknowns);
            assert!(displacement < 2.0 * tensor_displacement && stress < 2.0 * tensor_stress, "{displacement:.3e}, {stress:.3e}");
        }
    }
}

/// A finer grid at degree 3, with the grid moved off the part's corner.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn a_finer_grid_at_degree_three() {
    let options = Options { degree: 3, depth: 1, offset: [0.37, 0.21, 0.43], ..Options::new(0.05) };
    let (displacement, stress) = report("degree 3, cell 0.05", &tube(&options));
    assert!(displacement < 0.005, "displacement error {displacement:.3e}");
    assert!(stress < 0.03, "stress error {stress:.3e}");
}

/// Local refinement on the tube: the bore has the radius of one cell, so its cells and those
/// around them are divided, and the coarse grid's error falls.
#[test]
fn refinement_at_the_bore_lowers_the_error() {
    let coarse = report("degree 2, no refinement", &tube(&Options { degree: 2, depth: 1, ..Options::new(0.1) }));
    let solution = tube(&Options { degree: 2, depth: 1, refine: 1, ..Options::new(0.1) });
    let refined = report("degree 2, one level", &solution);
    assert!(solution.stats.levels[1] > 0, "{:?}", solution.stats.levels);
    assert_eq!(solution.stats.levels.iter().sum::<usize>(), solution.stats.inside_cells + solution.stats.cut_cells);
    assert!(refined.0 < 0.25 * coarse.0 && refined.1 < 0.6 * coarse.1, "{refined:?} after {coarse:?}");
    // The cuts still carry the pressure on the bore: p a L along each of x and y.
    let force = PRESSURE * INNER_RADIUS * LENGTH;
    for (face, axis) in [(AT_X0, 0), (AT_Y0, 1)] {
        let reaction = solution.reactions.iter().find(|r| r.face == face).unwrap().force[axis];
        assert!((reaction + force).abs() < 2e-3 * force, "face {face}: {reaction} against {force}");
    }
}

/// The tube with one and two levels of refinement against the uniform grids, at degrees 3 and 4:
/// errors, unknowns and time.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn refinement_against_uniform_grids() {
    let mut errors = Vec::new();
    for (cell, degree, refine) in
        [(0.1, 3, 0), (0.1, 3, 1), (0.1, 3, 2), (0.05, 3, 0), (0.025, 3, 0), (0.1, 4, 0), (0.1, 4, 1), (0.1, 4, 2), (0.05, 4, 0), (0.2, 4, 2)]
    {
        let started = std::time::Instant::now();
        let solution = tube(&Options { degree, refine, offset: [0.37, 0.21, 0.43], ..Options::new(cell) });
        let seconds = started.elapsed().as_secs_f64();
        let found = report(
            &format!("cell {cell}, degree {degree}, {refine} levels, {seconds:.1} s, cells by level {:?}", &solution.stats.levels[..=refine]),
            &solution,
        );
        errors.push(found);
    }
    // Refinement is no worse than the grid it starts from.
    assert!(errors[1].1 < errors[0].1 && errors[2].1 < errors[0].1, "degree 3: {errors:?}");
    assert!(errors[6].1 < errors[5].1 && errors[7].1 < errors[5].1, "degree 4: {errors:?}");
}

/// Refinement that follows the solution ([`Options::adapt`]): with no cell divided for the
/// bore's curvature, each round divides the cells the last solution is least resolved in, and
/// the errors against Lamé's solution fall round by round, as does what a round still changes.
#[test]
fn rounds_that_follow_the_solution_lower_the_error() {
    let grid = |adapt: usize| Options { degree: 2, depth: 1, adapt, ..Options::new(0.1) };
    let mut errors = Vec::new();
    let mut settled = Vec::new();
    for adapt in 0..=3 {
        let solution = tube(&grid(adapt));
        let stats = &solution.stats;
        println!(
            "{adapt} rounds asked: {} done, the last changed the strain energy by {:.2e}, cells by level {:?}",
            stats.rounds,
            stats.settled,
            &stats.levels[..=adapt]
        );
        assert_eq!(stats.rounds, adapt);
        assert!(adapt == 0 || stats.levels[adapt] > 0, "{:?}", stats.levels);
        errors.push(report(&format!("degree 2, {adapt} rounds, {} unknowns", stats.unknowns), &solution));
        settled.push(stats.settled);
        // The cuts still carry the pressure on the bore.
        let force = PRESSURE * INNER_RADIUS * LENGTH;
        for (face, axis) in [(AT_X0, 0), (AT_Y0, 1)] {
            let reaction = solution.reactions.iter().find(|r| r.face == face).unwrap().force[axis];
            assert!((reaction + force).abs() < 2e-3 * force, "face {face}: {reaction} against {force}");
        }
    }
    // Displacement and stress errors of 2.4 % and 15 % on the grid's own cells, 0.24 % and 7.6 %
    // after two rounds, 0.06 % and 1.9 % after three; no round makes them worse.
    for (before, after) in errors.iter().zip(errors.iter().skip(1)) {
        assert!(after.0 < before.0 && after.1 < before.1, "{errors:?}");
    }
    assert!(errors[2].0 < 0.2 * errors[0].0 && errors[3].1 < 0.2 * errors[0].1, "{errors:?}");
    // What a round changes is small against the solution, and none without a round.
    assert!(settled[0] == 0.0 && settled[1..].iter().all(|change| change.abs() < 1e-2 && *change != 0.0), "{settled:?}");
}

/// The rounds end once one has moved the solution by no more than [`Options::settle`], in the
/// energy norm, and a round whose system would be too large is left out: the result is then
/// the solution of the rounds before it, bit for bit.
#[test]
fn the_rounds_end_once_the_solution_has_settled() {
    let grid = |adapt: usize, settle: f64| Options { degree: 2, depth: 1, adapt, settle, ..Options::new(0.1) };
    let same = |a: &Solution, b: &Solution| {
        a.strain_energy.to_bits() == b.strain_energy.to_bits() && a.stats.unknowns == b.stats.unknowns && a.stats.levels == b.stats.levels
    };
    // What each of three rounds moves the solution by.
    let all: Vec<Solution> = (0..=3).map(|adapt| tube(&grid(adapt, 0.0))).collect();
    let moved: Vec<f64> = all.iter().map(|solution| solution.stats.settled.abs().sqrt()).collect();
    println!("the rounds move the solution by {:?} in the energy norm", &moved[1..]);
    assert!(moved[0] == 0.0 && moved[1..].iter().all(|m| *m > 0.0 && *m < 0.1), "{moved:?}");
    // Asked to settle within what a round moved it by, the rounds end with that round.
    for last in 1..=3 {
        let Some(ends) = (1..=3).find(|round| moved[*round] <= moved[last]) else { continue };
        let solution = tube(&grid(3, moved[last]));
        assert_eq!(solution.stats.rounds, ends, "settle within {}: {moved:?}", moved[last]);
        assert!(same(&solution, &all[ends]) && !solution.stats.left_out);
    }
    // Within 5 %, this tube is settled after two rounds (they move it by 5.5 %, 4.3 % and 3.4 %,
    // which are changes of the strain energy of 0.3 % and less); within nothing, never.
    assert_eq!(tube(&grid(3, Options::SETTLED)).stats.rounds, 2);
    assert_eq!(tube(&grid(3, 0.5 * moved[1..].iter().copied().fold(f64::INFINITY, f64::min))).stats.rounds, 3);
    // A round that does not fit is left out.
    let fits_two = Limits { max_dofs: all[2].stats.unknowns, ..Limits::default() };
    let capped = tube_within(&grid(3, 0.0), &fits_two).unwrap();
    assert!(capped.stats.left_out && capped.stats.rounds == 2 && same(&capped, &all[2]), "{:?}", capped.stats);
    let fits_none = Limits { max_dofs: all[0].stats.unknowns, ..Limits::default() };
    let capped = tube_within(&grid(3, 0.0), &fits_none).unwrap();
    assert!(capped.stats.left_out && capped.stats.rounds == 0 && same(&capped, &all[0]), "{:?}", capped.stats);
    // Not a change to end at.
    for settle in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(tube_within(&grid(1, settle), &Limits::default()), Err(fieldcraft_fcm::Error::Invalid(_))), "{settle}");
    }
}

/// The same bits on one thread and on several, with rounds that follow the solution, and the
/// same whether a round takes the integrals of the cut cells it leaves as they are from the
/// round before or integrates them again ([`Options::reuse`]).
#[test]
fn the_rounds_do_not_depend_on_the_threads() {
    let run_with = |threads: usize, reuse: bool| {
        fieldcraft_parallel::with_threads(threads, || {
            let solution = tube(&Options { degree: 2, depth: 2, adapt: 2, reuse, offset: [0.37, 0.21, 0.43], ..Options::new(0.1) });
            assert_eq!(solution.stats.rounds, 2);
            let mut bits = vec![solution.strain_energy.to_bits(), solution.stats.unknowns as u64, solution.stats.settled.to_bits()];
            bits.extend(solution.stats.levels.iter().map(|cells| *cells as u64));
            for x in [[0.1, 0.0, 0.025], [0.0, 0.2, 0.05], [0.11, 0.1, 0.02]] {
                bits.extend(solution.displacement(x).unwrap().map(f64::to_bits));
                bits.extend(solution.stress(x).unwrap().map(f64::to_bits));
            }
            bits
        })
    };
    let run = |threads: usize| run_with(threads, true);
    let one = run(1);
    assert_eq!(one, run(2));
    assert_eq!(one, run(7));
    assert_eq!(one, run_with(3, false));
}
