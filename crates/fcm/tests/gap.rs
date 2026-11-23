//! Theory: a fork pulled by one of its two prongs. The prong that is pulled is a prismatic bar
//! in uniaxial tension, with the stress of the load on every section and the stretch σ L / E
//! over any length L of it; the other prong carries nothing (S. P. Timoshenko, J. N. Goodier,
//! *Theory of Elasticity*, 3rd ed., McGraw-Hill 1970: the uniform tension of a prismatic bar,
//! and Saint-Venant's principle for how far from the root and the tip that holds).
//!
//! The gap between the prongs is a tenth of their thickness and the cells are wider than a
//! prong and the gap together, so every cell along the prongs holds both. One polynomial a
//! cell would join them: the prongs then share the load nine to one along their whole length
//! (`joined_the_prongs_share_the_load`). With the material of a cell told apart into its
//! separate pieces ([`Options::separate`]) the pulled prong has the stress of the load to five
//! digits and the other none, from two cells off the root on. At the root the two prongs are
//! one piece of material and the cell there still joins them over its own length, which
//! dividing the cells where the gap ends shortens ([`Options::gaps`]).

mod common;

use common::{AT_X0, AT_Y0, AT_Z0, AT_Z1, INNER, ROOT, TIP_UPPER, fork, quarter_tube};
use fieldcraft_fcm::{Load, Material, Options, Part, Solution, Support, assembly_unknowns, solve};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210e9, poisson: 0.3, density: 7850.0 };
/// The root block's length, the prongs' length and thickness, the gap between them and the
/// fork's depth.
const ROOT_LENGTH: f64 = 0.02;
const LENGTH: f64 = 0.1;
const THICKNESS: f64 = 0.01;
const GAP: f64 = 0.001;
const DEPTH: f64 = 0.02;
const STRESS: f64 = 100e6;
/// Wider than a prong and the gap: every cell along the prongs holds both.
const CELL: f64 = 0.017;

fn pulled(options: &Options) -> Solution {
    let surface = fork(ROOT_LENGTH, LENGTH, THICKNESS, GAP, DEPTH);
    let supports = [Support { face: ROOT, fixed: [true; 3], value: [0.0; 3] }];
    let loads = [Load::Traction { face: TIP_UPPER, traction: [STRESS, 0.0, 0.0] }];
    solve(&surface, &STEEL, &supports, &loads, options, &Limits::default()).unwrap()
}

/// The middle of the upper (pulled) and of the lower prong at the fraction `along` of their
/// length from the root.
fn middles(along: f64) -> [[f64; 3]; 2] {
    let x = ROOT_LENGTH + along * LENGTH;
    [[x, 1.5 * THICKNESS + GAP, 0.5 * DEPTH], [x, 0.5 * THICKNESS, 0.5 * DEPTH]]
}

/// The stress along the prongs in their middles at `along`, as fractions of the load: the
/// pulled prong's and the other's.
fn shares(solution: &Solution, along: f64) -> [f64; 2] {
    middles(along).map(|x| solution.stress(x).unwrap()[0] / STRESS)
}

fn options(gaps: usize) -> Options {
    Options { gaps, offset: [0.37, 0.21, 0.43], ..Options::new(CELL) }
}

#[test]
fn a_fork_pulled_by_one_prong_leaves_the_other_free() {
    let solution = pulled(&options(0));
    let stats = &solution.stats;
    assert!(stats.split_cells > 0 && stats.separated > 0 && !stats.joined, "{stats:?}");
    // From two cells off the root on: the stress of the load in the one prong, none in the other.
    for along in [0.4, 0.5, 0.65, 0.8, 0.9] {
        let [pulled, free] = shares(&solution, along);
        println!("at {along} of the prongs: {pulled:.6} of the load in the pulled prong, {free:.6} in the other");
        assert!((pulled - 1.0).abs() < 1e-3 && free.abs() < 1e-3, "at {along}: {pulled} and {free}");
        // And no other stress in either.
        for x in middles(along) {
            let stress = solution.stress(x).unwrap();
            assert!(stress[1..].iter().all(|s| s.abs() < 1e-3 * STRESS), "at {along}: {stress:?}");
        }
    }
    // The stretch of the pulled prong between two of its sections, and none of the other.
    let stretch = |from: f64, to: f64, prong: usize| {
        solution.displacement(middles(to)[prong]).unwrap()[0] - solution.displacement(middles(from)[prong]).unwrap()[0]
    };
    let exact = STRESS * 0.4 * LENGTH / STEEL.young;
    println!("stretch from 0.5 to 0.9 of the prongs: {:.6e} (exact {exact:.6e}), the other {:.3e}", stretch(0.5, 0.9, 0), stretch(0.5, 0.9, 1));
    assert!((stretch(0.5, 0.9, 0) / exact - 1.0).abs() < 1e-4);
    assert!(stretch(0.5, 0.9, 1).abs() < 1e-4 * exact);
    // The root carries the load.
    assert!((solution.reactions[0].force[0] + STRESS * THICKNESS * DEPTH).abs() < 1e-6 * STRESS * THICKNESS * DEPTH, "{:?}", solution.reactions);
}

/// What telling the pieces apart is for: one polynomial a cell holds the prongs together along
/// their whole length, and the prong that is not pulled takes a tenth of the load.
#[test]
fn joined_the_prongs_share_the_load() {
    let solution = pulled(&Options { separate: false, ..options(0) });
    assert_eq!((solution.stats.split_cells, solution.stats.separated), (0, 0));
    for along in [0.5, 0.8] {
        let [pulled, free] = shares(&solution, along);
        println!("joined, at {along} of the prongs: {pulled:.4} of the load in the pulled prong, {free:.4} in the other");
        assert!(pulled < 0.92 && free > 0.08, "at {along}: {pulled} and {free}");
    }
}

/// Where the gap ends, the cell joins the two prongs over its own length. Dividing the cells
/// there ([`Options::gaps`]) shortens that: a tenth of the prongs' length from the root, the
/// prong that is not pulled carries 4.7 % of the load on the grid's own cells and 0.3 % on
/// cells divided twice.
#[test]
fn cells_are_divided_where_the_gap_ends() {
    let (coarse, divided) = (pulled(&options(0)), pulled(&options(2)));
    assert_eq!(coarse.stats.levels[1..], [0; 6]);
    assert!(divided.stats.levels[1] > 0 && divided.stats.levels[2] > 0, "{:?}", divided.stats.levels);
    let ([_, before], [pulled, after]) = (shares(&coarse, 0.1), shares(&divided, 0.1));
    println!("a tenth along the prongs: {before:.4} of the load in the free prong, {after:.4} with the cells divided at the root");
    assert!(before.abs() > 0.03 && after.abs() < 0.01 && (pulled - 1.0).abs() < 0.01, "{before}, {after}, {pulled}");
    // Far from the root nothing changes.
    let [pulled, free] = shares(&divided, 0.8);
    assert!((pulled - 1.0).abs() < 1e-3 && free.abs() < 1e-3, "{pulled} and {free}");
}

/// The same bits on one thread and on several, with islands and cells divided where the gap ends.
/// A count of the unknowns is of the cells as they are before a cut cell is integrated: every
/// cell one piece of material, none divided where the gap ends, whatever the solve is asked
/// to do there. The solve has more: the prongs' unknowns apart, and the cells at the root.
#[test]
fn the_unknowns_are_counted_before_the_pieces_in_a_cell_are_told_apart() {
    let surface = fork(ROOT_LENGTH, LENGTH, THICKNESS, GAP, DEPTH);
    let supports = [Support { face: ROOT, fixed: [true; 3], value: [0.0; 3] }];
    let loads = [Load::Traction { face: TIP_UPPER, traction: [STRESS, 0.0, 0.0] }];
    let count = |options: Options| {
        let part = Part { surface: &surface, material: STEEL, supports: &supports, loads: &loads, options, curved: &[] };
        assembly_unknowns(&[part], &[], &Limits::default()).unwrap()
    };
    let joined = pulled(&Options { separate: false, ..options(0) }).stats.unknowns;
    assert_eq!([count(options(0)), count(options(3)), count(Options { separate: false, ..options(0) })], [joined; 3]);
    let (apart, divided) = (pulled(&options(0)).stats.unknowns, pulled(&options(3)).stats.unknowns);
    assert!(joined < apart && apart < divided, "{joined}, {apart} and {divided} unknowns");
}

#[test]
fn the_result_with_islands_does_not_depend_on_the_threads() {
    let run = |threads: usize| {
        fieldcraft_parallel::with_threads(threads, || {
            let solution = pulled(&options(1));
            let mut bits = vec![solution.strain_energy.to_bits(), solution.stats.unknowns as u64, solution.stats.separated as u64];
            for along in [0.05, 0.3, 0.9] {
                for x in middles(along) {
                    bits.extend(solution.displacement(x).unwrap().map(f64::to_bits));
                    bits.extend(solution.stress(x).unwrap().map(f64::to_bits));
                }
            }
            bits
        })
    };
    let one = run(1);
    assert_eq!(one, run(2));
    assert_eq!(one, run(7));
}

/// Rounds that follow the solution ([`Options::adapt`]) on a part with a gap: the prong that is
/// not pulled stays free, and a round that takes the integrals of the cut cells it leaves as
/// they are from the round before gives the bits of one that integrates them again.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn rounds_keep_the_pieces_apart_and_reuse_their_cells() {
    let run = |reuse: bool| {
        let solution = pulled(&Options { adapt: 2, reuse, ..options(1) });
        assert!(solution.stats.rounds >= 1 && solution.stats.separated > 0 && !solution.stats.joined, "{:?}", solution.stats);
        let [pulled, free] = shares(&solution, 0.8);
        assert!((pulled - 1.0).abs() < 1e-3 && free.abs() < 1e-3, "{pulled} and {free}");
        let mut bits = vec![solution.strain_energy.to_bits(), solution.stats.unknowns as u64, solution.stats.rounds as u64];
        bits.extend(solution.stats.levels.iter().map(|cells| *cells as u64));
        for along in [0.05, 0.3, 0.9] {
            for x in middles(along) {
                bits.extend(solution.displacement(x).unwrap().map(f64::to_bits));
                bits.extend(solution.stress(x).unwrap().map(f64::to_bits));
            }
        }
        bits
    };
    assert_eq!(run(true), run(false));
}

/// Where no cell holds a gap, telling pieces apart changes nothing: the same unknowns and the
/// same bits, on cells of several levels too.
#[test]
fn a_part_without_gaps_is_solved_as_it_was() {
    let tube = quarter_tube(0.1, 0.2, 0.05, 32);
    let supports = [
        Support { face: AT_X0, fixed: [true, false, false], value: [0.0; 3] },
        Support { face: AT_Y0, fixed: [false, true, false], value: [0.0; 3] },
        Support { face: AT_Z0, fixed: [false, false, true], value: [0.0; 3] },
        Support { face: AT_Z1, fixed: [false, false, true], value: [0.0; 3] },
    ];
    let loads = [Load::Pressure { face: INNER, pressure: STRESS }];
    let run = |separate: bool, gaps: usize| {
        let options = Options { degree: 2, refine: 2, separate, gaps, offset: [0.37, 0.21, 0.43], ..Options::new(0.1) };
        let solution = solve(&tube, &STEEL, &supports, &loads, &options, &Limits::default()).unwrap();
        assert_eq!((solution.stats.split_cells, solution.stats.separated, solution.stats.joined), (0, 0, false));
        let mut bits = vec![solution.strain_energy.to_bits(), solution.stats.unknowns as u64, solution.stats.factor_entries as u64];
        for x in [[0.1, 0.0, 0.0], [0.0, 0.2, 0.05], [0.11, 0.1, 0.02], [0.14, 0.03, 0.04]] {
            bits.extend(solution.displacement(x).unwrap().map(f64::to_bits));
            bits.extend(solution.stress(x).unwrap().map(f64::to_bits));
        }
        bits
    };
    let plain = run(false, 0);
    assert_eq!(plain, run(true, 0));
    assert_eq!(plain, run(true, 3));
}
