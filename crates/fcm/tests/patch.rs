//! Theory: a block in uniaxial tension has a constant stress and a linear displacement, which
//! every degree represents exactly, and the quadrature of cut cells is exact for flat faces along
//! the axes. What the grid solution misses is then the fictitious stiffness alone
//! ([`Options::FICTITIOUS`], 1e-8 of the material's): the strain energy is off by its order, and
//! the stress at a point of the cell that holds least material by about its square root.
//!
//! A block from the origin with sliding supports on its three lower faces and a traction σ₀ on
//! its upper x face: σ_xx = σ₀, every other stress zero, u = (σ₀ x, −ν σ₀ y, −ν σ₀ z) / E
//! (S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed., McGraw-Hill 1970, the
//! uniform tension of a prismatic bar).

mod common;

use common::{AT_X0, AT_Y0, AT_Z0, AT_Z1, INNER, X_HI, X_LO, Y_LO, Z_LO, block, quarter_tube};
use fieldcraft_fcm::{Basis, Error, Load, Material, Options, Solution, Support, solve};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210e9, poisson: 0.3, density: 7850.0 };
const SIZE: [f64; 3] = [0.75, 0.5, 0.25];
const STRESS: f64 = 100e6;

fn sliding() -> [Support; 3] {
    [
        Support { face: X_LO, fixed: [true, false, false], value: [0.0; 3] },
        Support { face: Y_LO, fixed: [false, true, false], value: [0.0; 3] },
        Support { face: Z_LO, fixed: [false, false, true], value: [0.0; 3] },
    ]
}

fn tension(options: &Options) -> Result<Solution, Error> {
    let loads = [Load::Traction { face: X_HI, traction: [STRESS, 0.0, 0.0] }];
    solve(&block([0.0; 3], SIZE), &STEEL, &sliding(), &loads, options, &Limits::default())
}

/// The largest errors of the displacement (relative to the block's elongation) and of the
/// stress (relative to σ₀) over points in the block and on its surface.
fn errors(solution: &Solution) -> (f64, f64) {
    let elongation = STRESS * SIZE[0] / STEEL.young;
    let (mut displacement, mut stress) = (0.0f64, 0.0f64);
    for fx in [0.0, 0.07, 0.31, 0.5, 0.88, 1.0] {
        for fy in [0.0, 0.23, 0.6, 1.0] {
            for fz in [0.0, 0.41, 1.0] {
                let x = [fx * SIZE[0], fy * SIZE[1], fz * SIZE[2]];
                let exact = [STRESS * x[0] / STEEL.young, -STEEL.poisson * STRESS * x[1] / STEEL.young, -STEEL.poisson * STRESS * x[2] / STEEL.young];
                let u = solution.displacement(x).unwrap();
                let s = solution.stress(x).unwrap();
                for i in 0..3 {
                    displacement = displacement.max((u[i] - exact[i]).abs() / elongation);
                }
                stress = stress.max((s[0] - STRESS).abs() / STRESS);
                for component in &s[1..] {
                    stress = stress.max(component.abs() / STRESS);
                }
            }
        }
    }
    (displacement, stress)
}

fn check(name: &str, solution: &Solution, tolerance: f64) {
    let volume = SIZE[0] * SIZE[1] * SIZE[2];
    let (displacement, stress) = errors(solution);
    let stats = &solution.stats;
    println!(
        "{name}: cells {:?} ({} inside, {} cut), {} unknowns, {} points, volume error {:.2e}, smallest fraction {:.2e}, displacement error {displacement:.2e}, stress error {stress:.2e}",
        stats.cells,
        stats.inside_cells,
        stats.cut_cells,
        stats.unknowns,
        stats.quadrature_points,
        (stats.volume - volume).abs() / volume,
        stats.smallest_fraction
    );
    assert!((stats.surface_volume - volume).abs() < 1e-12 * volume);
    assert!(displacement < tolerance, "{name}: displacement error {displacement:.3e}");
    assert!(stress < tolerance, "{name}: stress error {stress:.3e}");
    // The supports carry the load: the lower x face takes all of it, the others none.
    let force = STRESS * SIZE[1] * SIZE[2];
    let reaction = |face: u32| solution.reactions.iter().find(|r| r.face == face).unwrap().force;
    assert!((reaction(X_LO)[0] + force).abs() < 1e-6 * force, "{name}: reaction {:?}", reaction(X_LO));
    assert!(reaction(Y_LO)[1].abs() < tolerance * force && reaction(Z_LO)[2].abs() < tolerance * force);
    let energy = 0.5 * STRESS * STRESS / STEEL.young * volume;
    assert!((solution.strain_energy - energy).abs() < 2.0 * tolerance * energy, "{name}: energy {} vs {energy}", solution.strain_energy);
}

/// Faces a quarter, a half and three quarters of a cell into the cells they cut.
#[test]
fn constant_stress_with_faces_at_fractions_of_a_cell() {
    for degree in [1, 2] {
        let options = Options { degree, depth: 2, offset: [0.25, 0.5, 0.75], ..Options::new(0.25) };
        let solution = tension(&options).unwrap();
        assert!((solution.stats.volume - SIZE[0] * SIZE[1] * SIZE[2]).abs() < 1e-13);
        check(&format!("fractions, degree {degree}"), &solution, 5e-3);
    }
}

/// Every face of the block on a grid plane: each face belongs to the cell that holds its
/// material, and no cell is cut through.
#[test]
fn constant_stress_with_faces_on_grid_planes() {
    let options = Options { degree: 2, depth: 1, ..Options::new(0.25) };
    let solution = tension(&options).unwrap();
    assert_eq!(solution.stats.cells, [3, 2, 1]);
    assert!((solution.stats.volume - SIZE[0] * SIZE[1] * SIZE[2]).abs() < 1e-13);
    // No cell holds anything but material: nothing is fictitious, and the answer is exact.
    assert!((solution.stats.smallest_fraction - 1.0).abs() < 1e-12);
    check("on grid planes", &solution, 1e-10);
}

/// Faces anywhere in their cells, at any octree depth: the cell that holds least material holds
/// 1.5 % of its volume, and the stress at its corner of the block is the least accurate value.
#[test]
fn constant_stress_with_faces_anywhere() {
    for (degree, depth) in [(2, 0), (2, 3), (3, 1)] {
        let options = Options { degree, depth, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) };
        let solution = tension(&options).unwrap();
        assert!((solution.stats.volume - SIZE[0] * SIZE[1] * SIZE[2]).abs() < 1e-13);
        check(&format!("anywhere, degree {degree}, depth {depth}"), &solution, 5e-3);
    }
}

/// The trunk space holds the linear displacement too: the same patch with far fewer unknowns
/// than the tensor product of the same degree.
#[test]
fn constant_stress_in_the_trunk_space() {
    for degree in [2, 4] {
        let options = Options { degree, basis: Basis::Trunk, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) };
        let solution = tension(&options).unwrap();
        assert!((solution.stats.volume - SIZE[0] * SIZE[1] * SIZE[2]).abs() < 1e-13);
        let full = tension(&Options { basis: Basis::Tensor, ..options }).unwrap().stats.unknowns;
        assert!(solution.stats.unknowns < full, "degree {degree}: {} trunk unknowns, {full} of the tensor product", solution.stats.unknowns);
        check(&format!("trunk, degree {degree}"), &solution, 5e-3);
    }
    // On grid planes the answer is exact, as with the tensor product.
    let options = Options { degree: 3, basis: Basis::Trunk, ..Options::new(0.25) };
    check("trunk on grid planes", &tension(&options).unwrap(), 1e-10);
}

/// A nearly incompressible block cut by the grid: its constant stress is found at the default
/// fictitious stiffness, to a few tenths of a per cent where a cell holds 1.5 % material (the
/// pressure, bulk modulus times a change of volume, shows the fictitious stiffness first), and
/// a larger fictitious stiffness costs accuracy there.
#[test]
fn a_nearly_incompressible_block_keeps_its_constant_stress() {
    let rubber = Material { young: 2e6, poisson: 0.4999, density: 1000.0 };
    let loads = [Load::Traction { face: X_HI, traction: [STRESS, 0.0, 0.0] }];
    let errors = |degree: usize, fictitious: f64| -> (f64, f64) {
        let options = Options { degree, fictitious, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) };
        let solution = solve(&block([0.0; 3], SIZE), &rubber, &sliding(), &loads, &options, &Limits::default()).unwrap();
        assert_eq!(solution.stats.fictitious, fictitious, "solved at the fraction asked for");
        let (mut displacement, mut stress) = (0.0f64, 0.0f64);
        for fractions in [[0.0, 0.0, 0.0], [0.31, 0.6, 0.41], [1.0, 1.0, 1.0], [0.88, 0.23, 0.0]] {
            let x = [fractions[0] * SIZE[0], fractions[1] * SIZE[1], fractions[2] * SIZE[2]];
            let exact =
                [STRESS * x[0] / rubber.young, -rubber.poisson * STRESS * x[1] / rubber.young, -rubber.poisson * STRESS * x[2] / rubber.young];
            let (u, s) = (solution.displacement(x).unwrap(), solution.stress(x).unwrap());
            for i in 0..3 {
                displacement = displacement.max((u[i] - exact[i]).abs() / (STRESS * SIZE[0] / rubber.young));
            }
            stress = stress.max((s[0] - STRESS).abs() / STRESS);
            for component in &s[1..] {
                stress = stress.max(component.abs() / STRESS);
            }
        }
        println!(
            "nearly incompressible, degree {degree}, fictitious {fictitious:.1e}: displacement error {displacement:.2e}, stress error {stress:.2e}"
        );
        (displacement, stress)
    };
    for degree in [2, 4] {
        let (displacement, stress) = errors(degree, Options::FICTITIOUS);
        assert!(displacement < 1e-4 && stress < 5e-3, "degree {degree}: displacement error {displacement:.3e}, stress error {stress:.3e}");
    }
    let (_, raised) = errors(4, 2500.0 * Options::FICTITIOUS);
    assert!(raised > 0.02 && raised < 0.2, "the fictitious stiffness that keeps the pivot margin at ν = 0.4999: stress error {raised:.3e}");
}

/// The same bits on one thread and on several: cells are integrated in parallel and assembled in
/// order. With refinement too: a quarter tube under pressure, whose curved faces divide the cells
/// they cut twice, with supports on divided cut cells.
#[test]
fn the_result_does_not_depend_on_the_threads() {
    let bits = |solution: &Solution, points: &[[f64; 3]]| {
        let mut bits = vec![solution.strain_energy.to_bits(), solution.stats.volume.to_bits(), solution.stats.unknowns as u64];
        for reaction in &solution.reactions {
            bits.extend(reaction.force.map(f64::to_bits));
        }
        for x in points {
            bits.extend(solution.displacement(*x).unwrap().map(f64::to_bits));
            bits.extend(solution.stress(*x).unwrap().map(f64::to_bits));
        }
        bits
    };
    let run = |threads: usize| {
        fieldcraft_parallel::with_threads(threads, || {
            let options = Options { degree: 3, basis: Basis::Trunk, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) };
            bits(&tension(&options).unwrap(), &[[0.0; 3], [0.3, 0.2, 0.1], SIZE])
        })
    };
    let one = run(1);
    assert_eq!(one, run(2));
    assert_eq!(one, run(7));

    let tube = quarter_tube(0.1, 0.2, 0.05, 32);
    let supports = [
        Support { face: AT_X0, fixed: [true, false, false], value: [0.0; 3] },
        Support { face: AT_Y0, fixed: [false, true, false], value: [0.0; 3] },
        Support { face: AT_Z0, fixed: [false, false, true], value: [0.0; 3] },
        Support { face: AT_Z1, fixed: [false, false, true], value: [0.0; 3] },
    ];
    let loads = [Load::Pressure { face: INNER, pressure: STRESS }];
    let refined = |threads: usize| {
        fieldcraft_parallel::with_threads(threads, || {
            let options = Options { degree: 2, refine: 2, offset: [0.37, 0.21, 0.43], ..Options::new(0.1) };
            let solution = solve(&tube, &STEEL, &supports, &loads, &options, &Limits::default()).unwrap();
            assert!(solution.stats.levels[1] > 0 && solution.stats.levels[2] > 0, "cells of every level: {:?}", solution.stats.levels);
            bits(&solution, &[[0.1, 0.0, 0.0], [0.0, 0.2, 0.05], [0.11, 0.1, 0.02], [0.14, 0.03, 0.04]])
        })
    };
    let one = refined(1);
    assert_eq!(one, refined(2));
    assert_eq!(one, refined(7));
}

#[test]
fn unusable_input_is_refused() {
    let part = block([0.0; 3], SIZE);
    let loads = [Load::Traction { face: X_HI, traction: [STRESS, 0.0, 0.0] }];
    let run = |material: &Material, supports: &[Support], loads: &[Load], options: &Options| {
        solve(&part, material, supports, loads, options, &Limits::default())
    };
    let options = Options { degree: 1, depth: 1, ..Options::new(0.25) };
    assert!(matches!(run(&STEEL, &[], &loads, &options), Err(Error::Invalid(_))), "no support");
    assert!(
        matches!(run(&STEEL, &[Support { face: 9, fixed: [true; 3], value: [0.0; 3] }], &loads, &options), Err(Error::Invalid(_))),
        "no such face"
    );
    assert!(matches!(run(&STEEL, &[Support { face: X_LO, fixed: [false; 3], value: [0.0; 3] }], &loads, &options), Err(Error::Invalid(_))));
    let twice = [Support { face: X_LO, fixed: [true; 3], value: [0.0; 3] }, Support { face: X_LO, fixed: [true, false, false], value: [0.0; 3] }];
    assert!(matches!(run(&STEEL, &twice, &loads, &options), Err(Error::Invalid(_))));
    let fixed = [Support { face: X_LO, fixed: [true; 3], value: [0.0; 3] }];
    assert!(matches!(run(&STEEL, &fixed, &[Load::Pressure { face: X_HI, pressure: f64::NAN }], &options), Err(Error::Invalid(_))));
    assert!(matches!(run(&Material { poisson: 0.5, ..STEEL }, &fixed, &loads, &options), Err(Error::Invalid(_))));
    for bad in [
        Options { degree: 0, ..options },
        Options { degree: 99, ..options },
        Options { depth: 99, ..options },
        Options { refine: 99, ..options },
        Options { fictitious: 0.0, ..options },
    ] {
        assert!(matches!(run(&STEEL, &fixed, &loads, &bad), Err(Error::Invalid(_))), "{bad:?}");
    }
    for size in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(run(&STEEL, &fixed, &loads, &Options { cell_size: size, ..options }), Err(Error::Invalid(_))), "cell size {size}");
    }
    assert!(matches!(run(&STEEL, &fixed, &loads, &Options { cell_size: 1e-4, ..options }), Err(Error::Limit(_))), "too many cells");
    let small = Limits { max_dofs: 10, ..Limits::default() };
    assert!(matches!(solve(&part, &STEEL, &fixed, &loads, &options, &small), Err(Error::Limit(_))));
}

/// Supports that leave a rigid motion free give an error, not a panic or a wrong answer.
#[test]
fn a_part_that_is_not_held_is_singular() {
    let options = Options { degree: 1, depth: 1, ..Options::new(0.25) };
    let loads = [Load::Traction { face: X_HI, traction: [STRESS, 0.0, 0.0] }];
    let only_x = [Support { face: X_LO, fixed: [true, false, false], value: [0.0; 3] }];
    let result = solve(&block([0.0; 3], SIZE), &STEEL, &only_x, &loads, &options, &Limits::default());
    assert!(matches!(result, Err(Error::Singular(_))), "{:?}", result.map(|s| s.stats));
}

/// Hostile cuts: a cell that holds a sliver of material a millionth of its size thick, with a
/// support on it, and a part far smaller than one cell. Both solve the tension. The stress in
/// the sliver's cell is read with the cells beside it (`Solution::stress`), which hold
/// nothing fictitious, so it is the stress to rounding; from the sliver's own polynomial it
/// was off by a per cent.
#[test]
fn slivers_and_parts_smaller_than_a_cell() {
    let sliver = Options { degree: 2, depth: 3, offset: [1.0 - 1e-6, 0.0, 0.0], ..Options::new(0.25) };
    let solution = tension(&sliver).unwrap();
    assert!((solution.stats.smallest_fraction - 1e-6).abs() < 1e-9, "{}", solution.stats.smallest_fraction);
    check("sliver", &solution, 1e-6);
    let coarse = Options { degree: 2, depth: 5, offset: [0.4, 0.3, 0.2], ..Options::new(4.0) };
    let solution = tension(&coarse).unwrap();
    assert_eq!(solution.stats.cells, [1, 1, 1]);
    check("smaller than a cell", &solution, 5e-3);
}
