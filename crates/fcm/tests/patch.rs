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

use common::{X_HI, X_LO, Y_LO, Z_LO, block};
use fieldcraft_fcm::{Error, Load, Material, Options, Solution, Support, solve};
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
/// support on it, and a part far smaller than one cell. Both solve the tension; the stress in
/// the sliver's cell is the least accurate.
#[test]
fn slivers_and_parts_smaller_than_a_cell() {
    let sliver = Options { degree: 2, depth: 3, offset: [1.0 - 1e-6, 0.0, 0.0], ..Options::new(0.25) };
    let solution = tension(&sliver).unwrap();
    assert!((solution.stats.smallest_fraction - 1e-6).abs() < 1e-9, "{}", solution.stats.smallest_fraction);
    check("sliver", &solution, 3e-2);
    let coarse = Options { degree: 2, depth: 5, offset: [0.4, 0.3, 0.2], ..Options::new(4.0) };
    let solution = tension(&coarse).unwrap();
    assert_eq!(solution.stats.cells, [1, 1, 1]);
    check("smaller than a cell", &solution, 5e-3);
}
