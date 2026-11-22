//! Several bodies bonded where they touch ([`fieldcraft_fcm::solve_assembly`]), against closed
//! forms: a bar of two blocks in tension, which is one block's constant stress whatever the two
//! grids; a bar of two materials in series; and a block standing on a part of a plate, whose
//! support must carry what the block is loaded with.
//!
//! The fields of the first two are polynomials of the cells' degree, so they are found to the
//! fictitious stiffness of the cut cells, as in `tests/patch.rs`: the bond is consistent, and
//! the cells of the two bodies need not match.

use fieldcraft_fcm::{
    Bond, Curved, Error, Grip, Load, Material, Options, Part, Solution, Support, assembly_unknowns, find_bonds, free_motions, solve_assembly,
};
use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210e9, poisson: 0.3, density: 7850.0 };

/// The box from `lo` to `hi`, the face at the lower end of axis a tagged 2a and the one at its
/// upper end 2a + 1.
fn block(lo: [f64; 3], hi: [f64; 3]) -> BoundarySurface {
    let at = |i: usize| [if i & 1 == 0 { lo[0] } else { hi[0] }, if i & 2 == 0 { lo[1] } else { hi[1] }, if i & 4 == 0 { lo[2] } else { hi[2] }];
    let points: Vec<[f64; 3]> = (0..8).map(at).collect();
    // Counter-clockwise seen from outside.
    let quads: [([u32; 4], u32); 6] =
        [([0, 4, 6, 2], 0), ([1, 3, 7, 5], 1), ([0, 1, 5, 4], 2), ([2, 6, 7, 3], 3), ([0, 2, 3, 1], 4), ([4, 5, 7, 6], 5)];
    let (mut triangles, mut tags) = (Vec::new(), Vec::new());
    for (q, tag) in quads {
        triangles.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
        tags.extend([tag, tag]);
    }
    BoundarySurface::new(&points, &triangles, &tags, &SurfaceOptions::default()).unwrap()
}

/// The largest error of the displacement (over `scale`) and of the stress (over `stress`)
/// at points of the box `lo`–`hi` of a solution, against a field with the stress σ_xx alone.
fn errors(solution: &Solution, lo: [f64; 3], hi: [f64; 3], exact: impl Fn([f64; 3]) -> [f64; 3], scale: f64, stress: f64) -> (f64, f64) {
    let (mut displacement, mut worst) = (0.0f64, 0.0f64);
    for fx in [0.02, 0.31, 0.5, 0.88, 0.98] {
        for fy in [0.05, 0.6, 0.95] {
            for fz in [0.07, 0.41, 0.93] {
                let x = [lo[0] + fx * (hi[0] - lo[0]), lo[1] + fy * (hi[1] - lo[1]), lo[2] + fz * (hi[2] - lo[2])];
                let (u, s) = (solution.displacement(x).unwrap(), solution.stress(x).unwrap());
                let expected = exact(x);
                for i in 0..3 {
                    displacement = displacement.max((u[i] - expected[i]).abs() / scale);
                }
                worst = worst.max((s[0] - stress).abs() / stress);
                for component in &s[1..] {
                    worst = worst.max(component.abs() / stress);
                }
            }
        }
    }
    (displacement, worst)
}

/// The supports of a bar along x that leave it free to contract: its faces across y and z at
/// their lower ends slide, and `first` holds its end.
fn sliding(first: bool) -> Vec<Support> {
    let mut supports =
        vec![Support { face: 2, fixed: [false, true, false], value: [0.0; 3] }, Support { face: 4, fixed: [false, false, true], value: [0.0; 3] }];
    if first {
        supports.push(Support { face: 0, fixed: [true, false, false], value: [0.0; 3] });
    }
    supports
}

/// Two blocks of one material, bonded end to end and pulled: the stress is σ everywhere and the
/// displacement that of one bar, on grids that share nothing (different cell sizes, different
/// places), at degrees 2 and 3, in the tensor product and in the trunk space.
#[test]
fn two_blocks_bonded_end_to_end_are_one_bar() {
    const STRESS: f64 = 100e6;
    let (first, second) = (block([0.0; 3], [1.0, 0.8, 0.6]), block([1.0, 0.0, 0.0], [2.3, 0.8, 0.6]));
    let (held, free) = (sliding(true), sliding(false));
    let pull = [Load::Traction { face: 1, traction: [STRESS, 0.0, 0.0] }];
    let exact =
        |x: [f64; 3]| [STRESS * x[0] / STEEL.young, -STEEL.poisson * STRESS * x[1] / STEEL.young, -STEEL.poisson * STRESS * x[2] / STEEL.young];
    let elongation = STRESS * 2.3 / STEEL.young;
    for (degree, basis) in [(2, fieldcraft_fcm::Basis::Tensor), (3, fieldcraft_fcm::Basis::Tensor), (3, fieldcraft_fcm::Basis::Trunk)] {
        let parts = [
            Part {
                surface: &first,
                material: STEEL,
                supports: &held,
                loads: &[],
                options: Options { degree, basis, offset: [0.31, 0.62, 0.13], ..Options::new(0.4) },
                curved: &[],
            },
            Part {
                surface: &second,
                material: STEEL,
                supports: &free,
                loads: &pull,
                options: Options { degree, basis, offset: [0.57, 0.11, 0.73], ..Options::new(0.55) },
                curved: &[],
            },
        ];
        for bond in [
            Bond { first: (0, vec![1]), second: (1, vec![0]), tolerance: 1e-6, grip: Grip::Bonded },
            // The other way round, with a face that does not touch named as well.
            Bond { first: (1, vec![0, 3]), second: (0, vec![1]), tolerance: 1e-6, grip: Grip::Bonded },
        ] {
            let solutions = solve_assembly(&parts, &[bond], &Limits::default()).unwrap();
            assert_eq!(solutions.len(), 2);
            let (a, b) = (
                errors(&solutions[0], [0.0; 3], [1.0, 0.8, 0.6], exact, elongation, STRESS),
                errors(&solutions[1], [1.0, 0.0, 0.0], [2.3, 0.8, 0.6], exact, elongation, STRESS),
            );
            println!(
                "degree {degree}, {basis:?}: first body {a:?}, second body {b:?}; {} and {} unknowns",
                solutions[0].stats.unknowns, solutions[1].stats.unknowns
            );
            for (displacement, stress) in [a, b] {
                assert!(displacement < 5e-6 && stress < 2e-4, "degree {degree}, {basis:?}: displacement {displacement:.3e}, stress {stress:.3e}");
            }
            // The held end carries the pull: σ A.
            let reaction = solutions[0].reactions.iter().find(|r| r.face == 0).unwrap().force[0];
            assert!((reaction + STRESS * 0.8 * 0.6).abs() < 1e-6 * STRESS, "{reaction}");
            // The tied face is no support: it reports no reaction.
            assert!(solutions.iter().all(|s| s.reactions.iter().all(|r| !(r.face == 1 && std::ptr::eq(s, &solutions[0])) || r.force == [0.0; 3])));
        }
    }
}

/// Two materials in series, without lateral contraction: the stress is σ in both and each
/// stretches by its own modulus, u = σ x / E₁ up to the bond and σ (L₁ / E₁ + (x − L₁) / E₂)
/// beyond it. The strain jumps at the bond, which one body's polynomials could not follow.
#[test]
fn two_materials_in_series_stretch_each_by_its_own_modulus() {
    const STRESS: f64 = 50e6;
    let (stiff, soft) = (Material { young: 210e9, poisson: 0.0, density: 0.0 }, Material { young: 70e9, poisson: 0.0, density: 0.0 });
    let (first, second) = (block([0.0; 3], [0.7, 0.5, 0.5]), block([0.7, 0.0, 0.0], [1.6, 0.5, 0.5]));
    let (held, free) = (sliding(true), sliding(false));
    let pull = [Load::Traction { face: 1, traction: [STRESS, 0.0, 0.0] }];
    let parts = [
        Part {
            surface: &first,
            material: stiff,
            supports: &held,
            loads: &[],
            options: Options { degree: 2, offset: [0.21, 0.4, 0.33], ..Options::new(0.3) },
            curved: &[],
        },
        Part {
            surface: &second,
            material: soft,
            supports: &free,
            loads: &pull,
            options: Options { degree: 2, offset: [0.6, 0.15, 0.5], ..Options::new(0.25) },
            curved: &[],
        },
    ];
    let solutions =
        solve_assembly(&parts, &[Bond { first: (0, vec![1]), second: (1, vec![0]), tolerance: 1e-6, grip: Grip::Bonded }], &Limits::default())
            .unwrap();
    let elongation = STRESS * (0.7 / stiff.young + 0.9 / soft.young);
    let (a, b) = (
        errors(&solutions[0], [0.0; 3], [0.7, 0.5, 0.5], |x| [STRESS * x[0] / stiff.young, 0.0, 0.0], elongation, STRESS),
        errors(
            &solutions[1],
            [0.7, 0.0, 0.0],
            [1.6, 0.5, 0.5],
            |x| [STRESS * (0.7 / stiff.young + (x[0] - 0.7) / soft.young), 0.0, 0.0],
            elongation,
            STRESS,
        ),
    );
    println!("steel {a:?}, aluminium {b:?}");
    for (displacement, stress) in [a, b] {
        assert!(displacement < 5e-6 && stress < 2e-4, "displacement {displacement:.3e}, stress {stress:.3e}");
    }
}

/// A block standing on a part of a plate: the bond covers the block's foot and a part of the
/// plate's top only. Only the plate is supported; its clamped underside carries exactly what
/// the block is loaded with, the block moves with the plate where they touch, and the plate's
/// top is free beside the block.
#[test]
fn a_block_on_a_plate_is_held_through_its_foot() {
    let (plate, post) = (block([0.0; 3], [2.0, 1.5, 0.3]), block([0.6, 0.5, 0.3], [1.1, 0.9, 1.0]));
    let clamp = [Support { face: 4, fixed: [true; 3], value: [0.0; 3] }];
    let push = [Load::Force { face: 1, force: [3.0e5, 0.0, 0.0] }, Load::Force { face: 5, force: [0.0, 1.0e5, -2.0e5] }];
    let parts = [
        Part {
            surface: &plate,
            material: STEEL,
            supports: &clamp,
            loads: &[],
            options: Options { degree: 2, offset: [0.37, 0.21, 0.43], ..Options::new(0.25) },
            curved: &[],
        },
        Part {
            surface: &post,
            material: STEEL,
            supports: &[],
            loads: &push,
            options: Options { degree: 2, offset: [0.17, 0.41, 0.29], ..Options::new(0.2) },
            curved: &[],
        },
    ];
    let bond = Bond { first: (0, vec![5]), second: (1, vec![4]), tolerance: 1e-6, grip: Grip::Bonded };
    let solutions = solve_assembly(&parts, std::slice::from_ref(&bond), &Limits::default()).unwrap();
    let reaction = solutions[0].reactions.iter().find(|r| r.face == 4).unwrap().force;
    let applied = [3.0e5, 1.0e5, -2.0e5];
    for i in 0..3 {
        assert!((reaction[i] + applied[i]).abs() < 1e-6 * 3.0e5, "component {i}: reaction {} against {}", reaction[i], applied[i]);
    }
    // Where they touch the two move together, to what the cells resolve.
    let largest = solutions[1].displacement([1.1, 0.7, 1.0]).unwrap().iter().fold(0.0f64, |m, u| m.max(u.abs()));
    for at in [[0.7, 0.6, 0.3], [0.85, 0.7, 0.3], [1.0, 0.8, 0.3]] {
        let (below, above) = (solutions[0].displacement(at).unwrap(), solutions[1].displacement(at).unwrap());
        let apart = (0..3).map(|i| (below[i] - above[i]).abs()).fold(0.0, f64::max);
        println!("at {at:?}: plate {below:?}, post {above:?}");
        assert!(apart < 0.05 * largest, "at {at:?}: {apart:.3e} apart of {largest:.3e}");
    }
    // The unknowns are counted without the solve, and a limit refuses the count as it does it.
    let unknowns: usize = solutions.iter().map(|solution| solution.stats.unknowns).sum();
    assert_eq!(assembly_unknowns(&parts, std::slice::from_ref(&bond), &Limits::default()).unwrap(), unknowns);
    let few = Limits { max_dofs: unknowns - 1, ..Limits::default() };
    assert!(matches!(solve_assembly(&parts, std::slice::from_ref(&bond), &few), Err(Error::Limit(_))));
    // The post bends over: its top moves more than its foot.
    assert!(largest > 2.0 * solutions[1].displacement([0.85, 0.7, 0.3]).unwrap()[0].abs());
    // Without the bond nothing holds the post.
    assert!(matches!(solve_assembly(&parts, &[], &Limits::default()), Err(Error::Invalid(_))));
    // With faces that do not touch, nothing holds it either: refused, not a wrong answer.
    let apart = Bond { first: (0, vec![5]), second: (1, vec![5]), tolerance: 1e-6, grip: Grip::Bonded };
    assert_eq!(free_motions(&parts, std::slice::from_ref(&apart)).unwrap().len(), 6);
    assert!(matches!(solve_assembly(&parts, &[apart], &Limits::default()), Err(Error::Invalid(_))));
    assert!(free_motions(&parts, std::slice::from_ref(&bond)).unwrap().is_empty());
    // The bond's faces touch over the post's foot, 0.5 by 0.4; faces apart touch over nothing.
    let apart = Bond { first: (0, vec![5]), second: (1, vec![5]), tolerance: 1e-6, grip: Grip::Bonded };
    let areas = fieldcraft_fcm::bond_areas(&parts, &[bond.clone(), apart]).unwrap();
    assert!((areas[0] - 0.2).abs() < 1e-12 && areas[1] == 0.0, "{areas:?}");
}

/// One body with no bond is the body's own solve, bit for bit, and what cannot be an assembly
/// is refused.
#[test]
fn one_body_and_hostile_assemblies() {
    let bar = block([0.0; 3], [1.0, 0.5, 0.5]);
    let supports = sliding(true);
    let loads = [Load::Traction { face: 1, traction: [1e6, 0.0, 0.0] }];
    let options = Options { degree: 2, offset: [0.31, 0.62, 0.13], ..Options::new(0.3) };
    let part = Part { surface: &bar, material: STEEL, supports: &supports, loads: &loads, options, curved: &[] };
    let alone = fieldcraft_fcm::solve(&bar, &STEEL, &supports, &loads, &options, &Limits::default()).unwrap();
    let one = solve_assembly(&[part], &[], &Limits::default()).unwrap();
    assert_eq!(one[0].strain_energy.to_bits(), alone.strain_energy.to_bits());
    let limits = Limits::default();
    let refused = |bonds: &[Bond]| matches!(solve_assembly(&[part, part], bonds, &limits), Err(Error::Invalid(_)));
    assert!(matches!(solve_assembly(&[], &[], &limits), Err(Error::Invalid(_))));
    assert!(refused(&[Bond { first: (0, vec![1]), second: (0, vec![0]), tolerance: 1e-6, grip: Grip::Bonded }]), "a body bonded to itself");
    assert!(refused(&[Bond { first: (0, vec![1]), second: (7, vec![0]), tolerance: 1e-6, grip: Grip::Bonded }]), "a body that is not there");
    assert!(refused(&[Bond { first: (0, vec![99]), second: (1, vec![0]), tolerance: 1e-6, grip: Grip::Bonded }]), "a face that is not there");
    assert!(refused(&[Bond { first: (0, vec![]), second: (1, vec![0]), tolerance: 1e-6, grip: Grip::Bonded }]), "no face");
    for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(refused(&[Bond { first: (0, vec![1]), second: (1, vec![0]), tolerance, grip: Grip::Bonded }]), "tolerance {tolerance}");
    }
    // A face that is supported and bonded.
    assert!(refused(&[Bond { first: (0, vec![0]), second: (1, vec![0]), tolerance: 1e-6, grip: Grip::Bonded }]));
}

/// The bond is consistent whatever the two grids: where the cells of both bodies end on the
/// faces that touch, the constant stress is found to rounding, on cells of one size and of two;
/// where they are cut by them or do not line up along them, to the fictitious stiffness of the
/// cut cells. A piece of the one body's face is integrated part by part of the other's cells:
/// by its own points alone the bar on cells of two sizes was 9 % off.
#[test]
fn the_bond_passes_the_patch_test_on_any_two_grids() {
    const STRESS: f64 = 100e6;
    let size = [2.0, 1.0, 1.0];
    let exact =
        |x: [f64; 3]| [STRESS * x[0] / STEEL.young, -STEEL.poisson * STRESS * x[1] / STEEL.young, -STEEL.poisson * STRESS * x[2] / STEEL.young];
    let (held, free) = (sliding(true), sliding(false));
    let pull = [Load::Traction { face: 1, traction: [STRESS, 0.0, 0.0] }];
    let (first, second) = (block([0.0; 3], [1.0, size[1], size[2]]), block([1.0, 0.0, 0.0], size));
    for (name, offsets, cells, tolerance) in [
        ("the same cells", [[0.0; 3], [0.0; 3]], [0.5, 0.5], 1e-11),
        ("cells of two sizes", [[0.0; 3], [0.0; 3]], [0.5, 0.25], 1e-11),
        ("the first body's cells cut by the faces", [[0.5, 0.0, 0.0], [0.0; 3]], [0.5, 0.5], 1e-6),
        ("the second body's cells cut by them", [[0.0; 3], [0.5, 0.0, 0.0]], [0.5, 0.5], 1e-6),
        ("the second body's cells half a cell along them", [[0.0; 3], [0.0, 0.5, 0.5]], [0.5, 0.5], 1e-5),
        ("the first body's cells half a cell along them", [[0.0, 0.5, 0.5], [0.0; 3]], [0.5, 0.5], 1e-5),
    ] {
        let parts = [
            Part {
                surface: &first,
                material: STEEL,
                supports: &held,
                loads: &[],
                options: Options { degree: 2, offset: offsets[0], ..Options::new(cells[0]) },
                curved: &[],
            },
            Part {
                surface: &second,
                material: STEEL,
                supports: &free,
                loads: &pull,
                options: Options { degree: 2, offset: offsets[1], ..Options::new(cells[1]) },
                curved: &[],
            },
        ];
        let solutions =
            solve_assembly(&parts, &[Bond { first: (0, vec![1]), second: (1, vec![0]), tolerance: 1e-6, grip: Grip::Bonded }], &Limits::default())
                .unwrap();
        let elongation = STRESS * size[0] / STEEL.young;
        let (a, b) = (
            errors(&solutions[0], [0.0; 3], [1.0, size[1], size[2]], exact, elongation, STRESS),
            errors(&solutions[1], [1.0, 0.0, 0.0], size, exact, elongation, STRESS),
        );
        println!("{name}: {a:?} {b:?}");
        for (displacement, stress) in [a, b] {
            assert!(displacement < tolerance && stress < tolerance, "{name}: displacement {displacement:.3e}, stress {stress:.3e}");
        }
    }
}

/// A prism of `sides` sides around the z axis from `from` to `to`, with a hole of the radius
/// `bore` through it when that is not zero. Faces: 0 the side, 1 the lower end, 2 the upper
/// end, 3 the hole's wall.
fn prism(sides: usize, radius: f64, bore: f64, from: f64, to: f64) -> BoundarySurface {
    let n = sides as u32;
    let ring = |r: f64, z: f64| {
        (0..sides).map(move |i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / sides as f64;
            [r * angle.cos(), r * angle.sin(), z]
        })
    };
    let mut points: Vec<[f64; 3]> = ring(radius, from).chain(ring(radius, to)).collect();
    let (mut triangles, mut tags) = (Vec::new(), Vec::new());
    let mut add = |faces: &[[u32; 3]], tag: u32| {
        triangles.extend_from_slice(faces);
        tags.extend(faces.iter().map(|_| tag));
    };
    for i in 0..n {
        let j = (i + 1) % n;
        add(&[[i, j, n + j], [i, n + j, n + i]], 0);
    }
    if bore > 0.0 {
        points.extend(ring(bore, from).chain(ring(bore, to)));
        for i in 0..n {
            let j = (i + 1) % n;
            add(&[[2 * n + j, 2 * n + i, 3 * n + i], [2 * n + j, 3 * n + i, 3 * n + j]], 3);
            add(&[[j, i, 2 * n + i], [j, 2 * n + i, 2 * n + j]], 1);
            add(&[[n + i, n + j, 3 * n + j], [n + i, 3 * n + j, 3 * n + i]], 2);
        }
    } else {
        points.extend([[0.0, 0.0, from], [0.0, 0.0, to]]);
        for i in 0..n {
            let j = (i + 1) % n;
            add(&[[2 * n, j, i]], 1);
            add(&[[2 * n + 1, n + i, n + j]], 2);
        }
    }
    BoundarySurface::new(&points, &triangles, &tags, &SurfaceOptions::default()).unwrap()
}

/// The faces of two bodies that touch are found: blocks end to end, a block on a part of a
/// plate (from the block's side: the plate's two triangles hardly sample it), a pin in a hole
/// whose facets do not match. Bodies apart, and bodies that meet in an edge only, do not touch.
#[test]
fn faces_that_touch_are_found() {
    let ends = [block([0.0; 3], [1.0, 0.8, 0.6]), block([1.0, 0.0, 0.0], [2.3, 0.8, 0.6])];
    let found = find_bonds(&[&ends[0], &ends[1]], 1e-6).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].pairs.iter().map(|(a, b, _)| (*a, *b)).collect::<Vec<_>>(), [(1, 0)]);
    assert_eq!((&found[0].bond.first, &found[0].bond.second), (&(0, vec![1]), &(1, vec![0])));
    assert!((found[0].area - 0.48).abs() < 1e-12, "{}", found[0].area);

    let (plate, post) = (block([0.0; 3], [2.0, 1.5, 0.3]), block([0.6, 0.5, 0.3], [1.1, 0.9, 1.0]));
    let found = find_bonds(&[&plate, &post], 1e-6).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!((&found[0].bond.first, &found[0].bond.second), (&(0, vec![5]), &(1, vec![4])));
    assert!((found[0].area - 0.2).abs() < 1e-12, "{}", found[0].area);

    // A gap wider than the tolerance, and an edge on an edge.
    let apart = block([1.001, 0.0, 0.0], [2.0, 0.8, 0.6]);
    assert!(find_bonds(&[&ends[0], &apart], 1e-6).unwrap().is_empty());
    let diagonal = block([1.0, 0.8, 0.0], [2.0, 1.6, 0.6]);
    assert!(find_bonds(&[&ends[0], &diagonal], 1e-6).unwrap().is_empty());
    // Three bodies: each pair that touches, in order.
    let third = block([2.3, 0.0, 0.0], [3.0, 0.8, 0.6]);
    let found = find_bonds(&[&ends[0], &ends[1], &third], 1e-6).unwrap();
    assert_eq!(found.iter().map(|t| (t.bond.first.0, t.bond.second.0)).collect::<Vec<_>>(), vec![(0, 1), (1, 2)]);

    // A pin of 24 sides in a hole of 40: the facets are up to 0.9 % of the radius apart.
    let (pin, sleeve) = (prism(24, 0.2, 0.0, -0.1, 0.9), prism(40, 0.5, 0.2, 0.0, 0.6));
    assert!(find_bonds(&[&pin, &sleeve], 1e-6).unwrap().is_empty(), "facets that do not match are apart to rounding");
    let found = find_bonds(&[&pin, &sleeve], 0.01 * 0.2).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!((&found[0].bond.first, &found[0].bond.second), (&(0, vec![0]), &(1, vec![3])));
    let wall = 2.0 * std::f64::consts::PI * 0.2 * 0.6;
    assert!((found[0].area / wall - 1.0).abs() < 0.03, "{} of {wall}", found[0].area);
    for tolerance in [0.0, -1.0, f64::NAN] {
        assert!(matches!(find_bonds(&[&pin, &sleeve], tolerance), Err(Error::Invalid(_))));
    }
}

/// A pin bonded in the hole of a sleeve that is clamped on its outside, pulled along its axis:
/// the curved faces touch facet against facet, and the sleeve's support carries the pull.
#[test]
fn a_pin_bonded_in_a_hole_carries_its_pull() {
    let (pin, sleeve) = (prism(24, 0.2, 0.0, -0.1, 0.9), prism(40, 0.5, 0.2, 0.0, 0.6));
    let clamp = [Support { face: 0, fixed: [true; 3], value: [0.0; 3] }];
    let pull = [Load::Force { face: 2, force: [2.0e4, 0.0, 1.0e5] }];
    let parts = [
        Part {
            surface: &pin,
            material: STEEL,
            supports: &[],
            loads: &pull,
            options: Options { degree: 2, offset: [0.37, 0.21, 0.43], ..Options::new(0.15) },
            curved: &[],
        },
        Part {
            surface: &sleeve,
            material: STEEL,
            supports: &clamp,
            loads: &[],
            options: Options { degree: 2, offset: [0.17, 0.41, 0.29], ..Options::new(0.2) },
            curved: &[],
        },
    ];
    let bonds: Vec<Bond> = find_bonds(&[&pin, &sleeve], 0.01 * 0.2).unwrap().into_iter().map(|touch| touch.bond).collect();
    let solutions = solve_assembly(&parts, &bonds, &Limits::default()).unwrap();
    let reaction = solutions[1].reactions.iter().find(|r| r.face == 0).unwrap().force;
    for (i, applied) in [2.0e4, 0.0, 1.0e5].into_iter().enumerate() {
        assert!((reaction[i] + applied).abs() < 1e-6 * 1.0e5, "component {i}: reaction {} against {applied}", reaction[i]);
    }
    // The pin's top moves up and sideways, and the pin moves with the sleeve in the hole.
    let top = solutions[0].displacement([0.0, 0.0, 0.9]).unwrap();
    assert!(top[2] > 0.0 && top[0] > 0.0, "{top:?}");
    let (inside, around) = (solutions[0].displacement([0.19, 0.0, 0.3]).unwrap(), solutions[1].displacement([0.21, 0.0, 0.3]).unwrap());
    println!("the pin's top {top:?}; in the hole the pin {inside:?}, the sleeve {around:?}");
    assert!((inside[2] - around[2]).abs() < 0.2 * top[2], "{inside:?} {around:?}");
}

/// Two blocks of two materials, one on the other and free to slide on it, pressed together:
/// each is in uniaxial compression and widens by its own Poisson's ratio, so they slip where
/// they touch. The fields are linear, and a sliding connection that is consistent finds them
/// like the bond finds a constant stress; bonded, the two could not slip and would shear.
#[test]
fn two_blocks_that_slide_are_pressed_without_shear() {
    const PRESSURE: f64 = 80e6;
    let soft = Material { young: 70e9, poisson: 0.2, density: 2700.0 };
    let (below, above) = (block([0.0; 3], [1.0, 0.8, 0.5]), block([0.0, 0.0, 0.5], [1.0, 0.8, 1.2]));
    // Each keeps its faces at x = 0 and y = 0 in their planes; the lower one stands on z = 0,
    // and the upper one is held along z by the sliding connection alone.
    let sides = |standing: bool| {
        let mut supports = vec![
            Support { face: 0, fixed: [true, false, false], value: [0.0; 3] },
            Support { face: 2, fixed: [false, true, false], value: [0.0; 3] },
        ];
        if standing {
            supports.push(Support { face: 4, fixed: [false, false, true], value: [0.0; 3] });
        }
        supports
    };
    let (standing, lying) = (sides(true), sides(false));
    let press = [Load::Pressure { face: 5, pressure: PRESSURE }];
    let exact = |x: [f64; 3]| {
        if x[2] <= 0.5 {
            [STEEL.poisson * PRESSURE * x[0] / STEEL.young, STEEL.poisson * PRESSURE * x[1] / STEEL.young, -PRESSURE * x[2] / STEEL.young]
        } else {
            [
                soft.poisson * PRESSURE * x[0] / soft.young,
                soft.poisson * PRESSURE * x[1] / soft.young,
                -PRESSURE * 0.5 / STEEL.young - PRESSURE * (x[2] - 0.5) / soft.young,
            ]
        }
    };
    let scale = PRESSURE * (0.5 / STEEL.young + 0.7 / soft.young);
    for (degree, basis) in [(2, fieldcraft_fcm::Basis::Tensor), (3, fieldcraft_fcm::Basis::Trunk)] {
        let parts = [
            Part {
                surface: &below,
                material: STEEL,
                supports: &standing,
                loads: &[],
                options: Options { degree, basis, offset: [0.31, 0.62, 0.13], ..Options::new(0.4) },
                curved: &[],
            },
            Part {
                surface: &above,
                material: soft,
                supports: &lying,
                loads: &press,
                options: Options { degree, basis, offset: [0.57, 0.11, 0.73], ..Options::new(0.3) },
                curved: &[],
            },
        ];
        // The traction is taken from the first body named, the faces being equal: both ways.
        for bond in [
            Bond { first: (0, vec![5]), second: (1, vec![4]), tolerance: 1e-6, grip: Grip::Sliding },
            Bond { first: (1, vec![4]), second: (0, vec![5]), tolerance: 1e-6, grip: Grip::Sliding },
        ] {
            assert!(free_motions(&parts, std::slice::from_ref(&bond)).unwrap().is_empty());
            let solutions = solve_assembly(&parts, &[bond], &Limits::default()).unwrap();
            let (mut displacement, mut stress) = (0.0f64, 0.0f64);
            for (solution, from, to) in [(&solutions[0], 0.0, 0.5), (&solutions[1], 0.5, 1.2)] {
                for fx in [0.03, 0.4, 0.97] {
                    for fy in [0.05, 0.6, 0.95] {
                        for fz in [0.02, 0.5, 0.98] {
                            let x = [fx, 0.8 * fy, from + fz * (to - from)];
                            let (u, s, expected) = (solution.displacement(x).unwrap(), solution.stress(x).unwrap(), exact(x));
                            for i in 0..3 {
                                displacement = displacement.max((u[i] - expected[i]).abs() / scale);
                            }
                            stress = stress.max((s[2] + PRESSURE).abs() / PRESSURE);
                            for component in [s[0], s[1], s[3], s[4], s[5]] {
                                stress = stress.max(component.abs() / PRESSURE);
                            }
                        }
                    }
                }
            }
            println!("degree {degree}, {basis:?}: displacement {displacement:.3e}, stress {stress:.3e}");
            assert!(displacement < 5e-6 && stress < 2e-4, "degree {degree}, {basis:?}: displacement {displacement:.3e}, stress {stress:.3e}");
            // They slip: at the far edge of the faces that touch, by the difference of what the
            // two widen by.
            let at = [0.97, 0.4, 0.5];
            let slip = solutions[1].displacement(at).unwrap()[0] - solutions[0].displacement(at).unwrap()[0];
            let expected = 0.97 * PRESSURE * (soft.poisson / soft.young - STEEL.poisson / STEEL.young);
            assert!((slip - expected).abs() < 1e-4 * expected.abs(), "slip {slip:.6e} against {expected:.6e}");
            // The stand carries the pressure, and the sides nothing.
            let reaction = |solution: &Solution, face: u32| solution.reactions.iter().find(|r| r.face == face).unwrap().force;
            assert!((reaction(&solutions[0], 4)[2] - PRESSURE * 0.8).abs() < 1e-6 * PRESSURE, "{:?}", reaction(&solutions[0], 4));
            for solution in &solutions {
                assert!(reaction(solution, 0)[0].abs() < 1e-6 * PRESSURE && reaction(solution, 2)[1].abs() < 1e-6 * PRESSURE);
            }
        }
    }
}

/// A pin in the hole of a clamped sleeve, free to slide in it. Nothing but the pin's own
/// support keeps it from sliding along the hole and turning in it, which is found before any
/// solve; with the normals of the facets in place of the cylinder's, the hole's facets would
/// keep it from turning. Held at its lower end and pulled, the pin stretches as a bar on its
/// own (no Poisson's ratio: it does not widen), and the sleeve carries nothing along the hole.
#[test]
fn a_pin_slides_and_turns_in_its_hole() {
    const PULL: f64 = 1.0e5;
    let (pin, sleeve) = (prism(24, 0.2, 0.0, -0.1, 0.9), prism(40, 0.5, 0.2, 0.0, 0.6));
    let slender = Material { poisson: 0.0, ..STEEL };
    let round =
        [(0u32, Curved::Cylinder { origin: [0.0; 3], axis: [0.0, 0.0, 1.0] }), (3u32, Curved::Cylinder { origin: [0.0; 3], axis: [0.0, 0.0, 2.0] })];
    let clamp = [Support { face: 0, fixed: [true; 3], value: [0.0; 3] }];
    let foot = [Support { face: 1, fixed: [true; 3], value: [0.0; 3] }];
    let bonds: Vec<Bond> =
        find_bonds(&[&pin, &sleeve], 0.01 * 0.2).unwrap().into_iter().map(|touch| Bond { grip: Grip::Sliding, ..touch.bond }).collect();
    assert_eq!(bonds.len(), 1);
    let assembly = |held: bool, loads: &[Load], curved: bool| -> (Vec<fieldcraft_fcm::Motion>, Result<Vec<Solution>, Error>) {
        let parts = [
            Part {
                surface: &pin,
                material: slender,
                supports: if held { &foot } else { &[] },
                loads,
                options: Options { degree: 2, offset: [0.37, 0.21, 0.43], ..Options::new(0.15) },
                curved: if curved { &round } else { &[] },
            },
            Part {
                surface: &sleeve,
                material: STEEL,
                supports: &clamp,
                loads: &[],
                options: Options { degree: 2, offset: [0.17, 0.41, 0.29], ..Options::new(0.2) },
                curved: if curved { &round } else { &[] },
            },
        ];
        (free_motions(&parts, &bonds).unwrap(), solve_assembly(&parts, &bonds, &Limits::default()))
    };
    // Free: the pin alone moves, along the hole and about its axis, and the solve is refused.
    let (free, refused) = assembly(false, &[], true);
    assert!(matches!(refused, Err(Error::Invalid(_))));
    assert_eq!(free.len(), 2, "{free:?}");
    let unit = |v: [f64; 3]| v.iter().fold(0.0f64, |m, c| m.max(c.abs()));
    for motion in &free {
        let [(0, slide, turn)] = motion.bodies[..] else { panic!("{motion:?}") };
        // Along z or about z, and nothing else.
        assert!(slide[0].abs() < 1e-9 && slide[1].abs() < 1e-9 && turn[0].abs() < 1e-9 && turn[1].abs() < 1e-9, "{motion:?}");
        assert!((unit(slide) - 1.0).abs() < 1e-9 || (unit(turn) - 1.0).abs() < 1e-9, "{motion:?}");
    }
    assert!(free.iter().any(|motion| motion.bodies[0].2[2].abs() > 0.5) && free.iter().any(|motion| motion.bodies[0].1[2].abs() > 0.5), "{free:?}");
    // On the facets' normals the pin could slide but not turn.
    assert_eq!(assembly(false, &[], false).0.len(), 1);

    // Pulled along the hole: a bar, F L / (E A) at its top, A the area of its 24 sides.
    let pull = [Load::Force { face: 2, force: [0.0, 0.0, PULL] }];
    let (free, solutions) = assembly(true, &pull, true);
    assert!(free.is_empty(), "{free:?}");
    let solutions = solutions.unwrap();
    let area = 0.5 * 24.0 * 0.2 * 0.2 * (2.0 * std::f64::consts::PI / 24.0).sin();
    let (top, expected) = (solutions[0].displacement([0.05, 0.02, 0.9]).unwrap(), PULL * 1.0 / (slender.young * area));
    println!("the pin's top {top:?} against {expected:.6e}");
    assert!((top[2] - expected).abs() < 1e-5 * expected && top[0].abs() < 1e-5 * expected, "{top:?} against {expected:.6e}");
    let around = solutions[1].displacement([0.25, 0.05, 0.3]).unwrap();
    assert!(unit(around) < 1e-5 * expected, "the sleeve moves: {around:?}");
    assert!(solutions[1].strain_energy.abs() < 1e-8 * solutions[0].strain_energy, "{} of {}", solutions[1].strain_energy, solutions[0].strain_energy);

    // Pushed sideways as well: the sleeve takes its share across the hole and nothing along it.
    let push = [Load::Force { face: 2, force: [2.0e4, 0.0, PULL] }];
    let solutions = assembly(true, &push, true).1.unwrap();
    let (on_sleeve, on_pin) =
        (solutions[1].reactions.iter().find(|r| r.face == 0).unwrap().force, solutions[0].reactions.iter().find(|r| r.face == 1).unwrap().force);
    println!("the sleeve's clamp {on_sleeve:?}, the pin's foot {on_pin:?}");
    assert!(on_sleeve[2].abs() < 1e-6 * PULL, "{on_sleeve:?}");
    assert!((on_pin[2] + PULL).abs() < 1e-6 * PULL && (on_pin[0] + on_sleeve[0] + 2.0e4).abs() < 1e-6 * PULL, "{on_pin:?} {on_sleeve:?}");
    assert!(on_sleeve[0].abs() > 0.1 * 2.0e4, "the sleeve carries none of the push: {on_sleeve:?}");
}

/// A block standing on a clamped plate and free to slide on it can still slide two ways and
/// turn about the plate's normal: found and refused. Held against those by its sides, it is
/// solved, and the plate carries what presses it down.
#[test]
fn a_body_a_sliding_connection_leaves_free_is_refused() {
    let (plate, post) = (block([0.0; 3], [2.0, 1.5, 0.3]), block([0.6, 0.5, 0.3], [1.1, 0.9, 1.0]));
    let clamp = [Support { face: 4, fixed: [true; 3], value: [0.0; 3] }];
    let guides =
        [Support { face: 0, fixed: [true, false, false], value: [0.0; 3] }, Support { face: 2, fixed: [false, true, false], value: [0.0; 3] }];
    let press = [Load::Force { face: 5, force: [0.0, 0.0, -2.0e5] }];
    let bond = Bond { first: (0, vec![5]), second: (1, vec![4]), tolerance: 1e-6, grip: Grip::Sliding };
    let assembly = |guided: bool| {
        let parts = [
            Part {
                surface: &plate,
                material: STEEL,
                supports: &clamp,
                loads: &[],
                options: Options { degree: 2, offset: [0.37, 0.21, 0.43], ..Options::new(0.25) },
                curved: &[],
            },
            Part {
                surface: &post,
                material: STEEL,
                supports: if guided { &guides } else { &[] },
                loads: &press,
                options: Options { degree: 2, offset: [0.17, 0.41, 0.29], ..Options::new(0.2) },
                curved: &[],
            },
        ];
        (free_motions(&parts, std::slice::from_ref(&bond)).unwrap(), solve_assembly(&parts, std::slice::from_ref(&bond), &Limits::default()))
    };
    let (free, refused) = assembly(false);
    assert!(matches!(&refused, Err(Error::Invalid(why)) if why.contains("body 1")), "{:?}", refused.err());
    assert_eq!(free.len(), 3, "{free:?}");
    for motion in &free {
        let [(1, slide, turn)] = motion.bodies[..] else { panic!("{motion:?}") };
        // In the plane of the plate, or about its normal.
        assert!(slide[2].abs() < 1e-9 && turn[0].abs() < 1e-9 && turn[1].abs() < 1e-9, "{motion:?}");
    }
    let (free, solutions) = assembly(true);
    assert!(free.is_empty(), "{free:?}");
    let solutions = solutions.unwrap();
    let reaction = solutions[0].reactions.iter().find(|r| r.face == 4).unwrap().force;
    assert!((reaction[2] - 2.0e5).abs() < 1e-6 * 2.0e5 && reaction[0].abs() < 1e-6 * 2.0e5 && reaction[1].abs() < 1e-6 * 2.0e5, "{reaction:?}");
}
