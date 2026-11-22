//! The unknowns by iteration ([`fieldcraft_fcm::Solver::Iterative`]) against the factorization:
//! the same displacements, stresses and reactions to the iteration's tolerance, on one body
//! with and without divided cells, on bodies that are bonded and that slide, on any number of
//! threads; and the factorization after all for a system the iteration does not suit.

use fieldcraft_fcm::{Bond, Error, Grip, Load, Material, Options, Part, Solution, Solver, Support, find_bonds, solve, solve_assembly};
use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};
use fieldcraft_solver::Limits;

const STEEL: Material = Material { young: 210e9, poisson: 0.3, density: 7850.0 };

/// The box from `lo` to `hi`, the face at the lower end of axis a tagged 2a and the one at its
/// upper end 2a + 1.
fn block(lo: [f64; 3], hi: [f64; 3]) -> BoundarySurface {
    let at = |i: usize| [if i & 1 == 0 { lo[0] } else { hi[0] }, if i & 2 == 0 { lo[1] } else { hi[1] }, if i & 4 == 0 { lo[2] } else { hi[2] }];
    let points: Vec<[f64; 3]> = (0..8).map(at).collect();
    let quads: [([u32; 4], u32); 6] =
        [([0, 4, 6, 2], 0), ([1, 3, 7, 5], 1), ([0, 1, 5, 4], 2), ([2, 6, 7, 3], 3), ([0, 2, 3, 1], 4), ([4, 5, 7, 6], 5)];
    let (mut triangles, mut tags) = (Vec::new(), Vec::new());
    for (q, tag) in quads {
        triangles.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
        tags.extend([tag, tag]);
    }
    BoundarySurface::new(&points, &triangles, &tags, &SurfaceOptions::default()).unwrap()
}

/// A tube of `sides` sides around the z axis from 0 to `height`: faces 0 the outside, 1 the
/// lower end, 2 the upper end, 3 the bore.
fn tube(sides: usize, radius: f64, bore: f64, height: f64) -> BoundarySurface {
    let n = sides as u32;
    let ring = |r: f64, z: f64| {
        (0..sides).map(move |i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / sides as f64;
            [r * angle.cos(), r * angle.sin(), z]
        })
    };
    let points: Vec<[f64; 3]> = ring(radius, 0.0).chain(ring(radius, height)).chain(ring(bore, 0.0)).chain(ring(bore, height)).collect();
    let (mut triangles, mut tags) = (Vec::new(), Vec::new());
    let mut add = |faces: &[[u32; 3]], tag: u32| {
        triangles.extend_from_slice(faces);
        tags.extend(faces.iter().map(|_| tag));
    };
    for i in 0..n {
        let j = (i + 1) % n;
        add(&[[i, j, n + j], [i, n + j, n + i]], 0);
        add(&[[2 * n + j, 2 * n + i, 3 * n + i], [2 * n + j, 3 * n + i, 3 * n + j]], 3);
        add(&[[j, i, 2 * n + i], [j, 2 * n + i, 2 * n + j]], 1);
        add(&[[n + i, n + j, 3 * n + j], [n + i, 3 * n + j, 3 * n + i]], 2);
    }
    BoundarySurface::new(&points, &triangles, &tags, &SurfaceOptions::default()).unwrap()
}

/// The largest difference of the displacement and of the stress of two solutions at `points`,
/// each over the largest value there.
fn apart(a: &Solution, b: &Solution, points: &[[f64; 3]]) -> (f64, f64) {
    let (mut du, mut ds, mut u_max, mut s_max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for x in points {
        let ((ua, sa), (ub, sb)) = (a.at(*x).unwrap(), b.at(*x).unwrap());
        for i in 0..3 {
            du = du.max((ua[i] - ub[i]).abs());
            u_max = u_max.max(ub[i].abs());
        }
        for i in 0..6 {
            ds = ds.max((sa[i] - sb[i]).abs());
            s_max = s_max.max(sb[i].abs());
        }
    }
    (du / u_max, ds / s_max)
}

/// A block clamped at one end and pushed across at the other, solved both ways: the iteration
/// takes its rounds and ends below its tolerance, with the factorization's displacements,
/// stresses and reaction; the same on one thread as on many; and it solves what is refused
/// when the factor may not be large enough.
#[test]
fn a_block_by_iteration_is_the_block_by_factorization() {
    let surface = block([0.0; 3], [1.0, 0.9, 0.8]);
    let supports = [Support { face: 0, fixed: [true; 3], value: [0.0; 3] }];
    let loads = [Load::Traction { face: 1, traction: [0.0, 2.0e5, 1.0e6] }];
    let points = [[0.97, 0.45, 0.4], [0.5, 0.1, 0.7], [0.05, 0.8, 0.1], [0.999, 0.899, 0.799], [0.3, 0.3, 0.3]];
    for (degree, basis) in [(2, fieldcraft_fcm::Basis::Tensor), (3, fieldcraft_fcm::Basis::Trunk), (3, fieldcraft_fcm::Basis::Tensor)] {
        let options = |solver: Solver| Options { degree, basis, solver, offset: [0.31, 0.62, 0.13], ..Options::new(0.2) };
        let direct = solve(&surface, &STEEL, &supports, &loads, &options(Solver::Direct), &Limits::default()).unwrap();
        let iterated = solve(&surface, &STEEL, &supports, &loads, &options(Solver::Iterative), &Limits::default()).unwrap();
        assert_eq!((direct.stats.iterations, direct.stats.residual), (0, 0.0));
        assert!(iterated.stats.iterations > 5 && iterated.stats.iterations < 150, "degree {degree}, {basis:?}: {} rounds", iterated.stats.iterations);
        assert!(iterated.stats.residual > 0.0 && iterated.stats.residual <= 1e-9, "{}", iterated.stats.residual);
        // The coarse level's factor is a small part of the whole one.
        assert!(iterated.stats.factor_entries > 0 && iterated.stats.factor_entries * 10 < direct.stats.factor_entries);
        let (du, ds) = apart(&iterated, &direct, &points);
        println!(
            "degree {degree}, {basis:?}: {} rounds to {:.1e}; displacement {du:.1e}, stress {ds:.1e} apart",
            iterated.stats.iterations, iterated.stats.residual
        );
        assert!(du < 1e-7 && ds < 1e-6, "degree {degree}, {basis:?}: displacement {du:.2e}, stress {ds:.2e}");
        assert!((iterated.strain_energy - direct.strain_energy).abs() < 1e-8 * direct.strain_energy);
        let reaction = |s: &Solution| s.reactions.iter().find(|r| r.face == 0).unwrap().force;
        for i in 0..3 {
            assert!((reaction(&iterated)[i] - reaction(&direct)[i]).abs() < 1e-7 * 1.0e6 * 0.72, "component {i}");
        }
        // Small as it is, left to the choice it is factored.
        let auto = solve(&surface, &STEEL, &supports, &loads, &options(Solver::Auto), &Limits::default()).unwrap();
        assert_eq!((auto.stats.iterations, auto.strain_energy.to_bits()), (0, direct.strain_energy.to_bits()));
        if degree == 3 && basis == fieldcraft_fcm::Basis::Tensor {
            // The same to the bit on one thread and on eight.
            let one =
                fieldcraft_parallel::with_threads(1, || solve(&surface, &STEEL, &supports, &loads, &options(Solver::Iterative), &Limits::default()))
                    .unwrap();
            let eight =
                fieldcraft_parallel::with_threads(8, || solve(&surface, &STEEL, &supports, &loads, &options(Solver::Iterative), &Limits::default()))
                    .unwrap();
            assert_eq!((one.strain_energy.to_bits(), one.stats.iterations), (eight.strain_energy.to_bits(), eight.stats.iterations));
            assert_eq!(one.strain_energy.to_bits(), iterated.strain_energy.to_bits());
            // A factor that may hold a tenth of what it needs: refused by the factorization, and
            // by the choice left to the iteration, whose coarse level fits.
            let small = Limits { max_factor_entries: direct.stats.factor_entries / 10, ..Limits::default() };
            assert!(matches!(solve(&surface, &STEEL, &supports, &loads, &options(Solver::Direct), &small), Err(Error::Limit(_))));
            let chosen = solve(&surface, &STEEL, &supports, &loads, &options(Solver::Auto), &small).unwrap();
            assert_eq!((chosen.stats.iterations, chosen.strain_energy.to_bits()), (iterated.stats.iterations, iterated.strain_energy.to_bits()));
        }
    }
}

/// A thick tube under pressure in its bore, its cells divided where the surface is curved: the
/// coarse level is the linear functions of every level, and the iteration settles on a grid of
/// several levels as on one. Nearly every cell of so coarse a grid is cut by the curved
/// surface, which is the iteration's hardest ground: it takes three times the rounds of a
/// grid whose cells are mostly whole.
#[test]
fn divided_cells_are_iterated_like_the_grids_own() {
    let surface = tube(48, 1.0, 0.5, 0.6);
    let supports =
        [Support { face: 1, fixed: [false, false, true], value: [0.0; 3] }, Support { face: 2, fixed: [false, false, true], value: [0.0; 3] }];
    // Held against the rigid motions the ends leave: two points' worth, by the outside.
    let held = [supports[0], supports[1], Support { face: 0, fixed: [true, true, false], value: [0.0; 3] }];
    let loads = [Load::Pressure { face: 3, pressure: 5.0e7 }];
    let points = [[0.6, 0.0, 0.3], [0.0, 0.75, 0.1], [-0.52, 0.0, 0.5], [0.4, 0.4, 0.3]];
    for refine in [0, 1, 2] {
        let options = |solver: Solver| Options { degree: 3, refine, solver, offset: [0.37, 0.21, 0.43], ..Options::new(0.4) };
        let direct = solve(&surface, &STEEL, &held, &loads, &options(Solver::Direct), &Limits::default()).unwrap();
        let iterated = solve(&surface, &STEEL, &held, &loads, &options(Solver::Iterative), &Limits::default()).unwrap();
        let (du, ds) = apart(&iterated, &direct, &points);
        println!(
            "{refine} levels, cells by level {:?}: {} unknowns, {} rounds; displacement {du:.1e}, stress {ds:.1e} apart",
            &direct.stats.levels[..refine + 1],
            direct.stats.unknowns,
            iterated.stats.iterations
        );
        assert!(iterated.stats.iterations > 0 && iterated.stats.iterations < 390, "{refine} levels: {} rounds", iterated.stats.iterations);
        assert!(du < 1e-7 && ds < 1e-6, "{refine} levels: displacement {du:.2e}, stress {ds:.2e}");
        assert!(refine == 0 || direct.stats.levels[refine] > 0, "no cell of level {refine}");
    }
}

/// Bodies bonded and bodies that slide, solved both ways: a base with two blocks standing on
/// it. The blocks of two cells a connection couples are part of the local level.
#[test]
fn connected_bodies_by_iteration_are_those_by_factorization() {
    let surfaces = [block([0.0; 3], [1.0, 0.9, 0.4]), block([0.1, 0.1, 0.4], [0.45, 0.8, 0.8]), block([0.55, 0.1, 0.4], [0.9, 0.8, 0.8])];
    let clamp = [Support { face: 4, fixed: [true; 3], value: [0.0; 3] }];
    let guides =
        [Support { face: 0, fixed: [true, false, false], value: [0.0; 3] }, Support { face: 2, fixed: [false, true, false], value: [0.0; 3] }];
    let push = [Load::Traction { face: 5, traction: [2e5, 1e5, -1e6] }];
    let points = [[[0.5, 0.45, 0.2], [0.9, 0.1, 0.39]], [[0.3, 0.4, 0.79], [0.12, 0.7, 0.45]], [[0.7, 0.4, 0.79], [0.88, 0.2, 0.5]]];
    for grip in [Grip::Bonded, Grip::Sliding] {
        let held: &[Support] = if grip == Grip::Sliding { &guides } else { &[] };
        let bonds: Vec<Bond> =
            find_bonds(&[&surfaces[0], &surfaces[1], &surfaces[2]], 1e-6).unwrap().into_iter().map(|touch| Bond { grip, ..touch.bond }).collect();
        assert_eq!(bonds.len(), 2);
        let run = |solver: Solver| {
            let options = |offset: [f64; 3]| Options { degree: 2, solver, offset, ..Options::new(0.14) };
            let parts = [
                Part { surface: &surfaces[0], material: STEEL, supports: &clamp, loads: &[], options: options([0.31, 0.62, 0.13]), curved: &[] },
                Part { surface: &surfaces[1], material: STEEL, supports: held, loads: &push, options: options([0.57, 0.11, 0.73]), curved: &[] },
                Part { surface: &surfaces[2], material: STEEL, supports: held, loads: &push, options: options([0.17, 0.41, 0.29]), curved: &[] },
            ];
            solve_assembly(&parts, &bonds, &Limits::default()).unwrap()
        };
        let (direct, iterated) = (run(Solver::Direct), run(Solver::Iterative));
        let rounds = iterated[0].stats.iterations;
        // Every body's solution counts the rounds of the one system.
        assert!(iterated.iter().all(|s| s.stats.iterations == rounds) && rounds > 0 && rounds < 300, "{grip:?}: {rounds} rounds");
        for (body, at) in points.iter().enumerate() {
            let (du, ds) = apart(&iterated[body], &direct[body], at);
            println!("{grip:?}, body {body}: {rounds} rounds; displacement {du:.1e}, stress {ds:.1e} apart");
            assert!(du < 1e-7 && ds < 1e-6, "{grip:?}, body {body}: displacement {du:.2e}, stress {ds:.2e}");
        }
        // What the base's clamp carries: the push on both blocks, or without what their guides take.
        let reaction = |s: &[Solution]| s[0].reactions.iter().find(|r| r.face == 4).unwrap().force;
        for i in 0..3 {
            assert!((reaction(&iterated)[i] - reaction(&direct)[i]).abs() < 1e-7 * 1e6 * 0.5, "{grip:?}, component {i}");
        }
    }
}

/// A material that hardly changes its volume is not the iteration's: asked for, the rounds do
/// not settle it and the factorization solves it after all, to the bit.
#[test]
fn what_the_iteration_does_not_settle_is_factored() {
    let surface = block([0.0; 3], [1.0, 0.5, 0.5]);
    let rubber = Material { young: 1.0e7, poisson: 0.4999, density: 1000.0 };
    let supports = [Support { face: 0, fixed: [true; 3], value: [0.0; 3] }];
    let loads = [Load::Traction { face: 1, traction: [1.0e4, 0.0, 2.0e3] }];
    let options = |solver: Solver| Options { degree: 3, solver, offset: [0.31, 0.62, 0.13], ..Options::new(0.25) };
    let direct = solve(&surface, &rubber, &supports, &loads, &options(Solver::Direct), &Limits::default()).unwrap();
    let asked = solve(&surface, &rubber, &supports, &loads, &options(Solver::Iterative), &Limits::default()).unwrap();
    assert_eq!((asked.stats.iterations, asked.strain_energy.to_bits()), (0, direct.strain_energy.to_bits()));
    // Where the factor may not be had either, the refusal says both.
    let none = Limits { max_factor_entries: 1000, ..Limits::default() };
    let refused = solve(&surface, &rubber, &supports, &loads, &options(Solver::Iterative), &none);
    assert!(matches!(&refused, Err(Error::Limit(why)) if why.contains("did not settle") && why.contains("factorization")), "{:?}", refused.err());
    // Asked for the factorization alone, the refusal is its own.
    let refused = solve(&surface, &rubber, &supports, &loads, &options(Solver::Direct), &none);
    assert!(matches!(&refused, Err(Error::Limit(why)) if why.starts_with("the factorization needs")), "{:?}", refused.err());
}
