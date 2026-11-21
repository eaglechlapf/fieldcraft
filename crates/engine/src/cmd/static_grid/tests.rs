//! `solve.static.grid` on a block built by SolidCraft: a closed-form case, what the results
//! display reads, what is refused and hostile parameters.

use serde_json::{Value, json};

use super::*;
use crate::tests::TempDir;
use crate::{Limits as SessionLimits, Session};

/// A 20 × 10 × 10 mm block from (0, 0, 0), body `Block`.
const BLOCK: &str = r#"{"commands": [
    {"command": "sketch.create", "params": {"plane": "XY", "name": "Profile"}},
    {"command": "sketch.line", "params": {"points": [[0, 0], [20, 0], [20, 10], [0, 10]], "closed": true}},
    {"command": "sketch.finish", "params": {}},
    {"command": "solid.extrude", "params": {"distance": 10, "body_name": "Block"}}
]}"#;

/// Steel, in the session's default units (mm, t, s, N, MPa).
const YOUNG: f64 = 210_000.0;
const POISSON: f64 = 0.3;
/// The pull on the block's end (MPa).
const PULL: f64 = 100.0;
/// How close the grid's stress is to a stress its cells hold exactly: what is left is the
/// stiffness of the empty part of the cut cells, and the solver's own check of this case
/// (`crates/fcm/tests/patch.rs`, faces anywhere in their cells) holds it to this.
const STRESS: f64 = 5e-3;
/// The displacement is an order closer.
const TRAVEL: f64 = 1e-3;

/// The name of the block's face whose centre is `at`.
fn face(s: &Session, at: [f64; 3]) -> String {
    let design = s.design.clone().unwrap();
    let block = design.body("Block").unwrap();
    let found = block.faces().iter().find(|f| {
        let c = block.face_centre(&f.name).unwrap();
        (0..3).all(|k| (c[k] - at[k]).abs() < 1e-6)
    });
    found.unwrap_or_else(|| panic!("no face at {at:?}")).name.clone()
}

/// The block opened as a design, in steel, with a set of each of four faces (`x0`, `x1`, `y0`,
/// `z0`) and of the body (`block`), and nothing that holds or loads it yet. Small work caps, so
/// what a test asks for by mistake is refused at once.
fn block() -> (TempDir, Session) {
    let bytes = fieldcraft_geometry::script_design(BLOCK).unwrap();
    let (dir, path) = crate::tests::design_file(&bytes);
    let mut s = Session { limits: SessionLimits { max_grid_dofs: 4000, ..SessionLimits::default() }, ..Session::default() };
    s.execute("geometry.open", &json!({"path": path})).unwrap();
    let mut script = vec![
        json!({"command": "model.set.create", "params": {"name": "block", "geometry": [{"body": "Block"}]}}),
        json!({"command": "model.material.create", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": YOUNG, "poisson": POISSON}, "density": 7.85e-9}}),
        json!({"command": "model.section.create", "params": {"name": "Solid", "set": "block", "material": "Steel", "kind": {"type": "solid"}}}),
    ];
    for (name, at) in [("x0", [0.0, 5.0, 5.0]), ("x1", [20.0, 5.0, 5.0]), ("y0", [10.0, 0.0, 5.0]), ("z0", [10.0, 5.0, 0.0])] {
        script.push(json!({"command": "model.set.create", "params": {"name": name, "geometry": [{"body": "Block", "face": face(&s, at)}]}}));
    }
    s.run_script(&Value::Array(script)).unwrap();
    (dir, s)
}

/// [`block`] held on three faces as by symmetry planes and pulled on its end with [`PULL`]: step
/// `Pull`. The exact solution is a uniform stress σ_xx = [`PULL`].
fn pulled() -> (TempDir, Session) {
    let (dir, mut s) = block();
    s.run_script(&json!([
        {"command": "model.boundary_condition.create", "params": {"name": "HoldX", "type": "symmetry", "target": {"set": "x0"}, "plane": "x"}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldY", "type": "displacement", "target": {"set": "y0"}, "values": [null, 0, null, null, null, null]}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldZ", "type": "symmetry", "target": {"set": "z0"}, "plane": "z"}},
        {"command": "model.load.create", "params": {"name": "Pull", "type": "pressure", "set": "x1", "magnitude": -PULL}},
        {"command": "model.step.create", "params": {"name": "Pull", "procedure": {"type": "static"}, "boundary_conditions": ["HoldX", "HoldY", "HoldZ"], "loads": ["Pull"]}},
    ]))
    .unwrap();
    (dir, s)
}

/// The solve of [`pulled`] the tests share: 5 mm cells of degree 2.
fn solve(s: &mut Session) -> Value {
    s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2})).unwrap()
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1e-300)
}

/// A block in tension (S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed.,
/// McGraw-Hill 1970, the uniform tension of a prismatic bar): the uniform stress, the stretch
/// σ L / E with its Poisson contraction, the reaction of the held end and the block's volume, to
/// the solver's precision (a linear field is in the cells' space: [`STRESS`]).
#[test]
fn a_block_in_tension_has_the_uniform_stress() {
    let (_dir, mut s) = pulled();
    let journal = s.journal.len();
    let r = solve(&mut s);
    assert_eq!((r["step"].as_str(), r["body"].as_str(), r["degree"].as_u64()), (Some("Pull"), Some("Block"), Some(2)), "{r}");
    assert_eq!(s.journal.len(), journal, "solving does not change the model");
    assert!(r["unknowns"].as_u64().unwrap() > 0 && r["grid"]["cut"].as_u64().unwrap() > 0 && r["cells"].as_u64().unwrap() > 0, "{r}");
    assert!(close(r["max_von_mises"]["value"].as_f64().unwrap(), PULL, STRESS), "{r}");
    // The far corner moves most: σ L / E along x, ν σ w / E back along y and z.
    let strain = PULL / YOUNG;
    let corner = strain * (20.0f64.powi(2) + 2.0 * (POISSON * 10.0).powi(2)).sqrt();
    assert!(close(r["max_displacement"]["value"].as_f64().unwrap(), corner, TRAVEL), "{r}");
    assert_eq!(r["max_displacement"]["at"], json!([20.0, 10.0, 10.0]), "{r}");
    // 100 MPa on 100 mm²: the held end carries it, the faces that only slide carry nothing.
    let force = PULL * 100.0;
    assert!(close(r["applied_total"][0].as_f64().unwrap(), force, 1e-9) && close(r["reaction_total"][0].as_f64().unwrap(), -force, 1e-6), "{r}");
    let reactions = r["reactions"].as_array().unwrap();
    assert_eq!(reactions.len(), 3, "{r}");
    let held = reactions.iter().find(|x| x["face"] == json!(face(&s, [0.0, 5.0, 5.0]))).unwrap();
    assert!(close(held["force"][0].as_f64().unwrap(), -force, 1e-6), "{r}");
    assert!(close(r["volume"].as_f64().unwrap(), 2000.0, 1e-9) && close(r["surface_volume"].as_f64().unwrap(), 2000.0, 1e-9), "{r}");
    assert_eq!((r["units"]["pressure"].as_str(), r["units"]["length"].as_str()), (Some("MPa"), Some("mm")), "{r}");
    // The faces that only slide carry nothing: round-off beside the pull is 0.
    assert_eq!(r["reaction_total"][1], 0.0, "{r}");
    assert!(r["convergence"].is_null(), "{r}");
    // Asked for, the same step one degree lower: both hold the linear field, so nothing changes
    // beyond the solver's precision.
    let checked = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2, "check_convergence": true})).unwrap();
    let change = &checked["convergence"];
    assert_eq!(change["against_degree"], 1, "{checked}");
    assert!(change["points"].as_u64().unwrap() > 100, "{checked}");
    for key in ["peak_von_mises", "von_mises", "strain_energy"] {
        assert!(change[key].as_f64().unwrap().abs() < 2.0 * STRESS, "{key}: {checked}");
    }
    let e = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 1, "check_convergence": true})).unwrap_err().to_string();
    assert!(e.contains("a degree of 2 or more"), "{e}");
    // The stress is uniform: the smallest on the surface is the largest.
    let shown = s.execute("results.contour", &json!({"field": "von_mises"})).unwrap();
    assert!(close(shown["contour"]["range"]["min"]["value"].as_f64().unwrap(), PULL, STRESS), "{shown}");
    let sxx = s.execute("results.contour", &json!({"field": "stress_xx"})).unwrap();
    assert!(close(sxx["contour"]["range"]["min"]["value"].as_f64().unwrap(), PULL, STRESS), "{sxx}");
}

/// The result is kept on the body's surface: the display draws its contours there, a probe reads
/// its points, reactions sum per set of faces, and what needs a mesh says so.
#[test]
fn the_result_is_shown_and_probed_on_the_surface() {
    use crate::view::{Palette, render_png};
    use fieldcraft_render::{Orientation, StandardView};
    let (_dir, mut s) = pulled();
    solve(&mut s);
    let result = s.result("Pull").unwrap().clone();
    let grid = result.grid().unwrap();
    assert!(result.nodes().iter().all(|n| *n >= GRID_LABELS), "no surface point is taken for a mesh node");
    assert_eq!(grid.display.tris.len(), grid.tri_face.len());
    // Sampled no farther apart than cell_size / degree, on a closed surface: every side twice.
    let longest = grid.display.tris.iter().flat_map(|t| [t.p[0].dist(t.p[1]), t.p[1].dist(t.p[2]), t.p[2].dist(t.p[0])]).fold(0.0, f64::max);
    assert!(longest <= 0.0025 + 1e-12, "{longest}");
    let mut sides: Vec<(u32, u32)> =
        grid.display.tris.iter().flat_map(|t| [(t.nodes[0], t.nodes[1]), (t.nodes[1], t.nodes[2]), (t.nodes[2], t.nodes[0])]).collect();
    assert!(sides.iter().all(|(a, b)| sides.contains(&(*b, *a))), "each side is shared by two triangles, turned alike");
    sides.sort_unstable();
    sides.dedup();
    assert_eq!(sides.len(), 3 * grid.display.tris.len());
    // The twelve edges of the block are its feature lines.
    let on_edges: f64 = grid.display.edges.iter().map(|e| e.p[0].dist(e.p[1])).sum();
    assert!(close(on_edges, 4.0 * (0.02 + 0.01 + 0.01), 1e-9), "{on_edges}");

    // Shown like a static step: von Mises contours on the undeformed surface.
    let view = s.results_view.clone().unwrap();
    assert_eq!((view.step.as_str(), view.contour, view.deformed), ("Pull", Some(crate::results::Field::VonMises), false));
    let (png, shown) = render_png(&s, &Orientation::of(StandardView::Iso), 64, 48, &Palette::LIGHT).unwrap();
    assert!(png.starts_with(&[0x89, b'P', b'N', b'G']));
    assert_eq!(shown.contour.as_ref().map(|c| c.0.as_str()), Some("von_mises"), "{shown:?}");
    let (scene, ..) = crate::view::scene(&s, &Palette::LIGHT);
    assert!(scene.tris.len() >= grid.display.tris.len(), "the contour is drawn on the sampled surface");
    s.execute("results.deformed", &json!({})).unwrap();
    let (_, shown) = render_png(&s, &Orientation::of(StandardView::Iso), 64, 48, &Palette::LIGHT).unwrap();
    assert!(shown.scale > 1.0, "the deformed shape is magnified: {shown:?}");

    // A probe by position reads the nearest surface point; by label, the same.
    let strain = PULL / YOUNG;
    let at = s.execute("solve.probe", &json!({"step": "Pull", "at": [[20.0, 10.0, 10.0]]})).unwrap();
    let probed = &at["nodes"][0];
    let u: [f64; 3] = serde_json::from_value(probed["displacement"].clone()).unwrap();
    assert!(
        close(u[0], strain * 20.0, TRAVEL) && close(u[1], -POISSON * strain * 10.0, TRAVEL) && close(u[2], -POISSON * strain * 10.0, TRAVEL),
        "{at}"
    );
    assert!(close(probed["von_mises"].as_f64().unwrap(), PULL, STRESS) && probed["distance"].as_f64().unwrap() < 1e-9, "{at}");
    assert!(probed["face"].is_string() && probed["xyz"] == json!([20.0, 10.0, 10.0]), "{at}");
    let label = probed["node"].as_u64().unwrap();
    let by_label = s.execute("solve.probe", &json!({"step": "Pull", "nodes": [label]})).unwrap();
    assert_eq!(by_label["nodes"][0]["displacement"], probed["displacement"]);
    // By set: the points of the faces it names, and the largest von Mises stress among them.
    let on_end = s.execute("solve.probe", &json!({"step": "Pull", "set": "x1"})).unwrap();
    assert!(on_end["nodes"].as_array().unwrap().iter().all(|n| n["xyz"][0] == json!(20.0)), "{on_end}");
    assert!(close(on_end["max_von_mises"]["value"].as_f64().unwrap(), PULL, STRESS), "{on_end}");
    for bad in [json!({"nodes": [1]}), json!({"nodes": [label + 10_000_000]}), json!({"set": "block"}), json!({"nodes": [label], "elements": [1]})] {
        let mut p = bad.clone();
        p["step"] = json!("Pull");
        assert!(s.execute("solve.probe", &p).is_err(), "{bad}");
    }

    // Reactions per set of faces; the pull balances the held end.
    let reactions = s.execute("results.reactions", &json!({})).unwrap();
    let x0 = reactions["sets"].as_array().unwrap().iter().find(|r| r["set"] == "x0").unwrap();
    assert!(close(x0["force"][0].as_f64().unwrap(), -PULL * 100.0, 1e-6) && x0["faces"] == 1 && x0["moment"].is_null(), "{reactions}");
    assert!(close(reactions["total"]["force"][0].as_f64().unwrap(), -PULL * 100.0, 1e-6), "{reactions}");

    // What reads a mesh's nodes and elements says where the result is instead.
    for (command, params) in [
        ("results.export.vtu", json!({"path": "nowhere.vtu"})),
        ("results.path", json!({"field": "von_mises", "points": [[0, 5, 5], [20, 5, 5]]})),
        ("results.linearize", json!({"from": [0, 5, 5], "to": [20, 5, 5]})),
        ("results.xy", json!({})),
    ] {
        let e = s.execute(command, &params).unwrap_err().to_string();
        assert!(e.contains("solved on the grid"), "{command}: {e}");
    }
    // Files keep no grid result: nothing of the solve is unsaved, and an edit makes it stale.
    assert!(s.stored_results().is_empty());
    s.execute("model.load.edit", &json!({"name": "Pull", "type": "pressure", "set": "x1", "magnitude": -50})).unwrap();
    assert!(s.result("Pull").is_none() && s.latest_result("Pull").is_some());
}

/// A prescribed stretch of the end face in place of the pull: σ = E δ / L, and the reactions of
/// the two ends balance with no load.
#[test]
fn a_prescribed_displacement_stretches_the_block() {
    let (_dir, mut s) = block();
    s.run_script(&json!([
        {"command": "model.boundary_condition.create", "params": {"name": "Hold", "type": "fixed", "target": {"set": "x0"}}},
        {"command": "model.boundary_condition.create", "params": {"name": "Stretch", "type": "displacement", "target": {"set": "x1"}, "values": [0.01, 0, 0, null, null, 0]}},
        {"command": "model.step.create", "params": {"name": "Stretch", "procedure": {"type": "static"}, "boundary_conditions": ["Hold", "Stretch"]}},
    ]))
    .unwrap();
    let r = s.execute("solve.static.grid", &json!({"cell_size": 10, "degree": 2})).unwrap();
    // (The supports are imposed weakly: the end's travel is the prescribed one to the cells' accuracy.)
    assert!(close(r["max_displacement"]["value"].as_f64().unwrap(), 0.01, 0.02), "{r}");
    // Held ends keep the section from contracting, so the force lies between that of a free bar
    // (E δ / L) and that of a bar that cannot contract anywhere (E (1 − ν) / ((1 + ν)(1 − 2ν)) δ / L).
    let pull =
        r["reactions"].as_array().unwrap().iter().find(|x| x["face"] == json!(face(&s, [20.0, 5.0, 5.0]))).unwrap()["force"][0].as_f64().unwrap();
    let uniaxial = YOUNG * 0.01 / 20.0 * 100.0;
    let confined = (1.0 - POISSON) / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON));
    assert!(pull > uniaxial && pull < confined * uniaxial, "{r}");
    assert_eq!(r["reaction_total"], json!([0.0, 0.0, 0.0]), "{r}");
    assert!(r["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("rotations")), "{r}");
}

/// Everything the solver cannot do is refused by name, and the model and its results stay.
#[test]
fn what_the_grid_cannot_solve_is_refused() {
    let (_dir, s) = pulled();
    let refused = |setup: Value, params: Value, says: &str| {
        let mut t = s.clone();
        t.run_script(&setup).unwrap();
        let before = t.model.clone();
        let e = t.execute("solve.static.grid", &params).unwrap_err().to_string();
        assert!(e.contains(says), "expected `{says}` in: {e}");
        assert_eq!(t.model, before);
        assert!(t.latest_result("Pull").is_none());
    };
    let step = |bcs: Value, loads: Value| json!({"command": "model.step.edit", "params": {"name": "Pull", "procedure": {"type": "static"}, "boundary_conditions": bcs, "loads": loads}});
    let all = json!(["HoldX", "HoldY", "HoldZ"]);
    // Targets that are not whole faces of the body.
    refused(
        json!([
            {"command": "mesh.node.add", "params": {"label": 1, "xyz": [0, 0, 0]}},
            {"command": "model.set.create", "params": {"name": "corner", "nodes": [1]}},
            {"command": "model.boundary_condition.create", "params": {"name": "Pin", "type": "fixed", "target": {"set": "corner"}}},
            step(json!(["HoldX", "HoldY", "HoldZ", "Pin"]), json!(["Pull"])),
        ]),
        json!({}),
        "takes whole faces of the body",
    );
    refused(
        json!([
            {"command": "model.boundary_condition.create", "params": {"name": "All", "type": "fixed", "target": {"set": "block"}}},
            step(json!(["All"]), json!(["Pull"])),
        ]),
        json!({}),
        "whole faces only",
    );
    refused(
        json!([
            {"command": "model.reference_point.create", "params": {"name": "RP", "xyz": [25, 5, 5]}},
            {"command": "model.load.create", "params": {"name": "Point", "type": "force", "target": {"reference_point": "RP"}, "vector": [1, 0, 0]}},
            step(all.clone(), json!(["Point"])),
        ]),
        json!({}),
        "no reference points",
    );
    // Kinds of boundary conditions and loads.
    refused(
        json!([
            {"command": "model.boundary_condition.create", "params": {"name": "Skew", "type": "symmetry", "target": {"set": "x0"}, "plane": {"normal": [1, 1, 0]}}},
            step(json!(["Skew", "HoldY", "HoldZ"]), json!(["Pull"])),
        ]),
        json!({}),
        "inclined",
    );
    refused(
        json!([
            {"command": "model.boundary_condition.create", "params": {"name": "Warm", "type": "temperature", "target": {"set": "x1"}, "value": 50}},
            step(json!(["HoldX", "HoldY", "HoldZ", "Warm"]), json!(["Pull"])),
        ]),
        json!({}),
        "temperature",
    );
    refused(
        json!([
            {"command": "model.load.create", "params": {"name": "Spin", "type": "centrifugal", "angular_velocity": 10, "point": [0, 0, 0], "axis": [0, 0, 1]}},
            step(all.clone(), json!(["Spin"])),
        ]),
        json!({}),
        "centrifugal load",
    );
    refused(
        json!([
            {"command": "model.load.create", "params": {"name": "Each", "type": "force", "target": {"set": "x1"}, "vector": [1, 0, 0], "distribution": "per_node"}},
            step(all.clone(), json!(["Each"])),
        ]),
        json!({}),
        "as a total over faces",
    );
    // The material, the section and what holds the body.
    refused(
        json!([{"command": "model.material.edit", "params": {"name": "Steel", "elastic": {"type": "engineering_constants", "young": [2e5, 1e5, 1e5], "poisson": [0.3, 0.3, 0.3], "shear": [8e4, 8e4, 8e4]}}}]),
        json!({}),
        "orthotropic",
    );
    refused(
        json!([
            {"command": "model.section.delete", "params": {"name": "Solid"}},
            {"command": "model.set.create", "params": {"name": "end", "geometry": [{"body": "Block", "face": face(&s, [20.0, 5.0, 5.0])}]}},
            {"command": "model.section.create", "params": {"name": "Skin", "set": "end", "material": "Steel", "kind": {"type": "shell", "thickness": 1}}},
        ]),
        json!({}),
        "no section is assigned to body `Block`",
    );
    refused(json!([step(json!([]), json!(["Pull"]))]), json!({}), "holds body `Block` nowhere");
    refused(
        json!([
            {"command": "model.boundary_condition.create", "params": {"name": "Push", "type": "displacement", "target": {"set": "x0"}, "values": [0.5, null, null, null, null, null]}},
            step(json!(["HoldX", "Push"]), json!(["Pull"])),
        ]),
        json!({}),
        "prescribe different values",
    );
    // A body the supports leave free to move: the solver's own refusal.
    refused(json!([step(json!(["HoldX"]), json!(["Pull"]))]), json!({"cell_size": 10, "degree": 1}), "singular");
    // What a solve of one body on its own would leave out.
    refused(
        json!([
            {"command": "model.reference_point.create", "params": {"name": "RP", "xyz": [25, 5, 5]}},
            {"command": "model.mass.create", "params": {"name": "Bob", "target": {"reference_point": "RP"}, "mass": 1}},
        ]),
        json!({}),
        "point mass `Bob`",
    );
    // The request itself.
    refused(json!([]), json!({"body": "Nope"}), "no body `Nope`");
    refused(json!([]), json!({"step": "Nope"}), "does not exist");
    refused(
        json!([{"command": "model.step.create", "params": {"name": "Modes", "procedure": {"type": "frequency", "modes": 3}}}]),
        json!({"step": "Modes"}),
        "not a static one",
    );
    refused(json!([]), json!({"cell_size": 0.25, "degree": 3}), "limit reached");
    refused(json!([]), json!({"cell_size": 5, "degree": 2, "max_dofs": 30}), "limit");
    // Gravity needs the density.
    refused(
        json!([
            {"command": "model.material.edit", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": YOUNG, "poisson": POISSON}}},
            {"command": "model.load.create", "params": {"name": "Weight", "type": "gravity", "acceleration": [0, 0, -9810]}},
            step(all, json!(["Weight"])),
        ]),
        json!({}),
        "needs the density",
    );
}

/// Under its own weight the block's supports carry ρ g V.
#[test]
fn gravity_loads_the_volume() {
    let (_dir, mut s) = pulled();
    s.run_script(&json!([
        {"command": "model.load.create", "params": {"name": "Weight", "type": "gravity", "acceleration": [0, 0, -9810], "set": "block"}},
        {"command": "model.step.edit", "params": {"name": "Pull", "procedure": {"type": "static"}, "boundary_conditions": ["HoldX", "HoldY", "HoldZ"], "loads": ["Weight"]}},
    ]))
    .unwrap();
    let r = s.execute("solve.static.grid", &json!({"cell_size": 10, "degree": 2})).unwrap();
    let weight = 7.85e-9 * 9810.0 * 2000.0;
    assert!(close(r["applied_total"][2].as_f64().unwrap(), -weight, 1e-9) && close(r["reaction_total"][2].as_f64().unwrap(), weight, 1e-6), "{r}");
}

#[test]
fn hostile_params_never_panic() {
    let (_dir, mut s) = pulled();
    solve(&mut s);
    let hostile = [
        json!(null),
        json!([]),
        json!("x"),
        json!({"step": 7}),
        json!({"step": "x".repeat(100_000)}),
        json!({"body": 7}),
        json!({"body": null}),
        json!({"body": "x".repeat(100_000)}),
        json!({"body": ["Block"]}),
        json!({"cell_size": 0}),
        json!({"cell_size": -1}),
        json!({"cell_size": 1e-300}),
        json!({"cell_size": 1e300}),
        json!({"cell_size": 1e-3}),
        json!({"cell_size": 0.03}),
        json!({"cell_size": "fine"}),
        json!({"cell_size": "5 parsecs"}),
        json!({"cell_size": [5]}),
        json!({"cell_size": null}),
        json!({"cell_size": 40}),
        json!({"cell_size": 5, "degree": 0}),
        json!({"cell_size": 5, "degree": 9}),
        json!({"cell_size": 5, "degree": 8}),
        json!({"cell_size": 5, "degree": -1}),
        json!({"cell_size": 5, "degree": 2.5}),
        json!({"cell_size": 5, "degree": 1e300}),
        json!({"cell_size": 5, "degree": "cubic"}),
        json!({"cell_size": 5, "degree": 2, "tolerance": 0}),
        json!({"cell_size": 5, "degree": 2, "tolerance": -1}),
        json!({"cell_size": 5, "degree": 2, "tolerance": 1e-300}),
        json!({"cell_size": 5, "degree": 2, "tolerance": 1e300}),
        json!({"cell_size": 5, "degree": 2, "tolerance": "tight"}),
        json!({"cell_size": 5, "degree": 2, "sample_size": 0}),
        json!({"cell_size": 5, "degree": 2, "sample_size": 1e-300}),
        json!({"cell_size": 5, "degree": 2, "sample_size": 1e300}),
        json!({"cell_size": 5, "degree": 2, "sample_size": 0.4}),
        json!({"cell_size": 5, "degree": 2, "sample_size": {}}),
        json!({"cell_size": 5, "degree": 2, "max_dofs": 0}),
        json!({"cell_size": 5, "degree": 2, "max_dofs": 1}),
        json!({"cell_size": 5, "degree": 2, "max_dofs": 1e300}),
        json!({"cell_size": 5, "degree": 2, "max_dofs": -5}),
        json!({"cell_size": 5, "degree": 2, "max_factor_entries": 0}),
        json!({"cell_size": 5, "degree": 2, "max_factor_entries": 10}),
        json!({"cell_size": 5, "degree": 2, "max_factor_entries": 1e300}),
        json!({"cell_size": 5, "degree": 2, "threads": 0}),
        json!({"cell_size": 5, "degree": 2, "threads": 257}),
        json!({"cell_size": 5, "degree": 2, "threads": "all"}),
        json!({"cell_size": 5, "degree": 2, "threads": 2}),
        json!({"cell_size": 5, "degree": 2, "no_such": true}),
        json!({"cell_size": 5, "degree": 2, "check_convergence": "yes"}),
        json!({"cell_size": 5, "degree": 2, "check_convergence": 1}),
        json!({"cell_size": 5, "degree": 1, "check_convergence": true}),
        json!({"cell_size": 10, "degree": 2, "check_convergence": true}),
    ];
    let mut solved = 0;
    for p in &hostile {
        let mut t = s.clone();
        let (before, results) = (t.model.clone(), t.solved_steps().count());
        match t.execute("solve.static.grid", p) {
            Ok(r) => {
                solved += 1;
                assert!(r["unknowns"].as_u64().is_some_and(|n| n <= 4000), "{p}: {r}");
            }
            Err(_) => assert_eq!(t.solved_steps().count(), results, "{p}"),
        }
        assert_eq!(t.model, before, "{p}");
    }
    assert!(solved >= 3, "some of them are requests that solve: {solved}");
    // Against sessions that lack what the command needs, and a design with nothing defined.
    let (_bare_dir, bare) = block();
    for p in &hostile {
        let _ = Session::default().execute("solve.static.grid", p);
        assert!(bare.clone().execute("solve.static.grid", p).is_err(), "{p}");
    }
}

/// The result reads the body's geometry, which the step's definition does not hold: cutting the
/// body in two makes it stale, and undoing the cut current again. A cell of the cut body is
/// refused: its cut face joins it to the other cell.
#[test]
fn a_change_of_the_geometry_makes_the_result_stale() {
    let (_dir, mut s) = pulled();
    solve(&mut s);
    let side = face(&s, [10.0, 10.0, 5.0]);
    s.execute("model.set.create", &json!({"name": "other", "geometry": [{"body": "Block", "face": side}]})).unwrap();
    s.execute("geometry.datum", &json!({"name": "Mid", "by": {"type": "offset", "plane": "yz", "offset": 10}})).unwrap();
    assert!(s.result("Pull").is_some(), "edits the step does not read keep the result current");
    s.execute("geometry.partition", &json!({"name": "Halves", "datum": "Mid"})).unwrap();
    assert!(s.result("Pull").is_none() && s.latest_result("Pull").is_some(), "the body was cut in two");
    let bodies: Vec<String> = s.design.as_ref().unwrap().bodies.iter().map(|b| b.name.clone()).collect();
    assert_eq!(bodies.len(), 2, "{bodies:?}");
    let e = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2})).unwrap_err().to_string();
    assert!(e.contains("name the body"), "{e}");
    let e = s.execute("solve.static.grid", &json!({"body": bodies[0], "cell_size": 5, "degree": 2})).unwrap_err().to_string();
    assert!(e.contains("is a cell of the partitioned body `Block`"), "{e}");
    s.execute("edit.undo", &json!({})).unwrap();
    assert!(s.result("Pull").is_some(), "the body is whole again");
}
