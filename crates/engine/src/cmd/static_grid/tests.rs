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

/// The rounds that follow the solution: asked for by number, and run without being asked on a
/// body that is thin for its grid (a cell more than four of its mean thicknesses, here 4 mm,
/// across). The block is held at one end and pushed across, a cantilever in bending.
#[test]
fn rounds_follow_the_solution_where_the_body_is_thin_for_the_grid() {
    let (_dir, mut s) = block();
    s.limits.max_grid_dofs = 40_000;
    s.run_script(&json!([
        {"command": "model.boundary_condition.create", "params": {"name": "Wall", "type": "fixed", "target": {"set": "x0"}}},
        {"command": "model.load.create", "params": {"name": "Push", "type": "pressure", "set": "z0", "magnitude": 1.0}},
        {"command": "model.step.create", "params": {"name": "Bend", "procedure": {"type": "static"}, "boundary_conditions": ["Wall"], "loads": ["Push"]}},
    ]))
    .unwrap();
    let thin_note = |r: &Value| r["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("thin for this grid"));
    // Cells of 5 mm: not thin, no round unless one is asked for.
    let plain = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2, "refine": 0})).unwrap();
    assert_eq!((plain["thin"].as_bool(), plain["adapt"].as_u64(), plain["basis"].as_str()), (Some(false), Some(0), Some("tensor")), "{plain}");
    assert!(plain["settled"].is_null() && !thin_note(&plain), "{plain}");
    let auto = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2, "refine": 0, "adapt": "auto"})).unwrap();
    assert_eq!((auto["unknowns"].as_u64(), auto["strain_energy"].as_f64()), (plain["unknowns"].as_u64(), plain["strain_energy"].as_f64()), "{auto}");
    // One round, to the end (`settle` 0): more unknowns, and how much the round changed the
    // strain energy (either way: the support is imposed weakly, more firmly on smaller cells).
    let asked = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2, "refine": 0, "adapt": 1, "settle": 0})).unwrap();
    assert_eq!(asked["adapt"].as_u64(), Some(1), "{asked}");
    assert!(asked["unknowns"].as_u64() > plain["unknowns"].as_u64() && asked["levels"][1].as_u64().unwrap() > 0, "{asked}");
    let settled = asked["settled"].as_f64().unwrap();
    assert!(settled != 0.0 && settled.abs() < 0.2, "{asked}");
    assert!(close(asked["strain_energy"].as_f64().unwrap() * (1.0 - settled), plain["strain_energy"].as_f64().unwrap(), 1e-9), "{asked}");
    // Cells of 17 mm: thin. Rounds run without being asked, and the reply says where the
    // result stands; with none, it says what to expect.
    let coarse = s.execute("solve.static.grid", &json!({"cell_size": 17, "degree": 2, "refine": 0})).unwrap();
    assert_eq!((coarse["thin"].as_bool(), coarse["basis"].as_str()), (Some(true), Some("trunk")), "{coarse}");
    // The full tensor product when it is asked for, with more unknowns on the same cells.
    let full = s.execute("solve.static.grid", &json!({"cell_size": 17, "degree": 2, "refine": 0, "adapt": 0, "basis": "tensor"})).unwrap();
    let reduced = s.execute("solve.static.grid", &json!({"cell_size": 17, "degree": 2, "refine": 0, "adapt": 0, "basis": "trunk"})).unwrap();
    assert_eq!((full["basis"].as_str(), reduced["basis"].as_str()), (Some("tensor"), Some("trunk")), "{full}");
    assert!(full["unknowns"].as_u64() > reduced["unknowns"].as_u64(), "{full} {reduced}");
    assert!((1..=2).contains(&coarse["adapt"].as_u64().unwrap()) && coarse["settled"].is_f64(), "{coarse}");
    let none = s.execute("solve.static.grid", &json!({"cell_size": 17, "degree": 2, "refine": 0, "adapt": 0})).unwrap();
    assert_eq!((none["thin"].as_bool(), none["adapt"].as_u64()), (Some(true), Some(0)), "{none}");
    assert!(thin_note(&none) && none["unknowns"].as_u64() < coarse["unknowns"].as_u64(), "{none}");
    // A round that would be too large is left out, and the reply says so.
    s.limits.max_grid_dofs = none["unknowns"].as_u64().unwrap() as usize + 10;
    let capped = s.execute("solve.static.grid", &json!({"cell_size": 17, "degree": 2, "refine": 0})).unwrap();
    assert_eq!((capped["adapt"].as_u64(), capped["unknowns"].as_u64()), (Some(0), none["unknowns"].as_u64()), "{capped}");
    assert!(capped["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("too large to solve")), "{capped}");
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
        json!({"cell_size": 5, "degree": 2, "refine": 7}),
        json!({"cell_size": 5, "degree": 2, "basis": "cubic"}),
        json!({"cell_size": 5, "degree": 2, "basis": 7}),
        json!({"cell_size": 5, "degree": 2, "basis": null}),
        json!({"cell_size": 5, "degree": 2, "basis": "x".repeat(100_000)}),
        json!({"cell_size": 5, "degree": 2, "adapt": 7}),
        json!({"cell_size": 5, "degree": 2, "adapt": -1}),
        json!({"cell_size": 5, "degree": 2, "adapt": 1.5}),
        json!({"cell_size": 5, "degree": 2, "adapt": "always"}),
        json!({"cell_size": 5, "degree": 2, "adapt": null}),
        json!({"cell_size": 5, "degree": 2, "adapt": 1e300}),
        json!({"cell_size": 5, "degree": 2, "adapt": 6, "settle": 0}),
        json!({"cell_size": 5, "degree": 2, "settle": -1}),
        json!({"cell_size": 5, "degree": 2, "settle": 2}),
        json!({"cell_size": 5, "degree": 2, "settle": 1e300}),
        json!({"cell_size": 5, "degree": 2, "settle": "soon"}),
        json!({"cell_size": 5, "degree": 2, "settle": null}),
        json!({"cell_size": 5, "degree": 2, "refine": -1}),
        json!({"cell_size": 5, "degree": 2, "refine": 1.5}),
        json!({"cell_size": 5, "degree": 2, "refine": "fine"}),
        json!({"cell_size": 5, "degree": 2, "refine": 2}),
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
    // Both cells together are refused as each is: a partition's cells are no assembly.
    let e = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2})).unwrap_err().to_string();
    assert!(e.contains("is a cell of the partitioned body `Block`"), "{e}");
    let e = s.execute("solve.static.grid", &json!({"body": bodies[0], "cell_size": 5, "degree": 2})).unwrap_err().to_string();
    assert!(e.contains("is a cell of the partitioned body `Block`"), "{e}");
    s.execute("edit.undo", &json!({})).unwrap();
    assert!(s.result("Pull").is_some(), "the body is whole again");
}

/// Two blocks end to end, 20 and 15 mm long: bodies `First` and `Second`.
const TWO: &str = r#"{"commands": [
    {"command": "solid.box", "params": {"corner": [0, 0, 0], "length": 20, "width": 10, "height": 10, "body_name": "First"}},
    {"command": "solid.box", "params": {"corner": [20, 0, 0], "length": 15, "width": 10, "height": 10, "body_name": "Second"}}
]}"#;

/// Several bodies are solved together, bonded where they touch, without anyone naming the
/// faces: two blocks end to end, held on three sides as by symmetry planes and pulled, are one
/// bar. The connection is found and listed, the stress is uniform in both, the far end moves by
/// σ L / E of the two lengths together, and the held end carries the pull. Sets may name faces
/// of both bodies.
#[test]
fn bodies_that_touch_are_solved_together() {
    let bytes = fieldcraft_geometry::script_design(TWO).unwrap();
    let (_dir, path) = crate::tests::design_file(&bytes);
    let mut s = Session { limits: SessionLimits { max_grid_dofs: 20_000, ..SessionLimits::default() }, ..Session::default() };
    s.execute("geometry.open", &json!({"path": path})).unwrap();
    let design = s.design.clone().unwrap();
    assert_eq!(design.bodies.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), vec!["First", "Second"]);
    let face = |body: &str, at: [f64; 3]| -> Value {
        let shape = design.body(body).unwrap();
        let found = shape.faces().iter().find(|f| {
            let c = shape.face_centre(&f.name).unwrap();
            (0..3).all(|k| (c[k] - at[k]).abs() < 1e-6)
        });
        json!({"body": body, "face": found.unwrap_or_else(|| panic!("no face of {body} at {at:?}")).name})
    };
    s.run_script(&json!([
        {"command": "model.set.create", "params": {"name": "both", "geometry": [{"body": "First"}, {"body": "Second"}]}},
        {"command": "model.material.create", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": YOUNG, "poisson": POISSON}, "density": 7.85e-9}},
        {"command": "model.section.create", "params": {"name": "Solid", "set": "both", "material": "Steel", "kind": {"type": "solid"}}},
        {"command": "model.set.create", "params": {"name": "x0", "geometry": [face("First", [0.0, 5.0, 5.0])]}},
        {"command": "model.set.create", "params": {"name": "end", "geometry": [face("Second", [35.0, 5.0, 5.0])]}},
        {"command": "model.set.create", "params": {"name": "y0", "geometry": [face("First", [10.0, 0.0, 5.0]), face("Second", [27.5, 0.0, 5.0])]}},
        {"command": "model.set.create", "params": {"name": "z0", "geometry": [face("First", [10.0, 5.0, 0.0]), face("Second", [27.5, 5.0, 0.0])]}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldX", "type": "symmetry", "target": {"set": "x0"}, "plane": "x"}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldY", "type": "symmetry", "target": {"set": "y0"}, "plane": "y"}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldZ", "type": "symmetry", "target": {"set": "z0"}, "plane": "z"}},
        {"command": "model.load.create", "params": {"name": "Pull", "type": "pressure", "set": "end", "magnitude": -PULL}},
        {"command": "model.step.create", "params": {"name": "Pull", "procedure": {"type": "static"}, "boundary_conditions": ["HoldX", "HoldY", "HoldZ"], "loads": ["Pull"]}},
    ]))
    .unwrap();
    let r = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2})).unwrap();
    let bodies = r["bodies"].as_array().unwrap();
    assert_eq!(bodies.iter().map(|b| b["body"].as_str().unwrap()).collect::<Vec<_>>(), vec!["First", "Second"]);
    let connections = r["connections"].as_array().unwrap();
    assert_eq!(connections.len(), 1, "{connections:?}");
    assert_eq!(connections[0]["bodies"], json!(["First", "Second"]));
    assert_eq!(connections[0]["kind"], "bonded");
    assert_eq!((connections[0]["name"].as_str(), r["connections_added"].clone()), (Some("First-Second"), json!(["First-Second"])));
    assert!(close(connections[0]["area"].as_f64().unwrap(), 100.0, 1e-9), "{}", connections[0]["area"]);
    assert_eq!(connections[0]["faces"][0], json!([face("First", [20.0, 5.0, 5.0])["face"]]));
    assert!(close(r["max_von_mises"]["value"].as_f64().unwrap(), PULL, STRESS), "{}", r["max_von_mises"]);
    assert!(close(r["max_displacement"]["value"].as_f64().unwrap(), PULL * 35.0 / YOUNG, 2e-2), "{}", r["max_displacement"]);
    let (applied, reaction) = (r["applied_total"][0].as_f64().unwrap(), r["reaction_total"][0].as_f64().unwrap());
    assert!(close(applied, PULL * 100.0, 1e-9) && close(-reaction, applied, 1e-6), "{applied} {reaction}");
    assert!(r["reactions"].as_array().unwrap().iter().any(|x| x["body"] == "First") && r["warnings"].as_array().unwrap().is_empty(), "{r}");
    // The far end, read on the second body's surface.
    let probe = s.execute("solve.probe", &json!({"step": "Pull", "at": [[35.0, 5.0, 5.0]]})).unwrap();
    assert!(close(probe["nodes"][0]["displacement"][0].as_f64().unwrap(), PULL * 35.0 / YOUNG, TRAVEL), "{probe}");
    let on_end = s.execute("solve.probe", &json!({"step": "Pull", "set": "end"})).unwrap();
    assert!(on_end["nodes"].as_array().is_some_and(|nodes| !nodes.is_empty()), "{on_end}");
    let reactions = s.execute("results.reactions", &json!({"step": "Pull", "sets": ["x0"]})).unwrap();
    assert!(close(-reactions["sets"][0]["force"][0].as_f64().unwrap(), PULL * 100.0, 1e-6), "{reactions}");

    // Cells left to the command: a twentieth of each body's diagonal, all larger by the one
    // factor that leaves the two bodies the unknowns of one compact body between them.
    let diagonals = [(20.0f64 * 20.0 + 200.0).sqrt(), (15.0f64 * 15.0 + 200.0).sqrt()];
    let cells = |r: &Value| -> Vec<f64> { r["bodies"].as_array().unwrap().iter().map(|b| b["cell_size"].as_f64().unwrap()).collect() };
    let coarse = json!({"degree": 1, "refine": 0});
    let ours = s.execute("solve.static.grid", &coarse).unwrap();
    let sizes = cells(&ours);
    let (unknowns, budget) = (ours["unknowns"].as_f64().unwrap(), 3.0 * (20.0f64 / 3f64.sqrt()).powi(3));
    assert!(unknowns <= budget && unknowns > 0.8 * budget, "{unknowns} of {budget}");
    assert!(sizes[0] > diagonals[0] / 20.0 && close(sizes[0] / diagonals[0], sizes[1] / diagonals[1], 1e-12), "{sizes:?}");
    assert!(ours["warnings"].as_array().unwrap().is_empty(), "{ours}");
    assert!(close(ours["max_displacement"]["value"].as_f64().unwrap(), PULL * 35.0 / YOUNG, 2e-2), "{}", ours["max_displacement"]);
    // Too large for a limit, they are made larger until the solve fits, and the reply says so;
    // cells of a size that was asked for are refused.
    let limit = s.limits.max_grid_dofs;
    s.limits.max_grid_dofs = ours["unknowns"].as_u64().unwrap() as usize - 1;
    let fitted = s.execute("solve.static.grid", &coarse).unwrap();
    assert!(close(cells(&fitted)[0], 2f64.cbrt() * sizes[0], 1e-12), "{fitted}");
    assert!(fitted["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("1.26 times larger")), "{fitted}");
    assert!(fitted["unknowns"].as_u64().unwrap() < ours["unknowns"].as_u64().unwrap());
    let e = s.execute("solve.static.grid", &json!({"degree": 1, "refine": 0, "cell_size": 1})).unwrap_err().to_string();
    assert!(e.contains("limit"), "{e}");
    s.limits.max_grid_dofs = limit;

    // Free of each other, nothing holds the second block along x; and one body can still be
    // solved on its own.
    let e = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2, "connections": "none"})).unwrap_err().to_string();
    assert!(e.contains("`Second` can still slide along (1, 0, 0)"), "{e}");
    let e = s.execute("solve.static.grid", &json!({"connections": "welded"})).unwrap_err().to_string();
    assert!(e.contains("connections"), "{e}");
    // A stress the cells hold leaves the rounds nothing to do: one finds that, and is the last.
    let settled = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2, "adapt": 3})).unwrap();
    assert!(settled["adapt"].as_u64().unwrap() <= 1 && settled["warnings"].as_array().unwrap().is_empty(), "{settled}");
    assert!(close(settled["max_displacement"]["value"].as_f64().unwrap(), PULL * 35.0 / YOUNG, 2e-2), "{}", settled["max_displacement"]);
    // One body on its own takes sets of its own faces only.
    let e = s.execute("solve.static.grid", &json!({"body": "First", "cell_size": 5, "degree": 2})).unwrap_err().to_string();
    assert!(e.contains("set `y0` names body `Second`"), "{e}");
}

/// Rounds that follow the solution and the convergence check run over several bodies together.
/// The two blocks end to end, the first clamped at its end and the second pushed sideways at
/// its own: a cantilever of two bonded bodies. The rounds divide cells in both, the reply
/// counts them and says how much the last one still moved the strain energy, and the tip moves
/// a little further, as a finer solution under a load must; it stays at the beam's,
/// F L³ / (3 E I) with the shear's share (Timoshenko), to the half percent by which a beam
/// stands for a short clamped block. The check compares with the solve of both bodies one
/// degree lower.
#[test]
fn rounds_and_the_check_run_over_several_bodies() {
    const FORCE: f64 = 100.0;
    let bytes = fieldcraft_geometry::script_design(TWO).unwrap();
    let (_dir, path) = crate::tests::design_file(&bytes);
    let mut s = Session { limits: SessionLimits { max_grid_dofs: 60_000, ..SessionLimits::default() }, ..Session::default() };
    s.execute("geometry.open", &json!({"path": path})).unwrap();
    let design = s.design.clone().unwrap();
    let face = |body: &str, at: [f64; 3]| -> Value {
        let shape = design.body(body).unwrap();
        let found = shape.faces().iter().find(|f| {
            let c = shape.face_centre(&f.name).unwrap();
            (0..3).all(|k| (c[k] - at[k]).abs() < 1e-6)
        });
        json!({"body": body, "face": found.unwrap_or_else(|| panic!("no face of {body} at {at:?}")).name})
    };
    s.run_script(&json!([
        {"command": "model.set.create", "params": {"name": "both", "geometry": [{"body": "First"}, {"body": "Second"}]}},
        {"command": "model.material.create", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": YOUNG, "poisson": POISSON}, "density": 7.85e-9}},
        {"command": "model.section.create", "params": {"name": "Solid", "set": "both", "material": "Steel", "kind": {"type": "solid"}}},
        {"command": "model.set.create", "params": {"name": "x0", "geometry": [face("First", [0.0, 5.0, 5.0])]}},
        {"command": "model.set.create", "params": {"name": "end", "geometry": [face("Second", [35.0, 5.0, 5.0])]}},
        {"command": "model.boundary_condition.create", "params": {"name": "Clamp", "type": "fixed", "target": {"set": "x0"}}},
        {"command": "model.load.create", "params": {"name": "Push", "type": "force", "target": {"set": "end"}, "vector": [0, 0, -FORCE]}},
        {"command": "model.step.create", "params": {"name": "Bend", "procedure": {"type": "static"}, "boundary_conditions": ["Clamp"], "loads": ["Push"]}},
    ]))
    .unwrap();
    // The beam's tip: bending, and shear with the rectangle's factor 5/6.
    let (length, inertia, area, shear) = (35.0f64, 10.0f64.powi(4) / 12.0, 100.0, YOUNG / (2.0 * (1.0 + POISSON)));
    let beam = FORCE * length.powi(3) / (3.0 * YOUNG * inertia) + FORCE * length / (5.0 / 6.0 * shear * area);
    let solve = |s: &mut Session, more: Value| {
        let mut params = json!({"cell_size": 5, "degree": 2});
        params.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
        s.execute("solve.static.grid", &params).unwrap()
    };
    let tip = |r: &Value| r["max_displacement"]["value"].as_f64().unwrap();
    let unknowns = |r: &Value, body: usize| r["bodies"][body]["unknowns"].as_u64().unwrap();

    // Left to the command, bodies that are not thin for their grids get no rounds.
    let plain = solve(&mut s, json!({}));
    assert_eq!((plain["adapt"].as_u64(), plain["settled"].is_null(), plain["convergence"].is_null()), (Some(0), true, true), "{plain}");
    let rounds = solve(&mut s, json!({"adapt": 2, "settle": 0}));
    assert_eq!(rounds["adapt"], 2);
    let settled = rounds["settled"].as_f64().unwrap();
    assert!(settled > 0.0 && settled < 0.05, "{settled}");
    assert!(unknowns(&rounds, 0) > unknowns(&plain, 0) && unknowns(&rounds, 1) > unknowns(&plain, 1), "cells are divided in both bodies: {rounds}");
    println!("the tip: {} without rounds, {} with two, the beam's {beam}", tip(&plain), tip(&rounds));
    assert!(tip(&rounds) > tip(&plain) && tip(&rounds) < 1.005 * tip(&plain), "the finer solution is the softer one, by little");
    assert!(close(tip(&plain), beam, 5e-3) && close(tip(&rounds), beam, 5e-3), "{} and {} against {beam}", tip(&plain), tip(&rounds));
    let (applied, reaction) = (rounds["applied_total"][2].as_f64().unwrap(), rounds["reaction_total"][2].as_f64().unwrap());
    assert!(close(applied, -FORCE, 1e-9) && close(-reaction, applied, 1e-6), "{applied} {reaction}");
    assert!(rounds["bodies"].as_array().unwrap().iter().all(|b| b["levels"].as_array().unwrap().len() >= 2), "{}", rounds["bodies"]);

    // A round that moves the strain energy by no more than `settle` is the last, without a
    // warning; one that leaves it moving more than the bar says so.
    let early = solve(&mut s, json!({"adapt": 4, "settle": 0.5}));
    assert!(early["adapt"] == 1 && early["warnings"].as_array().unwrap().is_empty(), "{early}");
    let moving = solve(&mut s, json!({"adapt": 1, "settle": 1e-6}));
    assert!(moving["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("still changing")), "{moving}");

    // The check: both bodies one degree lower, which bends far too little.
    let checked = solve(&mut s, json!({"check_convergence": true}));
    let convergence = &checked["convergence"];
    assert_eq!(convergence["against_degree"], 1);
    assert!(convergence["strain_energy"].as_f64().unwrap() > 0.05 && convergence["points"].as_u64().unwrap() > 100, "{convergence}");
    assert!(convergence["von_mises"].as_f64().unwrap() > 0.0, "{convergence}");
    let e = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 1, "check_convergence": true})).unwrap_err().to_string();
    assert!(e.contains("degree of 2 or more"), "{e}");

    // By iteration: the same tip and reactions to its tolerance, with the rounds it took; left
    // to the choice, a system this small is factored.
    let iterated = solve(&mut s, json!({"solver": "iterative"}));
    assert_eq!((iterated["solver"].as_str(), plain["solver"].as_str()), (Some("iteration"), Some("factorization")));
    assert!(iterated["iterations"].as_u64().unwrap() > 5 && iterated["residual"].as_f64().unwrap() <= 1e-8, "{iterated}");
    assert!(plain["iterations"].is_null() && plain["residual"].is_null());
    assert!(close(tip(&iterated), tip(&plain), 1e-7), "{} against {}", tip(&iterated), tip(&plain));
    assert!(close(iterated["reaction_total"][2].as_f64().unwrap(), plain["reaction_total"][2].as_f64().unwrap(), 1e-7));
    assert!(iterated["factor_entries"].as_u64().unwrap() < plain["factor_entries"].as_u64().unwrap());
    let one = s.execute("solve.static.grid", &json!({"body": "First", "cell_size": 5, "degree": 2, "solver": "iterative", "connections": "none"}));
    assert!(one.is_err(), "the first block alone is held, but its sets name the second: {one:?}");
    assert!(s.execute("solve.static.grid", &json!({"solver": "fastest"})).unwrap_err().to_string().contains("solver"));

    // A round too large for a limit is left out, and the reply says so.
    s.limits.max_grid_dofs = plain["unknowns"].as_u64().unwrap() as usize;
    let left = solve(&mut s, json!({"adapt": 2}));
    assert_eq!((left["adapt"].as_u64(), left["unknowns"].as_u64()), (Some(0), plain["unknowns"].as_u64()));
    assert!(left["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("no round of refinement was run")), "{left}");
    // Hostile rounds are refused.
    for bad in [json!({"adapt": 7}), json!({"adapt": -1}), json!({"adapt": "often"}), json!({"settle": 2}), json!({"settle": -0.1})] {
        assert!(s.execute("solve.static.grid", &bad).is_err(), "{bad}");
    }
}

/// Bodies that are imported surface meshes (`geometry.import.mesh`), not a design: steel, each
/// a block with its faces named by their tags (`solid::tests::block_between`), held as by
/// symmetry planes and pulled on the end of the last with [`PULL`]. The bodies lie end to end
/// along x from 0, each `length` long.
fn imported(blocks: usize, length: f64) -> Session {
    let mut s = Session { limits: SessionLimits { max_grid_dofs: 8000, ..SessionLimits::default() }, ..Session::default() };
    let name = |k: usize| if k == 0 { "Block".to_string() } else { format!("Block {}", k + 1) };
    for k in 0..blocks {
        let x = k as f64 * length;
        s.surfaces.push(std::sync::Arc::new(solid::tests::block_between(&name(k), [x, 0.0, 0.0], [x + length, 10.0, 10.0])));
    }
    let on = |tag: u32| -> Vec<Value> { (0..blocks).map(|k| json!({"body": name(k), "face": tag.to_string()})).collect() };
    let all: Vec<Value> = (0..blocks).map(|k| json!({"body": name(k)})).collect();
    s.run_script(&json!([
        {"command": "model.set.create", "params": {"name": "all", "geometry": all}},
        {"command": "model.material.create", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": YOUNG, "poisson": POISSON}, "density": 7.85e-9}},
        {"command": "model.section.create", "params": {"name": "Solid", "set": "all", "material": "Steel", "kind": {"type": "solid"}}},
        {"command": "model.set.create", "params": {"name": "x0", "geometry": [{"body": "Block", "face": "0"}]}},
        {"command": "model.set.create", "params": {"name": "x1", "geometry": [{"body": name(blocks - 1), "face": "1"}]}},
        {"command": "model.set.create", "params": {"name": "y0", "geometry": on(2)}},
        {"command": "model.set.create", "params": {"name": "z0", "geometry": on(4)}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldX", "type": "symmetry", "target": {"set": "x0"}, "plane": "x"}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldY", "type": "displacement", "target": {"set": "y0"}, "values": [null, 0, null, null, null, null]}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldZ", "type": "symmetry", "target": {"set": "z0"}, "plane": "z"}},
        {"command": "model.load.create", "params": {"name": "Pull", "type": "pressure", "set": "x1", "magnitude": -PULL}},
        {"command": "model.step.create", "params": {"name": "Pull", "procedure": {"type": "static"}, "boundary_conditions": ["HoldX", "HoldY", "HoldZ"], "loads": ["Pull"]}},
    ]))
    .unwrap();
    s
}

/// A body that is an imported surface mesh is solved like one of a design: the pulled block's
/// uniform stress and its stretch, with its faces by their tags in the reply.
#[test]
fn an_imported_surface_mesh_is_solved_like_a_design_body() {
    let mut s = imported(1, 20.0);
    assert!(s.design.is_none());
    let r = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2})).unwrap();
    assert_eq!(r["body"], "Block");
    let stress = r["max_von_mises"]["value"].as_f64().unwrap();
    assert!((stress - PULL).abs() < STRESS * PULL, "{stress}");
    let travel = r["max_displacement"]["value"].as_f64().unwrap();
    // The end moves by PULL/E of the length; the corner farthest from the three planes also
    // draws in by Poisson's ratio of that across the two widths.
    let (along, across) = (PULL / YOUNG * 20.0, POISSON * PULL / YOUNG * 10.0);
    let expected = (along * along + 2.0 * across * across).sqrt();
    assert!((travel - expected).abs() < TRAVEL * expected, "{travel} against {expected}");
    // The faces are named by their tags; the reply says what an imported mesh is.
    let reactions = r["reactions"].as_array().unwrap();
    assert!(reactions.iter().any(|x| x["face"] == "0") && reactions.iter().any(|x| x["face"] == "4"), "{reactions:?}");
    let warnings = r["warnings"].as_array().unwrap();
    assert!(warnings.iter().any(|w| w.as_str().is_some_and(|w| w.contains("imported surface mesh"))), "{warnings:?}");

    // What it does not have is refused by name: a body, a face that is no tag of it.
    let e = s.execute("solve.static.grid", &json!({"body": "Other"})).unwrap_err().to_string();
    assert!(e.contains("no body `Other`") && e.contains("Block"), "{e}");
    s.execute("model.set.create", &json!({"name": "nowhere", "geometry": [{"body": "Block", "face": "9"}]})).unwrap();
    s.execute("model.load.create", &json!({"name": "Lost", "type": "pressure", "set": "nowhere", "magnitude": 1})).unwrap();
    s.execute("model.step.edit", &json!({"name": "Pull", "loads": ["Pull", "Lost"]})).unwrap();
    let e = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2})).unwrap_err().to_string();
    assert!(e.contains("no face `9`"), "{e}");
    // With neither a design nor a surface there is nothing to solve.
    let mut none = Session::default();
    let e = none.execute("solve.static.grid", &json!({})).unwrap_err().to_string();
    assert!(e.contains("an open design or an imported surface"), "{e}");
}

/// Two imported surfaces that touch are bonded there like two bodies of a design: the
/// connection is found, named after them and kept, and the pull goes through it.
#[test]
fn imported_surface_meshes_are_bonded_where_they_touch() {
    let mut s = imported(2, 10.0);
    let r = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2})).unwrap();
    assert_eq!(r["bodies"].as_array().map(Vec::len), Some(2));
    assert_eq!(r["connections_added"].as_array().map(Vec::len), Some(1), "{}", r["connections_added"]);
    let connection = s.model.connections().values().next().unwrap();
    assert_eq!((connection.first.len(), connection.second.len()), (1, 1));
    let stress = r["max_von_mises"]["value"].as_f64().unwrap();
    assert!((stress - PULL).abs() < 0.02 * PULL, "{stress}");
    // The connections can be found without a solve as well.
    let mut again = imported(2, 10.0);
    let found = again.execute("interaction.connection.find", &json!({})).unwrap();
    assert_eq!(again.model.connections().len(), 1, "{found}");
}
