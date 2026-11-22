//! `interaction.connection*`, and what a solve of several bodies on a grid makes of the model's
//! connections: found once, kept as the user leaves them, bonded, sliding or free.

use serde_json::{Value, json};

use crate::tests::TempDir;
use crate::{Limits as SessionLimits, Session};

/// Steel, in the session's default units (mm, t, s, N, MPa).
const YOUNG: f64 = 210_000.0;
const POISSON: f64 = 0.3;
/// The pull on the far end (MPa).
const PULL: f64 = 100.0;

fn close(got: f64, expected: f64, tolerance: f64) -> bool {
    (got - expected).abs() <= tolerance * expected.abs()
}

/// Two blocks end to end, 20 and 15 mm long: bodies `First` and `Second`.
const TWO: &str = r#"{"commands": [
    {"command": "solid.box", "params": {"corner": [0, 0, 0], "length": 20, "width": 10, "height": 10, "body_name": "First"}},
    {"command": "solid.box", "params": {"corner": [20, 0, 0], "length": 15, "width": 10, "height": 10, "body_name": "Second"}}
]}"#;

/// A plate with a hole through it off its middle, and a pin of the hole's diameter standing in
/// the hole and out of it on both sides: bodies `Sleeve` and `Pin`.
const PIN: &str = r#"{"commands": [
    {"command": "solid.box", "params": {"corner": [0, 0, 0], "length": 40, "width": 30, "height": 10, "body_name": "Sleeve"}},
    {"command": "solid.hole", "params": {"position": [25, 12, 10], "diameter": 8, "targets": ["Sleeve"]}},
    {"command": "solid.cylinder", "params": {"base": [25, 12, -5], "diameter": 8, "height": 25, "body_name": "Pin"}}
]}"#;

/// The face of a body whose centre is `at`, as a set names it.
fn face(s: &Session, body: &str, at: [f64; 3]) -> Value {
    let design = s.design.clone().unwrap();
    let shape = design.body(body).unwrap();
    let found = shape.faces().iter().find(|f| {
        let c = shape.face_centre(&f.name).unwrap();
        (0..3).all(|k| (c[k] - at[k]).abs() < 1e-6)
    });
    json!({"body": body, "face": found.unwrap_or_else(|| panic!("no face of {body} at {at:?}")).name})
}

/// The flat face of a body at the height `z`.
fn level(s: &Session, body: &str, z: f64) -> Value {
    let design = s.design.clone().unwrap();
    let shape = design.body(body).unwrap();
    let found = shape.faces().iter().find(|f| (shape.face_centre(&f.name).unwrap()[2] - z).abs() < 1e-6);
    json!({"body": body, "face": found.unwrap_or_else(|| panic!("no face of {body} at z = {z}")).name})
}

/// A design opened, with both bodies in steel.
fn opened(design: &str, bodies: [&str; 2]) -> (TempDir, Session) {
    let bytes = fieldcraft_geometry::script_design(design).unwrap();
    let (dir, path) = crate::tests::design_file(&bytes);
    let mut s = Session { limits: SessionLimits { max_grid_dofs: 60_000, ..SessionLimits::default() }, ..Session::default() };
    s.execute("geometry.open", &json!({"path": path})).unwrap();
    s.run_script(&json!([
        {"command": "model.set.create", "params": {"name": "both", "geometry": [{"body": bodies[0]}, {"body": bodies[1]}]}},
        {"command": "model.material.create", "params": {"name": "Steel", "elastic": {"type": "isotropic", "young": YOUNG, "poisson": POISSON}, "density": 7.85e-9}},
        {"command": "model.section.create", "params": {"name": "Solid", "set": "both", "material": "Steel", "kind": {"type": "solid"}}},
    ]))
    .unwrap();
    (dir, s)
}

/// The two blocks, held on three sides as by symmetry planes and pulled at the far end: a bar,
/// if the first block holds the second along x.
fn two() -> (TempDir, Session) {
    let (dir, mut s) = opened(TWO, ["First", "Second"]);
    let script = json!([
        {"command": "model.set.create", "params": {"name": "x0", "geometry": [face(&s, "First", [0.0, 5.0, 5.0])]}},
        {"command": "model.set.create", "params": {"name": "end", "geometry": [face(&s, "Second", [35.0, 5.0, 5.0])]}},
        {"command": "model.set.create", "params": {"name": "y0", "geometry": [face(&s, "First", [10.0, 0.0, 5.0]), face(&s, "Second", [27.5, 0.0, 5.0])]}},
        {"command": "model.set.create", "params": {"name": "z0", "geometry": [face(&s, "First", [10.0, 5.0, 0.0]), face(&s, "Second", [27.5, 5.0, 0.0])]}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldX", "type": "symmetry", "target": {"set": "x0"}, "plane": "x"}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldY", "type": "symmetry", "target": {"set": "y0"}, "plane": "y"}},
        {"command": "model.boundary_condition.create", "params": {"name": "HoldZ", "type": "symmetry", "target": {"set": "z0"}, "plane": "z"}},
        {"command": "model.load.create", "params": {"name": "Pull", "type": "pressure", "set": "end", "magnitude": -PULL}},
        {"command": "model.step.create", "params": {"name": "Pull", "procedure": {"type": "static"}, "boundary_conditions": ["HoldX", "HoldY", "HoldZ"], "loads": ["Pull"]}},
    ]);
    s.run_script(&script).unwrap();
    (dir, s)
}

const SOLVE: &str = r#"{"cell_size": 5, "degree": 2}"#;

fn solve(s: &mut Session) -> crate::Result<Value> {
    s.execute("solve.static.grid", &serde_json::from_str(SOLVE).unwrap())
}

/// The first solve of several bodies looks for the faces that touch, bonds them and leaves the
/// connections in the model; the result is of the model as it is then. With `"none"` the model
/// is left alone.
#[test]
fn the_first_solve_finds_the_connections_and_keeps_them() {
    let (_dir, mut s) = two();
    assert!(s.model.connections().is_empty() && !s.model.connections_found());
    let e = s.execute("solve.static.grid", &json!({"cell_size": 5, "degree": 2, "connections": "none"})).unwrap_err().to_string();
    assert!(e.contains("`Second` can still slide along (1, 0, 0)") && e.contains("every body needs supports of its own"), "{e}");
    assert!(s.model.connections().is_empty() && !s.model.connections_found());

    let r = solve(&mut s).unwrap();
    assert_eq!(r["connections_added"], json!(["First-Second"]));
    let found = &r["connections"][0];
    assert_eq!((found["name"].as_str(), found["kind"].as_str()), (Some("First-Second"), Some("bonded")));
    assert_eq!(found["bodies"], json!(["First", "Second"]));
    assert_eq!(found["faces"], json!([[face(&s, "First", [20.0, 5.0, 5.0])["face"]], [face(&s, "Second", [20.0, 5.0, 5.0])["face"]]]));
    assert!(close(found["area"].as_f64().unwrap(), 100.0, 1e-9), "{found}");
    assert!(close(r["max_displacement"]["value"].as_f64().unwrap(), PULL * 35.0 / YOUNG, 2e-2), "{}", r["max_displacement"]);
    assert!(s.model.connections_found() && s.model.connection("First-Second").is_some());
    assert!(s.result("Pull").is_some(), "the result is of the model with the connections the solve found");

    // The next solve finds none: the list is the model's.
    let again = solve(&mut s).unwrap();
    assert_eq!(again["connections_added"], json!([]));
    assert_eq!(again["connections"], r["connections"]);
    // Undoing the first solve takes its connections out of the model again.
    s.execute("edit.undo", &json!({})).unwrap();
    assert!(s.model.connections().is_empty() && !s.model.connections_found());
}

/// Connections are found, changed, deleted and made by hand, and each solve joins the bodies by
/// what the model holds: a deleted connection stays deleted until it is found or made again.
#[test]
fn connections_are_found_changed_deleted_and_made_by_hand() {
    let (_dir, mut s) = two();
    let r = s.execute("interaction.connection.find", &json!({})).unwrap();
    assert_eq!(r["added"], json!(["First-Second"]));
    assert_eq!(r["connections"][0]["kind"], "bonded");
    assert!(s.model.connections_found());
    let again = s.execute("interaction.connection.find", &json!({"kind": "sliding"})).unwrap();
    assert_eq!((again["added"].clone(), again["connections"].as_array().unwrap().len()), (json!([]), 1), "what is there stays as it is");
    let bonded = solve(&mut s).unwrap();
    assert_eq!(bonded["connections_added"], json!([]));
    let travel = PULL * 35.0 / YOUNG;
    assert!(close(bonded["max_displacement"]["value"].as_f64().unwrap(), travel, 2e-2), "{}", bonded["max_displacement"]);

    // Sliding: the faces lie across the pull, so the pull crosses them as before; the solve is
    // linear, and a sliding connection pulls as well as it pushes.
    s.execute("interaction.connection.edit", &json!({"name": "First-Second", "kind": "sliding"})).unwrap();
    assert!(s.result("Pull").is_none() && s.latest_result("Pull").is_some(), "a changed connection makes the result stale");
    let sliding = solve(&mut s).unwrap();
    assert_eq!(sliding["connections"][0]["kind"], "sliding");
    assert!(close(sliding["max_displacement"]["value"].as_f64().unwrap(), travel, 2e-2), "{}", sliding["max_displacement"]);
    assert!(close(sliding["max_von_mises"]["value"].as_f64().unwrap(), PULL, 5e-3), "{}", sliding["max_von_mises"]);

    // Free: listed, and nothing holds the second block along x.
    s.execute("interaction.connection.edit", &json!({"name": "First-Second", "kind": "free"})).unwrap();
    let e = solve(&mut s).unwrap_err().to_string();
    assert!(e.contains("`Second` can still slide along (1, 0, 0)") && e.contains("their connection `First-Second` is free"), "{e}");

    // Deleted: it stays deleted, and the refusal says how to get it back.
    s.execute("interaction.connection.delete", &json!({"name": "First-Second"})).unwrap();
    assert!(s.model.connections().is_empty() && s.model.connections_found());
    let e = solve(&mut s).unwrap_err().to_string();
    assert!(e.contains("it touches `First`, and no connection joins the two: interaction.connection.find"), "{e}");
    assert!(s.model.connections().is_empty(), "a refused solve adds nothing");

    // Made by hand, of faces the user names.
    let (a, b) = (face(&s, "First", [20.0, 5.0, 5.0]), face(&s, "Second", [20.0, 5.0, 5.0]));
    s.execute("interaction.connection", &json!({"name": "Mine", "first": [a.clone()], "second": [b.clone()]})).unwrap();
    assert_eq!(s.model.connection("Mine").unwrap().kind, fieldcraft_model::ConnectionKind::Bonded);
    let mine = solve(&mut s).unwrap();
    assert_eq!((mine["connections"][0]["name"].as_str(), mine["connections_added"].clone()), (Some("Mine"), json!([])));
    assert!(close(mine["max_displacement"]["value"].as_f64().unwrap(), travel, 2e-2), "{}", mine["max_displacement"]);
    // The faces are joined already: a search adds nothing, and one that says the same again is refused.
    assert_eq!(s.execute("interaction.connection.find", &json!({})).unwrap()["added"], json!([]));
    s.execute("interaction.connection", &json!({"name": "Twice", "kind": "sliding", "first": [b.clone()], "second": [a.clone()]})).unwrap();
    let e = solve(&mut s).unwrap_err().to_string();
    assert!(e.contains("connections `Mine` and `Twice` both join"), "{e}");
    s.execute("interaction.connection.delete", &json!({"name": "Twice"})).unwrap();

    // Faces that do not touch join nothing, and the solve says so.
    let far = face(&s, "Second", [35.0, 5.0, 5.0]);
    s.execute("interaction.connection.edit", &json!({"name": "Mine", "second": [far]})).unwrap();
    let e = solve(&mut s).unwrap_err().to_string();
    assert!(e.contains("`Second` can still slide along (1, 0, 0)"), "{e}");
    s.execute("interaction.connection.edit", &json!({"name": "Mine", "second": [b]})).unwrap();
    s.execute("interaction.connection", &json!({"name": "Apart", "first": [a], "second": [face(&s, "Second", [35.0, 5.0, 5.0])]})).unwrap();
    let r = solve(&mut s).unwrap();
    assert!(r["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("connection `Apart` joins nothing")), "{r}");
    assert_eq!(r["connections"].as_array().unwrap().iter().map(|c| c["area"].as_f64().unwrap() > 0.0).collect::<Vec<_>>(), [false, true]);

    // A found connection is named after its two bodies, whatever stands in their names.
    assert_eq!(super::tidy("Clamp body (Clamp:2)"), "Clamp_body_Clamp_2");
    // A renamed connection is the same connection.
    s.execute("model.rename", &json!({"kind": "connection", "from": "Mine", "to": "Glued"})).unwrap();
    assert!(s.model.connection("Glued").is_some() && s.model.connection("Mine").is_none());
}

/// A pin in a hole, free to slide: it can still slide along the hole and turn in it, by name,
/// until it has a support of its own. The hole is off the origin and the geometry is in
/// millimetres: the cylinder the hole's wall lies on must reach the solver in the model's unit,
/// or the facets would hold the pin against turning. Held and pulled, the pin's foot carries the
/// pull and the plate nothing of it along the hole; bonded, the plate carries its share.
#[test]
fn a_pin_slides_and_turns_in_its_hole_until_it_is_held() {
    const FORCE: f64 = 1000.0;
    let (_dir, mut s) = opened(PIN, ["Sleeve", "Pin"]);
    let script = json!([
        {"command": "model.set.create", "params": {"name": "under", "geometry": [level(&s, "Sleeve", 0.0)]}},
        {"command": "model.set.create", "params": {"name": "foot", "geometry": [level(&s, "Pin", -5.0)]}},
        {"command": "model.set.create", "params": {"name": "top", "geometry": [level(&s, "Pin", 20.0)]}},
        {"command": "model.boundary_condition.create", "params": {"name": "Clamp", "type": "fixed", "target": {"set": "under"}}},
        {"command": "model.boundary_condition.create", "params": {"name": "Foot", "type": "fixed", "target": {"set": "foot"}}},
        {"command": "model.load.create", "params": {"name": "Pull", "type": "force", "target": {"set": "top"}, "vector": [0, 0, FORCE]}},
        {"command": "model.step.create", "params": {"name": "Loose", "procedure": {"type": "static"}, "boundary_conditions": ["Clamp"], "loads": ["Pull"]}},
        {"command": "model.step.create", "params": {"name": "Held", "procedure": {"type": "static"}, "boundary_conditions": ["Clamp", "Foot"], "loads": ["Pull"]}},
    ]);
    s.run_script(&script).unwrap();
    let found = s.execute("interaction.connection.find", &json!({})).unwrap();
    assert_eq!(found["added"], json!(["Sleeve-Pin"]), "{found}");
    assert_eq!(found["connections"][0]["faces"].as_array().unwrap().iter().map(|side| side.as_array().unwrap().len()).collect::<Vec<_>>(), [1, 1]);

    let reaction = |r: &Value, body: &str| -> f64 {
        r["reactions"].as_array().unwrap().iter().filter(|x| x["body"] == body).map(|x| x["force"][2].as_f64().unwrap()).sum()
    };
    let grid = |step: &str| json!({"step": step, "cell_size": 4, "degree": 2});
    // Bonded, the plate takes its share of the pull, with or without the pin's own support.
    let bonded = s.execute("solve.static.grid", &grid("Held")).unwrap();
    assert!(reaction(&bonded, "Sleeve") < -0.1 * FORCE, "{}", bonded["reactions"]);
    s.execute("solve.static.grid", &grid("Loose")).unwrap();

    s.execute("interaction.connection.edit", &json!({"name": "Sleeve-Pin", "kind": "sliding"})).unwrap();
    let e = s.execute("solve.static.grid", &grid("Loose")).unwrap_err().to_string();
    assert!(e.contains("`Pin` can still ") && e.contains("slide along (0, 0, 1)") && e.contains("turn about an axis along (0, 0, 1)"), "{e}");
    let sliding = s.execute("solve.static.grid", &grid("Held")).unwrap();
    let (on_plate, on_foot) = (reaction(&sliding, "Sleeve"), reaction(&sliding, "Pin"));
    assert!(on_plate.abs() < 1e-6 * FORCE && close(-on_foot, FORCE, 1e-6), "plate {on_plate}, foot {on_foot}");
    assert!(sliding["warnings"].as_array().unwrap().is_empty(), "{sliding}");
}

/// Whatever the parameters, the commands refuse or run; none panics, and what is refused
/// changes nothing.
#[test]
fn hostile_connections_never_panic() {
    let (_dir, mut s) = two();
    s.execute("interaction.connection.find", &json!({})).unwrap();
    let (a, b) = (face(&s, "First", [20.0, 5.0, 5.0]), face(&s, "Second", [20.0, 5.0, 5.0]));
    let before = s.model.clone();
    let long = "x".repeat(100_000);
    let refused = [
        ("interaction.connection", json!(null)),
        ("interaction.connection", json!([])),
        ("interaction.connection", json!({})),
        ("interaction.connection", json!({"name": "Ok"})),
        ("interaction.connection", json!({"name": "Ok", "first": [a.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [a.clone()], "second": [a.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [a.clone(), b.clone()], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [a.clone(), a.clone()], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [{"body": "First"}], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [{"body": "First", "edge": "E1"}], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [{"body": "Nobody", "face": "F1"}], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [{"body": "First", "face": "no such face"}], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": [{"body": long, "face": long}], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "first": "F1", "second": 7})),
        ("interaction.connection", json!({"name": "Ok", "kind": "welded", "first": [a.clone()], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "Ok", "kind": 7, "first": [a.clone()], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "not a name", "first": [a.clone()], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": long, "first": [a.clone()], "second": [b.clone()]})),
        ("interaction.connection", json!({"name": "First-Second", "first": [a.clone()], "second": [b.clone()]})),
        ("interaction.connection.edit", json!({})),
        ("interaction.connection.edit", json!({"name": "Nobody", "kind": "free"})),
        ("interaction.connection.edit", json!({"name": "First-Second", "kind": "glued"})),
        ("interaction.connection.edit", json!({"name": "First-Second", "first": []})),
        ("interaction.connection.edit", json!({"name": "First-Second", "second": [a.clone()]})),
        ("interaction.connection.edit", json!({"name": "First-Second", "first": [{"body": "First", "face": long}]})),
        ("interaction.connection.delete", json!({})),
        ("interaction.connection.delete", json!({"name": "Nobody"})),
        ("interaction.connection.delete", json!({"name": 7})),
        ("interaction.connection.find", json!({"tolerance": 0})),
        ("interaction.connection.find", json!({"tolerance": -1})),
        ("interaction.connection.find", json!({"tolerance": "near"})),
        ("interaction.connection.find", json!({"kind": "welded"})),
        ("interaction.connection.find", json!({"kind": null, "tolerance": [1]})),
    ];
    for (command, params) in &refused {
        let shown: String = params.to_string().chars().take(200).collect();
        assert!(s.execute(command, params).is_err(), "{command} {shown}");
        assert!(s.model == before, "{command} {shown} changed the model");
    }
    // Tolerances from far too small to far too large find nothing new and fail on nothing.
    for tolerance in [1e-300, 1e-9, 1e-3, 1.0, 1e3, 1e300] {
        let r = s.execute("interaction.connection.find", &json!({"tolerance": tolerance}));
        assert!(r.is_ok_and(|r| r["added"] == json!([])) || tolerance > 1.0 || tolerance < 1e-6, "tolerance {tolerance}");
    }
    // A model that names what a design does not have is refused by the solve, by name.
    let mut other = before.clone();
    let gone = fieldcraft_model::Connection {
        kind: fieldcraft_model::ConnectionKind::Bonded,
        first: vec![fieldcraft_model::GeometryRef::face("First", "gone").unwrap()],
        second: vec![fieldcraft_model::GeometryRef::face("Second", "gone").unwrap()],
    };
    other.add_connection("Stale", gone).unwrap();
    s.model = other;
    let e = solve(&mut s).unwrap_err().to_string();
    assert!(e.contains("connection `Stale` names face `gone` of body `First`"), "{e}");
}
