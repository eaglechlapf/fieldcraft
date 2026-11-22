//! `interaction.connection*`: how the bodies of the open design hold each other where faces of
//! theirs touch ([`fieldcraft_model::connections`]): bonded, sliding or free.
//!
//! `find` looks for the faces of every two bodies that lie on each other
//! ([`fieldcraft_fcm::find_bonds`]) and adds a connection for those that none joins yet; a solve
//! of several bodies on a grid (`solve.static.grid`) does the same the first time, so a design is
//! solved on the first click with every two bodies that touch bonded. From then on the list is
//! the user's: `edit` changes a connection's kind or faces, `delete` removes it, and
//! `interaction.connection` makes one of faces the user names.

use std::collections::BTreeSet;

use fieldcraft_fcm::{Touch, find_bonds};
use fieldcraft_geometry::Shape;
use fieldcraft_mesh::surface::BoundarySurface;
use fieldcraft_model::{Connection, ConnectionKind, GeometryRef, MAX_NAME_LEN, Model, suggest_name};
use serde_json::{Value, json};

use super::{name, opt, req};
use crate::requirement::Requirement;
use crate::{CommandKind, CommandSpec, EngineError, Result, Session};

pub const CONNECTION: CommandSpec = CommandSpec {
    id: "interaction.connection",
    label: "Connection",
    params: r#"{"name": "Clamp_on_base", "kind": "bonded" | "sliding" | "free" (default bonded), "first": [{"body": "Base", "face": "F1:+z"}] (faces of one body of the open design), "second": [{"body": "Clamp", "face": "F4:-z"}] (faces of another)} — connects two bodies where the named faces touch, for a solve of several bodies on a grid (solve.static.grid). `bonded`: the two move together there, as if glued or welded. `sliding`: they keep touching and slide along each other without friction (a part in a pocket, a pin in a hole): neither leaves the other or enters it, and nothing is carried along the faces; the solve is linear, so a sliding connection pulls as well as it pushes. `free`: they do not hold each other. Whole faces are named; where they touch is found by the solve, and faces that touch in part are connected over the part. Fails if the name is taken, a body or face is not the design's, or both sides name one body"#,
    form: Some(super::forms::CONNECTION),
    kind: CommandKind::Model,
    requires: &[Requirement::Design],
    run: |s: &mut Session, p: &Value| {
        let n = name(p)?;
        let connection =
            Connection { kind: opt::<ConnectionKind>(p, "kind")?.unwrap_or_default(), first: req(p, "first")?, second: req(p, "second")? };
        in_design(s, &connection)?;
        s.model.add_connection(n, connection)?;
        Ok(json!({"name": n}))
    },
};

pub const EDIT: CommandSpec = CommandSpec {
    id: "interaction.connection.edit",
    label: "Edit Connection",
    params: r#"{"name": "Clamp_on_base", "kind"?: "bonded" | "sliding" | "free", "first"?: [{"body": "Base", "face": "F1:+z"}], "second"?: [{"body": "Clamp", "face": "F4:-z"}]} — changes a connection: the parameters given replace its own, the rest stay (see interaction.connection). `{"name": …, "kind": "sliding"}` lets two bonded bodies slide"#,
    form: Some(super::forms::CONNECTION_EDIT),
    kind: CommandKind::Model,
    requires: &[Requirement::Connection],
    run: |s: &mut Session, p: &Value| {
        let n = name(p)?;
        let mut connection = s.model.connection(n).cloned().ok_or_else(|| EngineError::Params(format!("no connection `{n}`")))?;
        if let Some(kind) = opt::<ConnectionKind>(p, "kind")? {
            connection.kind = kind;
        }
        if let Some(first) = opt::<Vec<GeometryRef>>(p, "first")? {
            connection.first = first;
        }
        if let Some(second) = opt::<Vec<GeometryRef>>(p, "second")? {
            connection.second = second;
        }
        in_design(s, &connection)?;
        s.model.replace_connection(n, connection)?;
        Ok(json!({"name": n}))
    },
};

pub const DELETE: CommandSpec = CommandSpec {
    id: "interaction.connection.delete",
    label: "Delete Connection",
    params: r#"{"name": "Clamp_on_base"} — removes the connection: the two bodies no longer hold each other there. It stays deleted: a solve looks for the connections of a design once and adds none after that (interaction.connection.find adds those that are missing again)"#,
    form: Some(super::forms::CONNECTION_DELETE),
    kind: CommandKind::Model,
    requires: &[Requirement::Connection],
    run: |s: &mut Session, p: &Value| {
        s.model.remove_connection(name(p)?)?;
        // What is left is what the user wants: no solve searches the design again.
        s.model.set_connections_found(true);
        Ok(Value::Null)
    },
};

pub const FIND: CommandSpec = CommandSpec {
    id: "interaction.connection.find",
    label: "Find Connections",
    params: r#"{"tolerance": 0.05 (a length: faces closer than this touch; default 1/5000 of the largest body's bounding-box diagonal, twice what a solve at its defaults triangulates that body within), "kind": "bonded" | "sliding" | "free" (of the connections it adds; default bonded)} — looks for the faces of every two bodies of the open design that lie on each other, facing each other: flat faces that touch, also in part, and curved ones (a pin in its hole) within the tolerance; an edge or a corner resting on a face is none. For the touching faces of two bodies that no connection joins yet a connection is added, named after the two bodies; connections that are there stay as they are, whatever their kind. Replies the names `added` and every connection of the model (`name`, `kind`, `bodies`, `faces`). A solve of several bodies on a grid runs this once by itself, on a model that was never searched; run it again after the design changed, or to get back connections that were deleted"#,
    form: Some(super::forms::CONNECTION_FIND),
    kind: CommandKind::Model,
    requires: &[Requirement::Design],
    run: find,
};

/// Refuse a connection that names a body or a face the open design does not have.
fn in_design(s: &Session, connection: &Connection) -> Result<()> {
    let Some(design) = s.design.as_deref() else { return Ok(()) };
    for (key, side) in [("first", &connection.first), ("second", &connection.second)] {
        for face in side {
            let Some(shape) = design.body(&face.body) else {
                return Err(EngineError::invalid(key, format!("the open design has no body `{}`", cut(&face.body))));
            };
            if let Some(named) = face.entity.name()
                && !shape.faces().is_empty()
                && shape.face_index(named).is_none()
            {
                return Err(EngineError::invalid(key, format!("body `{}` has no face `{}`", cut(&face.body), cut(named))));
            }
        }
    }
    Ok(())
}

/// A name from the model or the request, cut for a message.
fn cut(name: &str) -> String {
    name.chars().take(100).collect()
}

/// A body's name as a part of a definition's: its letters and digits, with one `_` for whatever
/// stands between them (`Clamp body (Clamp:2)` is `Clamp_body_Clamp_2`).
fn tidy(name: &str) -> String {
    let words: Vec<&str> = name.split(|c: char| !c.is_ascii_alphanumeric()).filter(|word| !word.is_empty()).collect();
    words.join("_")
}

/// A connection as replies list it.
pub(super) fn listed(name: &str, connection: &Connection) -> Value {
    json!({"name": name, "kind": connection.kind.word(), "bodies": connection.bodies(), "faces": connection.faces()})
}

/// The connections to add for faces that touch: for every two bodies of `shapes` (in the order
/// of the surfaces the `touches` were found on), one that joins the pairs of faces no
/// connection of the model joins yet, of the kind `kind`, under a name of its own.
pub(super) fn new_connections(model: &Model, shapes: &[&Shape], touches: &[Touch], kind: ConnectionKind) -> Result<Vec<(String, Connection)>> {
    let mut added: Vec<(String, Connection)> = Vec::new();
    for touch in touches {
        let (Some(a), Some(b)) = (shapes.get(touch.bond.first.0), shapes.get(touch.bond.second.0)) else { continue };
        let face = |shape: &Shape, tag: u32| {
            shape.faces().get(tag as usize).and_then(|face| GeometryRef::face(shape.name.as_str(), face.name.as_str()).ok())
        };
        let (mut first, mut second) = (BTreeSet::new(), BTreeSet::new());
        for (fa, fb, _) in &touch.pairs {
            let (Some(on_a), Some(on_b)) = (face(a, *fa), face(b, *fb)) else { continue };
            if model.connections().values().any(|connection| connection.joins(&on_a, &on_b)) {
                continue;
            }
            first.insert((*fa, on_a));
            second.insert((*fb, on_b));
        }
        if first.is_empty() {
            continue;
        }
        // A name after the two bodies, with a number when it is taken.
        let base = suggest_name(&format!("{}-{}", tidy(&a.name), tidy(&b.name))).unwrap_or_else(|| "Connection".into());
        let taken = |name: &str| model.connection(name).is_some() || added.iter().any(|(other, _)| other == name);
        let mut name = base.clone();
        // One more number than there are names taken ends it.
        for number in 2..model.connections().len() + added.len() + 3 {
            if !taken(&name) {
                break;
            }
            let suffix = format!("-{number}");
            name = format!("{}{suffix}", base.chars().take(MAX_NAME_LEN.saturating_sub(suffix.len())).collect::<String>());
        }
        let connection =
            Connection { kind, first: first.into_iter().map(|(_, face)| face).collect(), second: second.into_iter().map(|(_, face)| face).collect() };
        connection.validate()?;
        added.push((name, connection));
    }
    Ok(added)
}

fn find(s: &mut Session, p: &Value) -> Result<Value> {
    fieldcraft_model::progress::check()?;
    let kind = opt::<ConnectionKind>(p, "kind")?.unwrap_or_default();
    let asked = opt::<f64>(p, "tolerance")?;
    if let Some(tolerance) = asked
        && !(tolerance > 0.0 && tolerance.is_finite())
    {
        return Err(EngineError::invalid("tolerance", "must be a positive length".to_string()));
    }
    let added = {
        let design = s.design.as_deref().ok_or_else(|| EngineError::Params("no design is open (geometry.open)".into()))?;
        let shapes: Vec<&Shape> = design.bodies.iter().collect();
        if shapes.len() > fieldcraft_fcm::assembly::MAX_PARTS {
            return Err(EngineError::Params(format!(
                "the design has {} bodies; connections are found among up to {}",
                shapes.len(),
                fieldcraft_fcm::assembly::MAX_PARTS
            )));
        }
        // Faces within `reach` of each other touch. Two triangulations of one curved face lie
        // each within its own tolerance of it, so each body is triangulated within half the
        // reach; with none asked for, within the tolerance of a solve at its defaults, and the
        // reach is twice the largest of those, as it is in the solve.
        let (mut surfaces, mut reach): (Vec<BoundarySurface>, f64) = (Vec::with_capacity(shapes.len()), asked.unwrap_or(0.0));
        for shape in &shapes {
            fieldcraft_model::progress::stage(&format!("{}: surface", shape.name), None, "");
            fieldcraft_model::progress::check()?;
            let diagonal = super::static_grid::diagonal(shape)?;
            // No finer than a body can be triangulated.
            let tolerance = asked
                .map_or(super::static_grid::default_tolerance(diagonal), |reach| 0.5 * reach)
                .max(super::static_grid::MIN_RELATIVE_TOLERANCE * diagonal);
            if asked.is_none() {
                reach = reach.max(2.0 * tolerance);
            }
            surfaces.push(super::static_grid::surface(shape, tolerance)?);
        }
        fieldcraft_model::progress::stage("connections", None, "");
        fieldcraft_model::progress::check()?;
        let touches = find_bonds(&surfaces.iter().collect::<Vec<_>>(), reach).map_err(super::static_grid::grid_error)?;
        new_connections(&s.model, &shapes, &touches, kind)?
    };
    let names: Vec<String> = added.iter().map(|(name, _)| name.clone()).collect();
    for (name, connection) in added {
        s.model.add_connection(&name, connection)?;
    }
    s.model.set_connections_found(true);
    Ok(json!({
        "added": names,
        "connections": s.model.connections().iter().map(|(name, connection)| listed(name, connection)).collect::<Vec<_>>(),
    }))
}

#[cfg(test)]
mod tests;
