//! Connections: how two bodies of an assembly hold each other where faces of theirs touch.
//!
//! A connection names faces of one body and faces of another, by their persistent names, and a
//! kind: `bonded` (the two move together where the faces touch, as if glued or welded there),
//! `sliding` (they keep touching and slide along each other without friction) or `free` (they
//! do not hold each other: the connection records that the faces were looked at and left
//! apart). Where the faces touch is for the solver to find, so a connection may name whole
//! faces of which only parts touch.
//!
//! The solve on a grid joins the bodies of a design by its connections; the solvers on a mesh
//! join element faces, by ties and contacts. The connections are found from the geometry once
//! ([`Model::connections_found`] says that they were), and from then on the list is the user's:
//! a connection that was deleted stays deleted.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::geometry::{Entity, GeometryRef};
use crate::{Error, Model, Result, context, insert_new, remove_existing, replace_existing};

/// Most connections in a model.
pub const MAX_CONNECTIONS: usize = 4096;

/// Most faces a connection names of one body.
pub const MAX_CONNECTION_FACES: usize = 10_000;

/// How two bodies hold each other where their faces touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionKind {
    /// They move together.
    #[default]
    Bonded,
    /// They keep touching and slide along each other without friction: neither can leave the
    /// other or enter it, and nothing is carried along the faces.
    Sliding,
    /// They do not hold each other.
    Free,
}

impl ConnectionKind {
    /// The kind as commands and files name it.
    pub fn word(self) -> &'static str {
        match self {
            ConnectionKind::Bonded => "bonded",
            ConnectionKind::Sliding => "sliding",
            ConnectionKind::Free => "free",
        }
    }
}

/// A connection between faces of two bodies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Connection {
    #[serde(default)]
    pub kind: ConnectionKind,
    /// Faces of one body.
    pub first: Vec<GeometryRef>,
    /// Faces of another body.
    pub second: Vec<GeometryRef>,
}

/// The body all the faces of one side belong to.
fn side_body<'a>(which: &str, faces: &'a [GeometryRef]) -> Result<&'a str> {
    let Some(one) = faces.first() else {
        return Err(Error::Invalid(format!("a connection names faces of two bodies; `{which}` is empty")));
    };
    if faces.len() > MAX_CONNECTION_FACES {
        return Err(Error::Limit(format!("a connection names at most {MAX_CONNECTION_FACES} faces of a body")));
    }
    let mut seen = BTreeSet::new();
    for face in faces {
        if !matches!(face.entity, Entity::Face(_)) {
            return Err(Error::Invalid(format!("a connection joins faces; `{which}` names {face}")));
        }
        if face.body != one.body {
            return Err(Error::Invalid(format!("the faces in `{which}` are of one body; they name `{}` and `{}`", one.body, face.body)));
        }
        if !seen.insert(face) {
            return Err(Error::Invalid(format!("`{which}` names {face} twice")));
        }
    }
    Ok(&one.body)
}

impl Connection {
    pub fn validate(&self) -> Result<()> {
        let (first, second) = (side_body("first", &self.first)?, side_body("second", &self.second)?);
        if first == second {
            return Err(Error::Invalid(format!("a connection joins two bodies; both sides name faces of `{first}`")));
        }
        Ok(())
    }

    /// The two bodies (empty names for a connection that is not valid).
    pub fn bodies(&self) -> [&str; 2] {
        [&self.first, &self.second].map(|side| side.first().map_or("", |face| face.body.as_str()))
    }

    /// The names of the faces of each body.
    pub fn faces(&self) -> [Vec<&str>; 2] {
        [&self.first, &self.second].map(|side| side.iter().filter_map(|face| face.entity.name()).collect())
    }

    /// Whether the connection has this face of this body on the one side and that face of that
    /// body on the other, whichever is first.
    pub fn joins(&self, a: &GeometryRef, b: &GeometryRef) -> bool {
        (self.first.contains(a) && self.second.contains(b)) || (self.first.contains(b) && self.second.contains(a))
    }
}

impl Model {
    pub fn connections(&self) -> &BTreeMap<String, Connection> {
        &self.connections
    }

    pub fn connection(&self, name: &str) -> Option<&Connection> {
        self.connections.get(name)
    }

    /// Whether the design's touching faces were searched for connections: the list is then the
    /// user's, and a solve joins the bodies by what it holds and adds nothing to it.
    pub fn connections_found(&self) -> bool {
        self.connections_found
    }

    pub fn set_connections_found(&mut self, found: bool) {
        self.connections_found = found;
    }

    pub fn add_connection(&mut self, name: &str, connection: Connection) -> Result<()> {
        connection.validate().map_err(|e| context("connection", name, e))?;
        if self.connections.len() >= MAX_CONNECTIONS {
            return Err(Error::Limit(format!("a model holds at most {MAX_CONNECTIONS} connections")));
        }
        insert_new(&mut self.connections, "connection", name, connection)
    }

    pub fn replace_connection(&mut self, name: &str, connection: Connection) -> Result<()> {
        connection.validate().map_err(|e| context("connection", name, e))?;
        replace_existing(&mut self.connections, "connection", name, connection)
    }

    pub fn remove_connection(&mut self, name: &str) -> Result<()> {
        remove_existing(&mut self.connections, "connection", name).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(body: &str, name: &str) -> GeometryRef {
        GeometryRef::face(body, name).unwrap()
    }

    #[test]
    fn connections_validate_and_round_trip() {
        let mut m = Model::default();
        let glue = Connection { kind: ConnectionKind::Bonded, first: vec![face("Base", "F1:+z")], second: vec![face("Clamp", "F4:-z")] };
        m.add_connection("glue", glue.clone()).unwrap();
        assert!(m.add_connection("glue", glue.clone()).is_err(), "names stay unique");
        assert_eq!(glue.bodies(), ["Base", "Clamp"]);
        assert_eq!(glue.faces(), [vec!["F1:+z"], vec!["F4:-z"]]);
        assert!(glue.joins(&face("Clamp", "F4:-z"), &face("Base", "F1:+z")) && !glue.joins(&face("Base", "F1:+z"), &face("Clamp", "F9")));
        let bad = [
            Connection { first: vec![], ..glue.clone() },
            Connection { second: vec![], ..glue.clone() },
            Connection { second: vec![face("Base", "F2")], ..glue.clone() },
            Connection { first: vec![face("Base", "F1:+z"), face("Clamp", "F2")], ..glue.clone() },
            Connection { first: vec![face("Base", "F1:+z"), face("Base", "F1:+z")], ..glue.clone() },
            Connection { first: vec![GeometryRef::body("Base").unwrap()], ..glue.clone() },
            Connection { first: vec![GeometryRef::new("Base", Entity::Edge("E1".into())).unwrap()], ..glue.clone() },
        ];
        for connection in bad {
            assert!(matches!(m.add_connection("bad", connection.clone()), Err(Error::Invalid(_))), "{connection:?}");
            assert!(m.replace_connection("glue", connection).is_err());
        }
        assert!(!m.connections_found());
        m.set_connections_found(true);
        m.replace_connection("glue", Connection { kind: ConnectionKind::Sliding, ..glue.clone() }).unwrap();
        let text = serde_json::to_string(&m).unwrap();
        assert!(text.contains(r#""kind":"sliding""#) && text.contains(r#""connections_found":true"#), "{text}");
        let back: Model = serde_json::from_str(&text).unwrap();
        assert_eq!(back, m);
        // A kind left out is bonded, and a model without connections writes nothing of them.
        let read: Connection = serde_json::from_str(r#"{"first": [{"body": "A", "face": "F1"}], "second": [{"body": "B", "face": "F2"}]}"#).unwrap();
        assert_eq!(read.kind, ConnectionKind::Bonded);
        assert!(!serde_json::to_string(&Model::default()).unwrap().contains("connection"));
        m.remove_connection("glue").unwrap();
        assert!(m.remove_connection("glue").is_err() && m.connections_found(), "deleting keeps the search made");
    }

    #[test]
    fn a_renamed_face_or_connection_is_followed() {
        let mut m = Model::default();
        let glue = Connection { kind: ConnectionKind::Free, first: vec![face("Base", "F1")], second: vec![face("Clamp", "F4"), face("Clamp", "F5")] };
        m.add_connection("glue", glue).unwrap();
        assert_eq!(m.name_uses().iter().filter(|u| u.owner == "connection `glue`").count(), 3);
        assert_eq!(m.rename_entity(&face("Clamp", "F4"), &face("Clamp", "F9")).unwrap(), 1);
        assert_eq!(m.connection("glue").unwrap().faces()[1], ["F9", "F5"]);
        // Two names that would become one are refused.
        assert!(m.rename_entity(&face("Clamp", "F9"), &face("Clamp", "F5")).is_err());
        m.rename(crate::rename::Kind::Connection, "glue", "weld").unwrap();
        assert!(m.connection("weld").is_some() && m.connection("glue").is_none());
    }
}
