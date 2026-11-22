//! What the viewport hides: `view.hide`, `view.show`, `view.isolate` and `view.show_all`.
//!
//! The hidden items (bodies, imported surfaces, datums, boundary conditions, loads, reference
//! points, couplings) are view state ([`fieldcraft_io::View`]): saved with the model and marking
//! it unsaved, but not a definition, so they are neither journalled nor undone and they leave
//! results current. A hidden item is not drawn, so it cannot be picked either. An item that goes
//! away (deleted, or a body cut into cells) is forgotten.

use std::collections::BTreeSet;

use fieldcraft_io::{Item, ItemKind, MAX_HIDDEN};
use fieldcraft_render::Vec3;
use serde_json::Value;

use crate::{CommandKind, CommandSpec, EngineError, Result, Session};

pub const HIDE: CommandSpec = CommandSpec {
    id: "view.hide",
    label: "Hide",
    params: concat!(
        r#"{"items": [{"kind": "body" | "surface" | "datum" | "boundary_condition" | "load" | "reference_point" | "coupling", "name": "Sym"}]}"#,
        " — hides the items in the viewport (and from picking); view state, saved with the model"
    ),
    form: Some(super::forms::VIEW_ITEMS),
    kind: CommandKind::Session,
    requires: &[],
    run: |s, p| change(s, p, Change::Hide),
};

pub const SHOW: CommandSpec = CommandSpec {
    id: "view.show",
    label: "Show",
    params: concat!(
        r#"{"items": [{"kind": "body" | "surface" | "datum" | "boundary_condition" | "load" | "reference_point" | "coupling", "name": "Sym"}]}"#,
        " — shows hidden items again"
    ),
    form: Some(super::forms::VIEW_ITEMS),
    kind: CommandKind::Session,
    requires: &[],
    run: |s, p| change(s, p, Change::Show),
};

pub const ISOLATE: CommandSpec = CommandSpec {
    id: "view.isolate",
    label: "Isolate",
    params: concat!(
        r#"{"items": [{"kind": "body" | "surface" | "datum" | "boundary_condition" | "load" | "reference_point" | "coupling", "name": "Sym"}]}"#,
        " — shows only these items: every other one is hidden (view.show_all brings them back)"
    ),
    form: Some(super::forms::VIEW_ITEMS),
    kind: CommandKind::Session,
    requires: &[],
    run: |s, p| change(s, p, Change::Isolate),
};

pub const SHOW_ALL: CommandSpec = CommandSpec {
    id: "view.show_all",
    label: "Show All",
    params: "{}",
    form: Some(&[]),
    kind: CommandKind::Session,
    requires: &[],
    run: |s, _| {
        s.view.hidden.clear();
        Ok(Value::Null)
    },
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Change {
    Hide,
    Show,
    Isolate,
}

/// Every item of `kind` the session has, by name.
pub(crate) fn names(s: &Session, kind: ItemKind) -> Vec<String> {
    let m = &s.model;
    match kind {
        ItemKind::Body => s.design.iter().flat_map(|d| d.bodies.iter().map(|b| b.name.clone())).collect(),
        ItemKind::Surface => s.surfaces.iter().map(|x| x.name.clone()).collect(),
        ItemKind::Datum => m.datums().keys().cloned().collect(),
        ItemKind::BoundaryCondition => m.boundary_conditions().keys().cloned().collect(),
        ItemKind::Load => m.loads().keys().cloned().collect(),
        ItemKind::ReferencePoint => m.reference_points().keys().cloned().collect(),
        ItemKind::Coupling => m.couplings().keys().cloned().collect(),
    }
}

fn exists(s: &Session, item: &Item) -> bool {
    let m = &s.model;
    match item.kind {
        ItemKind::Body => s.design.as_ref().is_some_and(|d| d.body(&item.name).is_some()),
        ItemKind::Surface => s.surfaces.iter().any(|x| x.name == item.name),
        ItemKind::Datum => m.datum(&item.name).is_some(),
        ItemKind::BoundaryCondition => m.boundary_conditions().contains_key(&item.name),
        ItemKind::Load => m.loads().contains_key(&item.name),
        ItemKind::ReferencePoint => m.reference_points().contains_key(&item.name),
        ItemKind::Coupling => m.couplings().contains_key(&item.name),
    }
}

/// Forget hidden items the session no longer has.
pub(crate) fn forget_missing(s: &mut Session) {
    if s.view.hidden.is_empty() {
        return;
    }
    let gone: Vec<Item> = s.view.hidden.iter().filter(|i| !exists(s, i)).cloned().collect();
    for i in gone {
        s.view.hidden.remove(&i);
    }
}

/// A body's surface as the caller draws it, tessellated, in the model's metres: where the
/// symbols of a setup go on a body with no mesh ([`Session::glyphs_on`]).
#[derive(Clone, Copy)]
pub struct BodySurface<'a> {
    pub body: &'a str,
    /// The names of the body's faces, by the index its triangles carry.
    pub faces: &'a [String],
    pub positions: &'a [[f64; 3]],
    /// Counter-clockwise seen from outside.
    pub triangles: &'a [[u32; 3]],
    pub tri_face: &'a [u32],
}

/// The places a set's symbols may go on the faces it names, about: more than are ever drawn of
/// one definition ([`fieldcraft_render::glyphs::MAX_PER_DEFINITION`]), so that a close view of
/// a part of the faces still finds some.
const PLACES: usize = 1500;
/// Places along a triangle's side, at most, and triangles of a set's faces that get any.
const MOST_ALONG: usize = 48;
const MOST_TRIANGLES: usize = 20_000;

/// Points spread evenly over the faces set `set` names on `surfaces`, each with the face's
/// outward direction there: about [`PLACES`] of them, by a lattice on every triangle as fine as
/// the triangle's share of the area asks.
fn places(model: &fieldcraft_model::Model, surfaces: &[BodySurface], set: &str) -> Vec<(Vec3, Vec3)> {
    use fieldcraft_model::Entity;
    let Some(definition) = model.sets().get(set) else { return Vec::new() };
    let mut triangles: Vec<([Vec3; 3], Vec3, f64)> = Vec::new();
    for surface in surfaces {
        let mut wanted = vec![false; surface.faces.len()];
        for reference in definition.geometry.iter().filter(|g| g.body == surface.body) {
            match &reference.entity {
                Entity::Body => wanted.fill(true),
                Entity::Face(name) => {
                    if let Some(flag) = surface.faces.iter().position(|face| face == name).and_then(|at| wanted.get_mut(at)) {
                        *flag = true;
                    }
                }
                Entity::Edge(_) | Entity::Vertex(_) => {}
            }
        }
        if !wanted.contains(&true) {
            continue;
        }
        let on_faces = surface.triangles.iter().zip(surface.tri_face).filter(|(_, face)| wanted.get(**face as usize) == Some(&true));
        for (triangle, _) in on_faces.take(MOST_TRIANGLES.saturating_sub(triangles.len())) {
            let corner = |i: u32| surface.positions.get(i as usize).map(|p| Vec3::from(*p)).filter(|p| p.is_finite());
            let (Some(a), Some(b), Some(c)) = (corner(triangle[0]), corner(triangle[1]), corner(triangle[2])) else { continue };
            let cross = (b - a).cross(c - a);
            if let Some(outward) = cross.normalized() {
                triangles.push(([a, b, c], outward, 0.5 * cross.len()));
            }
        }
    }
    let area: f64 = triangles.iter().map(|(_, _, area)| area).sum();
    if !(area > 0.0) || !area.is_finite() {
        return Vec::new();
    }
    // A lattice of this step over all the faces has about that many points.
    let step = (area / PLACES as f64).sqrt();
    let mut found = Vec::new();
    for ([a, b, c], outward, area) in &triangles {
        let along = (((2.0 * area).sqrt() / step).ceil() as usize).clamp(1, MOST_ALONG);
        for i in 0..along {
            for j in 0..along - i {
                let (u, v) = ((i as f64 + 1.0 / 3.0) / along as f64, (j as f64 + 1.0 / 3.0) / along as f64);
                found.push((*a + (*b - *a) * u + (*c - *a) * v, *outward));
            }
        }
    }
    found
}

impl Session {
    /// The analysis setup's symbols, without the definitions the view hides.
    pub fn glyphs(&self) -> fieldcraft_render::glyphs::Glyphs {
        self.glyphs_of(None, &[])
    }

    /// [`Session::glyphs`] with the symbols of bodies that have no mesh on their surfaces as
    /// drawn (`surfaces`: the open design's bodies, tessellated): a setup that is solved
    /// without a mesh shows its supports and loads on the faces their sets name.
    pub fn glyphs_on(&self, surfaces: &[BodySurface]) -> fieldcraft_render::glyphs::Glyphs {
        self.glyphs_of(None, surfaces)
    }

    /// The symbols of what solved step `step` used, over its results: its own boundary
    /// conditions and loads (none for a frequency step, which has no loads) and its own initial
    /// velocities and rigid walls; the interactions as always. Without the view's hidden items.
    pub fn step_glyphs(&self, step: &str) -> fieldcraft_render::glyphs::Glyphs {
        self.glyphs_of(Some(step), &[])
    }

    /// [`Session::step_glyphs`] with the symbols of bodies that have no mesh on `surfaces`
    /// ([`Session::glyphs_on`]).
    pub fn step_glyphs_on(&self, step: &str, surfaces: &[BodySurface]) -> fieldcraft_render::glyphs::Glyphs {
        self.glyphs_of(Some(step), surfaces)
    }

    fn glyphs_of(&self, step: Option<&str>, surfaces: &[BodySurface]) -> fieldcraft_render::glyphs::Glyphs {
        use fieldcraft_model::Procedure;
        use fieldcraft_render::glyphs::{Definition, Places};
        let step = step.map(|name| self.model.step(name));
        let frequency = matches!(step, Some(Some(s)) if matches!(s.procedure, Procedure::Frequency { .. }));
        // The bodies shown that have no mesh: their symbols go on their surfaces.
        let bare: Vec<BodySurface> = surfaces
            .iter()
            .filter(|surface| self.shows(ItemKind::Body, surface.body) && self.model.body_elements(surface.body).is_empty())
            .copied()
            .collect();
        let mut bounds = fieldcraft_render::Aabb3::EMPTY;
        for p in bare.iter().flat_map(|surface| surface.positions) {
            bounds.add(Vec3::from(*p));
        }
        let of_set = |set: &str| places(&self.model, &bare, set);
        fieldcraft_render::glyphs::Glyphs::from_model_on_hiding(&self.model, &Places { bounds, of_set: &of_set }, &|definition, name| {
            let (kind, used) = match definition {
                Definition::BoundaryCondition => (ItemKind::BoundaryCondition, step.map(|s| s.is_some_and(|s| s.boundary_conditions.contains(name)))),
                Definition::Load => (ItemKind::Load, step.map(|s| !frequency && s.is_some_and(|s| s.loads.contains(name)))),
                Definition::ReferencePoint => (ItemKind::ReferencePoint, None),
                Definition::Coupling => (ItemKind::Coupling, None),
                Definition::Step => return step.is_some_and(|s| s.is_none_or(|s| s.name != name)),
                Definition::Contact | Definition::Tie => return false,
            };
            used == Some(false) || self.view.is_hidden(kind, name)
        })
    }

    /// The elements of the bodies the view hides: hiding a body hides its mesh and its results.
    pub fn hidden_elements(&self) -> std::collections::BTreeSet<u32> {
        self.view.hidden.iter().filter(|i| i.kind == ItemKind::Body).flat_map(|i| self.model.body_elements(&i.name)).collect()
    }

    /// Whether the view shows item `name` of `kind`.
    pub fn shows(&self, kind: ItemKind, name: &str) -> bool {
        !self.view.is_hidden(kind, name)
    }
}

fn change(s: &mut Session, p: &Value, how: Change) -> Result<Value> {
    let items: Vec<Item> = super::req(p, "items")?;
    if items.len() > MAX_HIDDEN {
        return Err(EngineError::Params(format!("at most {MAX_HIDDEN} items")));
    }
    if let Some(i) = items.iter().find(|i| !exists(s, i)) {
        let kind = serde_json::to_value(i.kind).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        return Err(EngineError::Params(format!("no {} `{}`", kind.replace('_', " "), i.name.chars().take(100).collect::<String>())));
    }
    match how {
        Change::Hide => s.view.hidden.extend(items),
        Change::Show => {
            for i in &items {
                s.view.hidden.remove(i);
            }
        }
        Change::Isolate => {
            let keep: BTreeSet<Item> = items.into_iter().collect();
            let all = ItemKind::ALL.into_iter().flat_map(|kind| names(s, kind).into_iter().map(move |name| Item { kind, name }));
            let rest: BTreeSet<Item> = all.filter(|i| !keep.contains(i)).take(MAX_HIDDEN).collect();
            s.view.hidden = rest;
        }
    }
    Ok(Value::Null)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::cmd::datum::tests::plate;

    fn item(kind: ItemKind, name: &str) -> Value {
        json!({"kind": kind, "name": name})
    }

    /// A block from (0, 0, 0) to (2, 1, 1) as two faces of two triangles each: its top and its
    /// foot.
    fn block_faces() -> (Vec<String>, Vec<[f64; 3]>, Vec<[u32; 3]>, Vec<u32>) {
        let positions = vec![
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        (vec!["top".into(), "foot".into()], positions, vec![[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]], vec![0, 0, 1, 1])
    }

    #[test]
    fn a_setup_with_no_mesh_has_its_symbols_on_the_bodies_faces() {
        use fieldcraft_model::{BoundaryCondition, GeometryRef, Load, Set, Target};
        use fieldcraft_render::Scene;
        use fieldcraft_render::glyphs::{GlyphColors, GlyphShow, GlyphView};
        let (faces, positions, triangles, tri_face) = block_faces();
        let surface = BodySurface { body: "Block", faces: &faces, positions: &positions, triangles: &triangles, tri_face: &tri_face };
        let mut s = Session::default();
        let on = |face: &str| Set { geometry: vec![GeometryRef::face("Block", face).unwrap()], ..Set::default() };
        s.model.add_set("top", on("top")).unwrap();
        s.model.add_set("foot", on("foot")).unwrap();
        s.model.add_set("all", Set { geometry: vec![GeometryRef::body("Block").unwrap()], ..Set::default() }).unwrap();
        s.model.add_set("elsewhere", Set { geometry: vec![GeometryRef::face("Other", "top").unwrap()], ..Set::default() }).unwrap();

        // About PLACES points spread over the face, inside it, each with its outward direction.
        let top = places(&s.model, &[surface], "top");
        assert!(top.len() > PLACES / 2 && top.len() < 2 * PLACES, "{}", top.len());
        assert!(top.iter().all(|(p, n)| p.z == 1.0 && (0.0..=2.0).contains(&p.x) && (0.0..=1.0).contains(&p.y) && *n == Vec3::Z));
        // Spread: every tenth of the face holds some.
        for i in 0..10 {
            assert!(top.iter().any(|(p, _)| (p.x * 5.0).floor() as usize == i), "none in tenth {i}");
        }
        let foot = places(&s.model, &[surface], "foot");
        assert!(!foot.is_empty() && foot.iter().all(|(p, n)| p.z == 0.0 && *n == -Vec3::Z));
        // A whole body is all its faces; a set of another body, or no set, has none.
        assert!(
            places(&s.model, &[surface], "all").iter().any(|(p, _)| p.z == 0.0)
                && places(&s.model, &[surface], "all").iter().any(|(p, _)| p.z == 1.0)
        );
        assert!(places(&s.model, &[surface], "elsewhere").is_empty() && places(&s.model, &[surface], "none").is_empty());
        // A surface that is not one: indices and tags out of range, points that are no numbers.
        let bad = BodySurface {
            body: "Block",
            faces: &faces,
            positions: &[[f64::NAN; 3], [0.0; 3]],
            triangles: &[[0, 1, 9], [1, 1, 1]],
            tri_face: &[0, 7],
        };
        assert!(places(&s.model, &[bad], "all").is_empty());

        // The session draws a clamp and a pressure there, and nothing without the surface or
        // with the body hidden.
        s.model.add_boundary_condition("Clamp", BoundaryCondition::Fixed { target: Target::Set("foot".into()) }).unwrap();
        s.model.add_load("Press", Load::Pressure { set: "top".into(), magnitude: 2.0 }).unwrap();
        let colors = GlyphColors {
            constraint: fieldcraft_render::Rgb(0, 0, 255),
            load: fieldcraft_render::Rgb(255, 0, 0),
            interaction: fieldcraft_render::Rgb(0, 255, 0),
            contact: fieldcraft_render::Rgb(200, 200, 0),
            thermal: fieldcraft_render::Rgb(255, 128, 0),
            label: fieldcraft_render::Rgb(0, 0, 0),
            datum: fieldcraft_render::Rgb(0, 128, 128),
            highlight: fieldcraft_render::Rgb(255, 0, 255),
        };
        let lines = |glyphs: &fieldcraft_render::glyphs::Glyphs| {
            let mut scene = Scene::default();
            glyphs.draw(|_| None, &GlyphView { show: GlyphShow::ALL, colors: &colors, unit: 0.1, highlight: None }, &mut scene);
            (scene.lines.iter().filter(|l| l.color == colors.constraint).count(), scene.lines.iter().filter(|l| l.color == colors.load).count())
        };
        let (clamps, loads) = lines(&s.glyphs_on(&[surface]));
        assert!(clamps >= 3 * 6 && loads >= 5, "{clamps} lines of the clamp, {loads} of the pressure");
        assert_eq!(lines(&s.step_glyphs_on("none", &[surface])), (0, 0), "a step that used neither");
        assert_eq!(lines(&s.glyphs()), (0, 0));
        s.view.hidden.insert(Item { kind: ItemKind::Body, name: "Block".into() });
        assert_eq!(lines(&s.glyphs_on(&[surface])), (0, 0));
    }

    #[test]
    fn hiding_is_saved_view_state_not_an_edit() {
        let (mut s, dir) = plate();
        s.execute("geometry.datum", &json!({"name": "Sym", "by": {"type": "offset", "plane": "zx"}})).unwrap();
        s.execute("geometry.datum.point", &json!({"name": "P", "by": {"type": "coordinates", "point": [1, 2, 3]}})).unwrap();
        let file = dir.join("hide.fieldcraft");
        s.execute("file.save_as", &json!({"path": file.to_str().unwrap()})).unwrap();
        let (journal, revision) = (s.journal.len(), s.revision());
        assert!(!s.is_dirty());

        s.execute("view.hide", &json!({"items": [item(ItemKind::Datum, "Sym")]})).unwrap();
        assert_eq!(s.view.hidden.iter().collect::<Vec<_>>(), [&Item { kind: ItemKind::Datum, name: "Sym".into() }]);
        // Unsaved, but no edit: nothing journalled, the revision (and so any result) unchanged.
        assert!(s.is_dirty());
        assert_eq!((s.journal.len(), s.revision()), (journal, revision));
        // Showing it again is back to the saved state.
        s.execute("view.show", &json!({"items": [item(ItemKind::Datum, "Sym")]})).unwrap();
        assert!(!s.is_dirty());

        // Hidden state survives save and open.
        s.execute("view.hide", &json!({"items": [item(ItemKind::Datum, "Sym")]})).unwrap();
        s.execute("file.save", &json!({})).unwrap();
        let mut t = Session::default();
        t.execute("file.open", &json!({"path": file.to_str().unwrap()})).unwrap();
        assert!(t.view.is_hidden(ItemKind::Datum, "Sym") && !t.view.is_hidden(ItemKind::Datum, "P"));
        assert!(!t.is_dirty());

        // Isolate hides every other item; show_all brings them back.
        let body = names(&t, ItemKind::Body).remove(0);
        t.execute("view.isolate", &json!({"items": [item(ItemKind::Datum, "P")]})).unwrap();
        assert!(t.view.is_hidden(ItemKind::Body, &body) && t.view.is_hidden(ItemKind::Datum, "Sym") && !t.view.is_hidden(ItemKind::Datum, "P"));
        t.execute("view.show_all", &json!({})).unwrap();
        assert!(t.view.is_empty());

        // An item deleted is forgotten.
        t.execute("view.hide", &json!({"items": [item(ItemKind::Datum, "P")]})).unwrap();
        t.execute("geometry.datum.delete", &json!({"name": "P"})).unwrap();
        assert!(t.view.is_empty());

        // Unknown items and hostile parameters change nothing.
        for p in [json!({}), json!({"items": "x"}), json!({"items": [item(ItemKind::Load, "Nope")]}), json!({"items": [{"kind": "x", "name": "A"}]})]
        {
            assert!(t.execute("view.hide", &p).is_err(), "{p}");
        }
        assert!(t.view.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
