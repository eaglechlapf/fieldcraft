//! `solve.static.grid`: a linear static step of one body of the open design solved with no
//! volume mesh, by the finite cell solver on a background grid ([`fieldcraft_fcm`]).
//!
//! The command reads what a mesh solve reads, without the mesh: the body's tessellation (the
//! solver's whole knowledge of the geometry), the material of the body's solid section, and the
//! step's boundary conditions and loads, whose sets must name whole faces of the body. Anything
//! the solver cannot do is refused by name, never left out in silence.
//!
//! The solver gives displacement and stress at any point of the part. They are sampled at the
//! vertices of the body's surface, refined until no triangle edge is longer than the sampling
//! size, and kept as a [`GridResult`]: a static result on that surface, which the results display
//! draws as it draws a mesh solve's (contours, the deformed shape, the legend, probes).

use std::collections::BTreeMap;

use fieldcraft_fcm::{Load as GridLoad, Material as GridMaterial, Options, Solution, Support};
use fieldcraft_geometry::Shape;
use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};
use fieldcraft_model::units::Quantity;
use fieldcraft_model::{BoundaryCondition, Elastic, ElementKind, Entity, GeometryRef, Load, Model, Procedure, SectionKind, Step, Target};
use fieldcraft_parallel as parallel;
use fieldcraft_render::{Aabb3, DisplayEdge, DisplayMesh, DisplayNode, DisplayTri, Vec3};
use fieldcraft_solver::{Limits, StaticResult, Stats};
use serde_json::{Value, json};

use super::opt;
use super::solve::{cap, limits, settled, threads, with_threads};
use crate::requirement::Requirement;
use crate::results::{FaceReaction, GRID_LABELS, GridResult, magnitude};
use crate::{CommandKind, CommandSpec, EngineError, Result, Session, StepResult};

pub const STATIC_GRID: CommandSpec = CommandSpec {
    id: "solve.static.grid",
    label: "Solve Static Step on a Grid",
    params: r#"{"step": "Load", "body": "Bracket", "cell_size": 4.0, "degree": 3, "refine": 0, "check_convergence": false, "tolerance": 0.05, "sample_size": 1.0, "max_dofs": 6000000, "max_factor_entries": 400000000, "threads": 8} — solves a linear static step (the only one when `step` is left out) for one body of the open design (the only one when `body` is left out) with no volume mesh, by the finite cell method: the body's surface, triangulated within `tolerance` (a length; default 1e-3 of the body's bounding-box diagonal), is embedded in a grid of cubic cells of `cell_size` (a length; default 1/20 of the diagonal) carrying shape functions of polynomial `degree` (1 to 8, default 3). With `refine` (0 to 6 levels, default 0) a cell across which the surface is curved more tightly than it resolves (fewer than two cells to the radius of a fillet or hole at degree 4, four at degree 3) is divided into eight, level by level, so a coarse grid resolves a small feature without fine cells everywhere; `levels` in the reply counts the cells of each level. The body takes the isotropic elastic material of its solid section (a section on a set that names the body); the step's boundary conditions fix displacement components of whole faces (fixed, displacement, symmetry about a global plane) and its loads act on whole faces (force as a total, traction, pressure) or on the body (gravity): each must act on a set of geometry faces of this body. Refused, changing nothing: another kind of boundary condition or load, a set of nodes, elements, edges or vertices, a reference point, an orthotropic material, a cell of a partitioned body, and a model with contacts, ties, couplings, point masses or springs. Keeps the result in the session, sampled at the vertices of the body's surface refined to `sample_size` (a length; default `cell_size`/`degree`), where results.contour, results.deformed and solve.probe read it (its points are labelled above every mesh label); it is not written to model files. Returns the `unknowns`, the `cells` that hold material (under `grid` the grid's `cells` along x, y and z, those `inside` the part and those its surface `cut`s), the run time in `seconds`, the largest von Mises stress and displacement with where they are (`at`, and the `face`), the reaction of every supported face, the applied and reaction totals, the `volume` the solver integrated against the surface's (`surface_volume`), the strain energy and `warnings`. With `check_convergence` (degree 2 or more) the step is also solved one degree lower on the same grid, and `convergence` gives how far the two solutions are apart, each a fraction: `peak_von_mises` (the change of the largest von Mises stress over the sampled points), `von_mises` (the largest change at any one point, of the peak) and `strain_energy`; an indicator of convergence, not a bound of the error. `max_dofs` caps the unknowns and may only be lowered; `max_factor_entries` and `threads` as for solve.static. The grid is the same everywhere: a stress peak at a hole or a fillet is within about 1 % with two cells per radius of the smallest stressed feature at degree 4, or four at degree 3; displacements and reactions need far less (docs/grid-static.md)"#,
    form: Some(super::forms::SOLVE_STATIC_GRID),
    kind: CommandKind::Model,
    requires: &[Requirement::Design, Requirement::Material, Requirement::Section, Requirement::Step],
    run,
};

/// The highest polynomial degree the solver takes (`fieldcraft_fcm::Options::degree`).
pub(super) const MAX_DEGREE: i64 = 8;
/// The most levels of local refinement the solver takes (`fieldcraft_fcm::Options::refine`).
pub(super) const MAX_REFINE: i64 = fieldcraft_fcm::grid::MAX_REFINE as i64;
/// The default cell size and chordal tolerance, relative to the body's bounding-box diagonal:
/// the meshers' defaults (an element a twentieth of the part, the tessellation a thousandth).
const DEFAULT_RELATIVE_CELL: f64 = 1.0 / 20.0;
const DEFAULT_RELATIVE_TOLERANCE: f64 = super::hex::DEFAULT_RELATIVE_TOLERANCE;
/// The smallest chordal tolerance, relative to the diagonal (the hex meshers' floor).
const MIN_RELATIVE_TOLERANCE: f64 = 1e-5;
/// The smallest cell, relative to the diagonal: far below what the solver's grid can hold.
const MIN_RELATIVE_CELL: f64 = 1e-3; // unit-factors: relative to the bounding-box diagonal
/// A cell is at most this many diagonals across: beyond the body's own size one cell holds all
/// of it, and a larger number says nothing more.
const MAX_RELATIVE_CELL: f64 = 2.0;
/// The finest sampling of the surface, in samples along a cell edge per polynomial degree: a
/// polynomial of degree p has no detail finer than a few points per degree.
const MAX_SAMPLES_PER_DEGREE: f64 = 8.0;
/// Surface points the solution is sampled at, at most; refinement stops there.
const MAX_SAMPLES: usize = 1_000_000;
/// Refinement passes at most (each halves every edge longer than the sampling size).
const MAX_PASSES: usize = 40;
/// Surface points evaluated per parallel task.
const GRAIN: usize = 2048;
/// Where the grid starts, in cells before the lowest corner of the body's bounding box: a
/// different fraction along each axis, so the faces of a part modelled at round numbers (a foot
/// at z = 0, a web 8 mm thick on 4 mm cells) do not lie on grid planes, where a cell would hold
/// no material on one side. Any generic fractions do; these are the ones the solver's own
/// verification of a real part runs with (`crates/fcm/tests/bracket.rs`).
const OFFSET: [f64; 3] = [0.37, 0.21, 0.43];
/// Names shown of a list in a message.
const NAMED: usize = 8;

/// A name from the model or the request, cut for a message.
fn short(name: &str) -> String {
    name.chars().take(100).collect()
}

/// `names`, the first few, for a message.
fn listed<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let all: Vec<&str> = names.collect();
    let mut shown: Vec<String> = all.iter().take(NAMED).map(|n| format!("`{}`", short(n))).collect();
    if all.len() > NAMED {
        shown.push(format!("{} more", all.len() - NAMED));
    }
    shown.join(", ")
}

fn refused(message: String) -> EngineError {
    EngineError::Solver(fieldcraft_solver::Error::NotSupported(message))
}

/// The solver's error as the engine's: its kinds are the implicit solver's.
fn grid_error(e: fieldcraft_fcm::Error) -> EngineError {
    use fieldcraft_fcm::Error as G;
    use fieldcraft_solver::Error as S;
    // A cancelled job stops inside the solver, which reports it among its own failures.
    if let Err(cancelled) = fieldcraft_model::progress::check() {
        return EngineError::Model(cancelled);
    }
    match e {
        G::Invalid(m) => EngineError::Solver(S::Invalid(m)),
        G::Limit(m) => EngineError::Solver(S::Limit(m)),
        G::Singular(m) => EngineError::Solver(S::Singular(m)),
        G::Internal(m) => EngineError::Solver(S::Internal(m)),
        G::Surface(e) => EngineError::Mesh(e),
        G::Solver(e) => EngineError::Solver(e),
        // The solver's errors may grow: one this does not know is passed on as it reads.
        #[allow(unreachable_patterns)]
        other => EngineError::Solver(S::Invalid(other.to_string())),
    }
}

/// The body to solve: `body`, or the design's only one. A cell of a partitioned body is
/// refused: its cut faces join it to its neighbours, which a solve of the cell alone leaves free.
fn body<'a>(s: &'a Session, p: &Value) -> Result<&'a Shape> {
    let design = s.design.as_deref().ok_or_else(|| EngineError::Params("no design is open (geometry.open)".into()))?;
    let shape = match opt::<String>(p, "body")? {
        Some(name) => design.body(&name).ok_or_else(|| {
            EngineError::invalid(
                "body",
                format!("the open design has no body `{}` (its bodies: {})", short(&name), listed(design.bodies.iter().map(|b| b.name.as_str()))),
            )
        })?,
        None => match design.bodies.as_slice() {
            [one] => one,
            [] => return Err(EngineError::Params("the open design has no bodies".into())),
            many => {
                return Err(EngineError::invalid(
                    "body",
                    format!(
                        "name the body: the grid solves one body and the design has {} ({})",
                        many.len(),
                        listed(many.iter().map(|b| b.name.as_str()))
                    ),
                ));
            }
        },
    };
    match design.source_of(&shape.name) {
        Some(source) if source != shape.name => Err(refused(format!(
            "body `{}` is a cell of the partitioned body `{}`: its cut faces join it to the other cells, and the grid solves a whole body on its own (suppress or delete the partition)",
            short(&shape.name),
            short(source)
        ))),
        _ => Ok(shape),
    }
}

/// What the model holds that a solve of one body on its own would leave out.
fn refuse_interactions(model: &Model) -> Result<()> {
    let kinds: [(&str, Vec<&str>); 5] = [
        ("contact", model.contacts().keys().map(String::as_str).collect()),
        ("tie", model.ties().keys().map(String::as_str).collect()),
        ("coupling", model.couplings().keys().map(String::as_str).collect()),
        ("point mass", model.masses().keys().map(String::as_str).collect()),
        ("spring", model.springs().keys().map(String::as_str).collect()),
    ];
    for (what, names) in kinds {
        if !names.is_empty() {
            return Err(refused(format!(
                "the model has {what} {}: the grid solves one body on its own, without contacts, ties, couplings, point masses or springs (solve.static solves them on a mesh)",
                listed(names.into_iter())
            )));
        }
    }
    Ok(())
}

/// The isotropic material of the body's solid section, and the material's name.
fn material<'a>(model: &'a Model, body: &str, warnings: &mut Vec<String>) -> Result<(GridMaterial, &'a str)> {
    let on_body = |set: &str| model.set(set).is_some_and(|set| set.geometry.iter().any(|g| g.body == body && g.entity == Entity::Body));
    let sections: Vec<(&String, &fieldcraft_model::Section)> = model.sections().iter().filter(|(_, section)| on_body(&section.set)).collect();
    let (name, section) = match sections.as_slice() {
        [one] => *one,
        [] => {
            return Err(EngineError::Params(format!(
                "no section is assigned to body `{}`: give it one on a set that names the body (model.set.create {{\"geometry\": [{{\"body\": …}}]}}, then model.section.create)",
                short(body)
            )));
        }
        many => {
            return Err(EngineError::Params(format!(
                "body `{}` has the sections {}: one section gives the body its material",
                short(body),
                listed(many.iter().map(|(n, _)| n.as_str()))
            )));
        }
    };
    if section.kind != (SectionKind::Solid {}) {
        return Err(refused(format!(
            "section `{}` of body `{}` is a {} section: the grid solves solids",
            short(name),
            short(body),
            section.kind.name()
        )));
    }
    let found = model.material(&section.material).ok_or_else(|| EngineError::Params(format!("section `{}` names no material", short(name))))?;
    let (young, poisson) = match found.elastic {
        Some(Elastic::Isotropic { young, poisson }) => (young, poisson),
        Some(Elastic::EngineeringConstants { .. }) => {
            return Err(refused(format!("material `{}` is orthotropic: the grid solves isotropic elasticity only", short(&section.material))));
        }
        None => return Err(EngineError::Params(format!("material `{}` has no elastic constants", short(&section.material)))),
    };
    if found.plastic.is_some() || found.hyperelastic.is_some() {
        warnings.push(format!("material `{}`: a linear solve uses its elastic constants alone", short(&section.material)));
    }
    Ok((GridMaterial { young, poisson, density: found.density.unwrap_or(0.0) }, section.material.as_str()))
}

/// The faces of `shape` (their indices, the tessellation's tags) that set `set` names, for
/// `what` ("boundary condition `Clamp`"). The set must name whole faces of this body and
/// nothing else.
fn faces(model: &Model, shape: &Shape, set: &str, what: &str) -> Result<Vec<u32>> {
    let definition = model.set(set).ok_or_else(|| EngineError::Params(format!("{what}: no set `{}`", short(set))))?;
    if !(definition.nodes.is_empty() && definition.elements.is_empty() && definition.faces.is_empty() && definition.edges.is_empty()) {
        return Err(refused(format!(
            "{what}: set `{}` lists nodes, elements or element faces of a mesh; the grid has none and takes whole faces of the body (a set of geometry faces)",
            short(set)
        )));
    }
    let mut out = Vec::new();
    for g in &definition.geometry {
        if g.body != shape.name {
            return Err(refused(format!(
                "{what}: set `{}` names body `{}`; the grid solves body `{}` on its own",
                short(set),
                short(&g.body),
                short(&shape.name)
            )));
        }
        let Entity::Face(face) = &g.entity else {
            return Err(refused(format!(
                "{what}: set `{}` names {}; the grid supports and loads whole faces only",
                short(set),
                short(&g.to_string())
            )));
        };
        let before = out.len();
        for (index, f) in shape.faces().iter().enumerate() {
            if GeometryRef::face(shape.name.as_str(), f.name.as_str()).is_ok_and(|tag| g.selects(&tag)) {
                out.push(u32::try_from(index).map_err(|_| EngineError::Params("the body has too many faces".into()))?);
            }
        }
        if out.len() == before {
            return Err(EngineError::Params(format!("{what}: body `{}` has no face `{}`", short(&shape.name), short(face))));
        }
    }
    if out.is_empty() {
        return Err(refused(format!("{what}: set `{}` names no geometry face", short(set))));
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// The set a boundary condition or a force acts on; a reference point is refused.
fn target_set<'a>(target: &'a Target, what: &str) -> Result<&'a str> {
    match target {
        Target::Set(set) => Ok(set),
        Target::ReferencePoint(point) => Err(refused(format!("{what} acts on reference point `{}`; the grid has no reference points", short(point)))),
    }
}

/// A face's prescribed components: whether each is prescribed, its value, and the boundary
/// condition that prescribes it.
#[derive(Default)]
struct Held<'a> {
    fixed: [bool; 3],
    value: [f64; 3],
    by: [&'a str; 3],
}

/// The step's boundary conditions as supports of faces, one per face.
fn supports(model: &Model, shape: &Shape, step: &Step, warnings: &mut Vec<String>) -> Result<Vec<Support>> {
    let mut held: BTreeMap<u32, Held> = BTreeMap::new();
    for name in &step.boundary_conditions {
        let Some(bc) = model.boundary_condition(name) else { continue };
        let what = format!("boundary condition `{}`", short(name));
        let components: Vec<(usize, f64)> = match bc {
            BoundaryCondition::Fixed { .. } => vec![(0, 0.0), (1, 0.0), (2, 0.0)],
            BoundaryCondition::Displacement { values, .. } => {
                if values.iter().skip(3).any(Option::is_some) {
                    warnings.push(format!("{what}: its rotations are left out (a solid has none)"));
                }
                let translations: Vec<(usize, f64)> = values.iter().take(3).enumerate().filter_map(|(k, v)| v.map(|v| (k, v))).collect();
                if translations.is_empty() {
                    return Err(refused(format!("{what} prescribes rotations only; a solid has none")));
                }
                translations
            }
            BoundaryCondition::Symmetry { plane, .. } => match plane.axis() {
                Some(axis) => vec![(axis, 0.0)],
                None => {
                    return Err(refused(format!(
                        "{what} is a symmetry plane inclined to the axes; the grid fixes displacement components along X, Y and Z only"
                    )));
                }
            },
            BoundaryCondition::Temperature { .. } => {
                return Err(refused(format!("{what} prescribes a temperature; the grid solves the displacements of a static step only")));
            }
        };
        for face in faces(model, shape, target_set(bc.target(), &what)?, &what)? {
            let entry = held.entry(face).or_default();
            for &(k, v) in &components {
                let (Some(fixed), Some(value), Some(by)) = (entry.fixed.get_mut(k), entry.value.get_mut(k), entry.by.get_mut(k)) else { continue };
                if *fixed && *value != v {
                    return Err(EngineError::Params(format!(
                        "boundary conditions `{}` and `{}` prescribe different values for displacement component {} of face `{}`",
                        short(by),
                        short(name),
                        k + 1,
                        short(shape.faces().get(face as usize).map_or("", |f| f.name.as_str()))
                    )));
                }
                (*fixed, *value, *by) = (true, v, name.as_str());
            }
        }
    }
    if held.is_empty() {
        return Err(EngineError::Params(format!(
            "step `{}` holds body `{}` nowhere: none of its boundary conditions supports a face of the body",
            short(&step.name),
            short(&shape.name)
        )));
    }
    Ok(held.into_iter().map(|(face, h)| Support { face, fixed: h.fixed, value: h.value }).collect())
}

/// The area of every face of the surface, by tag, and the sum of its triangles' area vectors
/// (normal times area), which a pressure's resultant is along.
fn face_areas(surface: &BoundarySurface) -> BTreeMap<u32, (f64, [f64; 3])> {
    let mut out: BTreeMap<u32, (f64, [f64; 3])> = BTreeMap::new();
    let points = surface.points();
    for (t, tag) in surface.triangles().iter().zip(surface.tri_face()) {
        let (Some(a), Some(b), Some(c)) = (points.get(t[0] as usize), points.get(t[1] as usize), points.get(t[2] as usize)) else { continue };
        let (a, b, c) = (Vec3::from(*a), Vec3::from(*b), Vec3::from(*c));
        let n = (b - a).cross(c - a) * 0.5;
        let entry = out.entry(*tag).or_insert((0.0, [0.0; 3]));
        entry.0 += n.dot(n).sqrt();
        entry.1 = [entry.1[0] + n.x, entry.1[1] + n.y, entry.1[2] + n.z];
    }
    out
}

/// The step's loads as the solver's, and the terms of their resultant apart from gravity's
/// (which needs the volume the solver integrates): one force per loaded face.
fn loads(
    model: &Model,
    shape: &Shape,
    step: &Step,
    areas: &BTreeMap<u32, (f64, [f64; 3])>,
    density: f64,
    material_name: &str,
) -> Result<(Vec<GridLoad>, Vec<[f64; 3]>)> {
    let mut out = Vec::new();
    let mut applied = Vec::new();
    let area = |face: u32| areas.get(&face).copied().unwrap_or((0.0, [0.0; 3]));
    for name in &step.loads {
        let Some(load) = model.load(name) else { continue };
        let what = format!("load `{}`", short(name));
        match load {
            Load::Force { target, vector, distribution } => {
                if *distribution != fieldcraft_model::Distribution::Total {
                    return Err(refused(format!("{what} is a force at every node of a mesh; the grid takes a force as a total over faces")));
                }
                // The total, spread evenly over the area of all the set's faces together.
                let on = faces(model, shape, target_set(target, &what)?, &what)?;
                let total: f64 = on.iter().map(|f| area(*f).0).sum();
                if !(total > 0.0) {
                    return Err(EngineError::Params(format!("{what}: its faces have no area")));
                }
                for face in on {
                    let share = area(face).0 / total;
                    let force = vector.map(|c| c * share);
                    out.push(GridLoad::Force { face, force });
                    applied.push(force);
                }
            }
            Load::Traction { set, vector } => {
                for face in faces(model, shape, set, &what)? {
                    out.push(GridLoad::Traction { face, traction: *vector });
                    applied.push(vector.map(|c| c * area(face).0));
                }
            }
            Load::Pressure { set, magnitude } => {
                for face in faces(model, shape, set, &what)? {
                    out.push(GridLoad::Pressure { face, pressure: *magnitude });
                    // Pushing into the part: against the outward normal.
                    applied.push(area(face).1.map(|c| -magnitude * c));
                }
            }
            Load::Gravity { acceleration, set } => {
                if let Some(set) = set {
                    let definition = model.set(set).ok_or_else(|| EngineError::Params(format!("{what}: no set `{}`", short(set))))?;
                    let whole =
                        definition.nodes.is_empty() && definition.elements.is_empty() && definition.faces.is_empty() && definition.edges.is_empty();
                    if !whole || definition.geometry.iter().any(|g| g.entity != Entity::Body) {
                        return Err(refused(format!(
                            "{what}: set `{}` is not a set of whole bodies; on the grid gravity acts on the whole body",
                            short(set)
                        )));
                    }
                    // Gravity of other bodies does not load this one.
                    if !definition.geometry.iter().any(|g| g.body == shape.name) {
                        continue;
                    }
                }
                if !(density > 0.0) {
                    return Err(EngineError::Params(format!("{what} needs the density of material `{}`", short(material_name))));
                }
                out.push(GridLoad::Gravity { acceleration: *acceleration });
            }
            Load::Moment { .. }
            | Load::BodyForce { .. }
            | Load::Centrifugal { .. }
            | Load::SurfaceFlux { .. }
            | Load::Convection { .. }
            | Load::Radiation { .. }
            | Load::HeatSource { .. }
            | Load::Temperature { .. } => {
                return Err(refused(format!(
                    "{what} is a {} load; the grid takes forces, tractions and pressures on whole faces, and gravity",
                    load.kind().replace('_', " ")
                )));
            }
        }
    }
    Ok((out, applied))
}

/// A triangulated surface the solution is sampled on.
struct Sampled {
    points: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    tri_face: Vec<u32>,
    /// The longest edge left, when refinement stopped at the cap on points.
    capped: Option<f64>,
}

fn edge_length(points: &[[f64; 3]], a: u32, b: u32) -> f64 {
    match (points.get(a as usize), points.get(b as usize)) {
        (Some(p), Some(q)) => Vec3::from(*p).dist(Vec3::from(*q)),
        _ => 0.0,
    }
}

/// The surface's triangles refined until no edge is longer than `size`: every pass halves each
/// longer edge and cuts the triangles on it into two, three or four, by the edges they have
/// halved. Whether an edge is halved depends on its length alone, so the two triangles on it
/// agree and the surface stays closed. Stops at `max_points` points.
fn refine(surface: &BoundarySurface, size: f64, max_points: usize) -> Sampled {
    let mut points = surface.points().to_vec();
    let mut triangles = surface.triangles().to_vec();
    let mut tri_face: Vec<u32> = (0..triangles.len()).map(|t| surface.tri_face().get(t).copied().unwrap_or(u32::MAX)).collect();
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let mut capped = None;
    for _ in 0..MAX_PASSES {
        let mut long: Vec<(u32, u32)> = Vec::new();
        let mut longest = 0.0f64;
        for t in &triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let length = edge_length(&points, a, b);
                if length > size {
                    long.push(key(a, b));
                    longest = longest.max(length);
                }
            }
        }
        long.sort_unstable();
        long.dedup();
        if long.is_empty() {
            break;
        }
        if points.len().saturating_add(long.len()) > max_points || u32::try_from(points.len() + long.len()).is_err() {
            capped = Some(longest);
            break;
        }
        let first = points.len();
        for &(a, b) in &long {
            let (p, q) = (points.get(a as usize).copied().unwrap_or([0.0; 3]), points.get(b as usize).copied().unwrap_or([0.0; 3]));
            points.push([0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])]);
        }
        let middle = |a: u32, b: u32| long.binary_search(&key(a, b)).ok().and_then(|i| u32::try_from(first + i).ok());
        let mut next = Vec::with_capacity(triangles.len() * 2);
        let mut next_face = Vec::with_capacity(triangles.len() * 2);
        for (t, face) in triangles.iter().zip(&tri_face) {
            let [a, b, c] = *t;
            let pieces: Vec<[u32; 3]> = match (middle(a, b), middle(b, c), middle(c, a)) {
                (None, None, None) => vec![[a, b, c]],
                (Some(ab), Some(bc), Some(ca)) => vec![[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]],
                (Some(ab), None, None) => vec![[a, ab, c], [ab, b, c]],
                (None, Some(bc), None) => vec![[b, bc, a], [bc, c, a]],
                (None, None, Some(ca)) => vec![[c, ca, b], [ca, a, b]],
                // Two halved edges: the corner between them, and the rest cut by its shorter diagonal.
                (Some(ab), Some(bc), None) => corner_and_quad(&points, [b, bc, ab], [ab, bc, c, a]),
                (None, Some(bc), Some(ca)) => corner_and_quad(&points, [c, ca, bc], [bc, ca, a, b]),
                (Some(ab), None, Some(ca)) => corner_and_quad(&points, [a, ab, ca], [ca, ab, b, c]),
            };
            for piece in pieces {
                next.push(piece);
                next_face.push(*face);
            }
        }
        (triangles, tri_face) = (next, next_face);
    }
    Sampled { points, triangles, tri_face, capped }
}

/// A corner triangle and the quad `q` (in order around it) beside it as two triangles, cut by the
/// quad's shorter diagonal (the first on a tie).
fn corner_and_quad(points: &[[f64; 3]], corner: [u32; 3], q: [u32; 4]) -> Vec<[u32; 3]> {
    if edge_length(points, q[0], q[2]) <= edge_length(points, q[1], q[3]) {
        vec![corner, [q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    } else {
        vec![corner, [q[0], q[1], q[3]], [q[1], q[2], q[3]]]
    }
}

/// The sampled surface as the results display draws it: its triangles as faces with their
/// vertices' labels (from [`GRID_LABELS`]), the vertices as nodes, and the sides between
/// triangles of different faces as feature lines. Returns the face of each triangle drawn too
/// (a degenerate triangle is not drawn).
fn display(sampled: &Sampled) -> (DisplayMesh, Vec<u32>) {
    let label = |v: u32| GRID_LABELS.saturating_add(v);
    let at = |v: u32| sampled.points.get(v as usize).map(|p| Vec3::from(*p)).filter(|p| p.is_finite());
    let mut out = DisplayMesh::default();
    let mut bounds = Aabb3::EMPTY;
    for (v, p) in sampled.points.iter().enumerate() {
        let (Ok(v), p) = (u32::try_from(v), Vec3::from(*p)) else { continue };
        if p.is_finite() {
            out.nodes.push(DisplayNode { label: label(v), p });
            bounds.add(p);
        }
    }
    let mut tri_face = Vec::with_capacity(sampled.triangles.len());
    // (low vertex, high vertex, face) per triangle side.
    let mut sides: Vec<(u32, u32, u32)> = Vec::with_capacity(sampled.triangles.len().saturating_mul(3));
    for (index, (t, face)) in sampled.triangles.iter().zip(&sampled.tri_face).enumerate() {
        let (Some(a), Some(b), Some(c)) = (at(t[0]), at(t[1]), at(t[2])) else { continue };
        let Some(normal) = (b - a).cross(c - a).normalized() else { continue };
        let element = GRID_LABELS.saturating_add(u32::try_from(index).unwrap_or(u32::MAX));
        out.tris.push(DisplayTri { p: [a, b, c], nodes: t.map(label), normal, kind: ElementKind::Tri3, element, face: 1 });
        tri_face.push(*face);
        for (u, v) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            sides.push((u.min(v), u.max(v), *face));
        }
    }
    sides.sort_unstable();
    for run in sides.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
        let Some(&(u, v, face)) = run.first() else { continue };
        if run.len() == 1 || run.iter().any(|s| s.2 != face) {
            let (Some(a), Some(b)) = (at(u), at(v)) else { continue };
            out.edges.push(DisplayEdge { p: [a, b], nodes: [label(u), label(v)], corners: [label(u), label(v)], feature: true });
        }
    }
    out.bounds = bounds;
    (out, tri_face)
}

/// The value with its node, where it is and on which face: `{node, value, at, face}`.
fn peak(result: &GridResult, values: &[f64]) -> Value {
    let Some((_, (node, value))) = crate::results::range(result.result.nodes(), values) else { return Value::Null };
    json!({
        "node": node,
        "value": value,
        "at": result.display.node_position(node).map(|p| [p.x, p.y, p.z]),
        "face": result.face_of(node),
    })
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let started = crate::usage::now_ms();
    fieldcraft_model::progress::check()?;
    let threads = threads(s, p)?;
    let limits = Limits { max_dofs: cap(p, "max_dofs", s.limits.max_grid_dofs.clamp(1, Limits::MAX_DOFS))?, ..limits(p)? };
    // The solver's default degree, read off its default options.
    let default_degree = Options::new(1.0).degree;
    let degree = super::mesh::optional_count(p, "degree", default_degree as u64, 1..=MAX_DEGREE as u64)?;
    let levels = super::mesh::optional_count(p, "refine", 0, 0..=MAX_REFINE as u64)?;
    let check_convergence = opt::<bool>(p, "check_convergence")?.unwrap_or(false);
    if check_convergence && degree < 2 {
        return Err(EngineError::invalid("check_convergence", "compares with the solution one degree lower: it needs a degree of 2 or more"));
    }
    let shape = body(s, p)?;
    let model = &s.model;
    let step = model.select_step(opt::<String>(p, "step")?.as_deref(), |procedure| matches!(procedure, Procedure::Static {}), "static")?;
    refuse_interactions(model)?;
    let mut warnings = Vec::new();
    let (material, material_name) = material(model, &shape.name, &mut warnings)?;
    let supports = supports(model, shape, step, &mut warnings)?;

    // Sizes: the geometry is in millimetres, the model and the solver in SI.
    let measure = shape.measure()?;
    let (lo, hi) = (crate::units::geometry_point(measure.bbox_min), crate::units::geometry_point(measure.bbox_max));
    let diagonal = magnitude([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]);
    if !(diagonal > 0.0 && diagonal.is_finite()) {
        return Err(EngineError::Params(format!("body `{}` has no size", short(&shape.name))));
    }
    // A length parameter within `range`, which the message shows in the display units with why.
    let shown = |v: f64| crate::units::shown(s.units(), crate::units::si(), Quantity::Length, v);
    let length = |key: &str, default: f64, range: std::ops::RangeInclusive<f64>, why: &str| -> Result<f64> {
        let v = super::optional_number(p, key, default)?;
        if range.contains(&v) {
            Ok(v)
        } else {
            Err(EngineError::invalid(key, format!("must be a length from {} to {} ({why})", shown(*range.start()), shown(*range.end()))))
        }
    };
    let tolerance = length(
        "tolerance",
        DEFAULT_RELATIVE_TOLERANCE * diagonal,
        MIN_RELATIVE_TOLERANCE * diagonal..=diagonal,
        "1e-5 of the body's bounding-box diagonal to all of it",
    )?;
    let cell_size = length(
        "cell_size",
        DEFAULT_RELATIVE_CELL * diagonal,
        MIN_RELATIVE_CELL * diagonal..=MAX_RELATIVE_CELL * diagonal,
        "1e-3 of the body's bounding-box diagonal to twice it",
    )?;
    let finest = cell_size / (MAX_SAMPLES_PER_DEGREE * degree as f64);
    let sample_size =
        length("sample_size", cell_size / degree as f64, finest..=diagonal, "an eighth of cell_size/degree to the body's bounding-box diagonal")?;

    fieldcraft_model::progress::stage(&format!("{}: surface", shape.name), None, "");
    let mesh = shape.tessellate(crate::units::to_geometry_length(tolerance))?;
    let positions: Vec<[f64; 3]> = mesh.positions.iter().map(|p| crate::units::geometry_point(*p)).collect();
    let options = SurfaceOptions { weld_distance: crate::units::geometry_length(mesh.weld_distance), ..SurfaceOptions::default() };
    let surface = BoundarySurface::new(&positions, &mesh.triangles, &mesh.tri_face, &options)
        .map_err(|e| EngineError::context(format!("body `{}`: its surface cannot bound a solid", short(&shape.name)), e))?;
    let areas = face_areas(&surface);
    let (loads, mut applied) = loads(model, shape, step, &areas, material.density, material_name)?;
    if loads.is_empty() && supports.iter().all(|s| s.value.iter().zip(s.fixed).all(|(v, fixed)| !fixed || *v == 0.0)) {
        warnings.push(format!("step `{}` neither loads body `{}` nor moves a support: nothing deforms it", step.name, shape.name));
    }

    // Before the solver starts on its grid: the unknowns of the cells the body fills alone (p³
    // modes a cell, three unknowns a mode) are a lower bound of the system's.
    let volume = crate::units::geometry().to_si(Quantity::Volume, measure.volume).unwrap_or(f64::NAN);
    let least = 3.0 * (degree as f64).powi(3) * volume / cell_size.powi(3);
    if least > limits.max_dofs as f64 {
        return Err(EngineError::Solver(fieldcraft_solver::Error::Limit(format!(
            "about {least:.0} unknowns or more at this cell size and degree; the limit is {} (use a larger cell size or a lower degree)",
            limits.max_dofs
        ))));
    }

    fieldcraft_model::progress::stage(&format!("{}: grid solve", shape.name), None, "");
    fieldcraft_model::progress::check()?;
    let grid = Options { degree, refine: levels, offset: OFFSET, ..Options::new(cell_size) };
    let (solution, sampled, values, change) = with_threads(threads, || -> Result<_> {
        let solution = fieldcraft_fcm::solve(&surface, &material, &supports, &loads, &grid, &limits).map_err(grid_error)?;
        fieldcraft_model::progress::stage(&format!("{}: results", shape.name), None, "");
        fieldcraft_model::progress::check()?;
        let sampled = refine(&surface, sample_size, MAX_SAMPLES);
        let values = sample(&solution, &sampled.points);
        // The p-extension's own check: the same step one degree lower on the same grid, and how
        // far the two solutions are apart at the sampled points.
        let change = if check_convergence {
            fieldcraft_model::progress::stage(&format!("{}: convergence check at degree {}", shape.name, degree - 1), None, "");
            fieldcraft_model::progress::check()?;
            let coarser = Options { degree: degree - 1, ..grid };
            let lower = fieldcraft_fcm::solve(&surface, &material, &supports, &loads, &coarser, &limits).map_err(grid_error)?;
            Some(solution.change_from(&lower, &sampled.points))
        } else {
            None
        };
        Ok((solution, sampled, values, change))
    })?;
    let stats = &solution.stats;

    // The resultants. Gravity loads the volume the solver integrated.
    for load in &loads {
        if let GridLoad::Gravity { acceleration } = load {
            applied.push(acceleration.map(|a| a * material.density * stats.volume));
        }
    }
    let sum = |terms: &[[f64; 3]]| terms.iter().fold([0.0; 3], |t, f| [t[0] + f[0], t[1] + f[1], t[2] + f[2]]);
    let face_name = |face: u32| shape.faces().get(face as usize).map_or_else(|| face.to_string(), |f| f.name.clone());
    // Round-off is read against everything that pulls on the part, loads and reactions alike,
    // whatever its direction: a component below a billionth of their magnitudes is 0.
    let pulls: Vec<[f64; 3]> = applied.iter().chain(solution.reactions.iter().map(|r| &r.force)).map(|f| [magnitude(*f); 3]).collect();
    let reactions: Vec<FaceReaction> =
        solution.reactions.iter().map(|r| FaceReaction { face: face_name(r.face), force: settled(r.force, &pulls) }).collect();
    let reaction_terms: Vec<[f64; 3]> = solution.reactions.iter().map(|r| r.force).collect();
    let (applied_total, reaction_total) = (settled(sum(&applied), &pulls), settled(sum(&reaction_terms), &pulls));

    let missing = values.iter().filter(|v| v.is_none()).count();
    if missing > 0 {
        warnings.push(format!("{missing} of {} surface points lie in no cell with material and have no result", values.len()));
    }
    if let Some(longest) = sampled.capped {
        warnings.push(format!(
            "the surface is sampled at {} points, the most kept: triangles up to {} across remain (the sampling size asked for is {})",
            sampled.points.len(),
            shown(longest),
            shown(sample_size),
        ));
    }
    let design_bodies = s.design.as_deref().map_or(0, |d| d.bodies.len());
    if design_bodies > 1 {
        warnings
            .push(format!("the design has {design_bodies} bodies: body `{}` was solved on its own, free wherever it touches another", shape.name));
    }

    let nodes: Vec<u32> = (0..values.len()).map(|v| GRID_LABELS.saturating_add(u32::try_from(v).unwrap_or(u32::MAX))).collect();
    let stress: Vec<[f64; 6]> = values.iter().map(|v| v.map_or([f64::NAN; 6], |(_, s)| s)).collect();
    let (shown, tri_face) = display(&sampled);
    let result = GridResult {
        body: shape.name.clone(),
        result: fieldcraft_solver::StepResult::Static(StaticResult {
            step: step.name.clone(),
            nodes,
            displacement: values.iter().map(|v| v.map_or([f64::NAN; 3], |(u, _)| u)).collect(),
            // Kept per face, not per surface point (`GridResult::reactions`); a solid has no rotations.
            rotation: Vec::new(),
            reaction: Vec::new(),
            reaction_moment: Vec::new(),
            von_mises: stress.iter().map(fieldcraft_fem::von_mises).collect(),
            stress,
            elements: Vec::new(),
            shells: Vec::new(),
            beams: Vec::new(),
            reference_points: Vec::new(),
            applied_total,
            reaction_total,
            stats: Stats { dofs: stats.unknowns, equations: stats.unknowns, factor_entries: stats.factor_entries, lanczos_vectors: 0 },
            warnings: warnings.clone(),
            temperature: None,
            nonlinear: None,
            contacts: Vec::new(),
            table_ranges: Vec::new(),
        }),
        display: shown,
        tri_face,
        faces: shape.faces().iter().map(|f| f.name.clone()).collect(),
        reactions,
    };
    let fieldcraft_solver::StepResult::Static(sampled_values) = &result.result else {
        return Err(EngineError::Internal { command: STATIC_GRID.id.into(), message: "the sampled result is not a static one".into() });
    };
    let displacement: Vec<f64> = sampled_values.displacement.iter().map(|u| magnitude(*u)).collect();
    let seconds = started.zip(crate::usage::now_ms()).map(|(from, to)| (to - from) / crate::usage::MS_PER_SECOND);
    let summary = json!({
        "step": step.name,
        "body": shape.name,
        "material": material_name,
        "cell_size": cell_size,
        "degree": degree,
        "refine": levels,
        // The cells that are not divided, by level: the grid's own first.
        "levels": stats.levels.iter().take(levels + 1).collect::<Vec<_>>(),
        "tolerance": tolerance,
        "sample_size": sample_size,
        "unknowns": stats.unknowns,
        "cells": stats.inside_cells + stats.cut_cells,
        "grid": {"cells": stats.cells, "inside": stats.inside_cells, "cut": stats.cut_cells},
        "quadrature_points": stats.quadrature_points,
        "factor_entries": stats.factor_entries,
        "seconds": seconds,
        "convergence": change.map(|c| json!({
            "against_degree": degree - 1,
            "peak_von_mises": c.peak_von_mises,
            "von_mises": c.von_mises,
            "strain_energy": c.strain_energy,
            "points": c.points,
        })),
        "max_von_mises": peak(&result, &sampled_values.von_mises),
        "max_displacement": peak(&result, &displacement),
        "applied_total": applied_total,
        "reaction_total": reaction_total,
        "reactions": result.reactions.iter().map(|r| json!({"face": r.face, "force": r.force})).collect::<Vec<_>>(),
        "volume": stats.volume,
        "surface_volume": stats.surface_volume,
        "volume_error": if stats.surface_volume > 0.0 { json!((stats.volume - stats.surface_volume) / stats.surface_volume) } else { Value::Null },
        "strain_energy": solution.strain_energy,
        "surface": {"triangles": surface.triangles().len(), "sampled_points": sampled.points.len(), "sampled_triangles": sampled.triangles.len()},
        "smallest_cell_fraction": stats.smallest_fraction,
        "warnings": warnings,
    });
    s.store_result(StepResult::Grid(Box::new(result)));
    Ok(summary)
}

/// The displacement and the stress of `solution` at every point; `None` where a point lies in no
/// cell that holds material.
fn sample(solution: &Solution, points: &[[f64; 3]]) -> Vec<Option<([f64; 3], [f64; 6])>> {
    parallel::map_grained(points.len(), GRAIN, |i| {
        let x = *points.get(i)?;
        Some((solution.displacement(x)?, solution.stress(x)?))
    })
}

#[cfg(test)]
mod tests;
