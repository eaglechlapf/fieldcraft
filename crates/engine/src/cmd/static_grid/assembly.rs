//! `solve.static.grid` on several bodies: every body of the design in one solve, each on a grid
//! of its own, joined by the model's connections ([`fieldcraft_model::connections`]): bonded or
//! sliding where the faces of two of them touch. A model that was never searched for them is
//! searched first (`fieldcraft_fcm::find_bonds`), every two bodies that touch are bonded, and
//! the connections are left in the model for the user to change (`interaction.connection*`).

use std::collections::BTreeSet;

use super::*;

/// How often a solve too large for a limit is run again on larger cells, when the cell sizes
/// were left to the command: each time with half the cells.
const ENLARGE: usize = 3;

/// How often the unknowns are counted to fit the default cells to their budget: a count shows
/// how far the cells are from it, and the next ones are that much larger.
const FIT: usize = 4;

/// The least a round of fitting enlarges the cells by: what the cells cut by the surfaces, whose
/// number falls more slowly than the others', leave over of the step before.
const FIT_STEP: f64 = 1.03;

/// The unknowns a compact body has at the default cell size: a cube's, with cells
/// [`DEFAULT_RELATIVE_CELL`] of its diagonal across, (1/(√3 · that))³ of them, each with
/// 3 · degree³. The default cells of several bodies are made larger together until the bodies
/// have no more than these between them, so that an assembly costs what one part does however
/// many bodies it has.
fn compact_unknowns(degree: usize) -> f64 {
    3.0 * (degree as f64).powi(3) * (1.0 / (3f64.sqrt() * DEFAULT_RELATIVE_CELL)).powi(3)
}

/// One body, ready for the solver.
struct Prepared<'a> {
    shape: Solid<'a>,
    material: GridMaterial,
    material_name: &'a str,
    supports: Vec<Support>,
    loads: Vec<GridLoad>,
    /// The loads' resultants apart from gravity's.
    applied: Vec<[f64; 3]>,
    surface: BoundarySurface,
    options: Options,
    tolerance: f64,
    sample_size: f64,
    thin: bool,
    /// The body's volume and the area of its surface.
    volume: f64,
    skin: f64,
    /// The round surfaces its faces lie on, for the normals of those that slide.
    curved: Vec<(u32, Curved)>,
}

/// The cylinders, cones, spheres and tori the faces of a body lie on, by face, in the solver's
/// unit: the geometry's are in millimetres.
fn curved(shape: Solid) -> Result<Vec<(u32, Curved)>> {
    let point = crate::units::geometry_point;
    let surfaces = shape.face_surfaces()?;
    Ok(surfaces
        .into_iter()
        .enumerate()
        .filter_map(|(face, surface)| {
            let surface = match surface? {
                FaceSurface::Plane { .. } => return None,
                FaceSurface::Cylinder { origin, axis, .. } => Curved::Cylinder { origin: point(origin), axis },
                FaceSurface::Cone { apex, axis, half_angle } => Curved::Cone { apex: point(apex), axis, half_angle },
                FaceSurface::Sphere { center, .. } => Curved::Sphere { centre: point(center) },
                FaceSurface::Torus { center, axis, major, .. } => {
                    Curved::Torus { centre: point(center), axis, major: crate::units::geometry_length(major) }
                }
            };
            Some((u32::try_from(face).ok()?, surface))
        })
        .collect())
}

/// The bodies as the solver takes them.
fn parts_of<'a>(prepared: &'a [Prepared]) -> Vec<Part<'a>> {
    prepared
        .iter()
        .map(|body| Part {
            surface: &body.surface,
            material: body.material,
            supports: &body.supports,
            loads: &body.loads,
            options: body.options,
            curved: &body.curved,
        })
        .collect()
}

/// The model's connections as the solver's bonds, in their order: each body by its place among
/// `shapes`, each face by its tag. A free connection comes as a bond too, for its area; the
/// solver is given none of those.
fn joints(connections: &[(&str, &Connection)], shapes: &[Solid], reach: f64) -> Result<Vec<Bond>> {
    let fix = "change the connection or delete it (interaction.connection.edit, interaction.connection.delete)";
    let side = |name: &str, faces: &[GeometryRef]| -> Result<(usize, Vec<u32>)> {
        let body = faces.first().map_or("", |face| face.body.as_str());
        let missing =
            || EngineError::Params(format!("connection `{}` names body `{}`, which the open design does not have: {fix}", short(name), short(body)));
        let index = shapes.iter().position(|shape| shape.name == body).ok_or_else(missing)?;
        let shape = shapes.get(index).ok_or_else(missing)?;
        let mut tags = Vec::with_capacity(faces.len());
        for face in faces {
            let named = face.entity.name().unwrap_or_default();
            tags.push(shape.face_index(named).ok_or_else(|| {
                EngineError::Params(format!(
                    "connection `{}` names face `{}` of body `{}`, which the body does not have: {fix}",
                    short(name),
                    short(named),
                    short(body)
                ))
            })?);
        }
        Ok((index, tags))
    };
    let mut bonds = Vec::with_capacity(connections.len());
    for (name, connection) in connections {
        let grip = if connection.kind == ConnectionKind::Sliding { Grip::Sliding } else { Grip::Bonded };
        bonds.push(Bond { first: side(name, &connection.first)?, second: side(name, &connection.second)?, tolerance: reach, grip });
    }
    // Two connections that join the same two faces say the same twice, or contradict each other.
    for (at, (name, connection)) in connections.iter().enumerate() {
        for (other_name, other) in connections.iter().skip(at + 1) {
            let (bodies, others) = (connection.bodies(), other.bodies());
            if bodies != others && bodies != [others[1], others[0]] {
                continue;
            }
            let twice = connection.first.iter().find_map(|a| connection.second.iter().find(|b| other.joins(a, b)).map(|b| (a, b)));
            if let Some((a, b)) = twice {
                return Err(EngineError::Params(format!(
                    "connections `{}` and `{}` both join {a} to {b}: delete one of them, or take the face out of one",
                    short(name),
                    short(other_name)
                )));
            }
        }
    }
    Ok(bonds)
}

/// A direction for a message: of unit length, its largest component positive, to two places.
fn direction(v: [f64; 3]) -> String {
    let length = magnitude(v);
    let mut unit = v.map(|c| if length > 0.0 { c / length } else { 0.0 });
    let largest = unit.iter().map(|c| c.abs()).fold(0.0, f64::max);
    if unit.iter().find(|c| c.abs() == largest).is_some_and(|c| *c < 0.0) {
        unit = unit.map(|c| -c);
    }
    // To two places, without the zeros at the end: 1, 0.71, 0.
    let shown = |c: f64| {
        let text = format!("{c:.2}");
        let text = text.trim_end_matches('0').trim_end_matches('.');
        if text.is_empty() || text == "-0" || text == "-" { "0".to_string() } else { text.to_string() }
    };
    format!("({}, {}, {})", shown(unit[0]), shown(unit[1]), shown(unit[2]))
}

/// Why bodies that can still move are not solved, in their names: `hint` adds what is known
/// about a body that moves on its own.
fn loose(motions: &[Motion], names: &[&str], step: &str, hint: &dyn Fn(usize) -> String) -> EngineError {
    let name = |body: usize| short(names.get(body).copied().unwrap_or("?"));
    let mut alone: BTreeMap<usize, Vec<&Motion>> = BTreeMap::new();
    let mut together: BTreeSet<Vec<usize>> = BTreeSet::new();
    for motion in motions {
        match motion.bodies.as_slice() {
            [(body, _, _)] => alone.entry(*body).or_default().push(motion),
            several => {
                together.insert(several.iter().map(|(body, _, _)| *body).collect());
            }
        }
    }
    let mut said = Vec::new();
    for (body, own) in &alone {
        // All six: neither a support nor a connection touches it.
        if own.len() >= 6 {
            said.push(format!(
                "nothing holds `{}`: no boundary condition of step `{}` supports a face of it, and no connection joins it to another body{}",
                name(*body),
                short(step),
                hint(*body)
            ));
            continue;
        }
        // The ways it can slide, then those it can turn.
        let mut ways: Vec<(bool, String)> = own
            .iter()
            .filter_map(|motion| motion.bodies.first())
            .map(|(_, slide, turn)| {
                // A turn about an axis off the body's middle moves the middle too.
                if magnitude(*turn) > f64::EPSILON.sqrt() * magnitude(*slide) {
                    (true, format!("turn about an axis along {}", direction(*turn)))
                } else {
                    (false, format!("slide along {}", direction(*slide)))
                }
            })
            .collect();
        ways.sort();
        let ways: Vec<String> = ways.into_iter().map(|(_, way)| way).collect();
        let ways = match ways.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
            _ => ways.join(""),
        };
        said.push(format!("`{}` can still {ways}{}", name(*body), hint(*body)));
    }
    for group in &together {
        let moving: Vec<String> = group.iter().take(NAMED).map(|body| format!("`{}`", name(*body))).collect();
        said.push(format!("{} can still move together", moving.join(", ")));
    }
    said.truncate(NAMED);
    EngineError::Params(format!(
        "{}. A static solve needs every body held against sliding and turning: a sliding connection holds only across its faces, so add a boundary condition, or make a connection bonded (interaction.connection.edit)",
        said.join("; ")
    ))
}

/// The step on every body of the design: what [`super::run`] does for one body, for each, with
/// the bodies joined by the model's connections (`"connections": "model"`, the default) or free
/// of each other (`"none"`).
pub(super) fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let started = crate::usage::now_ms();
    fieldcraft_model::progress::check()?;
    let threads = threads(s, p)?;
    let limits = Limits { max_dofs: cap(p, "max_dofs", s.limits.max_grid_dofs.clamp(1, Limits::MAX_DOFS))?, ..limits(p)? };
    let default_degree = Options::new(1.0).degree;
    let degree = super::super::mesh::optional_count(p, "degree", default_degree as u64, 1..=MAX_DEGREE as u64)?;
    let levels = super::super::mesh::optional_count(p, "refine", DEFAULT_REFINE, 0..=MAX_REFINE as u64)?;
    // The rounds that follow the solution, over all the bodies together: a number, or as many
    // as a body that is thin for its grid needs.
    let asked_rounds = match p.get("adapt") {
        None => None,
        Some(Value::String(auto)) if auto == "auto" => None,
        Some(_) => Some(super::super::mesh::optional_count(p, "adapt", 0, 0..=MAX_REFINE as u64)?),
    };
    let settle = super::super::optional_number(p, "settle", Options::SETTLED)?;
    if !(0.0..=1.0).contains(&settle) {
        return Err(EngineError::invalid("settle", "must be a fraction from 0 to 1 (0 runs every round)"));
    }
    // Refused at degree 1 by the caller.
    let check_convergence = opt::<bool>(p, "check_convergence")?.unwrap_or(false);
    let solver = solver(p)?;
    let asked_basis = match opt::<String>(p, "basis")?.as_deref() {
        None | Some("auto") => None,
        Some("tensor") => Some(Basis::Tensor),
        Some("trunk") => Some(Basis::Trunk),
        Some(other) => return Err(EngineError::invalid("basis", format!("`{}` is none of \"auto\", \"tensor\" and \"trunk\"", short(other)))),
    };
    let open_gaps = match opt::<String>(p, "gaps")?.as_deref() {
        None | Some("open") => true,
        Some("joined") => false,
        Some(other) => return Err(EngineError::invalid("gaps", format!("`{}` is neither \"open\" nor \"joined\"", short(other)))),
    };
    let joined = match opt::<String>(p, "connections")?.as_deref() {
        None | Some("model") => true,
        Some("none") => false,
        Some(other) => return Err(EngineError::invalid("connections", format!("`{}` is neither \"model\" nor \"none\"", short(other)))),
    };
    let shapes = bodies(s, p)?;
    let model = &s.model;
    let step = model.select_step(opt::<String>(p, "step")?.as_deref(), |procedure| matches!(procedure, Procedure::Static {}), "static")?;
    refuse_interactions(model)?;
    let mut warnings = Vec::new();
    let shown = |v: f64| crate::units::shown(s.units(), crate::units::si(), Quantity::Length, v);

    // Sizes: the geometry is in millimetres, the model and the solver in SI.
    let mut sizes = Vec::with_capacity(shapes.len());
    for shape in &shapes {
        let measure = shape.measure()?;
        let (lo, hi) = (crate::units::geometry_point(measure.bbox_min), crate::units::geometry_point(measure.bbox_max));
        let diagonal = magnitude([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]);
        if !(diagonal > 0.0 && diagonal.is_finite()) {
            return Err(EngineError::Params(format!("body `{}` has no size", short(shape.name))));
        }
        let volume = crate::units::geometry().to_si(Quantity::Volume, measure.volume).unwrap_or(f64::NAN);
        let skin = crate::units::geometry().to_si(Quantity::Area, measure.area).unwrap_or(f64::NAN);
        sizes.push((diagonal, volume, skin));
    }
    let default_cells = p.get("cell_size").is_none();

    let mut prepared: Vec<Prepared> = Vec::with_capacity(shapes.len());
    let mut least = 0.0;
    for (shape, (diagonal, volume, skin)) in shapes.into_iter().zip(sizes) {
        let (material, material_name) = material(model, shape.name, &mut warnings)?;
        let supports = supports(model, shape, step, &mut warnings, false)?;
        let length = |key: &str, default: f64, range: std::ops::RangeInclusive<f64>, why: &str| -> Result<f64> {
            let v = super::super::optional_number(p, key, default)?;
            if range.contains(&v) {
                Ok(v)
            } else {
                Err(EngineError::invalid(
                    key,
                    format!("must be a length from {} to {} for body `{}` ({why})", shown(*range.start()), shown(*range.end()), short(shape.name)),
                ))
            }
        };
        let cell_size = length(
            "cell_size",
            DEFAULT_RELATIVE_CELL * diagonal,
            MIN_RELATIVE_CELL * diagonal..=MAX_RELATIVE_CELL * diagonal,
            "1e-3 of the body's bounding-box diagonal to twice it; leave it out and each body takes a twentieth of its own, or more when the bodies are many",
        )?;
        let thin = skin > 0.0 && cell_size * skin > THIN * 2.0 * volume;
        let basis = asked_basis.unwrap_or(if thin { Basis::Trunk } else { Basis::Tensor });
        // The facets are far smaller than the smallest cell the rounds may make.
        let rounds = asked_rounds.unwrap_or(if thin { THIN_ROUNDS } else { 0 });
        let smallest_cell = cell_size / (1u64 << (levels + rounds).min(MAX_REFINE as usize)) as f64;
        let tolerance = length(
            "tolerance",
            (DEFAULT_TOLERANCE_PER_CELL * smallest_cell).clamp(MIN_RELATIVE_TOLERANCE * diagonal, DEFAULT_RELATIVE_TOLERANCE * diagonal),
            MIN_RELATIVE_TOLERANCE * diagonal..=diagonal,
            "1e-5 of the body's bounding-box diagonal to all of it",
        )?;
        let finest = cell_size / (MAX_SAMPLES_PER_DEGREE * degree as f64);
        let sample_size =
            length("sample_size", cell_size / degree as f64, finest..=diagonal, "an eighth of cell_size/degree to the body's bounding-box diagonal")?;

        fieldcraft_model::progress::stage(&format!("{}: surface", shape.name), None, "");
        fieldcraft_model::progress::check()?;
        let surface = surface(shape, tolerance)?;
        let areas = face_areas(&surface);
        let (loads, applied) = loads(model, shape, step, &areas, material.density, material_name, false)?;
        least += 3.0 * (degree as f64).powi(3) * volume / cell_size.powi(3);
        let options = Options {
            degree,
            refine: levels,
            settle,
            basis,
            solver,
            separate: open_gaps,
            gaps: if open_gaps { GAP_LEVELS } else { 0 },
            offset: OFFSET,
            ..Options::new(cell_size)
        };
        prepared.push(Prepared {
            shape,
            material,
            material_name,
            supports,
            loads,
            applied,
            surface,
            options,
            tolerance,
            sample_size,
            thin,
            volume,
            skin,
            curved: curved(shape)?,
        });
    }
    // Cells of a size that was asked for are refused before anything is built on them; ours
    // are few (the bodies share a compact body's), and a solve too large is run again below.
    if least > limits.max_dofs as f64 && !default_cells {
        return Err(EngineError::Solver(fieldcraft_solver::Error::Limit(format!(
            "about {least:.0} unknowns or more at these cell sizes and this degree; the limit is {} (use a larger cell size or a lower degree)",
            limits.max_dofs
        ))));
    }
    if prepared.iter().all(|body| body.supports.is_empty()) {
        return Err(EngineError::Params(format!(
            "step `{}` holds none of the bodies: none of its boundary conditions supports a face of {}",
            short(&step.name),
            listed(prepared.iter().map(|body| body.shape.name))
        )));
    }

    // The connections: the model's. A model that was never searched for them is searched now,
    // and every two bodies whose faces touch are bonded there: an assembly is solved on the
    // first click, and what joined it is in the model afterwards, to change.
    fieldcraft_model::progress::stage("connections", None, "");
    fieldcraft_model::progress::check()?;
    // Two triangulations of one curved face lie each within its own tolerance of it.
    let reach = 2.0 * prepared.iter().map(|body| body.tolerance).fold(0.0, f64::max);
    let surfaces: Vec<&BoundarySurface> = prepared.iter().map(|body| &body.surface).collect();
    let shapes: Vec<Solid> = prepared.iter().map(|body| body.shape).collect();
    let names: Vec<&str> = shapes.iter().map(|shape| shape.name).collect();
    let searched = joined && !model.connections_found();
    let found = if searched {
        let touches = find_bonds(&surfaces, reach).map_err(grid_error)?;
        super::super::connection::new_connections(model, &shapes, &touches, ConnectionKind::Bonded)?
    } else {
        Vec::new()
    };
    let connections: Vec<(&str, &Connection)> = if joined {
        let kept = model.connections().iter().map(|(name, connection)| (name.as_str(), connection));
        kept.chain(found.iter().map(|(name, connection)| (name.as_str(), connection))).collect()
    } else {
        Vec::new()
    };
    let joints = joints(&connections, &shapes, reach)?;
    let touching = bond_areas(&parts_of(&prepared), &joints).map_err(grid_error)?;
    // What the solver joins: not what is free, and not faces that do not touch.
    let mut bonds: Vec<Bond> = Vec::new();
    for (((name, connection), bond), area) in connections.iter().zip(&joints).zip(&touching) {
        if connection.kind == ConnectionKind::Free {
            continue;
        }
        if *area > 0.0 {
            bonds.push(bond.clone());
        } else {
            let [first, second] = connection.bodies();
            warnings.push(format!(
                "connection `{}` joins nothing: the faces it names of `{}` and `{}` do not touch (they are more than {} apart)",
                short(name),
                short(first),
                short(second),
                shown(reach)
            ));
        }
    }
    // Every body must be held against every way it can move as a rigid piece: by supports, and
    // by connections to bodies that are held.
    let free = free_motions(&parts_of(&prepared), &bonds).map_err(grid_error)?;
    if !free.is_empty() {
        // What a body that moves touches without being held by it.
        let touches = if joined { find_bonds(&surfaces, reach).unwrap_or_default() } else { Vec::new() };
        let hint = |body: usize| -> String {
            if !joined {
                return " (with `\"connections\": \"none\"` every body needs supports of its own)".into();
            }
            let own = names.get(body).copied().unwrap_or_default();
            for touch in &touches {
                let other = match (touch.bond.first.0, touch.bond.second.0) {
                    (a, b) if a == body => b,
                    (a, b) if b == body => a,
                    _ => continue,
                };
                let other = names.get(other).copied().unwrap_or_default();
                let between = |connection: &Connection| connection.bodies() == [own, other] || connection.bodies() == [other, own];
                let mut joining = connections.iter().zip(&touching).filter(|((_, connection), _)| between(connection));
                if joining.clone().any(|((_, connection), area)| connection.kind != ConnectionKind::Free && *area > 0.0) {
                    continue;
                }
                return match joining.find(|((_, connection), _)| connection.kind == ConnectionKind::Free) {
                    Some(((name, _), _)) => format!(" (it touches `{}`, and their connection `{}` is free)", short(other), short(name)),
                    None => format!(
                        " (it touches `{}`, and no connection joins the two: interaction.connection.find adds the connections that are missing)",
                        short(other)
                    ),
                };
            }
            String::new()
        };
        return Err(loose(&free, &names, &step.name, &hint));
    }
    if prepared.iter().all(|body| body.loads.is_empty())
        && prepared.iter().flat_map(|body| &body.supports).all(|s| s.value.iter().zip(s.fixed).all(|(v, fixed)| !fixed || *v == 0.0))
    {
        warnings.push(format!("step `{}` neither loads a body nor moves a support: nothing deforms them", step.name));
    }
    let imported = prepared.iter().filter(|body| body.shape.imported()).count();
    if imported > 0 {
        warnings.push(imported_note(imported));
    }

    fieldcraft_model::progress::stage("grid solve", None, "");
    fieldcraft_model::progress::check()?;
    // Cells that grow take the samples with them, and a body can become thin for them. The
    // rounds that follow the solution are the assembly's: as many as were asked for, or those of
    // a thin body when any of the bodies is thin for its grid.
    let own_samples = p.get("sample_size").is_none();
    let grow = |prepared: &mut [Prepared], by: f64| {
        for body in prepared.iter_mut() {
            body.options.cell_size *= by;
            if own_samples {
                body.sample_size *= by;
            }
            body.thin = body.skin > 0.0 && body.options.cell_size * body.skin > THIN * 2.0 * body.volume;
            if asked_basis.is_none() && body.thin {
                body.options.basis = Basis::Trunk;
            }
        }
        let rounds = asked_rounds.unwrap_or(if prepared.iter().any(|body| body.thin) { THIN_ROUNDS } else { 0 });
        for body in prepared.iter_mut() {
            body.options.adapt = rounds;
        }
    };
    // The cells as they are: the rounds of the bodies as they come.
    grow(&mut prepared, 1.0);
    // The default cells: a twentieth of each body's own diagonal, and all of them larger by one
    // factor when the bodies would have more unknowns between them than one compact body has.
    // The unknowns are counted on the bodies' grids, with nothing integrated.
    if default_cells {
        let budget = compact_unknowns(degree);
        // Counted against the solver's own limit: a lower one refuses the solve below, which
        // then says that it made the cells larger.
        let countable = Limits { max_dofs: Limits::MAX_DOFS, ..limits };
        for _ in 0..FIT {
            fieldcraft_model::progress::check()?;
            let parts = parts_of(&prepared);
            let by = match with_threads(threads, || assembly_unknowns(&parts, &bonds, &countable)) {
                Ok(unknowns) if unknowns as f64 <= budget => break,
                Ok(unknowns) => (unknowns as f64 / budget).cbrt().max(FIT_STEP),
                Err(fieldcraft_fcm::Error::Limit(_)) => 2f64.cbrt(),
                Err(e) => return Err(grid_error(e)),
            };
            grow(&mut prepared, by);
        }
    }
    // A solve still too large for a limit is run again with half the cells, when the sizes are
    // ours to choose: an assembly is solved on the first click, coarser, and says so.
    let (mut enlarged, mut refused_at) = (0, None);
    let solutions = loop {
        let parts = parts_of(&prepared);
        match with_threads(threads, || solve_assembly(&parts, &bonds, &limits)) {
            Err(fieldcraft_fcm::Error::Limit(limit)) if default_cells && enlarged < ENLARGE && fieldcraft_model::progress::check().is_ok() => {
                enlarged += 1;
                grow(&mut prepared, 2f64.cbrt());
                refused_at = Some(limit);
            }
            other => break other.map_err(grid_error)?,
        }
    };
    if let Some(limit) = refused_at {
        warnings.push(format!(
            "the cells are {:.2} times larger than the command chose at first, to fit a limit ({limit}): give `cell_size`, a lower `degree` or fewer levels of `refine` to choose what is given up instead",
            2f64.powf(enlarged as f64 / 3.0)
        ));
    }
    let most = MAX_SAMPLES / prepared.len().max(1);
    let (solutions, sampled, change) = with_threads(threads, || -> Result<_> {
        fieldcraft_model::progress::stage("results", None, "");
        fieldcraft_model::progress::check()?;
        let sampled: Vec<(Sampled, Vec<Option<([f64; 3], [f64; 6])>>)> = prepared
            .iter()
            .zip(&solutions)
            .map(|(body, solution)| {
                let cell_size = body.options.cell_size;
                let sampled = refine(&body.surface, body.sample_size, |x| solution.cell_size(x).map_or(1.0, |leaf| leaf / cell_size), most);
                let values = sample(solution, &sampled.points);
                (sampled, values)
            })
            .collect();
        // The p-extension's own check: the same step one degree lower on the same grids, and
        // how far the two solutions are apart at the sampled points of every body.
        let change = if check_convergence {
            fieldcraft_model::progress::stage(&format!("convergence check at degree {}", degree - 1), None, "");
            fieldcraft_model::progress::check()?;
            let coarser: Vec<Part> =
                parts_of(&prepared).into_iter().map(|part| Part { options: Options { degree: degree - 1, ..part.options }, ..part }).collect();
            let lower = solve_assembly(&coarser, &bonds, &limits).map_err(grid_error)?;
            let bodies: Vec<(&Solution, &Solution, &[[f64; 3]])> =
                solutions.iter().zip(&lower).zip(&sampled).map(|((finer, lower), (sampled, _))| (finer, lower, sampled.points.as_slice())).collect();
            Some(Solution::change_of(&bodies))
        } else {
            None
        };
        Ok((solutions, sampled, change))
    })?;

    // One surface of all the bodies: one body's points, triangles and faces after another's.
    let mut all = Sampled { points: Vec::new(), triangles: Vec::new(), tri_face: Vec::new(), capped: None };
    let (mut values, mut faces, mut face_bodies) = (Vec::new(), Vec::new(), Vec::new());
    for (body, (sampled, own)) in prepared.iter().zip(sampled) {
        let (first_point, first_face) = (u32::try_from(all.points.len()).unwrap_or(u32::MAX), u32::try_from(faces.len()).unwrap_or(u32::MAX));
        all.points.extend(sampled.points);
        all.triangles.extend(sampled.triangles.iter().map(|t| t.map(|v| v.saturating_add(first_point))));
        all.tri_face.extend(sampled.tri_face.iter().map(|face| face.saturating_add(first_face)));
        all.capped = match (all.capped, sampled.capped) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        values.extend(own);
        face_bodies.extend(body.shape.face_names().iter().map(|_| body.shape.name.to_string()));
        faces.extend(body.shape.face_names());
    }
    let missing = values.iter().filter(|v| v.is_none()).count();
    if missing > 0 {
        warnings.push(format!("{missing} of {} surface points lie in no cell with material and have no result", values.len()));
    }
    if let Some(longest) = all.capped {
        warnings.push(format!(
            "the surfaces are sampled at {} points, the most kept: triangles up to {} across remain",
            all.points.len(),
            shown(longest)
        ));
    }
    // How far the result is from settled: the rounds are the assembly's, and every body's
    // solution counts them. Every round asked for (`settle` 0) is read against the default.
    let (rounds, moved, left_out) =
        solutions.first().map_or((0, 0.0, false), |solution| (solution.stats.rounds, solution.stats.settled, solution.stats.left_out));
    let bar = if settle > 0.0 { settle } else { Options::SETTLED };
    let still_moving = rounds > 0 && moved.abs().sqrt() > bar;
    if still_moving || (left_out && rounds > 0) {
        warnings.push(format!(
            "the result was still changing when the rounds of refinement ended: the last one changed the strain energy by {:.2} %{}",
            100.0 * moved.abs(),
            if left_out {
                "; one more round would have been too large to solve (a smaller `cell_size` with fewer rounds, or a higher `max_factor_entries`, goes further)"
            } else {
                "; more rounds (`adapt`) or a smaller `cell_size` bring it closer"
            },
        ));
    } else if left_out {
        warnings.push(
            "no round of refinement was run: the first would have been too large to solve (a larger `cell_size`, a lower `degree` or a higher `max_factor_entries` leaves room for them)"
                .into(),
        );
    }
    for (body, solution) in prepared.iter().zip(&solutions) {
        if body.thin && rounds == 0 && !left_out {
            warnings.push(format!(
                "body `{}` is thin for its grid (cells of {}): where a bend or a thin wall carries the load, the deflection can be up to a tenth too small; rounds of refinement (`adapt`) or a smaller `cell_size` show how far the result is from settled",
                short(body.shape.name),
                shown(body.options.cell_size),
            ));
        }
        if solution.stats.holding > 0 && open_gaps {
            warnings.push(format!(
                "body `{}`: where a narrow gap ends, {} cells still join its two sides over more than five widths of the gap",
                short(body.shape.name),
                solution.stats.holding
            ));
        }
    }
    if joined && bonds.is_empty() {
        warnings.push("no connection joins two bodies: each is solved on its own supports".into());
    }

    // The resultants. Gravity loads the volume the solver integrated.
    let mut applied: Vec<[f64; 3]> = Vec::new();
    for (body, solution) in prepared.iter().zip(&solutions) {
        applied.extend(&body.applied);
        for load in &body.loads {
            if let GridLoad::Gravity { acceleration } = load {
                applied.push(acceleration.map(|a| a * body.material.density * solution.stats.volume));
            }
        }
    }
    let sum = |terms: &[[f64; 3]]| terms.iter().fold([0.0; 3], |t, f| [t[0] + f[0], t[1] + f[1], t[2] + f[2]]);
    let reaction_terms: Vec<[f64; 3]> = solutions.iter().flat_map(|solution| solution.reactions.iter().map(|r| r.force)).collect();
    let pulls: Vec<[f64; 3]> = applied.iter().chain(&reaction_terms).map(|f| [magnitude(*f); 3]).collect();
    let mut reactions = Vec::new();
    for (body, solution) in prepared.iter().zip(&solutions) {
        let face_name = |face: u32| body.shape.face_name(face).unwrap_or_else(|| face.to_string());
        reactions.extend(solution.reactions.iter().map(|r| FaceReaction {
            body: body.shape.name.to_string(),
            face: face_name(r.face),
            force: settled(r.force, &pulls),
        }));
    }
    let (applied_total, reaction_total) = (settled(sum(&applied), &pulls), settled(sum(&reaction_terms), &pulls));

    let unknowns: usize = solutions.iter().map(|solution| solution.stats.unknowns).sum();
    let factor_entries = solutions.first().map_or(0, |solution| solution.stats.factor_entries);
    let iterations = solutions.first().map_or(0, |solution| solution.stats.iterations);
    let nodes: Vec<u32> = (0..values.len()).map(|v| GRID_LABELS.saturating_add(u32::try_from(v).unwrap_or(u32::MAX))).collect();
    let stress: Vec<[f64; 6]> = values.iter().map(|v| v.map_or([f64::NAN; 6], |(_, s)| s)).collect();
    let (shown_mesh, tri_face) = display(&all);
    let result = GridResult {
        body: names.join(", "),
        result: fieldcraft_solver::StepResult::Static(StaticResult {
            step: step.name.clone(),
            nodes,
            displacement: values.iter().map(|v| v.map_or([f64::NAN; 3], |(u, _)| u)).collect(),
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
            stats: Stats { dofs: unknowns, equations: unknowns, factor_entries, lanczos_vectors: 0 },
            warnings: warnings.clone(),
            temperature: None,
            nonlinear: None,
            contacts: Vec::new(),
            table_ranges: Vec::new(),
        }),
        display: shown_mesh,
        tri_face,
        faces,
        face_bodies,
        reactions,
    };
    let fieldcraft_solver::StepResult::Static(sampled_values) = &result.result else {
        return Err(EngineError::Internal { command: STATIC_GRID.id.into(), message: "the sampled result is not a static one".into() });
    };
    let displacement: Vec<f64> = sampled_values.displacement.iter().map(|u| magnitude(*u)).collect();
    let seconds = started.zip(crate::usage::now_ms()).map(|(from, to)| (to - from) / crate::usage::MS_PER_SECOND);
    let summary = json!({
        "step": step.name,
        "bodies": prepared.iter().zip(&solutions).map(|(body, solution)| {
            let stats = &solution.stats;
            json!({
                "body": body.shape.name,
                "material": body.material_name,
                "cell_size": body.options.cell_size,
                "basis": if body.options.basis == Basis::Trunk { "trunk" } else { "tensor" },
                "thin": body.thin,
                "tolerance": body.tolerance,
                "unknowns": stats.unknowns,
                "cells": stats.inside_cells + stats.cut_cells,
                "levels": stats.levels.iter().take(stats.levels.iter().rposition(|cells| *cells > 0).map_or(1, |last| last + 1).max(levels + 1)).collect::<Vec<_>>(),
                "gap_cells": if stats.split_cells > 0 { json!(stats.split_cells) } else { Value::Null },
                "volume": stats.volume,
                "strain_energy": solution.strain_energy,
            })
        }).collect::<Vec<_>>(),
        "connections": connections.iter().zip(&touching).map(|((name, connection), area)| {
            let mut listed = super::super::connection::listed(name, connection);
            if let Some(listed) = listed.as_object_mut() {
                listed.insert("area".into(), json!(area));
            }
            listed
        }).collect::<Vec<_>>(),
        "connections_added": found.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(),
        "degree": degree,
        "refine": levels,
        // The rounds that followed the solution, and how much the last one still changed the
        // strain energy of all the bodies.
        "adapt": rounds,
        "settled": if rounds > 0 { json!(moved) } else { Value::Null },
        "convergence": change.map(|c| json!({
            "against_degree": degree - 1,
            "peak_von_mises": c.peak_von_mises,
            "von_mises": c.von_mises,
            "strain_energy": c.strain_energy,
            "points": c.points,
        })),
        "gaps": if open_gaps { "open" } else { "joined" },
        "unknowns": unknowns,
        "factor_entries": factor_entries,
        "solver": solved_by(iterations),
        "iterations": if iterations > 0 { json!(iterations) } else { Value::Null },
        "residual": if iterations > 0 { json!(solutions.first().map_or(0.0, |solution| solution.stats.residual)) } else { Value::Null },
        "seconds": seconds,
        "max_von_mises": peak(&result, &sampled_values.von_mises),
        "max_displacement": peak(&result, &displacement),
        "applied_total": applied_total,
        "reaction_total": reaction_total,
        "reactions": result.reactions.iter().map(|r| json!({"body": r.body, "face": r.face, "force": r.force})).collect::<Vec<_>>(),
        "strain_energy": solutions.iter().map(|solution| solution.strain_energy).sum::<f64>(),
        "warnings": warnings,
    });
    // The connections the search found are the model's from here on, and the result is of them.
    if searched {
        for (name, connection) in found {
            s.model.add_connection(&name, connection)?;
        }
        s.model.set_connections_found(true);
    }
    s.store_result(StepResult::Grid(Box::new(result)));
    Ok(summary)
}
