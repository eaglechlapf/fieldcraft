//! `solve.static.grid` on several bodies: every body of the design in one solve, each on a grid
//! of its own, bonded where the faces of two of them touch. The faces are found, not named
//! (`fieldcraft_fcm::find_bonds`), and the reply lists them as connections.

use std::collections::BTreeSet;

use super::*;

/// One body, ready for the solver.
struct Prepared<'a> {
    shape: &'a Shape,
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
}

/// The step on every body of the design: what [`super::run`] does for one body, for each, with
/// the bodies bonded where they touch (`"connections": "bonded"`, the default) or free of each
/// other (`"none"`).
pub(super) fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let started = crate::usage::now_ms();
    fieldcraft_model::progress::check()?;
    let threads = threads(s, p)?;
    let limits = Limits { max_dofs: cap(p, "max_dofs", s.limits.max_grid_dofs.clamp(1, Limits::MAX_DOFS))?, ..limits(p)? };
    let default_degree = Options::new(1.0).degree;
    let degree = super::super::mesh::optional_count(p, "degree", default_degree as u64, 1..=MAX_DEGREE as u64)?;
    let levels = super::super::mesh::optional_count(p, "refine", DEFAULT_REFINE, 0..=MAX_REFINE as u64)?;
    let not_yet = |what: &str| refused(format!("{what} for several bodies yet: name one `body`, or leave it out"));
    let rounds = match p.get("adapt") {
        None => 0,
        Some(Value::String(auto)) if auto == "auto" => 0,
        Some(_) => super::super::mesh::optional_count(p, "adapt", 0, 0..=MAX_REFINE as u64)?,
    };
    if rounds > 0 {
        return Err(not_yet("rounds of refinement that follow the solution (`adapt`) are not run"));
    }
    if opt::<bool>(p, "check_convergence")?.unwrap_or(false) {
        return Err(not_yet("`check_convergence` is not run"));
    }
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
    let bonded = match opt::<String>(p, "connections")?.as_deref() {
        None | Some("bonded") => true,
        Some("none") => false,
        Some(other) => return Err(EngineError::invalid("connections", format!("`{}` is neither \"bonded\" nor \"none\"", short(other)))),
    };
    let shapes = bodies(s, p)?;
    let model = &s.model;
    let step = model.select_step(opt::<String>(p, "step")?.as_deref(), |procedure| matches!(procedure, Procedure::Static {}), "static")?;
    refuse_interactions(model)?;
    let mut warnings = Vec::new();
    let shown = |v: f64| crate::units::shown(s.units(), crate::units::si(), Quantity::Length, v);

    let mut prepared: Vec<Prepared> = Vec::with_capacity(shapes.len());
    let mut least = 0.0;
    for shape in shapes {
        let (material, material_name) = material(model, &shape.name, &mut warnings)?;
        let supports = supports(model, shape, step, &mut warnings, false)?;
        // Sizes: the geometry is in millimetres, the model and the solver in SI.
        let measure = shape.measure()?;
        let (lo, hi) = (crate::units::geometry_point(measure.bbox_min), crate::units::geometry_point(measure.bbox_max));
        let diagonal = magnitude([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]);
        if !(diagonal > 0.0 && diagonal.is_finite()) {
            return Err(EngineError::Params(format!("body `{}` has no size", short(&shape.name))));
        }
        let length = |key: &str, default: f64, range: std::ops::RangeInclusive<f64>, why: &str| -> Result<f64> {
            let v = super::super::optional_number(p, key, default)?;
            if range.contains(&v) {
                Ok(v)
            } else {
                Err(EngineError::invalid(
                    key,
                    format!("must be a length from {} to {} for body `{}` ({why})", shown(*range.start()), shown(*range.end()), short(&shape.name)),
                ))
            }
        };
        let cell_size = length(
            "cell_size",
            DEFAULT_RELATIVE_CELL * diagonal,
            MIN_RELATIVE_CELL * diagonal..=MAX_RELATIVE_CELL * diagonal,
            "1e-3 of the body's bounding-box diagonal to twice it; leave it out and each body takes a twentieth of its own",
        )?;
        let volume = crate::units::geometry().to_si(Quantity::Volume, measure.volume).unwrap_or(f64::NAN);
        let skin = crate::units::geometry().to_si(Quantity::Area, measure.area).unwrap_or(f64::NAN);
        let thin = skin > 0.0 && cell_size * skin > THIN * 2.0 * volume;
        let basis = asked_basis.unwrap_or(if thin { Basis::Trunk } else { Basis::Tensor });
        let smallest_cell = cell_size / (1u64 << levels.min(MAX_REFINE as usize)) as f64;
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
        let mesh = shape.tessellate(crate::units::to_geometry_length(tolerance))?;
        let positions: Vec<[f64; 3]> = mesh.positions.iter().map(|p| crate::units::geometry_point(*p)).collect();
        let options = SurfaceOptions { weld_distance: crate::units::geometry_length(mesh.weld_distance), ..SurfaceOptions::default() };
        let surface = BoundarySurface::new(&positions, &mesh.triangles, &mesh.tri_face, &options)
            .map_err(|e| EngineError::context(format!("body `{}`: its surface cannot bound a solid", short(&shape.name)), e))?;
        let areas = face_areas(&surface);
        let (loads, applied) = loads(model, shape, step, &areas, material.density, material_name, false)?;
        least += 3.0 * (degree as f64).powi(3) * volume / cell_size.powi(3);
        let options = Options {
            degree,
            refine: levels,
            basis,
            separate: open_gaps,
            gaps: if open_gaps { GAP_LEVELS } else { 0 },
            offset: OFFSET,
            ..Options::new(cell_size)
        };
        prepared.push(Prepared { shape, material, material_name, supports, loads, applied, surface, options, tolerance, sample_size, thin });
    }
    if least > limits.max_dofs as f64 {
        return Err(EngineError::Solver(fieldcraft_solver::Error::Limit(format!(
            "about {least:.0} unknowns or more at these cell sizes and this degree; the limit is {} (use a larger cell size or a lower degree)",
            limits.max_dofs
        ))));
    }
    if prepared.iter().all(|body| body.supports.is_empty()) {
        return Err(EngineError::Params(format!(
            "step `{}` holds none of the bodies: none of its boundary conditions supports a face of {}",
            short(&step.name),
            listed(prepared.iter().map(|body| body.shape.name.as_str()))
        )));
    }

    // The connections: the faces of two bodies that touch. Two triangulations of one curved
    // face lie each within its own tolerance of it.
    fieldcraft_model::progress::stage("connections", None, "");
    fieldcraft_model::progress::check()?;
    let reach = 2.0 * prepared.iter().map(|body| body.tolerance).fold(0.0, f64::max);
    let surfaces: Vec<&BoundarySurface> = prepared.iter().map(|body| &body.surface).collect();
    let touches = if bonded { find_bonds(&surfaces, reach).map_err(grid_error)? } else { Vec::new() };
    // A body must be held: supported, or bonded to one that is held.
    let mut held: BTreeSet<usize> = prepared.iter().enumerate().filter(|(_, body)| !body.supports.is_empty()).map(|(index, _)| index).collect();
    // Every pass holds one body more, or is the last.
    for _ in 0..prepared.len() {
        let before = held.len();
        for touch in &touches {
            let (a, b) = (touch.bond.first.0, touch.bond.second.0);
            if held.contains(&a) || held.contains(&b) {
                held.extend([a, b]);
            }
        }
        if held.len() == before {
            break;
        }
    }
    let loose: Vec<&str> = prepared.iter().enumerate().filter(|(index, _)| !held.contains(index)).map(|(_, body)| body.shape.name.as_str()).collect();
    if !loose.is_empty() {
        return Err(EngineError::Params(format!(
            "nothing holds {}: no boundary condition of step `{}` supports a face of it, and it touches no body that is held{}",
            listed(loose.into_iter()),
            short(&step.name),
            if bonded { "" } else { " (with `\"connections\": \"none\"` every body needs a support of its own)" }
        )));
    }
    if prepared.iter().all(|body| body.loads.is_empty())
        && prepared.iter().flat_map(|body| &body.supports).all(|s| s.value.iter().zip(s.fixed).all(|(v, fixed)| !fixed || *v == 0.0))
    {
        warnings.push(format!("step `{}` neither loads a body nor moves a support: nothing deforms them", step.name));
    }

    fieldcraft_model::progress::stage("grid solve", None, "");
    fieldcraft_model::progress::check()?;
    let parts: Vec<Part> = prepared
        .iter()
        .map(|body| Part { surface: &body.surface, material: body.material, supports: &body.supports, loads: &body.loads, options: body.options })
        .collect();
    let bonds: Vec<Bond> = touches.iter().map(|touch| touch.bond.clone()).collect();
    let most = MAX_SAMPLES / prepared.len().max(1);
    let (solutions, sampled) = with_threads(threads, || -> Result<_> {
        let solutions = solve_assembly(&parts, &bonds, &limits).map_err(grid_error)?;
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
        Ok((solutions, sampled))
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
        faces.extend(body.shape.faces().iter().map(|f| f.name.clone()));
        face_bodies.extend(body.shape.faces().iter().map(|_| body.shape.name.clone()));
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
    for (body, solution) in prepared.iter().zip(&solutions) {
        if body.thin {
            warnings.push(format!(
                "body `{}` is thin for its grid (cells of {}): where a bend or a thin wall carries the load, the deflection can be up to a tenth too small; solved on its own (`body`), rounds of refinement show how far the result is from settled",
                short(&body.shape.name),
                shown(body.options.cell_size),
            ));
        }
        if solution.stats.holding > 0 && open_gaps {
            warnings.push(format!(
                "body `{}`: where a narrow gap ends, {} cells still join its two sides over more than five widths of the gap",
                short(&body.shape.name),
                solution.stats.holding
            ));
        }
    }
    if bonded && touches.is_empty() {
        warnings.push("no two bodies touch: each is solved on its own supports".into());
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
        let face_name = |face: u32| body.shape.faces().get(face as usize).map_or_else(|| face.to_string(), |f| f.name.clone());
        reactions.extend(solution.reactions.iter().map(|r| FaceReaction {
            body: body.shape.name.clone(),
            face: face_name(r.face),
            force: settled(r.force, &pulls),
        }));
    }
    let (applied_total, reaction_total) = (settled(sum(&applied), &pulls), settled(sum(&reaction_terms), &pulls));

    let unknowns: usize = solutions.iter().map(|solution| solution.stats.unknowns).sum();
    let factor_entries = solutions.first().map_or(0, |solution| solution.stats.factor_entries);
    let nodes: Vec<u32> = (0..values.len()).map(|v| GRID_LABELS.saturating_add(u32::try_from(v).unwrap_or(u32::MAX))).collect();
    let stress: Vec<[f64; 6]> = values.iter().map(|v| v.map_or([f64::NAN; 6], |(_, s)| s)).collect();
    let (shown_mesh, tri_face) = display(&all);
    let names: Vec<&str> = prepared.iter().map(|body| body.shape.name.as_str()).collect();
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
    let face_names = |body: usize, tags: &[u32]| -> Vec<String> {
        let faces = prepared.get(body).map(|body| body.shape.faces()).unwrap_or_default();
        tags.iter().map(|tag| faces.get(*tag as usize).map_or_else(|| tag.to_string(), |f| f.name.clone())).collect()
    };
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
        "connections": touches.iter().map(|touch| {
            let (a, b) = (&touch.bond.first, &touch.bond.second);
            json!({
                "kind": "bonded",
                "bodies": [names.get(a.0), names.get(b.0)],
                "faces": [face_names(a.0, &a.1), face_names(b.0, &b.1)],
                "area": touch.area,
            })
        }).collect::<Vec<_>>(),
        "degree": degree,
        "refine": levels,
        "gaps": if open_gaps { "open" } else { "joined" },
        "unknowns": unknowns,
        "factor_entries": factor_entries,
        "seconds": seconds,
        "max_von_mises": peak(&result, &sampled_values.von_mises),
        "max_displacement": peak(&result, &displacement),
        "applied_total": applied_total,
        "reaction_total": reaction_total,
        "reactions": result.reactions.iter().map(|r| json!({"body": r.body, "face": r.face, "force": r.force})).collect::<Vec<_>>(),
        "strain_energy": solutions.iter().map(|solution| solution.strain_energy).sum::<f64>(),
        "warnings": warnings,
    });
    s.store_result(StepResult::Grid(Box::new(result)));
    Ok(summary)
}
