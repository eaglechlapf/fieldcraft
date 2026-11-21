# Grid Static: a static solve without a mesh

**Solve ▸ Grid Static** (`solve.static.grid`) solves a linear static step of one body with no
volume mesh. The body's surface is embedded in a grid of cubic cells, each cell carries
polynomial shape functions of a chosen degree, and the equations are integrated over the part
only (the finite cell method, `crates/fcm`). Short edges, sliver faces and small fillets, which
can stop a mesher, are only places where a cell is cut.

The model is the one a mesh solve reads: a material and a solid section on the body, boundary
conditions and loads on sets of its faces, a static step. So the same model can be solved both
ways and compared.

## Try it

1. Start the app with the sample: `cargo run --release -p fieldcraft -- --sample bracket-grid`
   (or **File ▸ Samples ▸ Bracket Without a Mesh** in a running app). It opens a steel L-bracket
   with a fillet and two holes, clamped under its foot, 2 kN on the end of its upright, and no
   mesh.
2. Choose the **Analysis** workspace (the workspace switch at the left of the ribbon), tab
   **Solve**, and click **Grid Static** in the *Implicit* panel, beside *Static*. Or press
   Ctrl+K (⌘K) and type `grid`.
3. Type `4` for **Cell size** (the display units are millimetres), leave the rest as it is and
   click **OK**. The solve runs in the background, with Cancel, and takes about 3 seconds. Left
   empty, the cell size is 1/20 of the body's bounding-box diagonal (4.39 mm here); the degree
   is 3. Tick **Check convergence** to solve one degree lower as well and see how much
   the result still changed.
4. The Results workspace comes up with the von Mises stress on the bracket and its legend. The
   card on the right lists the summary: the largest displacement and stress and where they
   are, the applied and reaction forces, cell size, degree, unknowns, cells, the volume in the
   cells against the volume of the body, the run time, and the reaction of each supported face.
5. In the Results card, **Contours** changes the field (displacement, stress components),
   **Deformed** shows the deformed shape, **Reaction sums** lists the reaction of each
   supported set, and a click on the part probes the nearest surface point.

The reference for this bracket is a peak von Mises stress of 213 MPa on the fillet
(`tests/oracle/filleted-bracket-fillet-stress`); 4 mm cells of degree 3 give 212 MPa there, and
it is the largest stress of the part. That grid has only 1.5 cells in the fillet's radius, so
the close agreement is not to be relied on: other grids of this fineness are up to 7 % off (see
[Accuracy](#accuracy-how-many-cells-which-degree)), and a face beside the fillet or the edge of
the clamped foot, where the exact stress is singular, can show the largest value. To read the
fillet alone, click it, or run `solve.probe {"step": "Load", "set": "fillet"}`, whose
`max_von_mises` is the largest value on the fillet face.

## Your own part

1. **File ▸ Open** a `.solidcraft` design.
2. **Properties**: a material (isotropic elastic), a set that names the body, a solid section on
   that set.
3. Sets of faces: in the viewport choose *Select: Faces*, pick the faces, right click,
   **Create set from selection**.
4. **Loads & BCs**: boundary conditions and loads on those sets (see the table below).
5. **Steps**: a static step with them active.
6. **Solve ▸ Grid Static**.

Nothing needs to be meshed. If the body is meshed too, **Static** solves the same step on the
mesh; each solve replaces the step's result.

## From a script

```json
[
  {"command": "file.sample", "params": {"name": "bracket-grid"}},
  {"command": "solve.static.grid", "params": {"cell_size": 4, "degree": 3}},
  {"command": "view.image.save", "params": {"path": "bracket.png"}}
]
```

`fieldcraft-cli run script.json` runs it headless; over MCP or the control channel
(`engine.execute`) the command replies with its summary. It never opens a form.

## Parameters

| Parameter | Default | |
|---|---|---|
| `step` | the only static step | The step to solve. |
| `body` | the only body | The body of the open design to solve, on its own. |
| `cell_size` | 1/20 of the bounding-box diagonal | The size of the grid's cubic cells (a length). |
| `degree` | 3 | The polynomial degree of the shape functions, 1 to 8. |
| `check_convergence` | false | Also solve one degree lower on the same grid and report the change (`convergence`); needs degree 2 or more. |
| `tolerance` | 1e-3 of the diagonal | The chordal tolerance of the surface's triangulation, which is all the solver knows of the geometry. |
| `sample_size` | `cell_size` / `degree` | The results are read at points of the surface no farther apart than this. |
| `max_dofs` | 6 000 000 | A cap on the unknowns; may only be lowered. |
| `max_factor_entries` | 400 000 000 | A cap on the entries of the factorization (8 bytes each), as for `solve.static`. |
| `threads` | the session's | Changes the time only, never the result. |

## What it solves, and what it refuses

It solves one body: linear elastic, isotropic, static.

| | Solved | Refused, with the reason |
|---|---|---|
| Boundary conditions | *Fixed*, *Displacement* (its translations; a prescribed value may be non-zero) and *Symmetry* about a plane normal to X, Y or Z, on sets of whole faces | a temperature, an inclined symmetry plane, a reference point |
| Loads | *Force* (a total over the set's faces), *Traction* and *Pressure* on sets of whole faces; *Gravity* | moments, body forces, centrifugal and thermal loads, a force at every node |
| Targets | sets that name faces of the body | sets of nodes, elements, edges or vertices, and faces of another body |
| Material | isotropic elasticity; plasticity is not used | orthotropic elasticity |
| Model | one body | contacts, ties, couplings, point masses, springs; a cell of a partitioned body |

A design with several bodies needs the `body` named, and the reply warns that the body was
solved on its own.

## Results

The solver gives displacement and stress at any point of the part. They are kept on the
body's surface: at the vertices of its triangulation, refined until no triangle edge is longer
than `sample_size`. The results display reads them like a mesh solve's:

- contours, the deformed shape, the legend and its range (`results.contour`,
  `results.deformed`);
- a click probes the nearest surface point (`solve.probe` with `at`, with `nodes`, or with a
  `set` of faces);
- `results.reactions` sums the reactions over each set of supported faces. The solver gives a
  force per face and not where it acts, so there are no moments.

Not available yet: the stress inside the part (slices, paths, stress linearization), VTU export,
XY plots. A grid result is not written to the model file: solve again after opening it.

## Accuracy: how many cells, which degree

The grid is the same everywhere, so the smallest stressed feature decides the cell size, and
the degree decides how much one cell can carry. Measured on holes and fillets
(`crates/fcm/tests`):

- a stress peak at a hole or a fillet is within about 1 % with **two cells per radius** of the
  smallest stressed feature at degree 4, or **four cells per radius** at degree 3;
- with one cell per radius the peak is only good to about ±10 % at degree 3, 6.5 % at degree 4
  and 3 % at degree 5;
- displacements and reactions are much less sensitive.

**Check convergence** in the form (`check_convergence`) solves the step a second time, one
degree lower on the same grid, and reports under `convergence` how far the two solutions are
apart: the change of the largest von Mises stress, the largest change at any one point (as a
fraction of the peak) and the change of the strain energy. A small change is necessary for a
converged result and does not prove one: it is an indicator, not a bound of the error.

On the sample bracket (8 mm thick, R6 fillet, reference 213 MPa on the fillet face), with the
default tessellation:

| Cell size | Cells in the R6 | Degree | Unknowns | Time | Peak on the fillet | Against 213 MPa | Largest displacement |
|---|---|---|---|---|---|---|---|
| 8 mm | 0.75 | 2 | 5 265 | 0.1 s | 177.5 MPa | −16.6 % | 0.1495 mm |
| 8 mm | 0.75 | 3 | 15 960 | 0.5 s | 226.8 MPa | +6.5 % | 0.1534 mm |
| 8 mm | 0.75 | 4 | 35 775 | 1.5 s | 213.1 MPa | 0.0 % | 0.1541 mm |
| 8 mm | 0.75 | 5 | 67 518 | 4.7 s | 205.0 MPa | −3.7 % | 0.1544 mm |
| 4 mm | 1.5 | 2 | 25 557 | 0.5 s | 233.7 MPa | +9.7 % | 0.1538 mm |
| 4 mm | 1.5 | 3 | 80 340 | 2.5 s | 212.3 MPa | −0.3 % | 0.1544 mm |
| 4 mm | 1.5 | 4 | 183 573 | 10 s | 215.0 MPa | +0.9 % | 0.1544 mm |
| 3 mm | 2 | 3 | 153 699 | 7 s | 219.8 MPa | +3.2 % | 0.1544 mm |
| 4.39 mm (the default) | 1.4 | 3 | 67 890 | 2.0 s | 225.0 MPa | +5.6 % | 0.1544 mm |
| 4.39 mm (the default) | 1.4 | 4 | 155 103 | 8 s | 220.3 MPa | +3.4 % | 0.1544 mm |

(Times on a 24-core desktop.) The convergence check on these grids: 4 mm at degree 3 against
degree 2 changes the peak by 13 % and the strain energy by 0.34 %; at degree 4 against degree
3 by 1.2 % and 0.02 %; 8 mm at degree 5 against degree 4 by 4.9 % and 0.13 %.

Read with the guidance above: these grids have 0.75 to 2 cells in the fillet's radius, and from
degree 3 on the peak lies between −4 % and +7 % of the reference without settling steadily; a
single close hit (4 mm at degree 3, 8 mm at degree 4) is no proof of convergence. The grids the
guidance asks for this fillet, 3 mm at degree 4 or 1.5 mm at degree 3, are refused on this part:
their matrices (211 and 356 million entries) are over the solver's cap of 180 million, as is
degree 5 at the default cell size. The largest displacement has settled at 0.1544 mm to 0.2 %
from 8 mm at degree 4 or 4 mm at degree 3 on, and the clamped face carries the 2 kN on every
grid.

Where the answer can be poor:

- at a re-entrant edge with no fillet and at the edge of a clamped face the exact stress is
  infinite, so the peak there grows with every refinement, as it does on a mesh;
- in a cell that holds very little material the stress can be noisy; `smallest_cell_fraction`
  in the reply says how little the emptiest cut cell holds;
- `volume` against `surface_volume` shows how well the cut cells were integrated; they agree to
  the tessellation's accuracy.
