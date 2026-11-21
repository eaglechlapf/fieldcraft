# Grid Static: a static solve without a mesh

**Solve ▸ Grid Static** (`solve.static.grid`) solves a linear static step of one body with no
volume mesh. The body's surface is embedded in a grid of cubic cells, each cell carries
polynomial shape functions of a chosen degree, and the equations are integrated over the part
only (the finite cell method, `crates/fcm`). Short edges, sliver faces and small fillets, which
can stop a mesher, are only places where a cell is cut. Where the surface is curved tightly the
cells are divided into smaller ones, so a fillet or a hole is resolved without fine cells
everywhere.

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
3. Leave the form as it is and click **OK**. The solve runs in the background, with Cancel, and
   takes about 8 seconds. Left empty, the cell size is 1/20 of the body's bounding-box diagonal
   (4.39 mm here); the degree is 3, and **Refinement levels** is 1: the cells at the fillet and
   the holes are divided once, to 2.2 mm. Tick **Check convergence** to solve one degree lower
   as well and see how much the result still changed.
4. The Results workspace comes up with the von Mises stress on the bracket and its legend. The
   card on the right lists the summary: the largest displacement and stress and where they
   are, the applied and reaction forces, cell size, degree, unknowns, cells, the volume in the
   cells against the volume of the body, the run time, and the reaction of each supported face.
5. In the Results card, **Contours** changes the field (displacement, stress components),
   **Deformed** shows the deformed shape, **Reaction sums** lists the reaction of each
   supported set, and a click on the part probes the nearest surface point.

The reference for this bracket is a peak von Mises stress of 213 MPa on the fillet, within 3 %
(`tests/oracle/filleted-bracket-fillet-stress`). The form as it is gives 212.5 MPa there, and it
is the largest stress of the part. With **Refinement levels** at 0 the same cells give 226 MPa,
6 % too much: they hold only 1.4 cells in the fillet's radius (see
[Accuracy](#accuracy-how-many-cells-which-degree-how-many-levels)). On other grids a face beside
the fillet, or the edge of the clamped foot, where the exact stress is singular, can show the
largest value. To read the fillet alone, click it, or run
`solve.probe {"step": "Load", "set": "fillet"}`, whose `max_von_mises` is the largest value on
the fillet face.

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
  {"command": "solve.static.grid", "params": {"cell_size": 8, "degree": 3, "refine": 2}},
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
| `refine` | 1 | Levels of local refinement, 0 to 6: a cell across which the surface is curved more tightly than it resolves is divided into eight, this many times at most. `levels` in the reply counts the cells of each level. |
| `check_convergence` | false | Also solve one degree lower on the same grid and report the change (`convergence`); needs degree 2 or more. |
| `tolerance` | 1/250 of the smallest cell, at most 1e-3 of the diagonal | The chordal tolerance of the surface's triangulation, which is all the solver knows of the geometry. The smallest cell is `cell_size` halved `refine` times. |
| `sample_size` | `cell_size` / `degree` | The results are read at points of the surface no farther apart than this; where cells are divided, as much closer as the cells are smaller. |
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
than `sample_size` (where cells are divided, than that fraction of `sample_size` that the
divided cell is of `cell_size`). The results display reads them like a mesh solve's:

- contours, the deformed shape, the legend and its range (`results.contour`,
  `results.deformed`);
- a click probes the nearest surface point (`solve.probe` with `at`, with `nodes`, or with a
  `set` of faces);
- `results.reactions` sums the reactions over each set of supported faces. The solver gives a
  force per face and not where it acts, so there are no moments.

Not available yet: the stress inside the part (slices, paths, stress linearization), VTU export,
XY plots. A grid result is not written to the model file: solve again after opening it.

## Accuracy: how many cells, which degree, how many levels

The smallest stressed feature decides how small the cells must be there, and the degree decides
how much one cell can carry. Measured on holes and fillets (`crates/fcm/tests`):

- a stress peak at a hole or a fillet is within about 1 % with **two cells per radius** of the
  smallest stressed feature at degree 4, or **four cells per radius** at degree 3;
- with one cell per radius the peak is only good to about ±10 % at degree 3, 6.5 % at degree 4
  and 3 % at degree 5;
- displacements and reactions are much less sensitive.

**Refinement levels** (`refine`) puts those cells where they are needed. A cell across which the
surface is curved more tightly than that is divided into eight, and its parts again, as many
times as there are levels; the cells beside a divided one are divided too. A part with flat
faces only is not divided and costs nothing more. A part with curved faces takes two to five
times the unknowns for one level (`crates/fcm/tests/corpus.rs`, 52 bodies). Measured against
closed-form and published values (`crates/fcm/tests/benchmarks.rs`):

| Case | Cells | Degree | No levels | One level | Two levels |
|---|---|---|---|---|---|
| Plate with a hole in tension (Kirsch), stress at the top of the hole | one per radius | 3 | +16.6 % | +0.8 % | +0.5 % |
| the same, stress at the side of the hole | one per radius | 3 | +32.8 % | +4.4 % | +1.6 % |
| the same, top and side | one per radius | 4 | +4.4 %, +26.8 % | +0.1 %, +0.2 % | |
| Elliptic membrane (NAFEMS LE1), stress at D | 0.5 m | 3 | +10.4 % | +1.0 % | +0.03 % |

Refinement follows the curvature of the surface and nothing else. It does not look at the
solution: a stress that peaks at a sharp re-entrant edge or where a support ends is not refined
towards, and a curved face that carries no stress is refined all the same.

**The triangulation matters with small cells.** The solver knows the body by its triangles
alone, and a solution on small cells resolves the creases between the facets of a concave fillet
as the notches they are. On the sample bracket, grids with cells of 1.5 to 2.2 mm at the R6
fillet read 4 to 7 % high at a chordal tolerance of 0.088 mm (1e-3 of the diagonal), up to 4 %
high at 0.02 mm, and within 0.7 % from 0.01 mm on; the largest displacement moves by 0.4 % over
that range. So the default `tolerance` is 1/250 of the smallest cell. A finer triangulation
costs little: six times the triangles add about a tenth to the run time.

**Check convergence** in the form (`check_convergence`) solves the step a second time, one
degree lower on the same grid, and reports under `convergence` how far the two solutions are
apart: the change of the largest von Mises stress, the largest change at any one point (as a
fraction of the peak) and the change of the strain energy. A small change is necessary for a
converged result and does not prove one: it is an indicator, not a bound of the error. It reads
on the cautious side: for the sample bracket as the form opens, degree 3 against degree 2
changes the peak by 6.6 % and the strain energy by 0.07 %, while the peak is 0.2 % from the
reference.

On the sample bracket (8 mm thick, R6 fillet, reference 213 MPa on the fillet face, within 3 %),
with the default triangulation:

| Cell size | Levels | Cells at the fillet | Degree | Unknowns | Time | Peak on the fillet | Against 213 MPa |
|---|---|---|---|---|---|---|---|
| 8 mm | 0 | 8 mm | 2 | 5 265 | 0.2 s | 178.9 MPa | −16.0 % |
| 8 mm | 0 | 8 mm | 3 | 15 960 | 0.6 s | 228.3 MPa | +7.2 % |
| 8 mm | 0 | 8 mm | 4 | 35 775 | 1.7 s | 214.5 MPa | +0.7 % |
| 4.39 mm | 0 | 4.39 mm | 3 | 67 890 | 2.4 s | 226.2 MPa | +6.2 % |
| 4.39 mm | 0 | 4.39 mm | 4 | 155 103 | 9.0 s | 213.6 MPa | +0.3 % |
| 4 mm | 0 | 4 mm | 3 | 80 340 | 3.0 s | 210.8 MPa | −1.0 % |
| 4 mm | 0 | 4 mm | 4 | 183 573 | 10.7 s | 210.9 MPa | −1.0 % |
| 3 mm | 0 | 3 mm | 3 | 153 699 | 7.6 s | 208.1 MPa | −2.3 % |
| **4.39 mm (the default)** | **1** | **2.2 mm** | **3** | **145 077** | **7.6 s** | **212.5 MPa** | **−0.2 %** |
| 8 mm | 1 | 4 mm | 3 | 51 816 | 2.5 s | 201.3 MPa | −5.5 % |
| 8 mm | 2 | 2 mm | 3 | 144 348 | 8.8 s | 211.6 MPa | −0.7 % |
| 12 mm | 2 | 3 mm | 3 | 94 893 | 5.0 s | 217.5 MPa | +2.1 % |
| 12 mm | 3 | 1.5 mm | 3 | 256 731 | 20.0 s | 211.5 MPa | −0.7 % |
| 16 mm | 3 | 2 mm | 3 | 151 449 | 11.9 s | 213.3 MPa | +0.2 % |
| 8 mm | 1 | 4 mm | 4 | 120 147 | 8.3 s | 208.6 MPa | −2.1 % |
| 6 mm | 1 | 3 mm | 4 | 226 449 | 18.1 s | 210.6 MPa | −1.1 % |
| 12 mm | 2 | 3 mm | 4 | 222 009 | 18.2 s | 212.4 MPa | −0.3 % |
| 16 mm | 2 | 4 mm | 4 | 124 824 | 10.6 s | 217.2 MPa | +2.0 % |

(Times on a 24-core desktop.) Read with the guidance above, for a fillet of 6 mm:

- with cells of 2.2 mm or less at the fillet at degree 3 (four grids), the peak lies between
  −0.7 % and +0.2 % of the reference; with cells of 3 mm at degree 4 (two grids), −1.1 % and
  −0.3 %;
- with cells of 3 to 4.4 mm at degree 3 it lies anywhere between −5.5 % and +6.2 %, and with
  cells of 4 mm and more at degree 4 between −2.1 % and +2.0 %: a single close hit there is no
  proof of convergence;
- this bracket is small against its fillet and holes, so more than half of its cells are
  divided and refinement saves little against small cells everywhere (8 mm with two levels
  and the default with one have the unknowns of 3 mm cells). The saving grows with the part:
  the Kirsch plate above has 106 000 unknowns with no levels and 118 000 with two, where cells
  of a quarter of the radius everywhere would be 64 times as many cells;
- where the grid sits matters too: over six positions of the grid, 8 mm cells with two levels
  at degree 3 give 211 to 221 MPa, with none 184 to 243 (`crates/fcm/tests/bracket.rs`, at a
  tolerance of 0.05 mm).

What is refused on this part: two levels at the default cell size (a factorization of 510
million entries against the cap of 400 million), and degree 4 with one level at the default
cell size (a stiffness of 228 million entries against the solver's cap of 180 million). The
largest displacement is 0.1549 to 0.1550 mm on every grid of the table but the three of 8 mm
cells with no levels (0.1499, 0.1538 and 0.1546 mm), and the clamped face carries the 2 kN on
every grid.

Where the answer can be poor:

- at a re-entrant edge with no fillet and at the edge of a clamped face the exact stress is
  infinite, so the peak there grows with every refinement, as it does on a mesh;
- in a cell that holds very little material the cell's own stress is noisy (up to a quarter off
  in a cell one per cent full). In a cell that holds less than a tenth of its volume the stress
  is therefore read as a weighted mean of the polynomials of the cells around the point. That
  removes the outliers, and where the cells resolve the stress well it costs about a per cent
  at such a point. `smallest_cell_fraction` in the reply says how little the emptiest cut cell
  holds;
- `volume` against `surface_volume` shows how well the cut cells were integrated; they agree to
  the tessellation's accuracy.
