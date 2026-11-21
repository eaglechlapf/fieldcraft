# Grid Static: a static solve without a mesh

**Solve ▸ Grid Static** (`solve.static.grid`) solves a linear static step of one body with no
volume mesh. The body's surface is embedded in a grid of cubic cells, each cell carries
polynomial shape functions of a chosen degree, and the equations are integrated over the part
only (the finite cell method, `crates/fcm`). Short edges, sliver faces and small fillets, which
can stop a mesher, are only places where a cell is cut. Where the surface is curved tightly the
cells are divided into smaller ones, so a fillet or a hole is resolved without fine cells
everywhere. Material on the two sides of a gap that is narrower than a cell (the open corner of
a sheet metal part, a slit, the fins of a heat sink) is kept apart, as it is in the part.

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
| `basis` | `"auto"` | The shape functions of a cell. `"tensor"`: every product of the polynomials along the three axes up to the degree. `"trunk"`: only those of total degree up to it, half the unknowns a cell at degree 3. `"auto"`: the trunk space for a body that is thin for the grid, the tensor product otherwise. The reply's `basis` names the one used. See [Rounds that follow the solution](#rounds-that-follow-the-solution). |
| `refine` | 1 | Levels of local refinement, 0 to 6: a cell across which the surface is curved more tightly than it resolves is divided into eight, this many times at most. `levels` in the reply counts the cells of each level. |
| `adapt` | `"auto"` | Rounds of refinement that follow the solution, 0 to 6: the step is solved, the cells the solution is least resolved in are divided, and it is solved again. `"auto"`: up to two rounds for a body that is thin for the grid (a cell more than four of its mean thicknesses across), none otherwise. The reply's `adapt` counts the rounds run and `settled` is the change of the strain energy in the last. See [Rounds that follow the solution](#rounds-that-follow-the-solution). |
| `settle` | 0.05 | The rounds end once one moves the solution by no more than this fraction in the energy norm (0.05 is a change of the strain energy of 0.25 %). 0 runs every round. |
| `gaps` | `"open"` | `"open"`: material on the two sides of a gap narrower than a cell has unknowns of its own on each side, and the cells where such a gap ends are divided (four times at most). `"joined"`: the part is solved as if such gaps were welded shut wherever a cell reaches across them. See [Narrow gaps](#narrow-gaps-sheet-metal-corners-slits-fins). |
| `check_convergence` | false | Also solve one degree lower on the same grid and report the change (`convergence`); needs degree 2 or more. |
| `tolerance` | 1/250 of the smallest cell, at most 1e-3 of the diagonal | The chordal tolerance of the surface's triangulation, which is all the solver knows of the geometry. The smallest cell is `cell_size` halved once for every level of `refine` and every round of `adapt`. |
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

The reply's `gap_cells` counts the cells that hold material in separate pieces, each with
unknowns of its own there (`gap_unknowns` more than the grid alone has); neither is there when
no cell holds a gap. `levels` counts the cells of every level, those divided where gaps end
included.

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

These levels follow the curvature of the surface and nothing else. They do not look at the
solution: a stress that peaks at a sharp re-entrant edge or where a support ends is not refined
towards, and a curved face that carries no stress is refined all the same. The rounds below do
look at it.

### Rounds that follow the solution

With **Adaptive rounds** (`adapt`) the step is solved, the cells the solution is least resolved
in are divided into eight, and the step is solved again, as many times as there are rounds. How
well a cell resolves the solution is read from the solution itself: the strain energy of the
part of it that only the functions of the highest degree carry in that cell, which is none
where the lower degrees already represent the field (the hierarchic indicator of the p-version:
Szabó & Babuška, *Finite Element Analysis*, 1991). The cells that carry nine tenths of the sum
are divided, largest first (Dörfler, 1996). Each round is a whole solve on a larger system.

The rounds end early once one moves the solution by no more than `settle`: the distance between
two rounds' solutions in the energy norm, relative to the solution, which is the square root of
the change of the strain energy. The default of 5 % (a change of the strain energy of 0.25 %) is
the permissible error Zienkiewicz & Zhu take in the examples of their adaptive procedure (1987).
A round whose system would be too large is left out, and the result is the solution before it.
The reply's `settled` is the change of the strain energy in the last round, and a warning says
so when the result was still changing as the rounds ended. Like **Check convergence**, this
tells how much the last refinement still moved the result; it is not a bound of the error.

What the rounds are for, and what they are not for:

- **They find what the surface does not show.** A sheet metal part has a bend at every wall,
  and the load goes through one of them: `refine` divides the cells of all of them once, the
  rounds divide the cells of the one that carries the load again and again. On the tray of
  [Narrow gaps](#narrow-gaps-sheet-metal-corners-slits-fins), hung by one wall, the sag is 12 %
  low without a round, 7 % low after one and 4 % low after two (in the tensor product; the
  trunk space, below, gets there in half the time).
- **They do not replace `refine` at a hole or a fillet.** The indicator follows the strain
  energy, and a stress peak holds little of it. On a plate with a hole of the radius of one
  cell in tension (degree 3, `crates/fcm/tests/hole.rs`), the stress at the hole over six
  positions of the grid is within ± 10 % on the grid's own cells, still ± 7 % after two rounds
  and ± 1.6 % after three, where two levels of `refine` give ± 0.7 % in the time of the two
  rounds. On the sample bracket, one round on top of the default level changes the strain
  energy by 0.001 % and the peak on the fillet from 212.5 to 211.2 MPa (reference 213), for
  three to four times the run time.

So the form leaves **Adaptive rounds** empty, which is `"auto"`: up to two rounds when the body
is thin for the grid (a cell is more than four of its mean thicknesses across, the thickness
taken as twice the volume over the surface's area), where a bend or a thin wall is likely to
carry the load on cells that do not resolve it, and none otherwise. A number runs that many
rounds on any body; 0 runs none, for a first look at a sheet metal part in a fifth of the
time.

**The shape functions** (`basis`) go with the rounds. The tensor product carries every product
of the polynomials along the three axes up to the degree, 64 functions a cell at degree 3. The
trunk space of Szabó and Babuška carries only those of total degree up to the degree, 32 a
cell: it still holds every polynomial of that degree, and lacks the products beyond it. Cell
for cell it is the less accurate, and on cells that are the same everywhere it needs about two
degrees more for the same stress. Where the cells are divided as the part needs it, smaller
cells make up for what it lacks, and it is the cheaper way to the same accuracy:

| | Tensor product | Trunk space |
|---|---|---|
| Hole of one cell's radius, cells divided twice there, degree 3: stress at the hole over six grid positions | ± 0.7 %, 78 000 unknowns | ± 1.1 %, 24 000 unknowns |
| the same, trunk space at degree 4 | | ± 0.5 %, 42 000 unknowns |
| Tray hung by one wall, two rounds: sag | −3.8 %, 322 000 unknowns, 36 s | −4.3 %, 145 000 unknowns, 20 s |
| Tray bolted, two rounds: sag | −0.2 % after one round (the second does not fit), 27 s | 0.0 %, 156 000 unknowns, 20 s |
| Sample bracket as the form opens, degree 3: peak on the fillet | 212.5 MPa (−0.2 %), 145 000 unknowns, 7 s | 210.5 MPa (−1.2 %), 40 000 unknowns, 2 s |
| the same, trunk space at degree 4 | | 212.4 MPa (−0.3 %), 73 000 unknowns, 5 s |

The form's **Shape functions** is *Automatic*: the trunk space for a body that is thin for the
grid, where the rounds divide the cells anyway, and the tensor product otherwise, which is what
every other table on this page was measured with. For a part that does not fit the solver's
caps, or takes long, the trunk space at one degree more is worth trying.

A round takes the cells it leaves as they are from the round before: integrating a cell the
surface cuts is the most work a cell takes (on one thread, more than the factorization of a
sheet metal part), and a round leaves most cells as they were.

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

(Times on a 24-core desktop. The table was measured again with the narrow gaps open, the
default: every peak is the same to the digit shown, and a few rows have up to one per cent more
unknowns, where a cell holds the two sides of a hole.) Read with the guidance above, for a
fillet of 6 mm:

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

## Narrow gaps: sheet metal corners, slits, fins

A cell carries one polynomial, and a polynomial cannot tell the two sides of a narrow gap apart.
Where one cell holds the two flanges at the open corner of a sheet metal part, the two jaws of a
slit clamp or two fins of a heat sink, it would join them: the part would be solved as if the
gap were welded shut, with no warning, and far too stiff. So the solver sorts the material of
every cut cell into the pieces that are not connected inside the cell and gives each piece
shape functions of its own there; two cells share the functions of the face between them only
where material goes on through it (`crates/fcm/src/islands.rs`: the overlapping elements of
Hansbo & Hansbo, 2004). Nothing is to be set or drawn for it: it follows the body's surface,
and a part without such gaps is solved exactly as before, to the last bit.

Where a gap ends, its two sides are one piece of material again, and the cell at the root
joins them over its own length. The cells whose shape functions do that are divided, level by
level (four at most, and no more cells than the part had), until they are no more than two and
a half widths of the gap across. Nothing else is divided for it. When cells are left that join
the sides of a gap over more than five of its widths, the reply warns.

**In theory** (`crates/fcm/tests/gap.rs`): a fork whose prongs are 10 mm thick and 1 mm apart,
on cells of 17 mm, pulled by one prong. The pulled prong must carry the stress of the load and
the other nothing.

| | Stress in the pulled prong | In the other prong |
|---|---|---|
| Gaps joined | 89.2 % of the load, along its whole length | 10.8 % |
| Gaps open, from two cells off the root on | 100.000 % | 0.000 % |
| Gaps open, 10 mm from the root | 97.0 % | 4.7 % |
| The same with the cells divided twice where the gap ends | 100.3 % | 0.3 % |

**On a real part** (`crates/fcm/tests/tray.rs`): a tray of 2 mm steel, 400 × 300 mm with four
walls 50 mm high bent up from its floor (inside bend radius 2 mm), holes in the floor and the
walls, and four open corners, where the ends of two walls are 2 mm apart. The form as it opens
gives cells of 25 mm, a dozen sheet thicknesses. The reference is FieldCraft's own mesh solve of
the same model on quadratic tetrahedra of 4 mm (920 000 unknowns; 44 s to mesh and 26 s to
solve, and only with `max_factor_entries` raised to 1.5 billion: with the default the mesh
solve is refused).

| | Bolted at four holes, 1 kPa on the floor: sag at the passage | Hung by one wall, 100 N on the opposite wall: sag at the passage |
|---|---|---|
| Mesh solve (reference) | 0.1532 mm | 2.175 mm |
| Grid, gaps joined | 0.1508 mm (−1.6 %), 6 s | 0.027 mm (−98.8 %), 6 s |
| Grid, gaps open, no cells divided where they end (the solver's option, not the form's) | 0.1518 mm (−0.9 %) | 1.054 mm (−52 %) |
| Grid, gaps open, tensor product, no round that follows the solution (`"basis": "tensor", "adapt": 0`) | 0.1520 mm (−0.8 %), 11 s | 1.915 mm (−12 %), 9 s |
| the same on cells of 12.5 mm | | 2.044 mm (−6 %), about 20 s |
| Grid, gaps open, tensor product, rounds that follow the solution (`"basis": "tensor"`) | 0.1530 mm (−0.2 %), 27 s | 2.093 mm (−3.8 %), 36 s |
| Grid, gaps open, trunk space, no round (`"adapt": 0`) | 0.1497 mm (−2.3 %), 4 s | 1.834 mm (−15.7 %), 4 s |
| **Grid as the form opens** (gaps open, trunk space, rounds that follow the solution) | **0.1532 mm (0.0 %), 20 s** | **2.082 mm (−4.3 %), 20 s** |

Bolted, the corners carry little, and joined or open the floor sags the same to a per cent.
Hung by one wall, everything goes through the bend along that wall and the side walls end free
at its corners: joined to it they carry the load as ribs, and the tray sags an eightieth of
what it does.

**Thin sheet is the solver's hard case.** What is left of the error in the hung tray is the
bend along the clamped wall, 2 mm of sheet bent to an inside radius of 2 mm, in cells that hold
all of it: they resolve neither the bend nor the reliefs at its ends, where the stress
concentrates. A section through that bend alone (the wall, the bend and a strip of the floor;
measured once, not kept as a test) is 8 % too stiff on cells of 25 mm, 6 % with the cells at
the bend divided once, 2 % divided twice and 0.7 % divided three times. Dividing the cells of
every bend and hole that far is more than the solver holds (cells of 12.5 mm divided once are
6 % low with 400 000 unknowns), which is what the rounds that follow the solution are for:
they divide the cells of the bend that carries the load. In the trunk space, which a thin
body takes, the tray hung by one wall is 4.3 % low after two of them with 145 000 unknowns,
and 2.8 % low after three (`"adapt": 3`, 219 000 unknowns, 42 s); bolted, the sag is within
0.1 % after two. The von Mises stress at seven points of the floor (measured once, not kept as
a test) is within 1 % of the mesh solve's at six of them and 4 % low at one, where it is up
to 14 % low without the rounds.

Each round still divides a cell along all three axes where a bend needs it across the bend
only. So the rounds buy accuracy at a price: the default takes 20 s on this tray where a solve
without them takes 4 s, and the reply's warning says when the result was still changing as
they ended (here by 4 % of the strain energy in the last round). Cells that follow the sheet
(divided across a bend only) are the next step for it.

The gaps stay as wide as they are: the solve is linear, and two sides that would close and
touch under load pass through each other. `"gaps": "joined"` (the form's **Narrow gaps**) solves
the part as if every gap narrower than a cell were welded shut, which is what a corner seam
that is welded after bending wants.

## Assemblies: what comes next

The grid solves one body. What it needs for several is what the gaps already use: every piece
of material has unknowns of its own in the cells it shares with another, and pieces are joined
only where something joins them. Two bodies are two pieces by definition, so in one grid over
an assembly no cell joins them by accident, however close they are; what joins them is a
*connection*, and connections can be found from the geometry instead of being drawn by hand:

1. **Faces that touch.** Two faces of two bodies that lie on each other are found as the meshed
   solve finds contact pairs (`interaction.contact.find`), and are *bonded* unless told
   otherwise. On the grid a bond is cheap: the two bodies' pieces in the cells along the common
   face share their unknowns there (exact for one material; with two, the cell integrates each
   body with its own constants), or are tied in the weak form over the pieces of the common
   face (Nitsche's method, as the supports are), which also carries *sliding* (the normal
   component alone) and later contact.
2. **Holes in line.** Coaxial holes through bodies that touch are bolt candidates
   (`interaction.bolt` of the roadmap's M7): a beam with the bolt's section, its head and nut
   bearing on the annuli around the holes.
3. **Edges that end on a face.** A plate standing on another is a weld seam candidate
   (`interaction.weld`).

Each found connection is listed with its kind, its two faces and its area, drawn in the
viewport, and can be switched (bonded, sliding, free) before the solve: the engineer reviews a
list instead of building one. In this order: several bodies with their own materials and every
touching face bonded; the list and its switches; bolts and seams; sliding; contact, which needs
the nonlinear step.
