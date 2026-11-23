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

A body with no mesh shows its supports and loads on its surface: the same symbols a mesh
shows at its nodes, spread over the faces their sets name (triangles where a face is held,
arrows for forces, pressures and tractions), and thinned to the zoom. **Loads & BCs** in the
browser hides and shows them.

## Try it

1. Start the app with the sample: `cargo run --release -p fieldcraft -- --sample bracket-grid`
   (or **File ▸ Samples ▸ Bracket Without a Mesh** in a running app). It opens a steel L-bracket
   with a fillet and two holes, clamped under its foot, 2 kN on the end of its upright, and no
   mesh.
2. Choose the **Analysis** workspace (the workspace switch at the left of the ribbon), tab
   **Solve**, and click **Grid Static** in the *Implicit* panel, beside *Static*. Or press
   Ctrl+K (⌘K) and type `grid`.
3. Leave the form as it is and click **OK**. The solve runs in the background, with Cancel, and
   takes about 4 seconds. Left empty, the cell size is 1/20 of the body's bounding-box diagonal
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

1. **File ▸ Open** a `.solidcraft` design. A part or an assembly from another CAD system
   comes in as a face-tagged 3MF surface mesh instead (**Import mesh…** on the start page,
   `geometry.import.mesh`): each of its bodies is solved like a body of a design, its faces
   are named by their numbers, and its curved faces are as fine as the file's triangles.
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
| `body` | every body | A body of the open design to solve on its own. Left out: every body of the design, solved together ([Several bodies](#several-bodies-an-assembly)). |
| `connections` | `"model"` | With several bodies. `"model"`: the bodies hold each other by the model's connections, bonded, sliding or free; a model never searched for them is searched first, and what touches is bonded ([Several bodies](#several-bodies-an-assembly)). `"none"`: no body holds another. |
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
| `solver` | `"auto"` | How the unknowns are solved for: `"direct"` by factorization, `"iterative"` by conjugate gradients, `"auto"` whichever is predicted to be the less work. See [How large a model](#how-large-a-model). |
| `max_factor_entries` | 400 000 000 | A cap on the entries of the factorization (8 bytes each), as for `solve.static`, and on the entries of the stiffness the iteration keeps. See [How large a model](#how-large-a-model). |
| `threads` | the session's | Changes the time only, never the result. |

## What it solves, and what it refuses

It solves one body, or several that are bonded or slide where they touch: linear elastic, isotropic, static.

| | Solved | Refused, with the reason |
|---|---|---|
| Boundary conditions | *Fixed*, *Displacement* (its translations; a prescribed value may be non-zero) and *Symmetry* about a plane normal to X, Y or Z, on sets of whole faces | a temperature, an inclined symmetry plane, a reference point |
| Loads | *Force* (a total over the set's faces), *Traction* and *Pressure* on sets of whole faces; *Gravity* | moments, body forces, centrifugal and thermal loads, a force at every node |
| Targets | sets that name faces of the body, or of any of the bodies solved together | sets of nodes, elements, edges or vertices; for one body on its own, faces of another body |
| Material | isotropic elasticity; plasticity is not used | orthotropic elasticity |
| Model | one body, or every body of the design | contacts, ties, couplings, point masses, springs; a cell of a partitioned body |

With `body` named in a design of several, the reply warns that the body was solved on its own,
free wherever it touches another.

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
part of it that only the functions of the highest degree carry in that cell, which is none where
the lower degrees already represent the field (the hierarchic indicator of the p-version: Szabó
& Babuška, *Finite Element Analysis*, 1991), or the cell's whole strain energy where that is
less. Both bound what the lower degrees leave of the solution in the cell (by dropping the
highest functions, or all of them), and the second keeps a cell that holds a sliver of material
from taking every round: there the highest functions and the lower ones carry large parts that
cancel. The cells that carry nine tenths of the sum are divided, largest first (Dörfler, 1996).
Each round is a whole solve on a larger system.

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
  [Narrow gaps](#narrow-gaps-sheet-metal-corners-slits-fins), hung by one wall, the sag is 9 %
  low without a round, 4 % low after one and 0.6 % low after two (in the tensor product; the
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
rounds on any body; 0 runs none, for a first look at a sheet metal part in a quarter of the
time.

**The shape functions** (`basis`) go with the rounds. The tensor product carries every product
of the polynomials along the three axes up to the degree, 64 functions a cell at degree 3. The
trunk space of Szabó and Babuška carries only those of total degree up to the degree, 32 a
cell: it still holds every polynomial of that degree, and lacks the products beyond it. Cell
for cell it is the less accurate, and on cells that are the same everywhere it needs about two
degrees more for the same stress. Where the cells are divided as the part needs it, smaller
cells make up for what it lacks, and it is the cheaper way to about the same accuracy:

| | Tensor product | Trunk space |
|---|---|---|
| Hole of one cell's radius, cells divided twice there, degree 3: stress at the hole over six grid positions | ± 0.7 %, 78 000 unknowns | ± 1.1 %, 24 000 unknowns |
| the same, trunk space at degree 4 | | ± 0.5 %, 42 000 unknowns |
| Tray hung by one wall, two rounds: sag | −0.2 %, 513 000 unknowns, 20 s | −1.8 %, 165 000 unknowns, 9 s |
| Tray bolted, two rounds: sag | +0.4 %, 874 000 unknowns, 30 s | +0.2 %, 266 000 unknowns, 12 s |
| Sample bracket as the form opens, degree 3: peak on the fillet | 212.5 MPa (−0.2 %), 145 000 unknowns, 2.1 s | 210.5 MPa (−1.2 %), 40 000 unknowns, 1.3 s |
| the same, trunk space at degree 4 | | 212.4 MPa (−0.3 %), 73 000 unknowns, 1.5 s |

The form's **Shape functions** is *Automatic*: the trunk space for a body that is thin for the
grid, where the rounds divide the cells anyway, and the tensor product otherwise, which is what
every other table on this page was measured with. For a part that does not fit the solver's
caps, or takes long, the trunk space at one degree more is worth trying.

A round takes the cells it leaves as they are from the round before, and it leaves most cells as
they were.

**The cut cells are integrated exactly.** The solver knows the body by its triangles alone, so
the material in a cell is a polyhedron, and what a cell needs of it are integrals of
polynomials. They are taken from the triangles in the cell by the divergence theorem, to
rounding: there are no sampling points inside a cell and no setting for their number, and the
work grows with the number of triangles in a cell, not with its square. The reply's `volume`
therefore agrees with `surface_volume` to rounding (6e-13 on the sheet metal tray of
[Narrow gaps](#narrow-gaps-sheet-metal-corners-slits-fins), where sampling the cells left
1e-5). Sampling them took a fifth of that tray's solve on 24 threads and close to half of it
on one; the exact integrals take a tenth of that time. One kind of cell is still
sampled, on an octree of smaller boxes where the surface in it is curved: a cell whose highest
corner lies on the body's surface, as when a face of the part lies exactly in a plane of the
grid. The grid starts off the body's corner by an odd fraction of a cell so that this is rare;
it costs time, not accuracy.

**The unknowns are ordered cell by cell.** The equations are solved by a direct factorization,
whose size and time depend on the order in which the unknowns are eliminated. The functions of
a cell all couple with each other, so the part is divided along the faces between cells: the
cells are split into two groups, the functions the two groups share are eliminated last, and
each group is divided again (a nested dissection by elements, the cut chosen as the one that
shares the fewest functions among those that leave each group a third of the cells). On the
tray the factor then has 86 million entries where dividing the part by the positions of its
functions, through the cells, gave 150 million, and takes about a third of the work; over 300 solves
of other parts the factor is a quarter smaller on average and never larger. That is most of
why the tray as the form opens takes 9 s where it took 21 s, and why two levels at the sample
bracket's default cell size fit the default `max_factor_entries` where they did not.

**The triangulation matters with small cells.** A solution on small cells resolves the creases
between the facets of a concave fillet as the notches they are. On the sample bracket, grids with cells of 1.5 to 2.2 mm at the R6
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
| 8 mm | 0 | 8 mm | 2 | 5 265 | 0.1 s | 178.9 MPa | −16.0 % |
| 8 mm | 0 | 8 mm | 3 | 15 960 | 0.3 s | 228.3 MPa | +7.2 % |
| 8 mm | 0 | 8 mm | 4 | 35 775 | 0.8 s | 214.5 MPa | +0.7 % |
| 4.39 mm | 0 | 4.39 mm | 3 | 67 890 | 1.2 s | 226.2 MPa | +6.2 % |
| 4.39 mm | 0 | 4.39 mm | 4 | 155 103 | 4.5 s | 213.6 MPa | +0.3 % |
| 4 mm | 0 | 4 mm | 3 | 80 340 | 1.5 s | 210.8 MPa | −1.0 % |
| 4 mm | 0 | 4 mm | 4 | 183 573 | 5.5 s | 210.9 MPa | −1.0 % |
| 3 mm | 0 | 3 mm | 3 | 153 699 | 3.3 s | 208.1 MPa | −2.3 % |
| **4.39 mm (the default)** | **1** | **2.2 mm** | **3** | **145 077** | **2.1 s** | **212.5 MPa** | **−0.2 %** |
| 8 mm | 1 | 4 mm | 3 | 51 816 | 1.2 s | 201.3 MPa | −5.5 % |
| 8 mm | 2 | 2 mm | 3 | 144 348 | 4.1 s | 211.6 MPa | −0.6 % |
| 12 mm | 2 | 3 mm | 3 | 94 893 | 2.4 s | 217.7 MPa | +2.2 % |
| 12 mm | 3 | 1.5 mm | 3 | 256 731 | 9.9 s | 211.1 MPa | −0.9 % |
| 16 mm | 3 | 2 mm | 3 | 151 449 | 5.0 s | 213.4 MPa | +0.2 % |
| 8 mm | 1 | 4 mm | 4 | 120 147 | 4.0 s | 208.6 MPa | −2.1 % |
| 6 mm | 1 | 3 mm | 4 | 226 449 | 8.5 s | 210.6 MPa | −1.1 % |
| 12 mm | 2 | 3 mm | 4 | 222 009 | 8.7 s | 212.3 MPa | −0.3 % |
| 16 mm | 2 | 4 mm | 4 | 124 824 | 4.2 s | 217.2 MPa | +2.0 % |

(Times on a 24-core desktop. The table was measured again with the narrow gaps open, the
default: every peak is the same to the digit shown, and a few rows have up to one per cent more
unknowns, where a cell holds the two sides of a hole.) Read with the guidance above, for a
fillet of 6 mm:

- with cells of 2.2 mm or less at the fillet at degree 3 (four grids), the peak lies between
  −0.9 % and +0.2 % of the reference; with cells of 3 mm at degree 4 (two grids), −1.1 % and
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

Too large to factor on this part: degree 4 with one level at the default cell size (339 000
unknowns, a stiffness of 228 million entries against the 180 million a factorization takes).
It is solved by iteration (36 rounds, 5 s: see [How large a model](#how-large-a-model)), and
gives about 219 MPa (+3 %). Two levels at the default cell
size (cells of 1.1 mm at the fillet, 367 000 unknowns) take 13 s and give 211.2 MPa (−0.8 %). The
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
- `volume` against `surface_volume` shows how well the cut cells were integrated: they agree to
  rounding where every cut cell was integrated exactly, and to about 1e-5 where some were
  sampled.

## How large a model

The unknowns are solved all at once, in one of two ways, and the command chooses
(`"solver": "auto"`):

- **By factorization** of their matrix (`"direct"`). The result is exact to rounding, the same
  digits on every machine. Its time grows with the square of the unknowns for a compact part
  and its memory a little faster than the unknowns; a thin part, whose cells lie side by side
  in a sheet, costs far less.
- **By iteration** (`"iterative"`). First the unknowns every cell has to itself are taken
  out exactly (an eighth of them inside a part, up to half where it is thin). The rest is solved by conjugate gradients, each
  round made effective by solving two kinds of simpler problem exactly: the part on linear
  shape functions alone (for a part of thin walls with the quadratic ones of the cells' edges,
  without which a wall cannot bend), and the shape functions around every vertex of the grid
  on their own. Its time and memory grow with the unknowns, and the rounds do not grow with
  the grid (25 to 35 where no cell is divided). It ends when what is left is 10⁻⁹ of the
  solution in the energy norm, which is displacements and stresses to about nine digits; the
  reply's `residual` is that measure. A part of very thin walls is so stiff across them that
  rounding leaves more (up to 10⁻⁸, as it does for the factorization), and the reply says
  what was reached.

Left to the command, a system under 50,000 unknowns is factored, and so is a larger one
whose factorization is predicted to be the less work (a part of walls much thinner than its
cells, or one whose cells are divided three times, of moderate size); the others are iterated, and so is any whose factor would not fit
`max_factor_entries` or whose stiffness has more than 180 million entries. The choice is
made from the grid, before the stiffness is added up, because the two ways keep it
differently. What the iteration does not settle in 400 rounds, such as a material that
does not change its volume at all (it settles at a Poisson's ratio of 0.499999, in ten
times the rounds), is factored after all; a system too large to factor is iterated for up to 2000
rounds. The reply's `solver` says which it was, with the `iterations`.

Measured at degree 3 on a desktop processor with 24 threads (i7-13700KF), the whole solve:

| Part | Unknowns | Factorization | Iteration | Chosen |
|---|---|---|---|---|
| A solid block, 14 cells along its longest side | 220,000 | 7.4 s, 2.1 GB of factor | 1.1 s, 28 rounds | iteration |
| The same, 18 cells | 416,000 | 23 s, 5.1 GB | 2.3 s, 28 rounds | iteration |
| The same, 24 cells | 974,000 | too large | 5.5 s, 27 rounds | iteration |
| The same, 36 cells | 3,050,000 | too large | 23 s, 27 rounds | iteration, with `max_factor_entries` raised (refused at the default) |
| A thick tube, curved, 20 cells across | 340,000 | 10.8 s, 3.1 GB | 1.9 s, 30 rounds | iteration |
| Three blocks bonded together | 188,000 | 6.2 s, 1.7 GB | 2.3 s, 47 rounds | iteration |
| A plate one cell thick, 40 cells along | 292,000 | 4.4 s, 1.4 GB | 1.4 s, 24 rounds | iteration |
| A plate a fifth of a cell thick, 80 cells along, trunk space | 217,000 | 2.1 s, 0.9 GB | 2.8 s, 28 rounds | factorization |
| The same, 160 cells along | 848,000 | 11.0 s, 4.2 GB | 13.0 s, 28 rounds | iteration (the factor is over the default cap) |
| The 8 mm bracket above, 4.4 mm cells, none divided | 68,000 | 1.0 s | 0.4 s, 33 rounds | iteration |
| The same, cells divided once | 145,000 | 2.6 s | 1.3 s, 35 rounds | iteration |
| The same on 8 mm cells divided twice | 145,000 | 3.0 s | 2.3 s, 54 rounds | iteration |
| The same on 16 mm cells divided three times | 152,000 | 3.4 s | 2.9 s, 45 rounds | factorization |
| The same on 8 mm cells divided three times | 447,000 | too large | 11.9 s, 58 rounds | iteration |

The iteration keeps the stiffness by blocks of three by three, one for every two shape
functions that couple (76 bytes for nine entries, where the factorization's rows take 144),
and one small factor for every vertex: 1.9 GB in all for the block of 416,000 unknowns,
4.6 GB for 974,000 and 15 GB for 3 million, about 5 GB for every million unknowns of degree
3. Its time grows with the unknowns: 6 to 8 s for every million on that processor.

A part as thick as its cells takes no more rounds than a compact one, and cut cells none:
the bracket takes 32 to 33 rounds at degree 3 on cells of 2.2 and 4.4 mm. Cells of two sizes
side by side, where a cell is divided and the one beside it is not, take few more: the
functions of the larger cell reach over all the smaller ones beside it, and the solver takes
each together with the smaller cells' functions that follow it. The bracket with its cells
divided once at its fillet and holes takes 35 rounds at degree 3 and 36 at degree 4, divided
twice 54, and three times 45 to 58; what grows with the levels is the time to prepare that
(a third of the 11.9 s above). Bodies held together by connections are treated alike across
the faces that touch, and take half as many rounds again as one body (47 for three bonded
blocks, 48 where they slide), three times as many where one body's cells are three times
the other's.

A part of walls much thinner than its cells is the hardest for an iteration: a wall bends, which
linear shape functions cannot follow inside a cell, and it is so much stiffer across than in
bending that the size of what is left of the equations says little. So the solver finds such a
part by how little of its cells it fills (less than three tenths), adds the quadratic functions
of the cells' edges to the problem it solves exactly, and measures what is left in energy. A
plate a tenth of a cell thick then takes 21 rounds with the tensor product and 33 with the trunk
space (84 and 140 without; a thinner one did not settle at all). The trunk space takes more the
thinner the wall: 84 rounds a fiftieth of a cell thick, where the tensor product takes 27. A
part of thin walls is also cheap to factor, and up to a few hundred thousand unknowns the
command still does that.

Two limits stop a solve before it takes the machine's memory. `max_factor_entries` (8 bytes
an entry; the default of 400 million is 3.2 GB) caps the factor of the factorization, and a
system above it is iterated; and it caps the entries of the stiffness the iteration keeps,
which a compact part reaches at about a million unknowns of degree 3. Raised to 1.2 billion
on a machine with 32 GB, it lets 3 million unknowns through. The stiffness of a
factorization holds at most 180 million entries (about 480,000 unknowns of a compact part
at degree 3). A solve above its limit is refused with the number that was reached, or, for
several bodies on cells of the command's choosing, run again on larger cells with a
warning. The trunk space (`"basis": "trunk"`) halves the unknowns at degree 3, and a lower
`degree` with a level more of `refine` costs less than a smaller `cell_size` everywhere.

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
the same model on quadratic tetrahedra of 4 mm (920 000 unknowns; 44 s to mesh and 16 s to
solve, and only with `max_factor_entries` raised to 1.5 billion: with the default the mesh
solve is refused). Both are read at the same point, the mesh's node on the passage's edge at
(30, 0, 2): hung by one wall the sag changes by 0.02 mm for every millimetre there, and the
mesh solve gives 2.104 and 2.106 mm on tetrahedra of 3 and 2.5 mm.

| | Bolted at four holes, 1 kPa on the floor: sag at the passage | Hung by one wall, 100 N on the opposite wall: sag at the passage |
|---|---|---|
| Mesh solve (reference) | 0.1532 mm | 2.105 mm |
| Grid, gaps joined | 0.1508 mm (−1.6 %), 3 s | 0.027 mm (−98.7 %), 3 s |
| Grid, gaps open, no cells divided where they end (the solver's option, not the form's) | 0.1518 mm (−0.9 %) | 1.054 mm (−50 %) |
| Grid, gaps open, tensor product, no round that follows the solution (`"basis": "tensor", "adapt": 0`) | 0.1520 mm (−0.8 %), 3.9 s | 1.914 mm (−9.1 %), 3.7 s |
| the same on cells of 12.5 mm | | 2.042 mm (−3.0 %), 9 s |
| Grid, gaps open, tensor product, rounds that follow the solution (`"basis": "tensor"`) | 0.1538 mm (+0.4 %), 30 s | 2.100 mm (−0.2 %), 20 s |
| Grid, gaps open, trunk space, no round (`"adapt": 0`) | 0.1497 mm (−2.3 %), 2.0 s | 1.834 mm (−12.9 %), 2.1 s |
| **Grid as the form opens** (gaps open, trunk space, rounds that follow the solution) | **0.1536 mm (+0.2 %), 12 s** | **2.066 mm (−1.8 %), 9 s** |

**Against the mesh solve in time.** Asked for tetrahedra of 25 down to 4 mm, the mesher gives
this tray 156 000 quadratic elements every time: the sheet's thickness and its bends set the
size, not the request. Meshing takes 44 s and the solve 16 s (918 000 unknowns, with the
factor's cap raised), 60 s against the grid's 9 s as the form opens, for results 2 % apart. On
the sample bracket, a thick part, the two are close: tetrahedra of the default size
(4.39 mm, 37 000 unknowns) mesh and solve in 2.8 s and read 212.6 MPa on the fillet, where the
grid takes 2.1 s for 212.5 MPa; finer meshes of it read between −1.5 % and +1.5 % of 213 MPa
(tetrahedra of 3, 2, 1.5 and 1 mm, 4 to 140 s). The grid's case is the part a mesher has to
work for (thin sheet, small features on a large part), and the part that is not meshed at all.

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
3 % low with 400 000 unknowns), which is what the rounds that follow the solution are for:
they divide the cells of the bend that carries the load. In the trunk space, which a thin
body takes, the tray hung by one wall is 1.8 % low after two of them with 165 000 unknowns,
and 0.4 % above the mesh solve after three (`"adapt": 3`, 315 000 unknowns, 19 s), which is
about as well as the mesh solve itself is known; bolted, the sag is within 0.2 % after two. The von Mises stress at seven points of the floor (measured once, not kept as
a test) is within 1 % of the mesh solve's at six of them and 4 % low at one, where it is up
to 14 % low without the rounds.

Each round still divides a cell along all three axes where a bend needs it across the bend
only. So the rounds buy accuracy at a price: the default takes 9 s on this tray where a solve
without them takes 2.1 s, and the reply's warning says when the result was still changing as
they ended (here by 4 % of the strain energy in the last round). Cells that follow the sheet
(divided across a bend only) are the next step for it.

The gaps stay as wide as they are: the solve is linear, and two sides that would close and
touch under load pass through each other. `"gaps": "joined"` (the form's **Narrow gaps**) solves
the part as if every gap narrower than a cell were welded shut, which is what a corner seam
that is welded after bending wants.

## Several bodies: an assembly

Leave **Body** empty in a design of several bodies and all of them are solved in one step. Each
body has its own grid, with its own cell size and the material of its own section. Two bodies
share no unknown, however close they are: what joins them is a *connection*.

**Cell sizes.** With `cell_size` left out each body's cells are a fraction of its own diagonal,
so a small part gets small cells: a twentieth, as for one body, or more when the bodies are
many. The bodies together are given the unknowns one compact body has at a twentieth (a cube:
125,000 at degree 3), and their cells are all made larger by one factor until they fit. The
unknowns are those of the bodies' cells, counted on the grids before anything is integrated,
added up or solved, and what a count worked out of the surfaces is the next one's and the
solve's to use. Two counts show how fast the unknowns fall as the cells grow (as the cube of
their size in a compact body, as its square in sheet metal), which is how much larger the next
cells are made. Cells divided where a gap ends are not in the count: whether a gap's end needs
them depends on where the grid lies on the part, so their number jumps from one cell size to the
next, and cells fitted to it would jump with it. An assembly then takes about the time of one
part however many bodies it has. The reply gives each body's `cell_size`. A `cell_size` that is
given is the cell of every body, and is not changed: a solve too large for a limit is then
refused, where one on cells of the command's choosing is run again on larger cells and says so
in a warning.

A base plate of 100 × 60 × 5 mm with two clamp blocks of 20 × 10 × 8 mm standing on it, each
with a bore and a load on its top: cells of 11.4 mm in the plate and 2.3 mm in the blocks
(halved around the bores), 130,000 unknowns, 7 s, the reactions equal to the loads to eleven
digits.

**Connections.** A connection names faces of one body and faces of another, and says how the
two hold each other where those faces touch:

| Kind | What the two bodies do there | For |
|---|---|---|
| `bonded` | move together | glued, welded, pressed or bolted tight over the area |
| `sliding` | keep touching and slide along each other without friction | a part in a pocket, a pin or a shaft in its hole, a key in its slot |
| `free` | nothing: they do not hold each other | faces that only look as if they touched |

Whole faces are named; where they touch is found by the solve. Flat faces touch to rounding,
curved ones (a pin in a hole) within the triangulations' tolerance, and faces that overlap in
part are connected over the part.

**They are found for you, once.** The first solve of a design with several bodies looks for
the faces of every two bodies that lie on each other, facing each other (an edge or a corner
resting on a face is none), bonds them, and leaves the connections in the model, named after
the two bodies. The reply lists them, and `connections_added` names those this solve found:

```json
"connections": [{"name": "Base-Post", "kind": "bonded", "bodies": ["Base", "Post"], "faces": [["F1:+z"], ["F2:-z"]], "area": 2000.0}],
"connections_added": ["Base-Post"]
```

From then on the list is yours, under **Interactions** in the browser, and a solve joins the
bodies by exactly what it holds:

| To | Do | Command |
|---|---|---|
| let two bodies slide, or part them | open the connection and change its **Kind** | `interaction.connection.edit {"name": "Base-Post", "kind": "sliding"}` |
| take a connection out | delete it; it stays deleted | `interaction.connection.delete` |
| connect faces yourself | pick the faces of the two bodies in the view, **Connection**, **Use selection** | `interaction.connection {"name", "kind", "first": [{"body", "face"}], "second": […]}` |
| look again, after the design changed or to get a deleted one back | **Find Connections**: adds one for touching faces that none joins yet, and changes none that is there | `interaction.connection.find` |

Read the list before the results: a connection that is missing (two parts that should touch
and are modelled a hair apart) leaves a body held by less than intended. A connection whose
faces do not touch joins nothing, and the reply says so in a warning. `"connections": "none"`
solves the bodies free of each other, each on its own supports, and leaves the model as it is.

**Every body must be held.** A static solve has no answer for a body that can still move as a
rigid piece. Before anything is solved, the motions that no boundary condition and no
connection resists are found, and the solve is refused with the body and the way it can move:

> `Pin` can still slide along (0, 0, 1) and turn about an axis along (0, 0, 1). A static solve
> needs every body held against sliding and turning: a sliding connection holds only across
> its faces, so add a boundary condition, or make a connection bonded

A sliding connection holds a body only across its faces: a block standing on a plate can
still slide two ways and turn about the plate's normal, and a pin in a hole can slide along
the hole and turn in it. Give such a body what holds it in the machine (the shoulder it
stands on, the key that keeps it from turning) as a boundary condition or a second
connection. A body that touches another with no connection between them is told so, with the
command that finds the connection.

**Sliding is not contact.** The solve is linear: two bodies that slide keep touching, so a
sliding connection pulls as well as it pushes, and where the real parts would lift off each
other it holds them together. Check the sign of what crosses it; contact that opens needs the
nonlinear step and is not here yet.

**Supports and loads** are the step's, as for one body, and a set may name faces of several
bodies. A body needs a boundary condition of its own or a connection to a body that is held. A
*Force* is a total over the faces of one body.

**What it is.** Each body is solved by the method of one body, and a bond makes the
displacements of the two equal weakly where they touch (Nitsche's method for an interface, the
traction taken from the body with the smaller face). Different materials, cell sizes and
degrees of fill on the two sides are all the same to it: two blocks end to end are one bar to
the accuracy of one body's cells, on any two grids, and two materials in series stretch each by
its own modulus (`crates/fcm/tests/assembly.rs`). The unknowns of all bodies are ordered and
factored together.

A sliding connection is the same method with the traction and the jump of the displacement
replaced by their parts along the face's normal (the frictionless case of Annavarapu,
Hautefeuille and Dolbow's sliding interface, 2014, for bodies that do not part). It is as
consistent as the bond: two blocks of different materials pressed together, each widening by
its own Poisson's ratio and slipping on the other, are found to 10⁻⁶ of the exact
displacement on any two grids. On a curved face the normal is that of the cylinder, cone,
sphere or torus the face lies on, not of the triangles that stand for it: held across every
facet of its hole a pin could not turn, and with the surface's normal it slides and turns
freely, carrying nothing along the hole (2 · 10⁻¹⁵ of the pull in the test). A curved face of
any other kind keeps its triangles' normals, which is exact for a face swept along a line and
somewhat too stiff otherwise. The bodies that can still move are the null space of the
restraints on the six rigid motions of each body, found by a Cholesky decomposition with
complete pivoting (Higham 1990).

**The stress at the edge of a bond.** Where the side of one body meets the face of the other
at the edge of a bonded area, the two form a sharp inside corner, and the stress of the elastic
solution has no finite value there, as at any sharp inside corner of one part. The largest von
Mises stress of an assembly is often at such an edge, and it rises with every refinement
(27 to 39 MPa in the example above between coarse and fine cells, while the deflection of the
clamps' tops moves by 2 %). Read the stress a cell away from the edge, or model the fillet or
the weld that is there in the part.

**Rounds and the convergence check** work on all the bodies together. A round of **Adaptive
rounds** (`adapt`) ranks the cells of every body by the one indicator, divides those the
assembly is least resolved in, in whichever body they are, and solves all of it again; the
rounds end when one moves the strain energy of all the bodies by no more than `settle`. Left
to the command (`"auto"`), up to two rounds are run when any body is thin for its grid, and
none otherwise. **Check convergence** solves every body one degree lower and compares over
the sampled points of all of them.

On the plate with two clamps of the example above:

| Rounds | Unknowns | Time | Tip of the clamp | Strain energy moved by | Largest von Mises |
|---|---|---|---|---|---|
| 0 | 130,000 | 8 s | 0.518 µm | | 27 MPa |
| 1 | 152,000 | 19 s | 0.534 µm | 2.6 % | 44 MPa |
| 2 | 184,000 | 46 s | 0.537 µm | 0.5 % | 62 MPa |
| 3 | 230,000 | 89 s | 0.540 µm | 0.4 % | 89 MPa |

The first round goes almost wholly into the plate (7,000 to 19,000 unknowns), whose cells are
more than twice its thickness across: that is where the assembly was least resolved, and the
deflection gains 3 % from it. The deflection settles; the largest stress does not, because it
sits on the edge of a bonded area (above), and every round divides the cells there again. The
check at the default settings says the same in 3 s more: the strain energy is 2.5 % from the
solve one degree lower, the largest von Mises stress 32 %.

**Not yet:** bolts, welds and contact, which are the plan below. The solvers on a mesh do not
read connections: they join element faces by ties and contacts.

### What comes next

Bonded, sliding and free are the kinds a linear solve can have between faces. The others are
found from the geometry the same way, and join the same list:

1. **Holes in line.** Coaxial holes through bodies that touch are bolt candidates
   (`interaction.bolt` of the roadmap's M7): a beam with the bolt's section, its head and nut
   bearing on the annuli around the holes.
2. **Edges that end on a face.** A plate standing on another is a weld seam candidate
   (`interaction.weld`).
3. **Contact**, which needs the nonlinear step.
