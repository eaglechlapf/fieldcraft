//! The integrals of one cell: stiffness and body load over its quadrature points, and loads and
//! supports over the pieces of the surface it owns.
//!
//! # Stiffness
//! For an isotropic material with Lamé constants λ and μ, the stiffness block of two modes a, b
//! (components i, j) is
//! K = λ G[i][j] + μ G[j][i] + μ δᵢⱼ tr G, with G[i][j] = ∫ ∂ᵢNₐ ∂ⱼN_b dV,
//! so only the nine numbers G of a mode pair are needed. They are not summed point by point:
//! the quadrature points are summed into the Legendre moments of the cell's material, and G
//! follows from the moments one axis at a time ([`crate::moments`]). Points outside the part
//! are summed apart: the cell matrix is the stiffness of the part plus the fictitious stiffness
//! of the rest of the cell.
//!
//! # Supports: Nitsche's method
//! Shape functions do not vanish on the surface of the part, so a prescribed displacement cannot
//! be set on unknowns. It is imposed weakly instead (J. Nitsche, *Über ein Variationsprinzip zur
//! Lösung von Dirichlet-Problemen bei Verwendung von Teilräumen, die keinen Randbedingungen
//! unterworfen sind*, Abh. Math. Sem. Univ. Hamburg 36(1):9–15, 1971; for finite cells, M. Ruess,
//! D. Schillinger, Y. Bazilevs, V. Varduhn, E. Rank, *Weakly enforced essential boundary
//! conditions for NURBS-embedded and trimmed NURBS geometries on the basis of the finite cell
//! method*, Int. J. Numer. Meth. Engng 95(10):811–846, 2013). With t(u) = σ(u) n the traction of
//! a displacement field on the surface Γ and g the prescribed value, for the fixed components i
//! only:
//!
//! a(u, v) = ∫ ε(v) : C : ε(u) dV − ∫ tᵢ(u) vᵢ dΓ − ∫ tᵢ(v) uᵢ dΓ + β ∫ uᵢ vᵢ dΓ,
//! l(v) = (loads) − ∫ tᵢ(v) gᵢ dΓ + β ∫ vᵢ gᵢ dΓ.
//!
//! The first surface term is what integration by parts leaves on Γ, the second makes the form
//! symmetric and the third makes it positive. All three vanish for the exact solution, so the
//! method is consistent for any β.
//!
//! **The parameter β is computed, not chosen.** Young's inequality gives, for any v on a cell,
//! 2 ∫ tᵢ(v) vᵢ ≤ (1/β) ∫ tᵢ(v) tᵢ(v) + β ∫ vᵢ vᵢ, so a(v, v) ≥ (1 − Λ/β) ∫ ε : C : ε with Λ the
//! largest eigenvalue of H x = Λ V x, H the matrix of ∫ tᵢ(u) tᵢ(v) dΓ over the cell's supported
//! pieces and V the cell's stiffness. Every β > Λ keeps the form positive; β = 2Λ, the value of
//! A. Embar, J. Dolbow, I. Harari, *Imposing Dirichlet boundary conditions with Nitsche's method
//! and spline-based finite elements*, Int. J. Numer. Meth. Engng 83(7):877–898, 2010, leaves half
//! of the stiffness. V is singular on the six rigid motions of the cell, where H vanishes too;
//! they are shifted away before V is factored, which does not change Λ.
//!
//! **The stress on a supported face** is evaluated like anywhere else, from the cell's
//! polynomial. The traction the support exerts, tᵢ(u) − β (uᵢ − gᵢ) on the fixed components, is
//! what [`reaction`] integrates, and it balances the loads exactly. Correcting the stress at a
//! point of a supported face with it (adding d ⊗ n + n ⊗ d − (d · n) n ⊗ n for
//! dᵢ = −β (uᵢ − gᵢ)) was measured on Lamé's cylinder, Kirsch's plate and NAFEMS LE1 and is not
//! done: on coarse grids it turns errors of 3–18 % into 0.2–3.5 %, but where a supported face
//! ends it leaves 1–2.5 % that do not fall as the grid or the degree is refined, while the
//! stress of the polynomial converges (to 0.15 % on LE1). β is computed cell by cell and grows
//! without bound in a cell that holds a sliver of the face (F. de Prenter, C. Lehrenfeld,
//! A. Massing, *A note on the stability parameter in Nitsche's method for unfitted boundary
//! value problems*, Comput. Math. Appl. 75(12):4322–4336, 2018), and β (u − g) with it.
//!
//! # Surface quadrature
//! A piece is a flat triangle, integrated with a Gauss rule on the square collapsed onto it
//! (M. G. Duffy, *Quadrature over a pyramid or cube of integrands with a singularity at a
//! vertex*, SIAM J. Numer. Anal. 19(6):1260–1262, 1982): the point at (s, t) ∈ [0, 1]² is
//! (1 − s) A + s ((1 − t) B + t C) with weight 2 · area · s. The points of the pieces of one
//! face that share a normal are summed into Legendre moments like the volume's, and the support
//! matrices follow from the integrals of Nₐ N_b, ∂_c Nₐ N_b and ∂_c Nₐ ∂_d N_b of each mode pair
//! over them.

use std::collections::BTreeMap;

use crate::dense;
use crate::grid::{Piece, QuadPoint, cross, dot, sub};
use crate::legendre::shape;
use crate::moments::{self, Moments, SLOPE_SLOPE, SLOPE_VALUE, Scratch, Tables, VALUE_SLOPE, VALUE_VALUE};

/// Lamé's constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Lame {
    pub lambda: f64,
    pub mu: f64,
}

/// What a face of the part carries.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Face {
    /// The displacement components that are prescribed.
    pub fixed: [bool; 3],
    /// Their values.
    pub value: [f64; 3],
    /// A traction (force per area), in the global axes.
    pub traction: [f64; 3],
    /// A pressure, pushing against the outward normal.
    pub pressure: f64,
    /// Whether the face is tied to another body where the two touch ([`crate::assembly`]): it
    /// is then fixed in every component over the points [`Surface::touching`] keeps, and the
    /// value prescribed there is the other body's displacement, an unknown.
    pub tied: bool,
}

impl Face {
    pub fn supported(&self) -> bool {
        self.fixed.iter().any(|f| *f)
    }

    pub fn loaded(&self) -> bool {
        self.pressure != 0.0 || self.traction.iter().any(|t| *t != 0.0)
    }
}

/// The surface of the part as the cells see it.
pub(crate) struct Surface<'a> {
    /// The outward unit normal of each surface triangle.
    pub normals: &'a [[f64; 3]],
    /// The face of each surface triangle.
    pub tags: &'a [u32],
    pub faces: &'a BTreeMap<u32, Face>,
    /// Where a piece of a tied face ([`Face::tied`]) may touch the other body.
    pub touching: Option<&'a Touching<'a>>,
}

/// Hands the parts of a piece (a triangle) of the tied face with this tag to the caller: each
/// part lies in one cell of the other body, whose functions are polynomials there, and comes
/// with whether a point of it touches that body.
pub(crate) type Touching<'a> = dyn Fn(u32, &[[f64; 3]; 3], &mut dyn FnMut(&[[f64; 3]; 3], &dyn Fn([f64; 3]) -> bool)) + Sync + 'a;

impl Surface<'_> {
    /// The face a piece belongs to, when it carries anything, with the piece's normal.
    fn face(&self, piece: &Piece) -> Option<(u32, &Face, [f64; 3])> {
        let tag = self.tags.get(piece.triangle as usize).copied()?;
        let normal = self.normals.get(piece.triangle as usize).copied()?;
        self.faces.get(&tag).map(|face| (tag, face, normal))
    }

    /// Whether any of the pieces is supported.
    pub fn supports(&self, pieces: &[Piece]) -> bool {
        pieces.iter().any(|piece| self.face(piece).is_some_and(|(_, face, _)| face.supported()))
    }

    /// Whether any of the pieces is of a face tied to another body.
    pub fn ties(&self, pieces: &[Piece]) -> bool {
        pieces.iter().any(|piece| self.face(piece).is_some_and(|(_, face, _)| face.tied))
    }

    /// Visit the quadrature points of a piece at which its face's support acts, each with its
    /// natural coordinates in the cell at `lo` of size `h` and its weight: every point of a
    /// piece of a supported face; of a tied face, the points that touch the other body, taken
    /// part by part of the piece, each part within one of that body's cells.
    #[allow(clippy::too_many_arguments)]
    fn held_points(&self, tag: u32, face: &Face, piece: &Piece, lo: [f64; 3], h: f64, rule: &[(f64, f64)], mut visit: impl FnMut([f64; 3], f64)) {
        if !face.tied {
            piece_points(piece, lo, h, rule, visit);
            return;
        }
        let Some(touching) = self.touching else { return };
        touching(tag, &piece.corners, &mut |part, touches| {
            piece_points(&Piece { corners: *part, triangle: piece.triangle }, lo, h, rule, |xi, w| {
                if touches([0, 1, 2].map(|i| lo[i] + 0.5 * (xi[i] + 1.0) * h)) {
                    visit(xi, w);
                }
            });
        });
    }
}

/// The shape functions of a cell: the modes it carries of the tensor product of degree `p`
/// ([`crate::space::local_modes`]) and the tables of their products.
#[derive(Debug, Clone)]
pub(crate) struct Shapes {
    pub p: usize,
    /// The index along each axis of every mode.
    pub modes: Vec<[u8; 3]>,
    pub tables: Tables,
    /// The modes of the cell's eight vertices, the vertex at the upper end of axis d having bit
    /// d set.
    vertices: [usize; 8],
}

impl Shapes {
    /// `None` when `modes` lacks a vertex mode or names a function of a higher degree.
    pub fn new(p: usize, modes: Vec<[u8; 3]>) -> Option<Shapes> {
        if modes.iter().flatten().any(|index| usize::from(*index) > p) {
            return None;
        }
        let mut vertices = [0; 8];
        for (corner, vertex) in vertices.iter_mut().enumerate() {
            let bits = [(corner & 1) as u8, ((corner >> 1) & 1) as u8, ((corner >> 2) & 1) as u8];
            *vertex = modes.iter().position(|mode| *mode == bits)?;
        }
        Some(Shapes { p, modes, tables: Tables::new(p), vertices })
    }

    /// Modes of a cell.
    pub fn len(&self) -> usize {
        self.modes.len()
    }

    /// The mode of a vertex of the cell (bit d of `corner`: the upper end of axis d).
    pub fn vertex(&self, corner: usize) -> usize {
        self.vertices[corner % 8]
    }
}

/// The values and physical gradients of a cell's modes at natural coordinates `xi`; `scale` is
/// 2 / (cell size), the derivative of the natural coordinate.
pub(crate) fn eval(shapes: &Shapes, xi: [f64; 3], scale: f64, values: &mut [f64], gradients: &mut [[f64; 3]]) {
    let p = shapes.p;
    let (sx, sy, sz) = (shape(p, xi[0]), shape(p, xi[1]), shape(p, xi[2]));
    for ((mode, value), gradient) in shapes.modes.iter().zip(values.iter_mut()).zip(gradients.iter_mut()) {
        let [a, b, c] = mode.map(usize::from);
        *value = sx.n[a] * sy.n[b] * sz.n[c];
        *gradient = [scale * sx.d[a] * sy.n[b] * sz.n[c], scale * sx.n[a] * sy.d[b] * sz.n[c], scale * sx.n[a] * sy.n[b] * sz.d[c]];
    }
}

/// The Legendre moments of a cell's quadrature points ([`crate::moments`]), in natural
/// coordinates: those of the points inside the part and those of the points outside it.
pub(crate) struct Volume {
    pub inside: Moments,
    pub outside: Moments,
}

impl Volume {
    pub fn new(shapes: &Shapes) -> Volume {
        Volume { inside: Moments::new(&shapes.tables), outside: Moments::new(&shapes.tables) }
    }

    /// Add a quadrature point.
    pub fn add(&mut self, point: &QuadPoint) {
        if point.inside { self.inside.add(point.xi, point.weight) } else { self.outside.add(point.xi, point.weight) }
    }
}

/// Sum the quadrature points of a cell.
pub(crate) fn integrate(shapes: &Shapes, points: &[QuadPoint]) -> Volume {
    let mut volume = Volume::new(shapes);
    for point in points {
        volume.add(point);
    }
    volume
}

/// Which derivative the two factors carry along each axis, for ∂ᵢNₐ ∂ⱼN_b.
fn gradient_kinds(i: usize, j: usize) -> [usize; 3] {
    [0, 1, 2].map(|axis| match (i == axis, j == axis) {
        (false, false) => VALUE_VALUE,
        (false, true) => VALUE_SLOPE,
        (true, false) => SLOPE_VALUE,
        (true, true) => SLOPE_SLOPE,
    })
}

/// G[i][j] = ∫ ∂ᵢNₐ ∂ⱼN_b of every mode pair a ≤ b over the points of `moments`, in natural
/// coordinates, at `(a m + b) 9 + 3 i + j`.
fn gradient_products(shapes: &Shapes, moments: &Moments, scratch: &mut Scratch) -> Vec<f64> {
    let m = shapes.len();
    let mut g = vec![0.0; m * m * 9];
    for i in 0..3 {
        for j in 0..3 {
            moments::pairs(&shapes.tables, moments, &shapes.modes, gradient_kinds(i, j), true, &mut g, 9, 3 * i + j, scratch);
        }
    }
    g
}

/// The stiffness matrix (3m × 3m, row-major, mode by mode with its three components) of a cell
/// of size `h` whose material has the `moments`.
pub(crate) fn stiffness(shapes: &Shapes, h: f64, moments: &Moments, lame: &Lame, scratch: &mut Scratch) -> Vec<f64> {
    let m = shapes.len();
    let n = 3 * m;
    let g = gradient_products(shapes, moments, scratch);
    // The Jacobian h³/8 and two derivatives of the natural coordinates, 2/h each.
    let (lambda, mu) = (0.5 * h * lame.lambda, 0.5 * h * lame.mu);
    let mut k = vec![0.0; n * n];
    for a in 0..m {
        for b in a..m {
            let at = (a * m + b) * 9;
            // A mode with itself: G is symmetric but for rounding.
            let entry = |i: usize, j: usize| if a == b { 0.5 * (g[at + 3 * i + j] + g[at + 3 * j + i]) } else { g[at + 3 * i + j] };
            let trace = entry(0, 0) + entry(1, 1) + entry(2, 2);
            for i in 0..3 {
                for j in 0..3 {
                    let mut value = lambda * entry(i, j) + mu * entry(j, i);
                    if i == j {
                        value += mu * trace;
                    }
                    k[(3 * a + i) * n + 3 * b + j] = value;
                    k[(3 * b + j) * n + 3 * a + i] = value;
                }
            }
        }
    }
    k
}

/// ∫ Nₐ dV of every mode over the material of a cell of size `h`.
pub(crate) fn body(shapes: &Shapes, h: f64, moments: &Moments, scratch: &mut Scratch) -> Vec<f64> {
    let jacobian = 0.125 * h * h * h;
    let mut integrals = moments::singles(&shapes.tables, moments, &shapes.modes, None, scratch);
    integrals.iter_mut().for_each(|x| *x *= jacobian);
    integrals
}

/// Visit the quadrature points of a flat piece: the point's natural coordinates in the cell at
/// `lo` of size `h`, and its weight (an area).
fn piece_points(piece: &Piece, lo: [f64; 3], h: f64, rule: &[(f64, f64)], mut visit: impl FnMut([f64; 3], f64)) {
    let [a, b, c] = piece.corners;
    let doubled = cross(sub(b, a), sub(c, a));
    let doubled = dot(doubled, doubled).sqrt();
    for (gs, ws) in rule {
        let s = 0.5 * (gs + 1.0);
        for (gt, wt) in rule {
            let t = 0.5 * (gt + 1.0);
            let mut xi = [0.0; 3];
            for i in 0..3 {
                let x = (1.0 - s) * a[i] + s * ((1.0 - t) * b[i] + t * c[i]);
                xi[i] = 2.0 * (x - lo[i]) / h - 1.0;
            }
            visit(xi, 0.25 * ws * wt * doubled * s);
        }
    }
}

/// The traction coefficients of a surface with normal `n`: component i of the traction of a
/// displacement with the gradient ∂u_k/∂x_c is Σ C[i][k][c] ∂u_k/∂x_c, with
/// C[i][k][c] = λ nᵢ δ_kc + μ n_k δᵢc + μ δᵢ_k n_c, zero where component i is not fixed.
fn traction_coefficients(n: [f64; 3], fixed: [bool; 3], lame: &Lame) -> [[[f64; 3]; 3]; 3] {
    let mut c = [[[0.0; 3]; 3]; 3];
    for i in 0..3 {
        if !fixed[i] {
            continue;
        }
        for k in 0..3 {
            c[i][k][k] += lame.lambda * n[i];
            c[i][k][i] += lame.mu * n[k];
        }
        for along in 0..3 {
            c[i][i][along] += lame.mu * n[along];
        }
    }
    c
}

/// The traction components i of each displacement component k of a mode with gradient `d` on a
/// surface with normal `n`: tᵢ = λ nᵢ d_k + μ n_k dᵢ + μ δᵢ_k (d · n), zero where component i is
/// not fixed.
fn tractions(d: [f64; 3], n: [f64; 3], fixed: [bool; 3], lame: &Lame) -> [[f64; 3]; 3] {
    let along = dot(d, n);
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        if !fixed[i] {
            continue;
        }
        for k in 0..3 {
            t[i][k] = lame.lambda * n[i] * d[k] + lame.mu * n[k] * d[i];
        }
        t[i][i] += lame.mu * along;
    }
    t
}

/// The surface integrals of a cell.
pub(crate) struct Boundary {
    /// The load vector of the pressures and tractions.
    pub load: Vec<f64>,
    /// The Nitsche terms without β: −∫ tᵢ(u) vᵢ − ∫ tᵢ(v) uᵢ (3m × 3m) and −∫ tᵢ(v) gᵢ.
    pub consistency: Vec<f64>,
    pub consistency_load: Vec<f64>,
    /// The terms β multiplies: ∫ uᵢ vᵢ and ∫ vᵢ gᵢ.
    pub penalty: Vec<f64>,
    pub penalty_load: Vec<f64>,
    /// H: ∫ tᵢ(u) tᵢ(v).
    pub traction: Vec<f64>,
}

/// The pieces of one face that share a normal: every surface integral over them has the same
/// coefficients, so their quadrature points are summed into one set of moments.
struct Flat {
    tag: u32,
    normal: [f64; 3],
    face: Face,
    moments: Moments,
    /// Of a tied face: the moments of the points its support acts at
    /// ([`Surface::held_points`]).
    held: Option<Moments>,
}

/// The flat groups of the pieces that carry something, in the order the pieces name them.
fn flats(shapes: &Shapes, lo: [f64; 3], h: f64, pieces: &[Piece], surface: &Surface, rule: &[(f64, f64)]) -> Vec<Flat> {
    let mut groups: Vec<Flat> = Vec::new();
    for piece in pieces {
        let Some((tag, face, normal)) = surface.face(piece) else { continue };
        let same = |group: &Flat| group.tag == tag && group.normal.map(f64::to_bits) == normal.map(f64::to_bits);
        let at = match groups.iter().position(same) {
            Some(at) => at,
            None => {
                let held = face.tied.then(|| Moments::new(&shapes.tables));
                groups.push(Flat { tag, normal, face: *face, moments: Moments::new(&shapes.tables), held });
                groups.len() - 1
            }
        };
        if let Some(group) = groups.get_mut(at) {
            piece_points(piece, lo, h, rule, |xi, w| group.moments.add(xi, w));
            if let Some(held) = group.held.as_mut() {
                surface.held_points(tag, face, piece, lo, h, rule, |xi, w| held.add(xi, w));
            }
        }
    }
    groups
}

/// Integrate the loads and supports of the pieces a cell owns. The support matrices are empty
/// when none of the pieces is supported.
pub(crate) fn boundary(
    shapes: &Shapes,
    lo: [f64; 3],
    h: f64,
    pieces: &[Piece],
    surface: &Surface,
    lame: &Lame,
    rule: &[(f64, f64)],
    scratch: &mut Scratch,
) -> Boundary {
    let m = shapes.len();
    let n = 3 * m;
    let size = if surface.supports(pieces) { n * n } else { 0 };
    let mut out = Boundary {
        load: vec![0.0; n],
        consistency: vec![0.0; size],
        consistency_load: vec![0.0; n],
        penalty: vec![0.0; size],
        penalty_load: vec![0.0; n],
        traction: vec![0.0; size],
    };
    let scale = 2.0 / h;
    let tables = &shapes.tables;
    // Of each mode pair: ∫ Nₐ N_b, ∫ ∂_c Nₐ N_b and ∫ ∂_c Nₐ ∂_d N_b over a group.
    let (mut values, mut slopes, mut gradients) = (vec![0.0; size / 9], vec![0.0; size / 3], vec![0.0; size]);
    for group in flats(shapes, lo, h, pieces, surface, rule) {
        let (face, normal) = (&group.face, group.normal);
        if group.moments.is_zero() {
            continue;
        }
        let integrals = moments::singles(tables, &group.moments, &shapes.modes, None, scratch);
        if face.loaded() {
            for a in 0..m {
                for k in 0..3 {
                    out.load[3 * a + k] += integrals[a] * (face.traction[k] - face.pressure * normal[k]);
                }
            }
        }
        // The support acts over the whole of a face, or where a tied face touches the other body.
        let held = group.held.as_ref().unwrap_or(&group.moments);
        if !face.supported() || size == 0 || held.is_zero() {
            continue;
        }
        let integrals = if group.held.is_some() { moments::singles(tables, held, &shapes.modes, None, scratch) } else { integrals };
        let c = traction_coefficients(normal, face.fixed, lame);
        let g = face.value;
        // The loads of the prescribed values: −∫ tᵢ(v) gᵢ and ∫ vᵢ gᵢ.
        for along in 0..3 {
            let slope = moments::singles(tables, held, &shapes.modes, Some(along), scratch);
            for a in 0..m {
                for k in 0..3 {
                    out.consistency_load[3 * a + k] -= scale * slope[a] * (c[0][k][along] * g[0] + c[1][k][along] * g[1] + c[2][k][along] * g[2]);
                }
            }
        }
        for a in 0..m {
            for k in 0..3 {
                if face.fixed[k] {
                    out.penalty_load[3 * a + k] += integrals[a] * g[k];
                }
            }
        }
        // The matrices.
        moments::pairs(tables, held, &shapes.modes, [VALUE_VALUE; 3], true, &mut values, 1, 0, scratch);
        for along in 0..3 {
            let mut kinds = [VALUE_VALUE; 3];
            kinds[along] = SLOPE_VALUE;
            moments::pairs(tables, held, &shapes.modes, kinds, false, &mut slopes, 3, along, scratch);
            for other in 0..3 {
                moments::pairs(tables, held, &shapes.modes, gradient_kinds(along, other), true, &mut gradients, 9, 3 * along + other, scratch);
            }
        }
        // H of two displacement components k, l from the gradient products c, d:
        // Σᵢ C[i][k][c] C[i][l][d].
        let mut weights = [[[[0.0; 3]; 3]; 3]; 3];
        for k in 0..3 {
            for l in 0..3 {
                for along in 0..3 {
                    for other in 0..3 {
                        weights[k][l][along][other] =
                            scale * scale * (c[0][k][along] * c[0][l][other] + c[1][k][along] * c[1][l][other] + c[2][k][along] * c[2][l][other]);
                    }
                }
            }
        }
        for a in 0..m {
            for b in a..m {
                let (ab, ba) = (a * m + b, b * m + a);
                let mut products = [[0.0; 3]; 3];
                for along in 0..3 {
                    for other in 0..3 {
                        // A mode with itself: symmetric but for rounding.
                        products[along][other] = if a == b {
                            0.5 * (gradients[ab * 9 + 3 * along + other] + gradients[ab * 9 + 3 * other + along])
                        } else {
                            gradients[ab * 9 + 3 * along + other]
                        };
                    }
                }
                for k in 0..3 {
                    for l in 0..3 {
                        let (upper, lower) = ((3 * a + k) * n + 3 * b + l, (3 * b + l) * n + 3 * a + k);
                        let mut traction = 0.0;
                        let mut consistency = 0.0;
                        for along in 0..3 {
                            for other in 0..3 {
                                traction += weights[k][l][along][other] * products[along][other];
                            }
                            // t_l of mode a's component k against N_b, and t_k of mode b's
                            // component l against Nₐ.
                            consistency += c[l][k][along] * slopes[ab * 3 + along] + c[k][l][along] * slopes[ba * 3 + along];
                        }
                        let consistency = scale * consistency;
                        let penalty = if k == l && face.fixed[k] { values[ab] } else { 0.0 };
                        out.traction[upper] += traction;
                        out.consistency[upper] -= consistency;
                        out.penalty[upper] += penalty;
                        if a != b {
                            out.traction[lower] += traction;
                            out.consistency[lower] -= consistency;
                            out.penalty[lower] += penalty;
                        }
                    }
                }
            }
        }
    }
    out
}

/// The Nitsche parameter β = 2Λ of a cell with stiffness `k` and traction matrix `h` (both
/// 3m × 3m). `None` when the stiffness is not positive definite on the cell's deformations (a
/// fictitious stiffness too small for the cut).
pub(crate) fn nitsche_parameter(shapes: &Shapes, k: &[f64], h: &[f64]) -> Option<f64> {
    let n = 3 * shapes.len();
    if k.len() != n * n || h.len() != n * n {
        return None;
    }
    let scale = (0..n).map(|i| k[i * n + i]).fold(0.0, f64::max);
    if !(scale > 0.0) || !scale.is_finite() {
        return None;
    }
    // The six rigid motions, in the modes of the cell's eight vertices: three translations and
    // three rotations about the cell's centre. They are orthogonal, with the norms √8 and 4.
    let mut rigid = vec![[0.0; 6]; n];
    for corner in 0..8usize {
        let bits = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
        let mode = shapes.vertex(corner);
        let x = [2.0 * bits[0] as f64 - 1.0, 2.0 * bits[1] as f64 - 1.0, 2.0 * bits[2] as f64 - 1.0];
        for axis in 0..3 {
            rigid[3 * mode + axis][axis] = 1.0 / 8f64.sqrt();
            // The rotation about `axis`: e × x.
            let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
            rigid[3 * mode + b][3 + axis] = -x[c] / 4.0;
            rigid[3 * mode + c][3 + axis] = x[b] / 4.0;
        }
    }
    // Only the vertex modes take part in a rigid motion.
    let mut shifted = k.to_vec();
    for first in 0..8 {
        for second in 0..8 {
            for i in 3 * shapes.vertex(first)..3 * shapes.vertex(first) + 3 {
                for j in 3 * shapes.vertex(second)..3 * shapes.vertex(second) + 3 {
                    let overlap: f64 = (0..6).map(|r| rigid[i][r] * rigid[j][r]).sum();
                    shifted[i * n + j] += scale * overlap;
                }
            }
        }
    }
    if !dense::cholesky(&mut shifted, n) {
        return None;
    }
    Some(2.0 * dense::largest_eigenvalue(h, &shifted, n))
}

/// Add the force the supports of a cell exert on the part to `reactions`, by face:
/// ∫ (tᵢ(u) − β (uᵢ − gᵢ)) dΓ over the fixed components, for the cell's unknowns `u` (three per
/// mode).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reaction(
    shapes: &Shapes,
    lo: [f64; 3],
    h: f64,
    pieces: &[Piece],
    surface: &Surface,
    lame: &Lame,
    rule: &[(f64, f64)],
    beta: f64,
    u: &[f64],
    reactions: &mut BTreeMap<u32, [f64; 3]>,
) {
    let m = shapes.len();
    if u.len() != 3 * m {
        return;
    }
    let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
    for piece in pieces {
        let Some((tag, face, normal)) = surface.face(piece) else { continue };
        // What a tied face carries goes into the other body: it is no reaction of a support.
        if !face.supported() || face.tied {
            continue;
        }
        let mut sum = [0.0; 3];
        piece_points(piece, lo, h, rule, |xi, w| {
            eval(shapes, xi, 2.0 / h, &mut values, &mut gradients);
            let (mut displacement, mut traction) = ([0.0; 3], [0.0; 3]);
            for a in 0..m {
                let t = tractions(gradients[a], normal, face.fixed, lame);
                for k in 0..3 {
                    displacement[k] += values[a] * u[3 * a + k];
                    for i in 0..3 {
                        traction[i] += t[i][k] * u[3 * a + k];
                    }
                }
            }
            for i in 0..3 {
                if face.fixed[i] {
                    sum[i] += w * (traction[i] - beta * (displacement[i] - face.value[i]));
                }
            }
        });
        let total = reactions.entry(tag).or_insert([0.0; 3]);
        for i in 0..3 {
            total[i] += sum[i];
        }
    }
}

/// The points of the tied pieces of a cell at which they touch the other body
/// ([`Surface::held_points`]), the same points [`boundary`] sums: each with the point, its weight (an
/// face, the point, its weight (an area), the values of the cell's modes there, and the traction
/// of every mode's displacement component k on the piece, `tractions[a][i][k]` its component i.
pub(crate) fn tied_points(
    shapes: &Shapes,
    lo: [f64; 3],
    h: f64,
    pieces: &[Piece],
    surface: &Surface,
    lame: &Lame,
    rule: &[(f64, f64)],
    mut visit: impl FnMut(u32, [f64; 3], f64, &[f64], &[[[f64; 3]; 3]]),
) {
    let m = shapes.len();
    let (mut values, mut gradients, mut traction) = (vec![0.0; m], vec![[0.0; 3]; m], vec![[[0.0; 3]; 3]; m]);
    for piece in pieces {
        let Some((tag, face, normal)) = surface.face(piece) else { continue };
        if !face.tied {
            continue;
        }
        surface.held_points(tag, face, piece, lo, h, rule, |xi, w| {
            let x = [0, 1, 2].map(|i| lo[i] + 0.5 * (xi[i] + 1.0) * h);
            eval(shapes, xi, 2.0 / h, &mut values, &mut gradients);
            for (t, gradient) in traction.iter_mut().zip(&gradients) {
                *t = tractions(*gradient, normal, [true; 3], lame);
            }
            visit(tag, x, w, &values, &traction);
        });
    }
}

/// The displacement gradient ∂uᵢ/∂xⱼ and the displacement of a cell's unknowns `u` at `xi`.
pub(crate) fn field(shapes: &Shapes, xi: [f64; 3], h: f64, u: &[f64]) -> ([f64; 3], [[f64; 3]; 3]) {
    let m = shapes.len();
    let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
    eval(shapes, xi, 2.0 / h, &mut values, &mut gradients);
    let (mut displacement, mut gradient) = ([0.0; 3], [[0.0; 3]; 3]);
    for a in 0..m {
        for i in 0..3 {
            let ui = u.get(3 * a + i).copied().unwrap_or(0.0);
            displacement[i] += values[a] * ui;
            for j in 0..3 {
                gradient[i][j] += gradients[a][j] * ui;
            }
        }
    }
    (displacement, gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauss::gauss_legendre;
    use crate::space::local_modes;

    /// The Gauss points of a whole cell, all inside.
    fn whole(p: usize) -> Vec<QuadPoint> {
        let rule = gauss_legendre(p + 1);
        let mut points = Vec::new();
        for (z, wz) in &rule {
            for (y, wy) in &rule {
                for (x, wx) in &rule {
                    points.push(QuadPoint { xi: [*x, *y, *z], weight: wx * wy * wz, inside: true });
                }
            }
        }
        points
    }

    fn shapes(p: usize, trunk: bool) -> Shapes {
        Shapes::new(p, local_modes(p, trunk)).unwrap()
    }

    #[test]
    fn a_cell_matrix_is_symmetric_and_rigid_motions_cost_no_energy() {
        let lame = Lame { lambda: 1.2, mu: 0.8 };
        let mut scratch = Scratch::default();
        for (p, trunk) in [(1, false), (2, false), (3, false), (3, true), (5, true)] {
            let shapes = shapes(p, trunk);
            let (m, h) = (shapes.len(), 0.5);
            let volume = integrate(&shapes, &whole(p));
            assert!((0.125 * h * h * h * volume.inside.total() - h * h * h).abs() < 1e-14);
            assert!(volume.outside.is_zero());
            let integrals = body(&shapes, h, &volume.inside, &mut scratch);
            let vertices: f64 = (0..8).map(|corner| integrals[shapes.vertex(corner)]).sum();
            assert!((vertices - h * h * h).abs() < 1e-14, "the vertex modes sum to one");
            let k = stiffness(&shapes, h, &volume.inside, &lame, &mut scratch);
            let n = 3 * m;
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(k[i * n + j], k[j * n + i]);
                }
            }
            // A translation and a rotation about z, in the vertex modes.
            let (mut translation, mut rotation) = (vec![0.0; n], vec![0.0; n]);
            for corner in 0..8usize {
                let bits = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
                let mode = shapes.vertex(corner);
                translation[3 * mode + 1] = 1.0;
                rotation[3 * mode] = -(2.0 * bits[1] as f64 - 1.0);
                rotation[3 * mode + 1] = 2.0 * bits[0] as f64 - 1.0;
            }
            for motion in [&translation, &rotation] {
                for i in 0..n {
                    let force: f64 = (0..n).map(|j| k[i * n + j] * motion[j]).sum();
                    assert!(force.abs() < 1e-12, "degree {p}, row {i}: {force}");
                }
            }
            // Uniform strain ε_xx = 1 (u_x = x): the energy is ½ (λ + 2μ) V.
            let mut stretch = vec![0.0; n];
            for corner in 0..8usize {
                stretch[3 * shapes.vertex(corner)] = h * (corner & 1) as f64;
            }
            let energy: f64 = (0..n).map(|i| 0.5 * stretch[i] * (0..n).map(|j| k[i * n + j] * stretch[j]).sum::<f64>()).sum();
            assert!((energy - 0.5 * (lame.lambda + 2.0 * lame.mu) * h * h * h).abs() < 1e-13, "degree {p}: {energy}");
            let (u, gradient) = field(&shapes, [0.3, -0.2, 0.9], h, &stretch);
            assert!((u[0] - 0.5 * 1.3 * h).abs() < 1e-14 && (gradient[0][0] - 1.0).abs() < 1e-13 && gradient[1][1].abs() < 1e-13);
        }
    }

    /// The stiffness from the moments is the sum over the quadrature points, mode pair by mode
    /// pair, for points inside and outside the part.
    #[test]
    fn the_stiffness_from_moments_is_the_sum_over_the_points() {
        let lame = Lame { lambda: 1.7, mu: 0.6 };
        let (p, h) = (3, 0.8);
        let shapes = shapes(p, false);
        let (m, n) = (shapes.len(), 3 * shapes.len());
        // The points of a cell cut by the plane x = 0.2 h.
        let mut points = whole(p);
        for point in &mut points {
            point.inside = point.xi[0] < -0.6;
        }
        let volume = integrate(&shapes, &points);
        let mut scratch = Scratch::default();
        let fraction = 1e-3;
        let softer = Lame { lambda: fraction * lame.lambda, mu: fraction * lame.mu };
        let mut k = stiffness(&shapes, h, &volume.inside, &lame, &mut scratch);
        for (total, outside) in k.iter_mut().zip(stiffness(&shapes, h, &volume.outside, &softer, &mut scratch)) {
            *total += outside;
        }
        let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
        let mut direct = vec![0.0; n * n];
        for point in &points {
            eval(&shapes, point.xi, 2.0 / h, &mut values, &mut gradients);
            let w = point.weight * 0.125 * h * h * h * if point.inside { 1.0 } else { fraction };
            for a in 0..m {
                for b in 0..m {
                    let (ga, gb) = (gradients[a], gradients[b]);
                    for i in 0..3 {
                        for j in 0..3 {
                            let mut value = lame.lambda * ga[i] * gb[j] + lame.mu * ga[j] * gb[i];
                            if i == j {
                                value += lame.mu * dot(ga, gb);
                            }
                            direct[(3 * a + i) * n + 3 * b + j] += w * value;
                        }
                    }
                }
            }
        }
        let scale = direct.iter().fold(0.0f64, |largest, x| largest.max(x.abs()));
        for (value, exact) in k.iter().zip(&direct) {
            assert!((value - exact).abs() < 1e-12 * scale, "{value} vs {exact}");
        }
    }

    /// The support matrices from the moments of a face are the sums over its quadrature points.
    #[test]
    fn the_support_matrices_are_the_sums_over_the_surface_points() {
        let lame = Lame { lambda: 1.3, mu: 0.7 };
        let (p, h, lo) = (2, 0.5, [0.1, -0.2, 0.3]);
        let shapes = shapes(p, false);
        let (m, n) = (shapes.len(), 3 * shapes.len());
        // Two inclined triangles of two faces: one supported in x and z with a prescribed value
        // and under pressure, one loaded only.
        let faces = BTreeMap::from([
            (3u32, Face { fixed: [true, false, true], value: [0.02, 0.0, -0.01], traction: [0.0; 3], pressure: 4.0, tied: false }),
            (5u32, Face { traction: [1.0, -2.0, 0.5], ..Face::default() }),
        ]);
        let at = |x: f64, y: f64, z: f64| [lo[0] + x * h, lo[1] + y * h, lo[2] + z * h];
        let pieces = [
            Piece { corners: [at(0.1, 0.1, 0.2), at(0.9, 0.2, 0.3), at(0.3, 0.8, 0.9)], triangle: 0 },
            Piece { corners: [at(0.9, 0.2, 0.3), at(0.8, 0.9, 0.7), at(0.3, 0.8, 0.9)], triangle: 1 },
            Piece { corners: [at(0.0, 0.0, 0.5), at(1.0, 0.0, 0.5), at(0.0, 1.0, 0.6)], triangle: 2 },
        ];
        let unit = |piece: &Piece| {
            let normal = cross(sub(piece.corners[1], piece.corners[0]), sub(piece.corners[2], piece.corners[0]));
            let length = dot(normal, normal).sqrt();
            normal.map(|x| x / length)
        };
        let normals = [unit(&pieces[0]), unit(&pieces[1]), unit(&pieces[2])];
        let surface = Surface { normals: &normals, tags: &[3, 3, 5], faces: &faces, touching: None };
        let rule = gauss_legendre(2 * p + 1);
        let mut scratch = Scratch::default();
        let b = boundary(&shapes, lo, h, &pieces, &surface, &lame, &rule, &mut scratch);

        let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
        let (mut load, mut consistency_load, mut penalty_load) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let (mut consistency, mut penalty, mut traction) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
        for piece in &pieces {
            let (_, face, normal) = surface.face(piece).unwrap();
            piece_points(piece, lo, h, &rule, |xi, w| {
                eval(&shapes, xi, 2.0 / h, &mut values, &mut gradients);
                let t: Vec<[[f64; 3]; 3]> = gradients.iter().map(|d| tractions(*d, normal, face.fixed, &lame)).collect();
                let g = face.value;
                for a in 0..m {
                    for k in 0..3 {
                        let row = 3 * a + k;
                        load[row] += w * values[a] * (face.traction[k] - face.pressure * normal[k]);
                        if !face.supported() {
                            continue;
                        }
                        consistency_load[row] -= w * (t[a][0][k] * g[0] + t[a][1][k] * g[1] + t[a][2][k] * g[2]);
                        if face.fixed[k] {
                            penalty_load[row] += w * values[a] * g[k];
                        }
                        for b in 0..m {
                            for l in 0..3 {
                                let at = row * n + 3 * b + l;
                                traction[at] += w * (t[a][0][k] * t[b][0][l] + t[a][1][k] * t[b][1][l] + t[a][2][k] * t[b][2][l]);
                                consistency[at] -= w * (t[a][l][k] * values[b] + t[b][k][l] * values[a]);
                                if k == l && face.fixed[k] {
                                    penalty[at] += w * values[a] * values[b];
                                }
                            }
                        }
                    }
                }
            });
        }
        let close = |name: &str, got: &[f64], exact: &[f64]| {
            assert_eq!(got.len(), exact.len(), "{name}");
            let scale = exact.iter().fold(0.0f64, |largest, x| largest.max(x.abs()));
            assert!(scale > 0.0, "{name} is not exercised");
            for (value, exact) in got.iter().zip(exact) {
                assert!((value - exact).abs() < 1e-12 * scale, "{name}: {value} vs {exact}");
            }
        };
        close("load", &b.load, &load);
        close("consistency load", &b.consistency_load, &consistency_load);
        close("penalty load", &b.penalty_load, &penalty_load);
        close("consistency", &b.consistency, &consistency);
        close("penalty", &b.penalty, &penalty);
        close("traction", &b.traction, &traction);
    }

    #[test]
    fn the_nitsche_parameter_scales_with_the_stiffness_over_the_cell_size() {
        // A cell fixed on its whole lower x face: Λ is a property of the cell, proportional to
        // the modulus and inversely to the size.
        let parameter = |p: usize, h: f64, lame: Lame| -> f64 {
            let shapes = shapes(p, false);
            let faces = BTreeMap::from([(7u32, Face { fixed: [true; 3], ..Face::default() })]);
            let surface = Surface { normals: &[[-1.0, 0.0, 0.0]], tags: &[7], faces: &faces, touching: None };
            let corner = |y: f64, z: f64| [0.0, y * h, z * h];
            let pieces = [
                Piece { corners: [corner(0.0, 0.0), corner(0.0, 1.0), corner(1.0, 1.0)], triangle: 0 },
                Piece { corners: [corner(0.0, 0.0), corner(1.0, 1.0), corner(1.0, 0.0)], triangle: 0 },
            ];
            let mut scratch = Scratch::default();
            let k = stiffness(&shapes, h, &integrate(&shapes, &whole(p)).inside, &lame, &mut scratch);
            let b = boundary(&shapes, [0.0; 3], h, &pieces, &surface, &lame, &gauss_legendre(2 * p + 1), &mut scratch);
            // The vertex modes sum to one, so the x rows and columns of their penalty matrix
            // sum to the area of the face.
            let n = 3 * shapes.len();
            let vertex = |corner: usize| 3 * shapes.vertex(corner);
            let area: f64 = (0..8).map(|c| (0..8).map(|d| b.penalty[vertex(c) * n + vertex(d)]).sum::<f64>()).sum();
            assert!((area - h * h).abs() < 1e-13 * h * h, "degree {p}: {area}");
            nitsche_parameter(&shapes, &k, &b.traction).unwrap()
        };
        let lame = Lame { lambda: 1.2, mu: 0.8 };
        for p in 1..=2 {
            let base = parameter(p, 1.0, lame);
            assert!(base > 0.0 && base.is_finite());
            assert!((parameter(p, 0.5, lame) / base - 2.0).abs() < 1e-6, "half the size, twice the parameter");
            let stiffer = Lame { lambda: 12.0, mu: 8.0 };
            assert!((parameter(p, 1.0, stiffer) / base - 10.0).abs() < 1e-6, "ten times the modulus");
        }
        assert!(parameter(2, 1.0, lame) > parameter(1, 1.0, lame), "a higher degree needs a larger parameter");
    }
}
