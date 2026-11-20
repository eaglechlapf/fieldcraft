//! The integrals of one cell: stiffness and body load over its quadrature points, and loads and
//! supports over the pieces of the surface it owns.
//!
//! # Stiffness
//! For an isotropic material with Lamé constants λ and μ, the stiffness block of two modes a, b
//! (components i, j) is
//! K = λ G[i][j] + μ G[j][i] + μ δᵢⱼ tr G, with G[i][j] = ∫ ∂ᵢNₐ ∂ⱼN_b dV,
//! so only the nine numbers G of a mode pair are summed over the quadrature points. Points
//! outside the part are summed apart: the cell matrix is the stiffness of the part plus the
//! fictitious stiffness of the rest of the cell.
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
//! # Surface quadrature
//! A piece is a flat triangle, integrated with a Gauss rule on the square collapsed onto it
//! (M. G. Duffy, *Quadrature over a pyramid or cube of integrands with a singularity at a
//! vertex*, SIAM J. Numer. Anal. 19(6):1260–1262, 1982): the point at (s, t) ∈ [0, 1]² is
//! (1 − s) A + s ((1 − t) B + t C) with weight 2 · area · s.

use std::collections::BTreeMap;

use crate::dense;
use crate::grid::{Piece, QuadPoint, cross, dot, sub};
use crate::legendre::shape;
use crate::space::modes;

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
}

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
}

/// The values and physical gradients of a cell's modes at natural coordinates `xi`; `scale` is
/// 2 / (cell size), the derivative of the natural coordinate.
pub(crate) fn eval(p: usize, xi: [f64; 3], scale: f64, values: &mut [f64], gradients: &mut [[f64; 3]]) {
    let q = p + 1;
    let (sx, sy, sz) = (shape(p, xi[0]), shape(p, xi[1]), shape(p, xi[2]));
    for c in 0..q {
        for b in 0..q {
            let (n, dy, dz) = (sy.n[b] * sz.n[c], sy.d[b] * sz.n[c], sy.n[b] * sz.d[c]);
            for a in 0..q {
                let mode = a + q * (b + q * c);
                if let (Some(value), Some(gradient)) = (values.get_mut(mode), gradients.get_mut(mode)) {
                    *value = sx.n[a] * n;
                    *gradient = [scale * sx.d[a] * n, scale * sx.n[a] * dy, scale * sx.n[a] * dz];
                }
            }
        }
    }
}

/// The volume integrals of a cell.
pub(crate) struct Volume {
    /// G of each mode pair a ≤ b over the points inside the part, at `(a m + b) 9 + 3 i + j`.
    pub inside: Vec<f64>,
    /// The same over the points outside it.
    pub outside: Vec<f64>,
    /// ∫ Nₐ dV over the part.
    pub body: Vec<f64>,
    /// The volume of the part in the cell.
    pub volume: f64,
}

/// Integrate over the quadrature points of a cell of size `h`.
pub(crate) fn integrate(p: usize, h: f64, points: &[QuadPoint]) -> Volume {
    let m = modes(p);
    let (scale, jacobian) = (2.0 / h, 0.125 * h * h * h);
    let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
    let mut volume = Volume { inside: vec![0.0; m * m * 9], outside: vec![0.0; m * m * 9], body: vec![0.0; m], volume: 0.0 };
    for point in points {
        eval(p, point.xi, scale, &mut values, &mut gradients);
        let w = point.weight * jacobian;
        let g = if point.inside {
            volume.volume += w;
            for a in 0..m {
                volume.body[a] += w * values[a];
            }
            &mut volume.inside
        } else {
            &mut volume.outside
        };
        for a in 0..m {
            let ga = [w * gradients[a][0], w * gradients[a][1], w * gradients[a][2]];
            for b in a..m {
                let gb = gradients[b];
                let at = (a * m + b) * 9;
                for i in 0..3 {
                    for j in 0..3 {
                        g[at + 3 * i + j] += ga[i] * gb[j];
                    }
                }
            }
        }
    }
    volume
}

/// The stiffness matrix (3m × 3m, row-major, mode by mode with its three components) from the
/// sums `g` of [`integrate`].
pub(crate) fn stiffness(m: usize, g: &[f64], lame: &Lame) -> Vec<f64> {
    let n = 3 * m;
    let mut k = vec![0.0; n * n];
    if g.len() != m * m * 9 {
        return k;
    }
    for a in 0..m {
        for b in a..m {
            let at = (a * m + b) * 9;
            // A mode with itself: G is symmetric but for rounding.
            let entry = |i: usize, j: usize| if a == b { 0.5 * (g[at + 3 * i + j] + g[at + 3 * j + i]) } else { g[at + 3 * i + j] };
            let trace = entry(0, 0) + entry(1, 1) + entry(2, 2);
            for i in 0..3 {
                for j in 0..3 {
                    let mut value = lame.lambda * entry(i, j) + lame.mu * entry(j, i);
                    if i == j {
                        value += lame.mu * trace;
                    }
                    k[(3 * a + i) * n + 3 * b + j] = value;
                    k[(3 * b + j) * n + 3 * a + i] = value;
                }
            }
        }
    }
    k
}

/// Visit the quadrature points of the pieces that carry something: the piece's face tag, its
/// face and normal, the point's natural coordinates in the cell at `lo` of size `h`, and its
/// weight (an area).
fn surface_points(
    pieces: &[Piece],
    surface: &Surface,
    lo: [f64; 3],
    h: f64,
    rule: &[(f64, f64)],
    mut visit: impl FnMut(u32, &Face, [f64; 3], [f64; 3], f64),
) {
    for piece in pieces {
        let Some((tag, face, normal)) = surface.face(piece) else { continue };
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
                visit(tag, face, normal, xi, 0.25 * ws * wt * doubled * s);
            }
        }
    }
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

/// Integrate the loads and supports of the pieces a cell owns. The support matrices are empty
/// when none of the pieces is supported.
pub(crate) fn boundary(p: usize, lo: [f64; 3], h: f64, pieces: &[Piece], surface: &Surface, lame: &Lame, rule: &[(f64, f64)]) -> Boundary {
    let m = modes(p);
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
    let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
    let mut t = vec![[[0.0; 3]; 3]; m];
    surface_points(pieces, surface, lo, h, rule, |_, face, normal, xi, w| {
        eval(p, xi, 2.0 / h, &mut values, &mut gradients);
        if face.loaded() {
            for a in 0..m {
                for k in 0..3 {
                    out.load[3 * a + k] += w * values[a] * (face.traction[k] - face.pressure * normal[k]);
                }
            }
        }
        if !face.supported() || size == 0 {
            return;
        }
        for a in 0..m {
            t[a] = tractions(gradients[a], normal, face.fixed, lame);
        }
        let g = face.value;
        for a in 0..m {
            for k in 0..3 {
                let row = 3 * a + k;
                out.consistency_load[row] -= w * (t[a][0][k] * g[0] + t[a][1][k] * g[1] + t[a][2][k] * g[2]);
                if face.fixed[k] {
                    out.penalty_load[row] += w * values[a] * g[k];
                }
                for b in 0..m {
                    for l in 0..3 {
                        let at = row * n + 3 * b + l;
                        out.traction[at] += w * (t[a][0][k] * t[b][0][l] + t[a][1][k] * t[b][1][l] + t[a][2][k] * t[b][2][l]);
                        out.consistency[at] -= w * (t[a][l][k] * values[b] + t[b][k][l] * values[a]);
                        if k == l && face.fixed[k] {
                            out.penalty[at] += w * values[a] * values[b];
                        }
                    }
                }
            }
        }
    });
    out
}

/// The Nitsche parameter β = 2Λ of a cell of degree `p` with stiffness `k` and traction matrix
/// `h` (both 3m × 3m). `None` when the stiffness is not positive definite on the cell's
/// deformations (a fictitious stiffness too small for the cut).
pub(crate) fn nitsche_parameter(p: usize, k: &[f64], h: &[f64]) -> Option<f64> {
    let (q, n) = (p + 1, 3 * modes(p));
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
        let mode = bits[0] + q * (bits[1] + q * bits[2]);
        let x = [2.0 * bits[0] as f64 - 1.0, 2.0 * bits[1] as f64 - 1.0, 2.0 * bits[2] as f64 - 1.0];
        for axis in 0..3 {
            rigid[3 * mode + axis][axis] = 1.0 / 8f64.sqrt();
            // The rotation about `axis`: e × x.
            let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
            rigid[3 * mode + b][3 + axis] = -x[c] / 4.0;
            rigid[3 * mode + c][3 + axis] = x[b] / 4.0;
        }
    }
    let mut shifted = k.to_vec();
    for i in 0..n {
        for j in 0..n {
            let overlap: f64 = (0..6).map(|r| rigid[i][r] * rigid[j][r]).sum();
            shifted[i * n + j] += scale * overlap;
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
pub(crate) fn reaction(
    p: usize,
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
    let m = modes(p);
    if u.len() != 3 * m {
        return;
    }
    let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
    surface_points(pieces, surface, lo, h, rule, |tag, face, normal, xi, w| {
        if !face.supported() {
            return;
        }
        eval(p, xi, 2.0 / h, &mut values, &mut gradients);
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
        let total = reactions.entry(tag).or_insert([0.0; 3]);
        for i in 0..3 {
            if face.fixed[i] {
                total[i] += w * (traction[i] - beta * (displacement[i] - face.value[i]));
            }
        }
    });
}

/// The displacement gradient ∂uᵢ/∂xⱼ and the displacement of a cell's unknowns `u` at `xi`.
pub(crate) fn field(p: usize, xi: [f64; 3], h: f64, u: &[f64]) -> ([f64; 3], [[f64; 3]; 3]) {
    let m = modes(p);
    let (mut values, mut gradients) = (vec![0.0; m], vec![[0.0; 3]; m]);
    eval(p, xi, 2.0 / h, &mut values, &mut gradients);
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

    #[test]
    fn a_cell_matrix_is_symmetric_and_rigid_motions_cost_no_energy() {
        let lame = Lame { lambda: 1.2, mu: 0.8 };
        for p in 1..=3 {
            let (m, q, h) = (modes(p), p + 1, 0.5);
            let volume = integrate(p, h, &whole(p));
            assert!((volume.volume - h * h * h).abs() < 1e-14);
            let vertices: f64 = (0..8usize).map(|corner| volume.body[(corner & 1) + q * (((corner >> 1) & 1) + q * ((corner >> 2) & 1))]).sum();
            assert!((vertices - h * h * h).abs() < 1e-14, "the vertex modes sum to one");
            let k = stiffness(m, &volume.inside, &lame);
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
                let mode = bits[0] + q * (bits[1] + q * bits[2]);
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
                let bits = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
                stretch[3 * (bits[0] + q * (bits[1] + q * bits[2]))] = h * bits[0] as f64;
            }
            let energy: f64 = (0..n).map(|i| 0.5 * stretch[i] * (0..n).map(|j| k[i * n + j] * stretch[j]).sum::<f64>()).sum();
            assert!((energy - 0.5 * (lame.lambda + 2.0 * lame.mu) * h * h * h).abs() < 1e-13, "degree {p}: {energy}");
            let (u, gradient) = field(p, [0.3, -0.2, 0.9], h, &stretch);
            assert!((u[0] - 0.5 * 1.3 * h).abs() < 1e-14 && (gradient[0][0] - 1.0).abs() < 1e-13 && gradient[1][1].abs() < 1e-13);
        }
    }

    #[test]
    fn the_nitsche_parameter_scales_with_the_stiffness_over_the_cell_size() {
        // A cell fixed on its whole lower x face: Λ is a property of the cell, proportional to
        // the modulus and inversely to the size.
        let parameter = |p: usize, h: f64, lame: Lame| -> f64 {
            let faces = BTreeMap::from([(7u32, Face { fixed: [true; 3], ..Face::default() })]);
            let surface = Surface { normals: &[[-1.0, 0.0, 0.0]], tags: &[7], faces: &faces };
            let corner = |y: f64, z: f64| [0.0, y * h, z * h];
            let pieces = [
                Piece { corners: [corner(0.0, 0.0), corner(0.0, 1.0), corner(1.0, 1.0)], triangle: 0 },
                Piece { corners: [corner(0.0, 0.0), corner(1.0, 1.0), corner(1.0, 0.0)], triangle: 0 },
            ];
            let k = stiffness(modes(p), &integrate(p, h, &whole(p)).inside, &lame);
            let b = boundary(p, [0.0; 3], h, &pieces, &surface, &lame, &gauss_legendre(2 * p + 1));
            // The vertex modes sum to one, so the x rows and columns of their penalty matrix
            // sum to the area of the face.
            let (n, q) = (3 * modes(p), p + 1);
            let vertex = |corner: usize| 3 * ((corner & 1) + q * (((corner >> 1) & 1) + q * ((corner >> 2) & 1)));
            let area: f64 = (0..8).map(|c| (0..8).map(|d| b.penalty[vertex(c) * n + vertex(d)]).sum::<f64>()).sum();
            assert!((area - h * h).abs() < 1e-13 * h * h, "degree {p}: {area}");
            nitsche_parameter(p, &k, &b.traction).unwrap()
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
