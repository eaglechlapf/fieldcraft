//! Cell integrals from Legendre moments.
//!
//! Every integrand of a cell matrix is a product of two shape functions or their derivatives,
//! and on a Cartesian cell such a product is a polynomial of degree at most 2p along each axis.
//! Summing it mode pair by mode pair at every quadrature point costs (modes)² a point. Instead
//! each one-dimensional product is written once in Legendre polynomials,
//!
//! φₐ⁽ˢ⁾(ξ) φ_b⁽ᵗ⁾(ξ) = Σₖ T[s, t][a, b][k] Pₖ(ξ), k = 0 … 2p, s, t ∈ {0, 1} derivatives,
//!
//! and the quadrature points are summed into the *moments* of the integration domain,
//!
//! M[k₁, k₂, k₃] = Σ_q w_q P_k₁(ξ_q) P_k₂(η_q) P_k₃(ζ_q),
//!
//! which costs (2p + 1)³ a point whatever the number of modes. The integral of a product of two
//! modes a = (a₁, a₂, a₃), b = (b₁, b₂, b₃) is then the contraction
//!
//! Σ T₁[a₁, b₁][k₁] T₂[a₂, b₂][k₂] T₃[a₃, b₃][k₃] M[k₁, k₂, k₃],
//!
//! taken one axis at a time (sum factorisation: S. A. Orszag, *Spectral methods for problems in
//! complex geometries*, J. Comput. Phys. 37(1):70–92, 1980; for hp finite elements, J. M. Melenk,
//! K. Gerdes, C. Schwab, *Fully discrete hp-finite elements: fast quadrature*, Comput. Methods
//! Appl. Mech. Engrg. 190(32–33):4339–4364, 2001). This is the same sum over the same points in
//! another order: an identity of polynomials, not another quadrature. The Legendre basis keeps it
//! well conditioned (the coefficients T are computed by a Gauss rule that is exact for them, and
//! none of the sums cancels more than the integrals themselves do).
//!
//! The same moments serve any subset of the tensor-product modes ([`crate::space::Basis`]).

use crate::gauss::gauss_legendre;
use crate::legendre::{MAX_DEGREE, shape};

/// Legendre polynomials of a product of two shape functions, at most.
pub(crate) const MAX_PRODUCT: usize = 2 * MAX_DEGREE + 1;

/// P₀ … P_{count−1} at `x` (entries past `count` are zero).
pub(crate) fn legendre_values(count: usize, x: f64) -> [f64; MAX_PRODUCT] {
    let mut values = [0.0; MAX_PRODUCT];
    values[0] = 1.0;
    if count > 1 {
        values[1] = x;
    }
    for k in 1..count.min(MAX_PRODUCT).saturating_sub(1) {
        let kf = k as f64;
        values[k + 1] = ((2.0 * kf + 1.0) * x * values[k] - kf * values[k - 1]) / (kf + 1.0);
    }
    values
}

/// Which derivative each factor of a one-dimensional product carries: index `2 s + t` for
/// φₐ⁽ˢ⁾ φ_b⁽ᵗ⁾.
pub(crate) const VALUE_VALUE: usize = 0;
pub(crate) const VALUE_SLOPE: usize = 1;
pub(crate) const SLOPE_VALUE: usize = 2;
pub(crate) const SLOPE_SLOPE: usize = 3;

/// The Legendre coefficients of the one-dimensional shape functions of one degree and of their
/// products.
#[derive(Debug, Clone)]
pub(crate) struct Tables {
    /// Shape functions along an axis, p + 1.
    pub q: usize,
    /// Legendre polynomials of a product, 2p + 1.
    pub k: usize,
    /// `product[2 s + t][(a q + b) k + n]`: the coefficient of Pₙ in φₐ⁽ˢ⁾ φ_b⁽ᵗ⁾.
    product: [Vec<f64>; 4],
    /// `single[s][a k + n]`: the coefficient of Pₙ in φₐ⁽ˢ⁾ (zero past n = p).
    single: [Vec<f64>; 2],
}

impl Tables {
    pub fn new(p: usize) -> Tables {
        let p = p.clamp(1, MAX_DEGREE);
        let (q, k) = (p + 1, 2 * p + 1);
        // A product of two shape functions and a Legendre polynomial has degree at most 4p,
        // which 2p + 1 Gauss points integrate exactly.
        let rule = gauss_legendre(k);
        let mut product = [vec![0.0; q * q * k], vec![0.0; q * q * k], vec![0.0; q * q * k], vec![0.0; q * q * k]];
        let mut single = [vec![0.0; q * k], vec![0.0; q * k]];
        for (x, w) in &rule {
            let s = shape(p, *x);
            let legendre = legendre_values(k, *x);
            let factors = [&s.n, &s.d];
            for n in 0..k {
                // Pₙ has the norm² 2 / (2n + 1).
                let weight = w * legendre[n] * (n as f64 + 0.5);
                for a in 0..q {
                    for (order, table) in single.iter_mut().enumerate() {
                        table[a * k + n] += weight * factors[order][a];
                    }
                    for b in 0..q {
                        for (st, table) in product.iter_mut().enumerate() {
                            table[(a * q + b) * k + n] += weight * factors[st / 2][a] * factors[st % 2][b];
                        }
                    }
                }
            }
        }
        // What the degrees and parities of the factors make zero is zero exactly: the linear
        // functions have degree 1, a higher function φₐ has degree a and the parity of a, and a
        // derivative lowers both by one.
        let degree = |a: usize, order: usize| a.max(1) - order;
        for a in 0..q {
            for (order, table) in single.iter_mut().enumerate() {
                let d = degree(a, order);
                for n in 0..k {
                    if n > d || (a >= 2 && (d - n) % 2 == 1) {
                        table[a * k + n] = 0.0;
                    }
                }
            }
            for b in 0..q {
                for (st, table) in product.iter_mut().enumerate() {
                    let d = degree(a, st / 2) + degree(b, st % 2);
                    for n in 0..k {
                        if n > d || (a >= 2 && b >= 2 && (d - n) % 2 == 1) {
                            table[(a * q + b) * k + n] = 0.0;
                        }
                    }
                }
            }
        }
        Tables { q, k, product, single }
    }
}

/// The Legendre moments of a set of quadrature points in a cell's natural coordinates:
/// `m[k₁ + K (k₂ + K k₃)]`.
#[derive(Debug, Clone)]
pub(crate) struct Moments {
    k: usize,
    pub m: Vec<f64>,
}

impl Moments {
    pub fn new(tables: &Tables) -> Moments {
        Moments { k: tables.k, m: vec![0.0; tables.k * tables.k * tables.k] }
    }

    /// Add a point with its weight.
    pub fn add(&mut self, xi: [f64; 3], weight: f64) {
        let k = self.k;
        let (px, py, pz) = (legendre_values(k, xi[0]), legendre_values(k, xi[1]), legendre_values(k, xi[2]));
        let px = &px[..k.min(MAX_PRODUCT)];
        for (plane, z) in self.m.chunks_exact_mut(k * k).zip(&pz) {
            let wz = weight * z;
            for (row, y) in plane.chunks_exact_mut(k).zip(&py) {
                let wyz = wz * y;
                for (m, x) in row.iter_mut().zip(px) {
                    *m += wyz * x;
                }
            }
        }
    }

    /// The sum of the weights.
    pub fn total(&self) -> f64 {
        self.m.first().copied().unwrap_or(0.0)
    }

    pub fn is_zero(&self) -> bool {
        self.total() == 0.0
    }
}

/// Scratch arrays of the contractions of one cell.
#[derive(Debug, Clone, Default)]
pub(crate) struct Scratch {
    first: Vec<f64>,
    second: Vec<f64>,
}

/// Contract the third and then the second axis of `moments` with rows of coefficient tables:
/// `second[(c₂ r₃ + c₃) K + k₁] = Σ t₂[c₂ K + k₂] t₃[c₃ K + k₃] M[k₁, k₂, k₃]` for the `r₂` rows of
/// `t₂` and the `r₃` rows of `t₃`.
fn reduce(k: usize, moments: &[f64], t2: &[f64], r2: usize, t3: &[f64], r3: usize, scratch: &mut Scratch) {
    let kk = k * k;
    scratch.first.clear();
    scratch.first.resize(r3 * kk, 0.0);
    for (c3, out) in scratch.first.chunks_exact_mut(kk).enumerate() {
        for (k3, plane) in moments.chunks_exact(kk).enumerate() {
            let t = t3[c3 * k + k3];
            if t == 0.0 {
                continue;
            }
            for (o, m) in out.iter_mut().zip(plane) {
                *o += t * m;
            }
        }
    }
    scratch.second.clear();
    scratch.second.resize(r2 * r3 * k, 0.0);
    for c2 in 0..r2 {
        for c3 in 0..r3 {
            let out = &mut scratch.second[(c2 * r3 + c3) * k..(c2 * r3 + c3 + 1) * k];
            let plane = &scratch.first[c3 * kk..(c3 + 1) * kk];
            for (k2, row) in plane.chunks_exact(k).enumerate() {
                let t = t2[c2 * k + k2];
                if t == 0.0 {
                    continue;
                }
                for (o, m) in out.iter_mut().zip(row) {
                    *o += t * m;
                }
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// ∫ Nₐ⁽ˢ⁾ N_b⁽ᵗ⁾ over the points of `moments` for the pairs of `modes`, with the derivative
/// each factor carries along each axis in `kinds` (one of [`VALUE_VALUE`] … per axis), in
/// natural coordinates. The integral of pair (a, b) is written to `out[(a n + b) stride + at]`,
/// n the number of modes; with `upper`, only the pairs a ≤ b.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pairs(
    tables: &Tables,
    moments: &Moments,
    modes: &[[u8; 3]],
    kinds: [usize; 3],
    upper: bool,
    out: &mut [f64],
    stride: usize,
    at: usize,
    scratch: &mut Scratch,
) {
    let (q, k, n) = (tables.q, tables.k, modes.len());
    let (Some(t1), Some(t2), Some(t3)) = (tables.product.get(kinds[0]), tables.product.get(kinds[1]), tables.product.get(kinds[2])) else { return };
    if moments.m.len() != k * k * k || out.len() < n * n * stride || at >= stride {
        return;
    }
    let rows = q * q;
    reduce(k, &moments.m, t2, rows, t3, rows, scratch);
    for (ia, a) in modes.iter().enumerate() {
        let [a1, a2, a3] = a.map(usize::from);
        for (ib, b) in modes.iter().enumerate().skip(if upper { ia } else { 0 }) {
            let [b1, b2, b3] = b.map(usize::from);
            let (c1, c2, c3) = (a1 * q + b1, a2 * q + b2, a3 * q + b3);
            let (Some(t), Some(m)) = (t1.get(c1 * k..(c1 + 1) * k), scratch.second.get((c2 * rows + c3) * k..(c2 * rows + c3 + 1) * k)) else {
                continue;
            };
            out[(ia * n + ib) * stride + at] = dot(t, m);
        }
    }
}

/// ∫ Nₐ (or its derivative along `axis`, in natural coordinates) over the points of `moments`,
/// for each of `modes`.
pub(crate) fn singles(tables: &Tables, moments: &Moments, modes: &[[u8; 3]], axis: Option<usize>, scratch: &mut Scratch) -> Vec<f64> {
    let (q, k) = (tables.q, tables.k);
    let table = |a: usize| &tables.single[usize::from(axis == Some(a))];
    let mut out = vec![0.0; modes.len()];
    if moments.m.len() != k * k * k {
        return out;
    }
    reduce(k, &moments.m, table(1), q, table(2), q, scratch);
    let t1 = table(0);
    for (value, mode) in out.iter_mut().zip(modes) {
        let [a1, a2, a3] = mode.map(usize::from);
        if let (Some(t), Some(m)) = (t1.get(a1 * k..(a1 + 1) * k), scratch.second.get((a2 * q + a3) * k..(a2 * q + a3 + 1) * k)) {
            *value = dot(t, m);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed scatter of points in the cell, with weights.
    fn scatter(count: usize) -> Vec<([f64; 3], f64)> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..count).map(|_| ([2.0 * next() - 1.0, 2.0 * next() - 1.0, 2.0 * next() - 1.0], 0.1 + next())).collect()
    }

    #[test]
    fn products_expand_in_legendre_polynomials() {
        for p in [1, 2, 5, MAX_DEGREE] {
            let tables = Tables::new(p);
            let (q, k) = (tables.q, tables.k);
            for x in [-1.0, -0.63, 0.0, 0.4, 1.0] {
                let s = shape(p, x);
                let legendre = legendre_values(k, x);
                let factors = [&s.n, &s.d];
                for a in 0..q {
                    for order in 0..2 {
                        let sum: f64 = (0..k).map(|n| tables.single[order][a * k + n] * legendre[n]).sum();
                        assert!((sum - factors[order][a]).abs() < 1e-11, "degree {p}, φ{a} order {order} at {x}");
                    }
                    for b in 0..q {
                        for st in 0..4 {
                            let sum: f64 = (0..k).map(|n| tables.product[st][(a * q + b) * k + n] * legendre[n]).sum();
                            let exact = factors[st / 2][a] * factors[st % 2][b];
                            assert!((sum - exact).abs() < 1e-10 * (1.0 + exact.abs()), "degree {p}, ({a}, {b}) kind {st} at {x}: {sum} vs {exact}");
                        }
                    }
                }
            }
        }
    }

    /// The contraction of the moments is the sum over the points, pair by pair.
    #[test]
    fn contracted_moments_are_the_sums_over_the_points() {
        for p in [1, 3, 4] {
            let tables = Tables::new(p);
            let q = tables.q;
            // Every other mode of the tensor product, so a subset is exercised too.
            let modes: Vec<[u8; 3]> =
                (0..q * q * q).filter(|i| p == 1 || i % 2 == 0).map(|i| [(i % q) as u8, ((i / q) % q) as u8, (i / (q * q)) as u8]).collect();
            let n = modes.len();
            let points = scatter(40);
            let mut moments = Moments::new(&tables);
            for (xi, w) in &points {
                moments.add(*xi, *w);
            }
            assert!((moments.total() - points.iter().map(|(_, w)| w).sum::<f64>()).abs() < 1e-12);
            let mut scratch = Scratch::default();
            let factor = |mode: [u8; 3], xi: [f64; 3], orders: [usize; 3]| -> f64 {
                (0..3).map(|d| if orders[d] == 1 { shape(p, xi[d]).d[mode[d] as usize] } else { shape(p, xi[d]).n[mode[d] as usize] }).product()
            };
            for kinds in [[VALUE_VALUE; 3], [SLOPE_VALUE, VALUE_VALUE, VALUE_SLOPE], [VALUE_VALUE, SLOPE_SLOPE, VALUE_VALUE]] {
                let mut out = vec![0.0; n * n * 2];
                pairs(&tables, &moments, &modes, kinds, false, &mut out, 2, 1, &mut scratch);
                let (left, right) = (kinds.map(|kind| kind / 2), kinds.map(|kind| kind % 2));
                for (ia, a) in modes.iter().enumerate() {
                    for (ib, b) in modes.iter().enumerate() {
                        let exact: f64 = points.iter().map(|(xi, w)| w * factor(*a, *xi, left) * factor(*b, *xi, right)).sum();
                        let value = out[(ia * n + ib) * 2 + 1];
                        assert!(
                            (value - exact).abs() < 1e-10 * (1.0 + exact.abs()),
                            "degree {p}, pair ({ia}, {ib}), kinds {kinds:?}: {value} vs {exact}"
                        );
                        assert_eq!(out[(ia * n + ib) * 2], 0.0);
                    }
                }
            }
            for axis in [None, Some(0), Some(2)] {
                let orders = [0, 1, 2].map(|d| usize::from(axis == Some(d)));
                let values = singles(&tables, &moments, &modes, axis, &mut scratch);
                for (value, mode) in values.iter().zip(&modes) {
                    let exact: f64 = points.iter().map(|(xi, w)| w * factor(*mode, *xi, orders)).sum();
                    assert!((value - exact).abs() < 1e-10 * (1.0 + exact.abs()), "degree {p}, axis {axis:?}: {value} vs {exact}");
                }
            }
        }
    }
}
