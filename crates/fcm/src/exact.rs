//! Cut cells integrated exactly, from the triangles of the surface in them.
//!
//! The surface is a closed triangulation, so the material in a cell is a polyhedron, and what a
//! cell matrix needs of it are the integrals of polynomials: the Legendre moments
//! M[i, j, k] = ∫ Pᵢ(ξ) Pⱼ(η) Pₖ(ζ) over the material ([`crate::moments`]). The divergence
//! theorem turns each into an integral over the boundary of the material, with no quadrature
//! point inside it and no decision about which side a point is on (B. Mirtich, *Fast and
//! accurate computation of polyhedral mass properties*, J. Graphics Tools 1(2):31–50, 1996,
//! reduces the moments of a polyhedron to its faces and then to their edges this way; for cells
//! cut by an interface, Y. Sudhakar, J. P. Moitinho de Almeida, W. A. Wall, *An accurate, robust,
//! and easy-to-implement method for integration over arbitrary polyhedra: application to
//! embedded interface methods*, J. Comput. Phys. 273:393–415, 2014). What follows is that
//! reduction for the moments of a cell, written so that the boundary of the cell itself needs no
//! polygon of its own.
//!
//! # The reduction
//! Take ξ along one axis of the cell and Qᵢ(ξ) = ∫ Pᵢ from −1 to ξ, so Qᵢ(−1) = 0 for every i and
//! Qᵢ(1) = 0 for every i but Q₀(1) = 2. The field (Qᵢ(ξ) Pⱼ(η) Pₖ(ζ), 0, 0) has the divergence
//! Pᵢ Pⱼ Pₖ, so
//!
//! M[i, j, k] = Σ over the surface in the cell of ∫ Qᵢ Pⱼ Pₖ n_ξ dS + 2 δᵢ₀ A[j, k],
//!
//! with A[j, k] = ∫ Pⱼ Pₖ over the material in the cell's upper face across ξ: the faces along
//! ξ have no n_ξ, and Qᵢ vanishes on the lower face. The same step in the plane of that face,
//! along η, leaves the cuts of the triangles by the plane and the material on one edge of the
//! face; along that edge it leaves the points where the cuts cross it and the cell's highest
//! corner:
//!
//! A[j, k] = Σ over the cuts of ∫ Qⱼ Pₖ dζ + 2 δⱼ₀ E[k],
//! E[k] = Σ over the crossings of ± Qₖ(ζ) + 2 δₖ₀ [the corner is inside].
//!
//! Every term but the last belongs to a triangle that reaches the cell, and the last is one
//! exact inside test ([`crate::inside`]). On a flat triangle the integrands are polynomials: a
//! clipped triangle is integrated in strips along η between its corners' ζ, by Gauss rules of
//! 2p + 1 points along a strip and 3p + 2 across the strips (degrees 4p + 1 and 6p + 2 for the
//! products of two shape functions of degree p), and a cut by 2p + 1 points. So the moments are
//! those of the triangulated body to rounding, whatever the number and the directions of its
//! triangles, at a cost that grows with their number and not with its square: no octree, and no
//! depth to choose.
//!
//! The axis ξ is the one the fewest triangles face. A triangle along ξ has no n_ξ and takes no
//! surface integral, so a wall, a hole or a bend along an axis of the grid costs only its cuts
//! by the face, which are line integrals.
//!
//! # Coincidences
//! The sums above are over closed chains: the cuts of the triangles by the plane of the face
//! close into polygons, and their crossings of the edge's line come in pairs. They stay closed
//! in every position because each level is taken just above the cell (the plane at ξ = 1⁺, the
//! line at η = 1⁺): a corner counts as above a plane only when its coordinate is larger, and the
//! point where an edge of a triangle crosses a plane is computed from the same two corners
//! whichever triangle asks. That is the perturbation of H. Edelsbrunner, E. P. Mücke,
//! *Simulation of Simplicity*, ACM Trans. Graph. 9(1):66–104, 1990, with the order ζ ≫ η ≫ ξ;
//! the integrands are continuous where the levels meet, so a face of the part in a grid plane is
//! counted once. The inside test of the corner perturbs in another order, so a cell whose
//! highest corner lies on the surface is not integrated here; nor is one whose pieces of
//! material cannot be told apart with certainty (below). Those are left to the quadrature of
//! [`crate::grid`], which integrates any cell.
//!
//! # Islands
//! A cell that holds separate pieces of material ([`crate::islands`]) needs the moments of each.
//! Every term above belongs to a triangle, so to a sheet of surface (the triangles joined by the
//! edges they share in the cell), and the terms of one sheet alone are the moments of a region:
//! a sheet divides the cell in two, and counting its crossings on the way from a point to the
//! highest corner (up along ξ, then along η in the face, then along the edge) gives +1 on its
//! material side when the corner is on its other side, or −1 on its other side when the corner
//! is on its material side. The sign of a sheet's volume tells which. A positive sheet bounds a
//! piece of material from the corner's side and is an island; so is the material at the corner
//! when it is inside. A negative sheet cuts a hollow out of the island around it: the one whose
//! sheet is the smallest that holds a point just behind the negative sheet, or the corner's
//! when none does, which the same count of crossings from that point tells. The count is
//! checked (the point is in the material, so the counts of all sheets and the corner sum to
//! one, and each sheet's is 0 or its sign); when no point of a sheet passes, the cell goes to
//! the quadrature. So does a cell in which a triangle that owns no piece of it, and so belongs
//! to no sheet, adds to the sums: a face of the part in a face of the cell, with its material
//! in the next cell.

use crate::Result;
use crate::grid::{self, Piece, Triangles, cross, dot, sub};
use crate::inside::Solid;
use crate::islands::{CutLeaf, distance_squared, sheets};
use crate::moments::{MAX_PRODUCT, Moments, Tables, legendre_values};

/// A cell whose highest corner is this close to a triangle, in cell sizes, is left to the
/// quadrature: the inside test of the corner and the levels above the cell must agree.
const CORNER: f64 = 1e-7;
/// A triangle whose normal has less than this part along the axis is along it: what its surface
/// integral would add is below rounding.
const ALONG: f64 = 1e-14;
/// What triangles that own no piece of the cell may add to a moment, of the cell's volume 8 in
/// natural coordinates, before the cell counts as degenerate; also how far outside 0 to 8 the
/// volume of an island may fall.
const STRAY: f64 = 1e-9; // unit-factors: relative to the cell's volume
/// A sheet or an island of less than this part of the cell's volume holds no material of its
/// own: it goes with the largest.
const SLIVER: f64 = 1e-12; // unit-factors: relative to the cell's volume
/// How far behind a piece of surface, in cell sizes, the point lies from which the sheets around
/// it are counted.
const BEHIND: f64 = 1e-6; // unit-factors: relative to the cell size
/// Pieces of a sheet tried for such a point, the largest first.
const TRIES: usize = 6;
/// Edges of a clipped triangle that span one strip, at most (a triangle clipped by a box has
/// nine corners at most).
const MOST_EDGES: usize = 12;

/// Q₀ … Q_{count−1} at `x`, Qᵢ(x) = ∫ Pᵢ from −1 to x: Q₀ = x + 1 and
/// Qᵢ = (Pᵢ₊₁ − Pᵢ₋₁) / (2i + 1) (entries past `count` are zero).
fn primitives(count: usize, x: f64) -> [f64; MAX_PRODUCT] {
    let count = count.min(MAX_PRODUCT);
    let mut p = [0.0; MAX_PRODUCT + 1];
    p[0] = 1.0;
    p[1] = x;
    for k in 1..count {
        let kf = k as f64;
        p[k + 1] = ((2.0 * kf + 1.0) * x * p[k] - kf * p[k - 1]) / (kf + 1.0);
    }
    let mut q = [0.0; MAX_PRODUCT];
    if count > 0 {
        q[0] = x + 1.0;
    }
    for i in 1..count {
        q[i] = (p[i + 1] - p[i - 1]) / (2.0 * i as f64 + 1.0);
    }
    q
}

/// The cut of a triangle by the plane just above the cell, in the two coordinates of that plane
/// (η, ζ), directed so the material is on its left seen from above.
#[derive(Debug, Clone, Copy)]
struct Cut {
    from: [f64; 2],
    to: [f64; 2],
    triangle: u32,
}

/// Where a cut crosses the line just beyond the face's upper edge across η: the coordinate ζ,
/// and +1 when the material ends there going up ζ, −1 when it begins.
#[derive(Debug, Clone, Copy)]
struct Crossing {
    at: f64,
    sign: i32,
    triangle: u32,
}

/// The cell and the axes the reduction runs along.
struct Frame {
    lo: [f64; 3],
    hi: [f64; 3],
    /// ξ, η and ζ: the axis of the flux and the next two in cyclic order.
    axes: [usize; 3],
    /// Legendre polynomials along an axis, 2p + 1.
    k: usize,
}

impl Frame {
    /// A coordinate along axis `a` in the cell's natural coordinate.
    fn natural(&self, a: usize, x: f64) -> f64 {
        2.0 * (x - self.lo[a]) / (self.hi[a] - self.lo[a]) - 1.0
    }

    /// The surface integral of one clipped triangle, `polygon` (convex, in natural coordinates
    /// ordered ξ, η, ζ), added to `tensor[i + K (j + K k)]`: Σ ∫ Qᵢ Pⱼ Pₖ dη dζ over its
    /// projection along ξ, times `sign` (that of the triangle's normal along ξ). `rules` are the
    /// Gauss rules along a strip and across the strips. Returns the points used.
    fn flux(&self, polygon: &[[f64; 3]], sign: f64, rules: [&[(f64, f64)]; 2], tensor: &mut [f64], rows: &mut Vec<f64>) -> usize {
        let k = self.k;
        let mut levels: Vec<f64> = polygon.iter().map(|corner| corner[2]).collect();
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        rows.resize(k * k, 0.0);
        let mut points = 0;
        for strip in levels.windows(2) {
            let (z0, z1) = (strip[0], strip[1]);
            if !(z1 > z0) {
                continue;
            }
            // The edges that span the strip, each from its lower to its upper end: two of a
            // convex polygon, and more only where rounding doubles one.
            let mut spanning = [([0.0; 3], [0.0; 3]); MOST_EDGES];
            let mut count = 0;
            for (i, p) in polygon.iter().enumerate() {
                let Some(q) = polygon.get((i + 1) % polygon.len()) else { continue };
                let (lower, upper) = if p[2] <= q[2] { (p, q) } else { (q, p) };
                if lower[2] <= z0 && upper[2] >= z1 && count < MOST_EDGES {
                    spanning[count] = (*lower, *upper);
                    count += 1;
                }
            }
            if count < 2 {
                continue;
            }
            for (gz, wz) in rules[1] {
                let z = 0.5 * (z0 + z1) + 0.5 * (z1 - z0) * gz;
                // The strip's two ends at this ζ: (η, ξ), the lowest and the highest η.
                let (mut left, mut right) = ((f64::INFINITY, 0.0), (f64::NEG_INFINITY, 0.0));
                for (lower, upper) in &spanning[..count] {
                    let s = (z - lower[2]) / (upper[2] - lower[2]);
                    let (eta, xi) = (lower[1] + s * (upper[1] - lower[1]), lower[0] + s * (upper[0] - lower[0]));
                    if eta < left.0 {
                        left = (eta, xi);
                    }
                    if eta > right.0 {
                        right = (eta, xi);
                    }
                }
                let length = right.0 - left.0;
                if !(length > 0.0 && length.is_finite()) {
                    continue;
                }
                rows.fill(0.0);
                for (g, w) in rules[0] {
                    let s = 0.5 * (g + 1.0);
                    let (q, p) = (primitives(k, left.1 + s * (right.1 - left.1)), legendre_values(k, left.0 + s * length));
                    let weight = 0.5 * w * length;
                    for (row, pj) in rows.chunks_exact_mut(k).zip(&p) {
                        let wp = weight * pj;
                        for (sum, qi) in row.iter_mut().zip(&q) {
                            *sum += wp * qi;
                        }
                    }
                }
                let pz = legendre_values(k, z);
                let weight = sign * 0.5 * wz * (z1 - z0);
                for (plane, pk) in tensor.chunks_exact_mut(k * k).zip(&pz) {
                    let wp = weight * pk;
                    for (sum, row) in plane.iter_mut().zip(rows.iter()) {
                        *sum += wp * row;
                    }
                }
                points += rules[0].len();
            }
        }
        points
    }

    /// The cut of a triangle by the plane just above the cell: from where the triangle's
    /// boundary comes down through the plane to where it goes up again, which is the direction
    /// of the boundary of the part of the triangle above. `None` when the triangle lies wholly
    /// above or wholly not.
    fn above(&self, triangle: &[[f64; 3]; 3], t: u32) -> Option<Cut> {
        let [a, b, c] = self.axes;
        let up = triangle.map(|corner| corner[a] > self.hi[a]);
        // Where the edge from a corner at or below the plane to one above meets it: from the
        // same two corners in the same order for both triangles of the edge.
        let meet = |down: &[f64; 3], over: &[f64; 3]| {
            let s = (self.hi[a] - down[a]) / (over[a] - down[a]);
            [down[b] + s * (over[b] - down[b]), down[c] + s * (over[c] - down[c])]
        };
        let (mut from, mut to) = (None, None);
        for e in 0..3 {
            let (p, q) = (&triangle[e], &triangle[(e + 1) % 3]);
            match (up[e], up[(e + 1) % 3]) {
                (true, false) => from = Some(meet(q, p)),
                (false, true) => to = Some(meet(p, q)),
                _ => {}
            }
        }
        Some(Cut { from: from?, to: to?, triangle: t })
    }

    /// The line integral of one cut over its part in the face, 2 ∫ Qⱼ Pₖ dζ added to
    /// `tensor[K (j + K k)]`, and where the cut crosses the line just beyond the face's upper
    /// edge across η, 4 ± Qₖ added to `tensor[K² k]` when that is on the cell's edge. Returns
    /// the crossing, wherever it is, and the points used.
    fn face(&self, cut: &Cut, rule: &[(f64, f64)], tensor: &mut [f64]) -> (Option<Crossing>, usize) {
        let ([_, b, c], k) = (self.axes, self.k);
        let (u, v) = (cut.from, cut.to);
        let (u_over, v_over) = (u[0] > self.hi[b], v[0] > self.hi[b]);
        if u_over && v_over {
            return (None, 0);
        }
        // The part of the cut at or below the edge, as a range of its parameter.
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        let mut crossing = None;
        if u_over != v_over {
            let (down, over) = if u_over { (v, u) } else { (u, v) };
            let s = (self.hi[b] - down[0]) / (over[0] - down[0]);
            let at = down[1] + s * (over[1] - down[1]);
            // Coming down from beyond the edge, the cut has the material on the side of
            // larger ζ behind it: going up ζ along the edge's line, the material ends here.
            let sign = if u_over {
                t0 = 1.0 - s;
                1
            } else {
                t1 = s;
                -1
            };
            crossing = Some(Crossing { at, sign, triangle: cut.triangle });
            if at <= self.hi[c] && at > self.lo[c] {
                let q = primitives(k, self.natural(c, at));
                for (plane, qk) in tensor.chunks_exact_mut(k * k).zip(&q) {
                    if let Some(first) = plane.first_mut() {
                        *first += 4.0 * f64::from(sign) * qk;
                    }
                }
            }
        }
        // Inside the face: ζ between its two edges, η above its lower edge (below it Qⱼ is
        // zero, so where that cut falls needs no care).
        let mut keep = |from: f64, to: f64, bound: f64, below: bool| {
            let d = to - from;
            if d == 0.0 {
                if (below && from > bound) || (!below && from < bound) {
                    t1 = t0;
                }
                return;
            }
            let at = (bound - from) / d;
            if (d > 0.0) == below { t1 = t1.min(at) } else { t0 = t0.max(at) }
        };
        keep(u[1], v[1], self.hi[c], true);
        keep(u[1], v[1], self.lo[c], false);
        keep(u[0], v[0], self.lo[b], false);
        let rise = 2.0 * (v[1] - u[1]) / (self.hi[c] - self.lo[c]);
        if !(t1 > t0) || rise == 0.0 || !rise.is_finite() {
            return (crossing, 0);
        }
        for (g, w) in rule {
            let t = t0 + 0.5 * (g + 1.0) * (t1 - t0);
            let (eta, zeta) = (self.natural(b, u[0] + t * (v[0] - u[0])), self.natural(c, u[1] + t * (v[1] - u[1])));
            let (q, p) = (primitives(k, eta), legendre_values(k, zeta));
            let weight = 2.0 * 0.5 * w * (t1 - t0) * rise;
            for (plane, pk) in tensor.chunks_exact_mut(k * k).zip(&p) {
                let wp = weight * pk;
                for (row, qj) in plane.chunks_exact_mut(k).zip(&q) {
                    if let Some(first) = row.first_mut() {
                        *first += wp * qj;
                    }
                }
            }
        }
        (crossing, rule.len())
    }

    /// The way from the point `x` in the cell to just beyond its highest corner, up along ξ,
    /// then along η in the plane just above the cell, then along ζ on the line just beyond the
    /// face's edge: every crossing of the surface on it is handed to `add` with its triangle,
    /// +1 where the material ends and −1 where it begins.
    fn way(&self, x: [f64; 3], reach: &[u32], surface: &Triangles, cuts: &[Cut], crossings: &[Crossing], add: &mut impl FnMut(u32, i32)) {
        let [a, b, c] = self.axes;
        for &t in reach {
            let Some(triangle) = surface.corners.get(t as usize) else { continue };
            if let Some(at) = grid::pierce(triangle, a, x[b], x[c])
                && at > x[a]
                && at <= self.hi[a]
            {
                let facing = cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]))[a];
                if facing != 0.0 {
                    add(t, if facing > 0.0 { 1 } else { -1 });
                }
            }
        }
        for cut in cuts {
            let (u, v) = (cut.from, cut.to);
            if (u[1] > x[c]) != (v[1] > x[c]) {
                let at = u[0] + (x[c] - u[1]) / (v[1] - u[1]) * (v[0] - u[0]);
                if at > x[b] && at <= self.hi[b] {
                    add(cut.triangle, if v[1] > u[1] { 1 } else { -1 });
                }
            }
        }
        for crossing in crossings {
            if crossing.at > x[c] && crossing.at <= self.hi[c] {
                add(crossing.triangle, crossing.sign);
            }
        }
    }

    /// The moments in the order of [`Moments`], from a tensor in the order of the frame's axes.
    fn moments(&self, tables: &Tables, tensor: &[f64]) -> Moments {
        let (k, [a, b, c]) = (self.k, self.axes);
        let stride = [1, k, k * k];
        let mut moments = Moments::new(tables);
        for (index, value) in tensor.iter().enumerate() {
            let (i, j, l) = (index % k, (index / k) % k, index / (k * k));
            if let Some(slot) = moments.m.get_mut(i * stride[a] + j * stride[b] + l * stride[c]) {
                *slot = *value;
            }
        }
        moments
    }
}

/// What is not in `inside`, of a whole cell: its moments are 8 for the constant and none else.
fn rest(tables: &Tables, inside: &[Moments]) -> Moments {
    let mut outside = Moments::new(tables);
    for moments in inside {
        for (out, m) in outside.m.iter_mut().zip(&moments.m) {
            *out -= m;
        }
    }
    if let Some(first) = outside.m.first_mut() {
        *first += 8.0;
    }
    outside
}

/// The moments of the material in the cut cell from `lo` to `hi`, and of its islands with
/// `separate`, from the triangles `reach` that reach it (the module's reduction; `rules` are the
/// Gauss rules of 2p + 1 and of 3p + 2 points). `None` when the cell is in a position this does
/// not decide with certainty, which the quadrature of [`crate::grid`] then integrates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate(
    tables: &Tables,
    lo: [f64; 3],
    hi: [f64; 3],
    reach: &[u32],
    pieces: &[Piece],
    surface: &Triangles,
    solid: &Solid,
    rules: [&[(f64, f64)]; 2],
    separate: bool,
) -> Result<Option<CutLeaf>> {
    let k = tables.k;
    let h = hi[0] - lo[0];
    if !(0..3).all(|a| hi[a] - lo[a] > 0.0) || k == 0 || k > MAX_PRODUCT {
        return Ok(None);
    }
    let near = (CORNER * h) * (CORNER * h);
    if reach.iter().any(|t| surface.corners.get(*t as usize).is_some_and(|triangle| distance_squared(hi, triangle) <= near)) {
        return Ok(None);
    }
    // The axis the fewest triangles face, and of those the one they face least.
    let area = |triangle: &[[f64; 3]; 3]| cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]));
    let (mut facing, mut faced) = ([0usize; 3], [0.0; 3]);
    for t in reach {
        let Some(normal) = surface.corners.get(*t as usize).map(area) else { continue };
        let length = dot(normal, normal).sqrt();
        for a in 0..3 {
            if normal[a].abs() > ALONG * length {
                facing[a] += 1;
                faced[a] += normal[a].abs();
            }
        }
    }
    let axis = (0..3).min_by(|x, y| facing[*x].cmp(&facing[*y]).then(faced[*x].total_cmp(&faced[*y]))).unwrap_or(0);
    let frame = Frame { lo, hi, axes: [axis, (axis + 1) % 3, (axis + 2) % 3], k };
    let [a, b, c] = frame.axes;

    // The sheets of surface, each with a tensor of its own; one more for the triangles that own
    // no piece here. One tensor for all when the cell is one island whatever it holds.
    let (owned, mut sets, groups) = if separate { sheets(lo, hi, pieces, surface) } else { (Vec::new(), crate::islands::Sets::new(0), 1) };
    let apart = separate && groups > 1;
    let root_of: Vec<usize> = if apart { (0..owned.len()).map(|i| sets.find(i)).collect() } else { Vec::new() };
    let stray = owned.len();
    let slot_of = |t: u32| {
        if !apart {
            return 0;
        }
        owned.binary_search(&t).ok().and_then(|i| root_of.get(i).copied()).unwrap_or(stray)
    };
    let mut tensors: Vec<Vec<f64>> = vec![Vec::new(); if apart { stray + 1 } else { 1 }];
    let (mut cuts, mut crossings): (Vec<Cut>, Vec<Crossing>) = (Vec::new(), Vec::new());
    let mut rows = Vec::new();
    let mut points = 0usize;
    for &t in reach {
        let Some(triangle) = surface.corners.get(t as usize) else { continue };
        let Some(tensor) = tensors.get_mut(slot_of(t)) else { continue };
        tensor.resize(k * k * k, 0.0);
        let normal = area(triangle);
        if normal[a].abs() > ALONG * dot(normal, normal).sqrt() {
            let polygon: Vec<[f64; 3]> = grid::clip(triangle, lo, hi)
                .into_iter()
                .map(|(corner, _)| [frame.natural(a, corner[a]), frame.natural(b, corner[b]), frame.natural(c, corner[c])])
                .collect();
            if polygon.len() >= 3 {
                points += frame.flux(&polygon, normal[a].signum(), rules, tensor, &mut rows);
            }
        }
        if let Some(cut) = frame.above(triangle, t) {
            let (crossing, used) = frame.face(&cut, rules[0], tensor);
            points += used;
            cuts.push(cut);
            crossings.extend(crossing);
        }
    }
    let corner = solid.contains(hi)?;
    let volume = |tensor: &[f64]| tensor.first().copied().unwrap_or(0.0);
    let plausible = |volume: f64| (-8.0 * STRAY..=8.0 * (1.0 + STRAY)).contains(&volume);

    if !apart {
        let Some(tensor) = tensors.first_mut() else { return Ok(None) };
        tensor.resize(k * k * k, 0.0);
        if corner && let Some(first) = tensor.first_mut() {
            *first += 8.0;
        }
        if !plausible(volume(tensor)) {
            return Ok(None);
        }
        if volume(tensor) <= 8.0 * SLIVER {
            tensor.fill(0.0);
        }
        let inside = vec![frame.moments(tables, tensor)];
        let outside = rest(tables, &inside);
        return Ok(Some(CutLeaf { inside, outside, points, of_piece: Vec::new(), exact: true }));
    }

    // Triangles that own no piece of the cell add nothing, unless one lies in a face of it.
    if tensors.get(stray).is_some_and(|tensor| tensor.iter().any(|value| value.abs() > 8.0 * STRAY)) {
        return Ok(None);
    }
    let mut roots = root_of.clone();
    roots.sort_unstable();
    roots.dedup();
    let volumes: Vec<f64> = (0..owned.len()).map(|root| tensors.get(root).map_or(0.0, |tensor| volume(tensor))).collect();
    let kind = |root: usize| {
        let volume = volumes.get(root).copied().unwrap_or(0.0);
        if volume > 8.0 * SLIVER {
            1
        } else if volume < -8.0 * SLIVER {
            -1
        } else {
            0
        }
    };

    // The island of every sheet: a positive sheet is one; a negative sheet goes with the
    // smallest positive sheet around a point behind it (`Some`), or with the corner (`None`).
    let unit = |t: u32| surface.normals.get(t as usize).copied();
    let mut home: Vec<Option<Option<usize>>> = vec![None; owned.len()];
    for &root in &roots {
        match kind(root) {
            1 => {
                if let Some(slot) = home.get_mut(root) {
                    *slot = Some(Some(root));
                }
                continue;
            }
            0 => continue,
            _ => {}
        }
        // The largest pieces of the sheet.
        let mut tried: Vec<(f64, usize)> = pieces
            .iter()
            .enumerate()
            .filter(|(_, piece)| slot_of(piece.triangle) == root)
            .map(|(index, piece)| {
                let doubled = cross(sub(piece.corners[1], piece.corners[0]), sub(piece.corners[2], piece.corners[0]));
                (dot(doubled, doubled), index)
            })
            .collect();
        tried.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        let mut found = None;
        for (_, index) in tried.into_iter().take(TRIES) {
            let Some(piece) = pieces.get(index) else { continue };
            let Some(normal) = unit(piece.triangle) else { continue };
            let mut x = [0.0; 3];
            for axis in 0..3 {
                x[axis] = (piece.corners[0][axis] + piece.corners[1][axis] + piece.corners[2][axis]) / 3.0 - BEHIND * h * normal[axis];
            }
            if !(0..3).all(|axis| x[axis] > lo[axis] && x[axis] < hi[axis]) {
                continue;
            }
            // The crossings of each sheet on the way from the point to the corner.
            let mut count = vec![0i32; owned.len()];
            let mut sound = true;
            let mut add = |t: u32, sign: i32| match owned.binary_search(&t).ok().and_then(|i| root_of.get(i)).and_then(|root| count.get_mut(*root)) {
                Some(sum) => *sum += sign,
                None => sound = false,
            };
            frame.way(x, reach, surface, &cuts, &crossings, &mut add);
            // The point is in the material and behind its own sheet.
            let total: i32 = count.iter().sum::<i32>() + i32::from(corner);
            sound &= total == 1 && count.get(root) == Some(&0);
            sound &= roots.iter().all(|other| {
                let crossed = count.get(*other).copied().unwrap_or(0);
                match kind(*other) {
                    1 => crossed == 0 || crossed == 1,
                    -1 => crossed == 0 || crossed == -1,
                    _ => crossed.abs() <= 1,
                }
            });
            if !sound {
                continue;
            }
            let around = roots.iter().copied().filter(|other| kind(*other) == 1 && count.get(*other) == Some(&1)).min_by(|x, y| {
                let size = |root: &usize| volumes.get(*root).copied().unwrap_or(0.0);
                size(x).total_cmp(&size(y)).then(x.cmp(y))
            });
            if around.is_none() && !corner {
                continue;
            }
            found = Some(around);
            break;
        }
        let Some(found) = found else { return Ok(None) };
        if let Some(slot) = home.get_mut(root) {
            *slot = Some(found);
        }
    }

    // The islands: the positive sheets and the corner, each with the negative sheets it holds,
    // in the order of their lowest triangle.
    struct Island {
        /// The lowest sheet (the sheets are in the order of their lowest triangle).
        first: usize,
        /// The positive sheet, or the corner.
        key: Option<usize>,
        tensor: Vec<f64>,
    }
    let mut islands: Vec<Island> = Vec::new();
    if corner {
        let mut tensor = vec![0.0; k * k * k];
        if let Some(first) = tensor.first_mut() {
            *first = 8.0;
        }
        islands.push(Island { first: usize::MAX, key: None, tensor });
    }
    for &root in &roots {
        if kind(root) == 1 {
            islands.push(Island { first: root, key: Some(root), tensor: vec![0.0; k * k * k] });
        }
    }
    let mut slivers: Vec<usize> = Vec::new();
    for &root in &roots {
        let (Some(tensor), Some(place)) = (tensors.get(root), home.get(root)) else { continue };
        let Some(island) = place.and_then(|key| islands.iter_mut().find(|island| island.key == key)) else {
            slivers.push(root);
            continue;
        };
        island.first = island.first.min(root);
        for (sum, value) in island.tensor.iter_mut().zip(tensor) {
            *sum += value;
        }
    }
    if islands.iter().any(|island| !plausible(volume(&island.tensor))) {
        return Ok(None);
    }
    // What holds no material of its own goes with the largest island.
    let mut most = 0;
    for (index, island) in islands.iter().enumerate() {
        if islands.get(most).is_some_and(|kept| volume(&island.tensor) > volume(&kept.tensor)) {
            most = index;
        }
    }
    let gone: Vec<bool> = islands.iter().enumerate().map(|(index, island)| index != most && volume(&island.tensor) <= 8.0 * SLIVER).collect();
    let mut extra = vec![0.0; k * k * k];
    let mut lowest = usize::MAX;
    let absorbed = islands.iter().zip(&gone).filter(|(_, gone)| **gone).map(|(island, _)| (island.first, &island.tensor));
    for (first, tensor) in absorbed.chain(slivers.iter().filter_map(|root| Some((*root, tensors.get(*root)?)))) {
        lowest = lowest.min(first);
        for (sum, value) in extra.iter_mut().zip(tensor) {
            *sum += value;
        }
    }
    let Some(kept) = islands.get_mut(most) else {
        // No sheet with material behind it and none at the corner: the cell holds none.
        let inside = vec![Moments::new(tables)];
        let outside = rest(tables, &inside);
        return Ok(Some(CutLeaf { inside, outside, points, of_piece: Vec::new(), exact: true }));
    };
    kept.first = kept.first.min(lowest);
    for (sum, value) in kept.tensor.iter_mut().zip(&extra) {
        *sum += value;
    }
    // The islands that are left, in the order of their lowest sheet.
    let mut order: Vec<usize> = (0..islands.len()).filter(|index| gone.get(*index) == Some(&false)).collect();
    order.sort_by_key(|index| islands.get(*index).map_or(usize::MAX, |island| island.first));
    if order.len() > usize::from(u16::MAX) {
        return Err(crate::Error::Limit(format!("a cell holds {} separate pieces of material (use a smaller cell size)", order.len())));
    }
    let rank_of = |key: Option<usize>| {
        let index = islands.iter().position(|island| island.key == key).filter(|index| gone.get(*index) == Some(&false)).unwrap_or(most);
        order.iter().position(|at| *at == index).unwrap_or(0)
    };
    let of_piece: Vec<u16> = if order.len() <= 1 {
        Vec::new()
    } else {
        pieces
            .iter()
            .map(|piece| {
                let root = owned.binary_search(&piece.triangle).ok().and_then(|i| root_of.get(i).copied());
                let key = root.and_then(|root| home.get(root).copied().flatten());
                key.map_or_else(|| rank_of(islands.get(most).and_then(|island| island.key)), rank_of) as u16
            })
            .collect()
    };
    let mut inside = Vec::with_capacity(order.len());
    for index in &order {
        let Some(island) = islands.get_mut(*index) else { continue };
        if volume(&island.tensor) <= 8.0 * SLIVER {
            island.tensor.fill(0.0);
        }
        inside.push(frame.moments(tables, &island.tensor));
    }
    let outside = rest(tables, &inside);
    Ok(Some(CutLeaf { inside, outside, points, of_piece, exact: true }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauss::gauss_legendre;
    use crate::grid::{Grid, feature_flags, sort_into_box};
    use crate::islands;
    use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};

    /// A surface with what the integration of its cells reads.
    struct Body {
        surface: BoundarySurface,
        corners: Vec<[[f64; 3]; 3]>,
        normals: Vec<[f64; 3]>,
        feature: Vec<[bool; 3]>,
        solid: Solid,
    }

    impl Body {
        fn new(points: &[[f64; 3]], triangles: &[[u32; 3]]) -> Body {
            let surface = BoundarySurface::new(points, triangles, &[], &SurfaceOptions::default()).unwrap();
            let corners: Vec<[[f64; 3]; 3]> = surface.triangles().iter().map(|t| t.map(|v| surface.points()[v as usize])).collect();
            let normals = corners
                .iter()
                .map(|t| {
                    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
                    let length = dot(n, n).sqrt();
                    n.map(|x| x / length)
                })
                .collect();
            let feature = feature_flags(&surface);
            let solid = Solid::new(&surface).unwrap();
            Body { surface, corners, normals, feature, solid }
        }

        /// The cell from `lo` of size `h` at degree `p`: by the triangles (`None` when that
        /// leaves it to the quadrature) and by the quadrature, and whether it owns a piece.
        fn cell(&self, lo: [f64; 3], h: f64, p: usize, separate: bool) -> (Option<CutLeaf>, CutLeaf, bool) {
            let surface = Triangles { corners: &self.corners, normals: &self.normals, across: self.surface.neighbors(), feature: &self.feature };
            let hi = [lo[0] + h, lo[1] + h, lo[2] + h];
            let all: Vec<u32> = (0..self.corners.len() as u32).collect();
            let (reach, pieces) = sort_into_box(&self.corners, &all, lo, hi, h);
            let tables = Tables::new(p);
            let rules = [gauss_legendre(p + 1), gauss_legendre(2 * p + 1), gauss_legendre(3 * p + 2)];
            let exact = integrate(&tables, lo, hi, &reach, &pieces, &surface, &self.solid, [&rules[1], &rules[2]], separate).unwrap();
            let rules = [&rules[0][..], &rules[1][..], &rules[2][..]];
            let octree =
                islands::integrate(&tables, lo, hi, h, &reach, &pieces, &surface, &self.solid, rules, 3, 10_000_000, separate, false).unwrap();
            (exact, octree, !pieces.is_empty())
        }
    }

    /// The points and outward triangles of the box from `lo` to `hi`, its points numbered from
    /// `first`.
    fn block(lo: [f64; 3], hi: [f64; 3], first: u32) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let at = |i: usize| [if i & 1 == 0 { lo[0] } else { hi[0] }, if i & 2 == 0 { lo[1] } else { hi[1] }, if i & 4 == 0 { lo[2] } else { hi[2] }];
        let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [1, 3, 7, 5], [0, 4, 6, 2]];
        let triangles = quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).map(|t| t.map(|v| v + first)).collect();
        ((0..8).map(at).collect(), triangles)
    }

    /// A prism of `sides` sides around the z axis from z = 0 to `height`, with a hole of the
    /// radius `bore` through it when that is not zero: the points and outward triangles.
    fn prism(sides: usize, radius: f64, bore: f64, height: f64) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let n = sides as u32;
        let ring = |r: f64, z: f64| {
            (0..sides).map(move |i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / sides as f64;
                [r * angle.cos(), r * angle.sin(), z]
            })
        };
        let mut points: Vec<[f64; 3]> = ring(radius, 0.0).chain(ring(radius, height)).collect();
        let mut triangles = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            triangles.extend([[i, j, n + j], [i, n + j, n + i]]);
        }
        if bore > 0.0 {
            points.extend(ring(bore, 0.0).chain(ring(bore, height)));
            for i in 0..n {
                let j = (i + 1) % n;
                // The hole's wall faces the axis; the ends are rings between the two walls.
                triangles.extend([[2 * n + j, 2 * n + i, 3 * n + i], [2 * n + j, 3 * n + i, 3 * n + j]]);
                triangles.extend([[j, i, 2 * n + i], [j, 2 * n + i, 2 * n + j]]);
                triangles.extend([[n + i, n + j, 3 * n + j], [n + i, 3 * n + j, 3 * n + i]]);
            }
        } else {
            points.extend([[0.0, 0.0, 0.0], [0.0, 0.0, height]]);
            for i in 0..n {
                let j = (i + 1) % n;
                triangles.extend([[2 * n, j, i], [2 * n + 1, n + i, n + j]]);
            }
        }
        (points, triangles)
    }

    fn largest_difference(first: &Moments, second: &Moments) -> f64 {
        first.m.iter().zip(&second.m).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }

    /// Flat faces, which the quadrature integrates exactly as well: the two agree in every
    /// moment, with the cell's highest corner outside the part, inside it, and the part wholly
    /// in the cell.
    #[test]
    fn flat_faces_give_the_moments_of_the_quadrature() {
        let reach = [0.7, 0.6, 0.5];
        let body =
            Body::new(&[[0.0; 3], [reach[0], 0.0, 0.0], [0.0, reach[1], 0.0], [0.0, 0.0, reach[2]]], &[[0, 3, 2], [0, 1, 3], [0, 2, 1], [1, 2, 3]]);
        for p in [1, 2, 3, 4] {
            for (lo, h) in [([-0.1, -0.2, -0.15], 1.0), ([0.1, 0.1, 0.05], 0.3), ([-0.1, -0.1, -0.1], 0.25), ([0.21, -0.03, 0.11], 0.4)] {
                let (exact, octree, _) = body.cell(lo, h, p, true);
                let exact = exact.expect("a cell in no special position");
                assert_eq!(exact.inside.len(), 1);
                let difference = largest_difference(&exact.inside[0], &octree.inside[0]);
                assert!(difference < 1e-12, "degree {p}, cell at {lo:?}: the moments differ by {difference:.2e}");
                assert!(largest_difference(&exact.outside, &octree.outside) < 1e-12);
            }
        }
    }

    /// The same with the axes exchanged, so that the reduction runs along each of them.
    #[test]
    fn the_reduction_runs_along_any_axis() {
        let reach = [0.7, 0.6, 0.5];
        for turn in 0..3 {
            let at = |p: [f64; 3]| [p[turn], p[(turn + 1) % 3], p[(turn + 2) % 3]];
            let points = [[0.0; 3], [reach[0], 0.0, 0.0], [0.0, reach[1], 0.0], [0.0, 0.0, reach[2]]].map(at);
            let body = Body::new(&points, &[[0, 3, 2], [0, 1, 3], [0, 2, 1], [1, 2, 3]]);
            // The last two cells have faces of the part in their lower faces.
            for (lo, h) in [([0.1, 0.1, 0.05], 0.3), ([0.21, -0.03, 0.11], 0.4), ([0.0, 0.0, 0.0], 0.3), ([0.0, -0.1, 0.0], 0.3)] {
                let (exact, octree, _) = body.cell(at(lo), h, 3, true);
                let difference = largest_difference(&exact.expect("a cell in no special position").inside[0], &octree.inside[0]);
                assert!(difference < 1e-12, "turn {turn}, cell at {lo:?}: the moments differ by {difference:.2e}");
            }
        }
    }

    /// A prism of many sides, which the quadrature only approaches: over the cells of a grid the
    /// volume and the second moment are the prism's own to rounding, whatever its triangles'
    /// directions. The regular polygon's ∫ x² is A (R² / 12)(2 + cos(2π/n)), A its area.
    #[test]
    fn a_body_of_many_facets_is_integrated_to_rounding() {
        const SIDES: usize = 48;
        let (radius, height) = (0.8, 0.5);
        let (points, triangles) = prism(SIDES, radius, 0.0, height);
        // Turned out of the grid's axes, so every triangle faces every axis.
        let turned = |p: [f64; 3]| {
            let (c, s) = (0.8, 0.6);
            let q = [p[0], c * p[1] - s * p[2], s * p[1] + c * p[2]];
            [c * q[0] - s * q[1], s * q[0] + c * q[1], q[2]]
        };
        let turn = 2.0 * std::f64::consts::PI / SIDES as f64;
        let area = 0.5 * SIDES as f64 * radius * radius * turn.sin();
        for (tilted, p) in [(false, 2), (true, 2), (true, 3)] {
            let points: Vec<[f64; 3]> = if tilted { points.iter().map(|p| turned(*p)).collect() } else { points.clone() };
            let body = Body::new(&points, &triangles);
            let grid = Grid::over([-1.3; 3], [1.3; 3], 0.45, [0.37, 0.21, 0.43], 1000).unwrap();
            let (mut volume, mut second, mut left) = (0.0, 0.0, 0);
            for cell in 0..grid.len() {
                let (lo, h) = (grid.lo(grid.coords(cell)), grid.spacing);
                let (exact, _, cut) = body.cell(lo, h, p, true);
                let (m0, m1, m2) = match (cut, exact) {
                    (true, Some(leaf)) => (leaf.inside[0].m[0], leaf.inside[0].m[1], leaf.inside[0].m[2]),
                    (true, None) => {
                        left += 1;
                        continue;
                    }
                    (false, _) if body.solid.contains([lo[0] + 0.5 * h, lo[1] + 0.5 * h, lo[2] + 0.5 * h]).unwrap() => (8.0, 0.0, 0.0),
                    _ => continue,
                };
                // x = c + (h/2) ξ, and ξ² = (2 P₂ + 1) / 3.
                let (c, half) = (lo[0] + 0.5 * h, 0.5 * h);
                volume += half.powi(3) * m0;
                second += half.powi(3) * (c * c * m0 + 2.0 * c * half * m1 + half * half * (2.0 * m2 + m0) / 3.0);
            }
            assert_eq!(left, 0, "no cell is left to the quadrature");
            assert!((volume - area * height).abs() < 1e-13, "tilted {tilted}: volume {volume} vs {}", area * height);
            if !tilted {
                let exact = height * area * radius * radius / 12.0 * (2.0 + turn.cos());
                assert!((second - exact).abs() < 1e-13, "second moment {second} vs {exact}");
            }
        }
    }

    /// Two plates in one cell are two islands, each with the moments the quadrature gives it;
    /// a plate alone is one, though its two faces are two sheets of surface.
    #[test]
    fn the_sides_of_a_gap_are_islands() {
        let (mut points, mut triangles) = block([0.1, -1.0, -1.0], [0.3, 2.0, 2.0], 0);
        let (more, faces) = block([0.6, -1.0, -1.0], [0.8, 2.0, 2.0], 8);
        let plate = Body::new(&points, &triangles);
        points.extend(more);
        triangles.extend(faces);
        let plates = Body::new(&points, &triangles);
        for p in [2, 3] {
            for lo in [[0.013, 0.021, 0.017], [-0.21, 0.3, -0.4]] {
                let (exact, octree, _) = plates.cell(lo, 1.0, p, true);
                let exact = exact.expect("a cell in no special position");
                assert_eq!((exact.inside.len(), octree.inside.len()), (2, 2));
                assert_eq!(exact.of_piece, octree.of_piece);
                for (first, second) in exact.inside.iter().zip(&octree.inside) {
                    assert!(largest_difference(first, second) < 1e-12);
                }
                // Not told apart, they are one.
                let (joined, octree, _) = plates.cell(lo, 1.0, p, false);
                assert!(largest_difference(&joined.unwrap().inside[0], &octree.inside[0]) < 1e-12);
                let (exact, octree, _) = plate.cell(lo, 1.0, p, true);
                let exact = exact.unwrap();
                assert_eq!(exact.inside.len(), 1);
                assert!(exact.of_piece.is_empty());
                assert!(largest_difference(&exact.inside[0], &octree.inside[0]) < 1e-12);
            }
        }
    }

    /// A tube through a cell: its outer wall and the wall of its hole are two sheets and one
    /// island, the hole's volume taken out of the outer wall's, wherever the cell's highest
    /// corner is (outside the tube with all of it in the cell, in its wall, in its hole).
    #[test]
    fn a_hollow_is_taken_out_of_the_island_around_it() {
        const SIDES: usize = 40;
        let (radius, bore) = (0.4, 0.25);
        let (points, triangles) = prism(SIDES, radius, bore, 3.0);
        let body = Body::new(&points.iter().map(|p| [p[0], p[1], p[2] - 1.0]).collect::<Vec<_>>(), &triangles);
        let polygon = |r: f64| 0.5 * SIDES as f64 * r * r * (2.0 * std::f64::consts::PI / SIDES as f64).sin();
        for (lo, h, whole) in [([-0.52, -0.47, 0.1], 1.0, true), ([-0.67, -0.69, 0.1], 1.0, false), ([-0.93, -0.91, 0.1], 1.0, false)] {
            let (exact, octree, _) = body.cell(lo, h, 3, true);
            let exact = exact.expect("a cell in no special position");
            assert_eq!((exact.inside.len(), octree.inside.len()), (1, 1), "cell at {lo:?}");
            let volume = 0.125 * h * h * h * exact.inside[0].m[0];
            if whole {
                assert!((volume - (polygon(radius) - polygon(bore)) * h).abs() < 1e-13, "cell at {lo:?}: {volume}");
            }
            // The octree's quadrature approaches the same moments.
            let difference = largest_difference(&exact.inside[0], &octree.inside[0]);
            assert!(volume > 0.02 && difference < 2e-4, "cell at {lo:?}: volume {volume}, the moments differ by {difference:.2e}");
        }
    }

    /// A face of the part in the plane of the cell's highest corner is left to the quadrature.
    #[test]
    fn a_corner_on_the_surface_is_left_to_the_quadrature() {
        let (points, triangles) = block([0.0; 3], [1.0; 3], 0);
        let body = Body::new(&points, &triangles);
        assert!(body.cell([0.5; 3], 0.5, 2, true).0.is_none());
        // The lowest corner on the surface is no hindrance.
        let (exact, octree, _) = body.cell([0.0; 3], 0.5, 2, true);
        assert!(largest_difference(&exact.unwrap().inside[0], &octree.inside[0]) < 1e-12);
    }
}
