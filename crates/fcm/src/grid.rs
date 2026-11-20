//! The background grid: which cells the surface cuts, the pieces of the surface in each cell and
//! the quadrature points of a cut cell.
//!
//! # Surface pieces
//! Shape functions are polynomials on one cell, so a surface integral is taken cell by cell. Each
//! triangle is clipped against the box of every cell it reaches (I. E. Sutherland, G. W. Hodgman,
//! *Reentrant polygon clipping*, Commun. ACM 17(1):32–42, 1974) and the polygon left is split
//! into a fan of triangles. A piece that lies in the plane between two cells would be found in
//! both; it belongs to the cell on the material side, which the triangle's outward normal tells.
//! Vertices within [`SNAP`] of a cell size from a grid plane are moved onto it first, so a face
//! of the part that lies on a grid plane lies on it exactly and is never split between the two
//! cells by rounding.
//!
//! # Cut cells
//! A cell that owns a piece is cut. Any other cell lies wholly on one side of the surface (the
//! surface reaches at most its boundary), which its centre decides ([`crate::inside`]).
//!
//! # Quadrature of a cut cell
//! The integrand of a cut cell jumps at the surface, which Gauss quadrature over the whole cell
//! would integrate badly. The cell is divided into an octree instead (Düster et al. 2008):
//! a box the surface reaches is split into eight, down to a depth limit, and every leaf gets the
//! Gauss rule of the cell's degree. A leaf the surface does not reach is wholly inside or
//! outside by its centre. Whether a triangle reaches a box is the separating axis test for a
//! triangle and a box (T. Akenine-Möller, *Fast 3D triangle-box overlap testing*, J. Graphics
//! Tools 6(1):29–33, 2001) in floating point, with the box grown slightly: it decides only where
//! the octree refines, never which side a point is on.
//!
//! In a leaf the surface does reach, testing each Gauss point on its own would be first order in
//! the leaf size: the jump lies somewhere between two points. The leaf is integrated one axis at
//! a time instead (Fubini's theorem), each axis split where the integrand is not smooth:
//! - along a line, where it pierces a triangle, so every segment lies on one side of the surface.
//!   The segment's ends come from the triangles in floating point; which side it is on is the
//!   exact inside test of its midpoint;
//! - across the lines, the integral along a line is smooth except where a face ends or turns: at
//!   a feature edge of the surface (the boundary of a B-rep face, or a crease inside one), at an
//!   edge where the surface turns back along the lines, and where the surface leaves the leaf
//!   through one of the faces the lines end on. The second axis is split where such an edge
//!   crosses, and the third at the ends of those edges.
//!
//! Every interval gets the Gauss rule of the cell's degree. The lines run along the axis the
//! leaf's triangles face most. For a part with flat faces this gives the volume exactly (up to
//! sixteen intervals an axis of a leaf; beyond that, the closest split points are merged). The
//! edges between the facets of a smooth curved face are not split at, or the points would grow
//! with the square of the tessellation's fineness: the integral along a line differs from that
//! of the smooth face by at most the chordal tolerance, so what the rule misses there is of the
//! order of the tessellation's own error, and it falls with the leaf size.

use std::collections::BTreeSet;

use fieldcraft_mesh::surface::BoundarySurface;

use crate::inside::Solid;
use crate::{Error, Result};

/// Vertices this close to a grid plane, as a fraction of the cell size, are moved onto it.
pub const SNAP: f64 = 1e-9; // unit-factors: relative to the cell size
/// How much a box is grown for the overlap test, as a fraction of its size.
const GROW: f64 = 1e-9; // unit-factors: relative to the box
/// The deepest octree under a cut cell.
pub const MAX_DEPTH: usize = 8;

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A uniform grid of cubic cells: cell `(i, j, k)` spans `origin + (i, j, k) · spacing` to
/// `origin + (i + 1, j + 1, k + 1) · spacing` and has index `i + nx (j + ny k)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub origin: [f64; 3],
    pub spacing: f64,
    pub cells: [usize; 3],
}

impl Grid {
    /// The grid of cells of size `spacing` over the box `lo`–`hi`, starting `offset` cells
    /// (each component in `[0, 1)`) before `lo`.
    pub fn over(lo: [f64; 3], hi: [f64; 3], spacing: f64, offset: [f64; 3], max_cells: usize) -> Result<Grid> {
        if !(spacing > 0.0 && spacing.is_finite()) {
            return Err(Error::Invalid(format!("cell size {spacing} is not a positive length")));
        }
        let mut grid = Grid { origin: [0.0; 3], spacing, cells: [1; 3] };
        let mut total = 1usize;
        for a in 0..3 {
            if !(lo[a].is_finite() && hi[a].is_finite() && hi[a] >= lo[a]) {
                return Err(Error::Invalid("the surface has no finite bounding box".into()));
            }
            if !(offset[a] >= 0.0 && offset[a] < 1.0) {
                return Err(Error::Invalid(format!("grid offset {} is not in [0, 1)", offset[a])));
            }
            let origin = lo[a] - offset[a] * spacing;
            if !(origin + spacing > origin) {
                return Err(Error::Invalid(format!("cell size {spacing} is below the resolution of the coordinates")));
            }
            let span = (hi[a] - origin) / spacing;
            if !(span.is_finite() && span < max_cells as f64) {
                return Err(Error::Limit(format!("the grid would have more than {max_cells} cells (use a larger cell size)")));
            }
            let mut n = (span.ceil() as usize).max(1);
            // The last plane may fall short of the box by rounding.
            for _ in 0..2 {
                if origin + n as f64 * spacing < hi[a] {
                    n += 1;
                }
            }
            grid.origin[a] = origin;
            grid.cells[a] = n;
            total = total.saturating_mul(n);
        }
        if total > max_cells {
            return Err(Error::Limit(format!("the grid would have {total} cells; the limit is {max_cells} (use a larger cell size)")));
        }
        Ok(grid)
    }

    /// The number of cells.
    pub fn len(&self) -> usize {
        self.cells[0] * self.cells[1] * self.cells[2]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The index of cell `(i, j, k)`.
    pub fn index(&self, [i, j, k]: [usize; 3]) -> usize {
        i + self.cells[0] * (j + self.cells[1] * k)
    }

    /// The `(i, j, k)` of a cell index.
    pub fn coords(&self, c: usize) -> [usize; 3] {
        let (nx, ny) = (self.cells[0].max(1), self.cells[1].max(1));
        [c % nx, (c / nx) % ny, c / (nx * ny)]
    }

    /// The coordinate of grid plane `i` along `axis`.
    pub fn plane(&self, axis: usize, i: usize) -> f64 {
        self.origin[axis] + i as f64 * self.spacing
    }

    /// The lowest corner of a cell.
    pub fn lo(&self, [i, j, k]: [usize; 3]) -> [f64; 3] {
        [self.plane(0, i), self.plane(1, j), self.plane(2, k)]
    }

    /// The highest corner of a cell.
    pub fn hi(&self, [i, j, k]: [usize; 3]) -> [f64; 3] {
        [self.plane(0, i + 1), self.plane(1, j + 1), self.plane(2, k + 1)]
    }

    /// `p` with every coordinate within [`SNAP`] cells of a grid plane moved onto that plane.
    pub fn snap(&self, p: [f64; 3]) -> [f64; 3] {
        let mut q = p;
        for a in 0..3 {
            let t = (p[a] - self.origin[a]) / self.spacing;
            let r = t.round();
            if (t - r).abs() < SNAP && r >= 0.0 && r <= self.cells[a] as f64 {
                q[a] = self.plane(a, r as usize);
            }
        }
        q
    }

    /// The cells along `axis` a coordinate range can reach, widened by one on each side.
    fn range(&self, axis: usize, lo: f64, hi: f64) -> (usize, usize) {
        let n = self.cells[axis].max(1);
        let clamp = |t: f64| if !(t > 0.0) { 0 } else { (t as usize).min(n - 1) };
        (clamp(((lo - self.origin[axis]) / self.spacing).floor() - 1.0), clamp(((hi - self.origin[axis]) / self.spacing).floor() + 1.0))
    }
}

/// Whether a triangle and the box `centre ± half` overlap (touching counts).
pub fn overlaps(triangle: &[[f64; 3]; 3], centre: [f64; 3], half: [f64; 3]) -> bool {
    let v = [sub(triangle[0], centre), sub(triangle[1], centre), sub(triangle[2], centre)];
    // The box's face normals.
    for a in 0..3 {
        let (lo, hi) = (v[0][a].min(v[1][a]).min(v[2][a]), v[0][a].max(v[1][a]).max(v[2][a]));
        if lo > half[a] || hi < -half[a] {
            return false;
        }
    }
    // The triangle's normal.
    let edges = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];
    let normal = cross(edges[0], edges[1]);
    let reach = half[0] * normal[0].abs() + half[1] * normal[1].abs() + half[2] * normal[2].abs();
    if dot(normal, v[0]).abs() > reach {
        return false;
    }
    // The nine cross products of a box axis and a triangle edge.
    for edge in edges {
        for a in 0..3 {
            let (b, c) = ((a + 1) % 3, (a + 2) % 3);
            let mut axis = [0.0; 3];
            axis[b] = -edge[c];
            axis[c] = edge[b];
            let p = [dot(axis, v[0]), dot(axis, v[1]), dot(axis, v[2])];
            let reach = half[b] * axis[b].abs() + half[c] * axis[c].abs();
            if p[0].min(p[1]).min(p[2]) > reach || p[0].max(p[1]).max(p[2]) < -reach {
                return false;
            }
        }
    }
    true
}

/// Where an edge of a clipped triangle comes from: 0, 1 or 2 for the triangle's own edge from
/// its first to its second corner, its second to its third, or its third to its first; from
/// `CUT` on, the cut by a face of the box (`CUT + 2 axis`, plus one for the upper face).
const CUT: u8 = 3;

/// The part of a triangle inside the box `lo`–`hi`: a convex polygon (empty when they do not
/// overlap), each corner with where the edge that leaves it comes from. A corner made by a cut
/// has the cut plane's coordinate exactly.
fn clip(triangle: &[[f64; 3]; 3], lo: [f64; 3], hi: [f64; 3]) -> Vec<([f64; 3], u8)> {
    let mut polygon: Vec<([f64; 3], u8)> = vec![(triangle[0], 0), (triangle[1], 1), (triangle[2], 2)];
    for a in 0..3 {
        for (side, (bound, below)) in [(lo[a], false), (hi[a], true)].into_iter().enumerate() {
            let keeps = |p: &[f64; 3]| if below { p[a] <= bound } else { p[a] >= bound };
            let mut kept = Vec::with_capacity(polygon.len() + 2);
            for (i, (p, origin)) in polygon.iter().enumerate() {
                let Some((q, _)) = polygon.get((i + 1) % polygon.len()) else { continue };
                let (p_in, q_in) = (keeps(p), keeps(q));
                if p_in {
                    kept.push((*p, *origin));
                }
                if p_in != q_in {
                    // The two ends lie on different sides, so their coordinates differ.
                    let s = (bound - p[a]) / (q[a] - p[a]);
                    let mut x = [p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1]), p[2] + s * (q[2] - p[2])];
                    x[a] = bound;
                    // Leaving the box, the polygon goes on along the cut to where it comes back
                    // in; coming in, along the edge it was on.
                    kept.push((x, if p_in { CUT + 2 * a as u8 + side as u8 } else { *origin }));
                }
            }
            polygon = kept;
            if polygon.len() < 3 {
                return Vec::new();
            }
        }
    }
    polygon
}

/// A triangle of the surface inside one cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Piece {
    pub corners: [[f64; 3]; 3],
    /// The surface triangle it is a part of.
    pub triangle: u32,
}

/// The surface sorted into the grid's cells.
#[derive(Debug, Clone, Default)]
pub struct Sorted {
    /// For each cell, the surface triangles that reach its box (its boundary included).
    pub touching: Vec<Vec<u32>>,
    /// For each cell, the pieces of the surface it owns.
    pub pieces: Vec<Vec<Piece>>,
}

/// Sort triangles (already snapped to the grid, [`Grid::snap`]) into the grid's cells.
pub fn sort_surface(grid: &Grid, triangles: &[[[f64; 3]; 3]]) -> Sorted {
    let mut sorted = Sorted { touching: vec![Vec::new(); grid.len()], pieces: vec![Vec::new(); grid.len()] };
    let half = [0.5 * grid.spacing * (1.0 + GROW); 3];
    for (t, triangle) in triangles.iter().enumerate() {
        let normal = cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]));
        let mut ranges = [(0usize, 0usize); 3];
        for a in 0..3 {
            let (lo, hi) = (triangle[0][a].min(triangle[1][a]).min(triangle[2][a]), triangle[0][a].max(triangle[1][a]).max(triangle[2][a]));
            ranges[a] = grid.range(a, lo, hi);
        }
        for k in ranges[2].0..=ranges[2].1 {
            for j in ranges[1].0..=ranges[1].1 {
                for i in ranges[0].0..=ranges[0].1 {
                    let ijk = [i, j, k];
                    let (lo, hi) = (grid.lo(ijk), grid.hi(ijk));
                    let centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
                    if !overlaps(triangle, centre, half) {
                        continue;
                    }
                    let cell = grid.index(ijk);
                    if let Some(list) = sorted.touching.get_mut(cell) {
                        list.push(t as u32);
                    }
                    let polygon: Vec<[f64; 3]> = clip(triangle, lo, hi).into_iter().map(|(corner, _)| corner).collect();
                    // A polygon in the plane between two cells belongs to the one the material
                    // is in: the one the normal points away from.
                    let foreign = (0..3).any(|a| {
                        (polygon.iter().all(|p| p[a] == lo[a]) && !(normal[a] < 0.0)) || (polygon.iter().all(|p| p[a] == hi[a]) && !(normal[a] > 0.0))
                    });
                    if polygon.len() < 3 || foreign {
                        continue;
                    }
                    let Some(pieces) = sorted.pieces.get_mut(cell) else { continue };
                    for pair in polygon.get(1..).unwrap_or_default().windows(2) {
                        let corners = [polygon[0], pair[0], pair[1]];
                        let area = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
                        if dot(area, area) > 0.0 {
                            pieces.push(Piece { corners, triangle: t as u32 });
                        }
                    }
                }
            }
        }
    }
    sorted
}

/// A quadrature point of a cell: natural coordinates in [−1, 1]³, its weight there (the weights
/// of a cell sum to 8) and whether it lies inside the part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadPoint {
    pub xi: [f64; 3],
    pub weight: f64,
    pub inside: bool,
}

/// The surface as the quadrature of cut cells reads it, triangle by triangle.
#[derive(Debug, Clone, Copy)]
pub struct Triangles<'a> {
    /// The corners, snapped to the grid ([`Grid::snap`]).
    pub corners: &'a [[[f64; 3]; 3]],
    /// The outward unit normals.
    pub normals: &'a [[f64; 3]],
    /// The triangles across the edges from the first to the second corner, the second to the
    /// third and the third to the first ([`BoundarySurface::neighbors`]).
    pub across: &'a [[u32; 3]],
    /// Which of those edges are feature edges ([`feature_flags`]).
    pub feature: &'a [[bool; 3]],
}

/// For each triangle of a surface, which of its edges (from the first to the second corner, the
/// second to the third, the third to the first) are feature edges: the boundary of a B-rep face,
/// or a crease inside one.
pub fn feature_flags(surface: &BoundarySurface) -> Vec<[bool; 3]> {
    let features: BTreeSet<[u32; 2]> = surface.feature_edges().iter().map(|edge| edge.vertices).collect();
    surface.triangles().iter().map(|t| [0, 1, 2].map(|k| features.contains(&[t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3])]))).collect()
}

/// The quadrature points of the cell at `lo` with size `h`: an octree of at most `depth` levels
/// refined where the triangles `touching` (indices into the surface's) reach, with the Gauss
/// `rule` in every leaf.
pub fn cut_quadrature(
    lo: [f64; 3],
    h: f64,
    touching: &[u32],
    surface: &Triangles,
    solid: &Solid,
    rule: &[(f64, f64)],
    depth: usize,
    max_points: usize,
) -> Result<Vec<QuadPoint>> {
    let depth = depth.min(MAX_DEPTH);
    let triangles = surface.corners;
    let physical = |xi: [f64; 3]| [lo[0] + 0.5 * (xi[0] + 1.0) * h, lo[1] + 0.5 * (xi[1] + 1.0) * h, lo[2] + 0.5 * (xi[2] + 1.0) * h];
    let per_leaf = rule.len() * rule.len() * rule.len();
    let mut points = Vec::new();
    // Boxes still to do: the lowest corner in natural coordinates, the level and the triangles
    // that reach the parent. Children are pushed last to first, so they are done in order.
    let mut stack: Vec<([f64; 3], usize, Vec<u32>)> = vec![([-1.0; 3], 0, touching.to_vec())];
    while let Some((corner, level, candidates)) = stack.pop() {
        let size = 2.0 / (1u64 << level) as f64;
        let half = 0.5 * size;
        let centre = physical([corner[0] + half, corner[1] + half, corner[2] + half]);
        let reach = [0.5 * half * h * (1.0 + GROW); 3];
        let cut: Vec<u32> =
            candidates.into_iter().filter(|t| triangles.get(*t as usize).is_some_and(|triangle| overlaps(triangle, centre, reach))).collect();
        if !cut.is_empty() && level < depth {
            for child in (0..8usize).rev() {
                let shift = |bit: usize| if (child >> bit) & 1 == 1 { half } else { 0.0 };
                stack.push(([corner[0] + shift(0), corner[1] + shift(1), corner[2] + shift(2)], level + 1, cut.clone()));
            }
            continue;
        }
        let full = || Error::Limit(format!("a cut cell needs more than {max_points} quadrature points (lower the octree depth or the degree)"));
        if cut.is_empty() {
            // The surface does not reach the leaf: one side, the Gauss rule of the box.
            if points.len() + per_leaf > max_points {
                return Err(full());
            }
            let inside = solid.contains(centre)?;
            for (gz, wz) in rule {
                for (gy, wy) in rule {
                    for (gx, wx) in rule {
                        let xi = [corner[0] + half * (gx + 1.0), corner[1] + half * (gy + 1.0), corner[2] + half * (gz + 1.0)];
                        points.push(QuadPoint { xi, weight: wx * wy * wz * half * half * half, inside });
                    }
                }
            }
            continue;
        }
        // The lines run along the axis the leaf's triangles face most (by area).
        let mut facing = [0.0; 3];
        for t in &cut {
            let Some(triangle) = triangles.get(*t as usize) else { continue };
            let normal = cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]));
            for a in 0..3 {
                facing[a] += normal[a].abs();
            }
        }
        let axis = if facing[0] >= facing[1] && facing[0] >= facing[2] {
            0
        } else if facing[1] >= facing[2] {
            1
        } else {
            2
        };
        let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
        // The leaf's box, and the edges of the triangles inside it, seen along the lines.
        let leaf_lo = physical(corner);
        let leaf_hi = physical([corner[0] + size, corner[1] + size, corner[2] + size]);
        let tiny = MERGE * (leaf_hi[axis] - leaf_lo[axis]);
        // The edges where the integral along a line is not smooth from line to line: a feature
        // edge, an edge where the surface turns back along the lines (its two triangles face
        // opposite ways along them), and the cut where the surface leaves the leaf through one
        // of the faces the lines end on.
        let facing = |normal: &[f64; 3]| {
            if normal[axis] > 0.0 {
                1
            } else if normal[axis] < 0.0 {
                -1
            } else {
                0
            }
        };
        let mut edges: Vec<[[f64; 2]; 2]> = Vec::new();
        for t in &cut {
            let index = *t as usize;
            let Some(triangle) = triangles.get(index) else { continue };
            let own = surface.normals.get(index).map_or(0, facing);
            let polygon = clip(triangle, leaf_lo, leaf_hi);
            for (i, (from, origin)) in polygon.iter().enumerate() {
                let Some((to, _)) = polygon.get((i + 1) % polygon.len()) else { continue };
                let splits = match *origin as usize {
                    edge @ 0..=2 => {
                        surface.feature.get(index).is_some_and(|flags| flags[edge])
                            || surface
                                .across
                                .get(index)
                                .and_then(|across| surface.normals.get(across[edge] as usize))
                                .is_none_or(|other| facing(other) != own)
                    }
                    cut => (cut - CUT as usize) / 2 == axis,
                };
                if splits {
                    edges.push([[from[b], from[c]], [to[b], to[c]]]);
                }
            }
        }
        // Across the lines, the integral along a line is smooth between the edges: split the
        // third axis at the edges' ends, then the second where an edge crosses.
        let mut third = vec![leaf_lo[c], leaf_hi[c]];
        third.extend(edges.iter().flatten().map(|end| end[1]).filter(|at| *at > leaf_lo[c] && *at < leaf_hi[c]));
        thin(&mut third, tiny);
        let (mut second, mut first): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        for span_c in third.windows(2) {
            for (gc, wc) in rule {
                let at_c = 0.5 * (span_c[0] + span_c[1]) + 0.5 * (span_c[1] - span_c[0]) * gc;
                let weight_c = wc * (span_c[1] - span_c[0]) / h;
                second.clear();
                second.extend([leaf_lo[b], leaf_hi[b]]);
                for [[b0, c0], [b1, c1]] in &edges {
                    if c0 != c1 && (at_c - c0) * (at_c - c1) <= 0.0 {
                        let at = b0 + (at_c - c0) / (c1 - c0) * (b1 - b0);
                        if at > leaf_lo[b] && at < leaf_hi[b] {
                            second.push(at);
                        }
                    }
                }
                thin(&mut second, tiny);
                for span_b in second.windows(2) {
                    for (gb, wb) in rule {
                        let at_b = 0.5 * (span_b[0] + span_b[1]) + 0.5 * (span_b[1] - span_b[0]) * gb;
                        let weight_b = wb * (span_b[1] - span_b[0]) / h;
                        // Along the line, split where it pierces a triangle.
                        first.clear();
                        first.extend([leaf_lo[axis], leaf_hi[axis]]);
                        for t in &cut {
                            if let Some(at) = triangles.get(*t as usize).and_then(|triangle| pierce(triangle, axis, at_b, at_c))
                                && at > leaf_lo[axis]
                                && at < leaf_hi[axis]
                            {
                                first.push(at);
                            }
                        }
                        tidy(&mut first, tiny);
                        if points.len() + first.len() * rule.len() > max_points {
                            return Err(full());
                        }
                        let mut x = [0.0; 3];
                        x[b] = at_b;
                        x[c] = at_c;
                        for span in first.windows(2) {
                            x[axis] = 0.5 * (span[0] + span[1]);
                            let inside = solid.contains(x)?;
                            for (g, w) in rule {
                                let mut at = x;
                                at[axis] += 0.5 * (span[1] - span[0]) * g;
                                let xi = [2.0 * (at[0] - lo[0]) / h - 1.0, 2.0 * (at[1] - lo[1]) / h - 1.0, 2.0 * (at[2] - lo[2]) / h - 1.0];
                                points.push(QuadPoint { xi, weight: weight_b * weight_c * w * (span[1] - span[0]) / h, inside });
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(points)
}

/// Split points closer than this fraction of the leaf are one.
const MERGE: f64 = 1e-12; // unit-factors: relative to the leaf

/// Sort the split points of an interval, whose two ends are among them, and drop every one
/// within `tiny` of the one before it or of the upper end: what is left are the ends of
/// intervals of positive length (a single point when the interval has none).
fn tidy(cuts: &mut Vec<f64>, tiny: f64) {
    cuts.sort_by(f64::total_cmp);
    let (Some(&lower), Some(&upper)) = (cuts.first(), cuts.last()) else { return };
    let mut kept = vec![lower];
    let mut last = lower;
    for &at in cuts.iter() {
        if at - last > tiny && upper - at > tiny {
            kept.push(at);
            last = at;
        }
    }
    if upper > lower {
        kept.push(upper);
    }
    *cuts = kept;
}

/// Intervals an axis of a leaf is split into across the lines, at most.
const MOST: usize = 16;

/// [`tidy`], then no more than [`MOST`] intervals: with more, split points closer than one
/// [`MOST`]-th of the interval are one. Few split points are kept as they are, so a thin wall or
/// a sliver is integrated exactly. Many of them are the facets of a curved face seen edge-on
/// (a hole along the lines, tessellated in triangles that tilt either way), whose band is as
/// narrow as the chordal tolerance; thinning them bounds the points of a leaf whatever the
/// tessellation's fineness, and what it costs falls with the leaf size.
fn thin(cuts: &mut Vec<f64>, tiny: f64) {
    tidy(cuts, tiny);
    if cuts.len() > MOST + 1
        && let (Some(&lower), Some(&upper)) = (cuts.first(), cuts.last())
    {
        tidy(cuts, (upper - lower) / MOST as f64);
    }
}

/// Lines through a triangle within this much of its edges, in barycentric coordinates, pierce
/// it: a line along an edge two triangles share must not slip between them by rounding. A
/// crossing found twice only makes a segment of no length.
const ON_EDGE: f64 = 1e-9; // unit-factors: barycentric, relative to the triangle

/// Where the line along `axis` through the other two coordinates `(ub, uc)` (those of the next
/// two axes in cyclic order) pierces a triangle: the coordinate along `axis`. `None` when the
/// line misses the triangle or runs in its plane.
fn pierce(triangle: &[[f64; 3]; 3], axis: usize, ub: f64, uc: f64) -> Option<f64> {
    let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
    let [p, q, r] = triangle;
    let (qb, qc, rb, rc) = (q[b] - p[b], q[c] - p[c], r[b] - p[b], r[c] - p[c]);
    let doubled = qb * rc - rb * qc;
    if doubled == 0.0 || !doubled.is_finite() {
        return None;
    }
    let (db, dc) = (ub - p[b], uc - p[c]);
    let at_q = (db * rc - rb * dc) / doubled;
    let at_r = (qb * dc - db * qc) / doubled;
    let at_p = 1.0 - at_q - at_r;
    (at_p >= -ON_EDGE && at_q >= -ON_EDGE && at_r >= -ON_EDGE).then(|| at_p * p[axis] + at_q * q[axis] + at_r * r[axis])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_covers_its_box() {
        let grid = Grid::over([0.0, -1.0, 2.0], [1.0, 1.0, 2.5], 0.25, [0.0, 0.5, 0.25], 1000).unwrap();
        assert_eq!(grid.cells, [4, 9, 3]);
        assert_eq!(grid.origin, [0.0, -1.125, 1.9375]);
        assert_eq!(grid.coords(grid.index([3, 8, 2])), [3, 8, 2]);
        assert_eq!(grid.snap([0.25 + 1e-12, 0.3, 2.5]), [0.25, 0.3, 2.5]);
        assert!(matches!(Grid::over([0.0; 3], [1.0; 3], 0.001, [0.0; 3], 1000), Err(Error::Limit(_))));
        assert!(matches!(Grid::over([0.0; 3], [1.0; 3], f64::NAN, [0.0; 3], 1000), Err(Error::Invalid(_))));
        assert!(matches!(Grid::over([0.0; 3], [1.0; 3], 0.5, [1.0, 0.0, 0.0], 1000), Err(Error::Invalid(_))));
        assert!(matches!(Grid::over([1e20; 3], [1e20; 3], 1.0, [0.0; 3], 1000), Err(Error::Invalid(_))));
    }

    #[test]
    fn triangle_box_overlap() {
        let (centre, half) = ([0.0; 3], [1.0; 3]);
        // Through the box, touching a face, beside it, and past a corner on a diagonal plane.
        assert!(overlaps(&[[-2.0, 0.0, 0.0], [2.0, 0.1, 0.0], [0.0, 2.0, 0.3]], centre, half));
        assert!(overlaps(&[[1.0, -0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.0, 0.5]], centre, half));
        assert!(!overlaps(&[[1.1, -0.5, -0.5], [1.1, 0.5, -0.5], [1.1, 0.0, 0.5]], centre, half));
        assert!(!overlaps(&[[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]], centre, half));
        assert!(overlaps(&[[2.9, 0.0, 0.0], [0.0, 2.9, 0.0], [0.0, 0.0, 2.9]], centre, half));
    }

    #[test]
    fn pieces_keep_the_area_and_a_plane_between_cells_goes_to_one_of_them() {
        let grid = Grid::over([0.0; 3], [2.0, 2.0, 1.0], 1.0, [0.0; 3], 100).unwrap();
        let area = |pieces: &[Piece]| -> f64 {
            pieces.iter().map(|p| 0.5 * dot(cross(sub(p.corners[1], p.corners[0]), sub(p.corners[2], p.corners[0])), [1.0, 0.0, 0.0]).abs()).sum()
        };
        // A triangle in the plane x = 1 between cells, with its normal along +x: the material
        // is at lower x.
        let triangle = [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 0.0, 1.0]];
        let sorted = sort_surface(&grid, &[triangle]);
        let owned: Vec<f64> = sorted.pieces.iter().map(|p| area(p)).collect();
        assert!((owned.iter().sum::<f64>() - 1.0).abs() < 1e-14, "{owned:?}");
        assert!(owned[grid.index([0, 0, 0])] > 0.0 && owned[grid.index([0, 1, 0])] > 0.0, "{owned:?}");
        assert_eq!(owned[grid.index([1, 0, 0])], 0.0);
        assert_eq!(owned[grid.index([1, 1, 0])], 0.0);
        assert!(!sorted.touching[grid.index([1, 0, 0])].is_empty(), "the other cell is still reached");
        // Turned over, the other two cells own it.
        let turned = [triangle[0], triangle[2], triangle[1]];
        let owned: Vec<f64> = sort_surface(&grid, &[turned]).pieces.iter().map(|p| area(p)).collect();
        assert!((owned.iter().sum::<f64>() - 1.0).abs() < 1e-14, "{owned:?}");
        assert_eq!(owned[grid.index([0, 0, 0])], 0.0);
    }
}
