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
//! would integrate badly. A box the surface reaches is integrated one axis at a time instead
//! (Fubini's theorem), each axis split where the integrand is not smooth:
//! - along a line, where it pierces a triangle, so every segment lies on one side of the surface.
//!   The segment's ends come from the triangles in floating point; which side it is on is the
//!   exact inside test of its midpoint;
//! - across the lines, the integral along a line is smooth except where a face ends or turns: at
//!   a feature edge of the surface (the boundary of a B-rep face, or a crease inside one), at an
//!   edge where the surface turns back along the lines, and where the surface leaves the box
//!   through one of the faces the lines end on. The second axis is split where such an edge
//!   crosses, and the third at the ends of those edges and where two of them cross.
//!
//! The lines run along the axis the box's triangles face most.
//!
//! **Flat faces are integrated exactly.** The integrand of a cell matrix is a polynomial of
//! degree 2p along each axis. Along a line, a Gauss rule of p + 1 points integrates it exactly.
//! Where the surface is flat, the end of a line's segment is a linear function of the other two
//! coordinates, so the integral along the line is a polynomial of degree 4p + 1 in the second
//! coordinate; between two edges, which are straight, its integral is in turn a polynomial of
//! degree 6p + 2 in the third. Rules of 2p + 1 and 3p + 2 points across the lines therefore
//! integrate a cell cut by flat faces exactly, whatever their directions (up to sixteen
//! intervals an axis of a box; beyond that, the closest split points are merged). With p + 1
//! points across the lines only the volume and faces along the axes would be exact, and the
//! matrices of the higher modes of a cell cut at an angle would not.
//!
//! **Curved faces need the octree.** The edges between the facets of a smooth curved face are
//! not split at, or the points would grow with the square of the tessellation's fineness. The
//! integral across the lines is then smooth but no polynomial, and what the rules miss falls
//! with the size of the box. So a box in which the surface is curved (two of its triangles that
//! meet in an edge that is no feature edge face different ways) is divided into eight, down to
//! a depth limit (the octree of Düster et al. 2008, refined only where it is needed); a box of
//! flat faces is not divided. A box the surface does not reach is wholly inside or outside by
//! its centre and gets the Gauss rule of the cell's degree. Whether a triangle reaches a box is
//! the separating axis test for a triangle and a box (T. Akenine-Möller, *Fast 3D triangle-box
//! overlap testing*, J. Graphics Tools 6(1):29–33, 2001) in floating point, with the box grown
//! slightly: it decides only where the octree refines, never which side a point is on. With too
//! shallow an octree the matrices of a cell cut by a curved face are inexact, the more so the
//! higher the degree, and the stress there moves erratically from degree to degree.

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
/// The most levels of refinement under a cell of the grid (`Options::refine`): the finest cells
/// are a 64th of the grid's.
pub const MAX_REFINE: usize = 6;

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
        self.snap_to(p, 0)
    }

    /// The cells of level 0 are the grid's; a cell of level l + 1 is an eighth of one of level
    /// l. This is the size of a cell of `level`.
    pub fn step(&self, level: usize) -> f64 {
        self.spacing / (1u64 << level.min(MAX_REFINE)) as f64
    }

    /// The coordinate of plane `i` of the cells of `level` along `axis`. The steps are halves,
    /// so a plane that several levels share has the same coordinate on each of them, bit for
    /// bit, and plane `i` of level 0 is the grid's.
    pub fn fine_plane(&self, axis: usize, level: usize, i: usize) -> f64 {
        self.origin[axis] + i as f64 / (1u64 << level.min(MAX_REFINE)) as f64 * self.spacing
    }

    /// The lowest corner of cell `(i, j, k)` of `level`.
    pub fn fine_lo(&self, level: usize, [i, j, k]: [usize; 3]) -> [f64; 3] {
        [self.fine_plane(0, level, i), self.fine_plane(1, level, j), self.fine_plane(2, level, k)]
    }

    /// The highest corner of cell `(i, j, k)` of `level`.
    pub fn fine_hi(&self, level: usize, [i, j, k]: [usize; 3]) -> [f64; 3] {
        [self.fine_plane(0, level, i + 1), self.fine_plane(1, level, j + 1), self.fine_plane(2, level, k + 1)]
    }

    /// `p` with every coordinate within [`SNAP`] cells of `level` of a plane of that level moved
    /// onto the plane ([`Grid::snap`] for the cells that are refined so far).
    pub fn snap_to(&self, p: [f64; 3], level: usize) -> [f64; 3] {
        let steps = (1u64 << level.min(MAX_REFINE)) as f64;
        let mut q = p;
        for a in 0..3 {
            let t = (p[a] - self.origin[a]) / self.spacing * steps;
            let r = t.round();
            if (t - r).abs() < SNAP && r >= 0.0 && r <= self.cells[a] as f64 * steps {
                q[a] = self.origin[a] + r / steps * self.spacing;
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
                    if let Some(pieces) = sorted.pieces.get_mut(cell) {
                        own_pieces(triangle, t as u32, normal, lo, hi, pieces);
                    }
                }
            }
        }
    }
    sorted
}

/// Add the pieces of a triangle (with the normal `normal`, of any length) that the box `lo`–`hi`
/// owns to `pieces`.
fn own_pieces(triangle: &[[f64; 3]; 3], t: u32, normal: [f64; 3], lo: [f64; 3], hi: [f64; 3], pieces: &mut Vec<Piece>) {
    let polygon: Vec<[f64; 3]> = clip(triangle, lo, hi).into_iter().map(|(corner, _)| corner).collect();
    // A polygon in the plane between two cells belongs to the one the material is in: the one
    // the normal points away from.
    let foreign = (0..3)
        .any(|a| (polygon.iter().all(|p| p[a] == lo[a]) && !(normal[a] < 0.0)) || (polygon.iter().all(|p| p[a] == hi[a]) && !(normal[a] > 0.0)));
    if polygon.len() < 3 || foreign {
        return;
    }
    for pair in polygon.get(1..).unwrap_or_default().windows(2) {
        let corners = [polygon[0], pair[0], pair[1]];
        let area = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
        if dot(area, area) > 0.0 {
            pieces.push(Piece { corners, triangle: t });
        }
    }
}

/// Sort the triangles `candidates` (indices into `triangles`, in increasing order) into the cube
/// from `lo` to `hi` of size `size`, a cell of a refined level: the ones that reach it and the
/// pieces it owns, as [`sort_surface`] does for the cells of the grid.
pub fn sort_into_box(triangles: &[[[f64; 3]; 3]], candidates: &[u32], lo: [f64; 3], hi: [f64; 3], size: f64) -> (Vec<u32>, Vec<Piece>) {
    let half = [0.5 * size * (1.0 + GROW); 3];
    let centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
    let (mut touching, mut pieces) = (Vec::new(), Vec::new());
    for &t in candidates {
        let Some(triangle) = triangles.get(t as usize) else { continue };
        if !overlaps(triangle, centre, half) {
            continue;
        }
        touching.push(t);
        let normal = cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]));
        own_pieces(triangle, t, normal, lo, hi, &mut pieces);
    }
    (touching, pieces)
}

/// The curvature of the surface at each of its triangles, one over a length: the largest over
/// the triangle's smooth edges (the ones that are no feature edges, so a sharp crease is no
/// curvature). Two triangles that meet in a smooth edge are two flat facets of one curved face,
/// and the normal turns from the one's to the other's from the middle of the one to the middle
/// of the other, half the sum of their heights over the edge apart. Facets whose corners lie on
/// a circle of radius r and whose normals differ by the vector d are |d| r wide, so the
/// curvature across the edge is 2 |n₁ − n₂| / (h₁ + h₂). That is exact for the strips that a
/// cylinder, a fillet or any face tessellated on a grid of its parameters is made of, whichever
/// way a strip's quads are split; on an irregular triangulation of a doubly curved face it is
/// up to a third low. Triangles that are one flat face to the octree have none.
pub fn curvatures(surface: &Triangles) -> Vec<f64> {
    let doubled_area = |t: &[[f64; 3]; 3]| {
        let area = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        dot(area, area).sqrt()
    };
    let mut curvature = vec![0.0; surface.corners.len()];
    for (index, (triangle, value)) in surface.corners.iter().zip(&mut curvature).enumerate() {
        let (Some(normal), Some(across), Some(feature)) = (surface.normals.get(index), surface.across.get(index), surface.feature.get(index)) else {
            continue;
        };
        let own = doubled_area(triangle);
        for edge in 0..3 {
            let (Some(other), Some(other_normal)) = (surface.corners.get(across[edge] as usize), surface.normals.get(across[edge] as usize)) else {
                continue;
            };
            if feature[edge] {
                continue;
            }
            let turn = cross(*normal, *other_normal);
            let (sine, cosine) = (dot(turn, turn).sqrt(), dot(*normal, *other_normal));
            let along = sub(triangle[(edge + 1) % 3], triangle[edge]);
            // The two heights over the edge, times the edge's length.
            let heights = own + doubled_area(other);
            if (sine * sine <= FLAT && cosine > 0.0) || !(heights > 0.0) {
                continue;
            }
            let apart = sub(*normal, *other_normal);
            let found = 2.0 * dot(apart, apart).sqrt() * dot(along, along).sqrt() / heights;
            if found.is_finite() && found > *value {
                *value = found;
            }
        }
    }
    curvature
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
/// refined where the surface of the triangles `touching` (indices into the surface's) is curved,
/// with the Gauss `rule` in every leaf and along and across the lines of a leaf the surface
/// reaches (which is exact for flat faces along the axes only: see [`visit_cut_quadrature`]).
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
    let mut points = Vec::new();
    visit_cut_quadrature(lo, h, touching, surface, solid, [rule, rule, rule], depth, max_points, |point| points.push(point))?;
    Ok(points)
}

/// [`cut_quadrature`] with a Gauss rule for each direction of a leaf, handing every point to
/// `visit` instead of collecting them; the number of points is returned. `rules[0]` is the rule
/// of a leaf the surface does not reach and of the lines through one it does; `rules[1]` and
/// `rules[2]` are the rules across the lines, along the second and the third axis.
#[allow(clippy::too_many_arguments)]
pub fn visit_cut_quadrature(
    lo: [f64; 3],
    h: f64,
    touching: &[u32],
    surface: &Triangles,
    solid: &Solid,
    rules: [&[(f64, f64)]; 3],
    depth: usize,
    max_points: usize,
    mut visit: impl FnMut(QuadPoint),
) -> Result<usize> {
    let depth = depth.min(MAX_DEPTH);
    let triangles = surface.corners;
    let physical = |xi: [f64; 3]| [lo[0] + 0.5 * (xi[0] + 1.0) * h, lo[1] + 0.5 * (xi[1] + 1.0) * h, lo[2] + 0.5 * (xi[2] + 1.0) * h];
    let [rule, second_rule, third_rule] = rules;
    let per_leaf = rule.len() * rule.len() * rule.len();
    let mut count = 0usize;
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
        if level < depth && curved(&cut, surface) {
            for child in (0..8usize).rev() {
                let shift = |bit: usize| if (child >> bit) & 1 == 1 { half } else { 0.0 };
                stack.push(([corner[0] + shift(0), corner[1] + shift(1), corner[2] + shift(2)], level + 1, cut.clone()));
            }
            continue;
        }
        let full = || Error::Limit(format!("a cut cell needs more than {max_points} quadrature points (lower the octree depth or the degree)"));
        if cut.is_empty() {
            // The surface does not reach the leaf: one side, the Gauss rule of the box.
            if count + per_leaf > max_points {
                return Err(full());
            }
            count += per_leaf;
            let inside = solid.contains(centre)?;
            for (gz, wz) in rule {
                for (gy, wy) in rule {
                    for (gx, wx) in rule {
                        let xi = [corner[0] + half * (gx + 1.0), corner[1] + half * (gy + 1.0), corner[2] + half * (gz + 1.0)];
                        visit(QuadPoint { xi, weight: wx * wy * wz * half * half * half, inside });
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
        // And where two of the edges cross: the order in which a line across meets them changes
        // there.
        if edges.len() <= MOST_CROSSING {
            for (i, first) in edges.iter().enumerate() {
                for second in edges.iter().skip(i + 1) {
                    if let Some(at) = crossing(first, second).filter(|at| *at > leaf_lo[c] && *at < leaf_hi[c]) {
                        third.push(at);
                    }
                }
            }
        }
        thin(&mut third, tiny);
        let (mut second, mut first): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        for span_c in third.windows(2) {
            for (gc, wc) in third_rule {
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
                    for (gb, wb) in second_rule {
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
                        if count + first.len() * rule.len() > max_points {
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
                                visit(QuadPoint { xi, weight: weight_b * weight_c * w * (span[1] - span[0]) / h, inside });
                            }
                            count += rule.len();
                        }
                    }
                }
            }
        }
    }
    Ok(count)
}

/// Leaves with more edges than this are not searched for crossing edges (the search is
/// quadratic): so many edges are the facets of a curved face seen edge-on, which [`thin`] merges.
const MOST_CROSSING: usize = 64;

/// Where two segments in the plane across the lines cross, strictly inside both: the second
/// coordinate of the point.
fn crossing(first: &[[f64; 2]; 2], second: &[[f64; 2]; 2]) -> Option<f64> {
    let [p, q] = *first;
    let [r, s] = *second;
    let (d, e) = ([q[0] - p[0], q[1] - p[1]], [s[0] - r[0], s[1] - r[1]]);
    let denominator = d[0] * e[1] - d[1] * e[0];
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }
    let along_first = ((r[0] - p[0]) * e[1] - (r[1] - p[1]) * e[0]) / denominator;
    let along_second = ((r[0] - p[0]) * d[1] - (r[1] - p[1]) * d[0]) / denominator;
    (along_first > 0.0 && along_first < 1.0 && along_second > 0.0 && along_second < 1.0).then(|| p[1] + along_first * d[1])
}

/// Two triangles whose unit normals differ by less than this angle (its square, in radians)
/// across a smooth edge are one flat face to the octree.
const FLAT: f64 = 1e-12; // unit-factors: an angle squared

/// Whether the surface is curved where these triangles are: two of them that meet in an edge
/// that is not a feature edge face different ways. A leaf of flat faces only is integrated
/// exactly without refining it; this decides only where the octree refines.
fn curved(cut: &[u32], surface: &Triangles) -> bool {
    cut.iter().any(|t| {
        let index = *t as usize;
        let (Some(normal), Some(across), Some(feature)) = (surface.normals.get(index), surface.across.get(index), surface.feature.get(index)) else {
            return false;
        };
        (0..3).any(|edge| {
            !feature[edge]
                && surface.normals.get(across[edge] as usize).is_some_and(|other| {
                    let turn = cross(*normal, *other);
                    dot(turn, turn) > FLAT
                })
        })
    })
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

    /// The tetrahedron between the coordinate planes and the plane through `reach` on the axes,
    /// as a surface with outward triangles.
    fn corner(reach: [f64; 3]) -> BoundarySurface {
        let points = [[0.0; 3], [reach[0], 0.0, 0.0], [0.0, reach[1], 0.0], [0.0, 0.0, reach[2]]];
        let triangles = [[0, 3, 2], [0, 1, 3], [0, 2, 1], [1, 2, 3]];
        BoundarySurface::new(&points, &triangles, &[0, 1, 2, 3], &fieldcraft_mesh::surface::SurfaceOptions::default()).unwrap()
    }

    /// ∫ xᵃ yᵇ zᶜ over the part of a cell inside `surface`, by the cut-cell quadrature for
    /// degree `p` (or with the rule of p + 1 points in every direction), and the points used.
    fn monomial(surface: &BoundarySurface, lo: [f64; 3], h: f64, p: usize, powers: [i32; 3], depth: usize, plain: bool) -> (f64, usize) {
        let corners: Vec<[[f64; 3]; 3]> = surface.triangles().iter().map(|t| t.map(|v| surface.points()[v as usize])).collect();
        let normals: Vec<[f64; 3]> = corners
            .iter()
            .map(|t| {
                let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
                let length = dot(n, n).sqrt();
                n.map(|x| x / length)
            })
            .collect();
        let feature = feature_flags(surface);
        let triangles = Triangles { corners: &corners, normals: &normals, across: surface.neighbors(), feature: &feature };
        let solid = Solid::new(surface).unwrap();
        let touching: Vec<u32> = (0..corners.len() as u32).collect();
        let rules = [crate::gauss::gauss_legendre(p + 1), crate::gauss::gauss_legendre(2 * p + 1), crate::gauss::gauss_legendre(3 * p + 2)];
        let rules = if plain { [&rules[0][..], &rules[0][..], &rules[0][..]] } else { [&rules[0][..], &rules[1][..], &rules[2][..]] };
        let mut sum = 0.0;
        let count = visit_cut_quadrature(lo, h, &touching, &triangles, &solid, rules, depth, 10_000_000, |point| {
            if point.inside {
                let x = [0, 1, 2].map(|a| lo[a] + 0.5 * (point.xi[a] + 1.0) * h);
                sum += point.weight * 0.125 * h * h * h * x[0].powi(powers[0]) * x[1].powi(powers[1]) * x[2].powi(powers[2]);
            }
        })
        .unwrap();
        (sum, count)
    }

    /// A cell cut by an inclined flat face: every monomial of degree 2p along each axis, which
    /// is what a product of two shape functions holds, is integrated exactly, with no octree.
    /// ∫ xᵃ yᵇ zᶜ over the tetrahedron is Xᵃ⁺¹ Yᵇ⁺¹ Zᶜ⁺¹ a! b! c! / (a + b + c + 3)!.
    #[test]
    fn a_cell_cut_by_an_inclined_flat_face_is_integrated_exactly() {
        let reach = [0.7, 0.6, 0.5];
        let surface = corner(reach);
        let (lo, h) = ([-0.1, -0.2, -0.15], 1.0);
        let factorial = |n: i32| (1..=n).map(f64::from).product::<f64>();
        for p in [1, 2, 3] {
            let top = 2 * p as i32;
            for powers in [[0, 0, 0], [top, 0, 0], [0, top, 1], [1, 2, top], [top, top, top], [top - 1, top, top - 1]] {
                let [a, b, c] = powers;
                let exact = reach[0].powi(a + 1) * reach[1].powi(b + 1) * reach[2].powi(c + 1) * factorial(a) * factorial(b) * factorial(c)
                    / factorial(a + b + c + 3);
                let (sum, count) = monomial(&surface, lo, h, p, powers, 0, false);
                assert!((sum - exact).abs() < 1e-12 * exact, "degree {p}, powers {powers:?}: {sum} vs {exact}");
                // Flat faces: a deeper octree is not used.
                assert_eq!(monomial(&surface, lo, h, p, powers, 3, false).1, count);
            }
            // The rule of p + 1 points across the lines gives the volume and no more.
            let (volume, _) = monomial(&surface, lo, h, p, [0, 0, 0], 0, true);
            assert!((volume - reach[0] * reach[1] * reach[2] / 6.0).abs() < 1e-14);
            let exact = (reach[0] * reach[1] * reach[2]).powi(top + 1) * factorial(top).powi(3) / factorial(3 * top + 3);
            let (plain, _) = monomial(&surface, lo, h, p, [top, top, top], 0, true);
            assert!((plain - exact).abs() > 1e-6 * exact, "degree {p}: the plain rule is not exact, {plain} vs {exact}");
        }
    }

    /// A curved face (a prism of many sides, its side one smooth face) refines the octree down
    /// to the depth limit.
    #[test]
    fn the_octree_refines_where_the_surface_is_curved() {
        const SIDES: usize = 48;
        let (radius, height) = (0.8, 0.5);
        let surface = prism(SIDES, radius, height);
        // The regular polygon's x² over its area: A (R² / 12)(2 + cos(2π/n)), A its area.
        let turn = 2.0 * std::f64::consts::PI / SIDES as f64;
        let area = 0.5 * SIDES as f64 * radius * radius * turn.sin();
        let exact = height * area * radius * radius / 12.0 * (2.0 + turn.cos());
        let (lo, h) = ([-1.0, -1.1, -0.2], 2.0);
        // The facets' edges are not split at, so the octree's answer is not the prism's own
        // integral to rounding, but it is far closer to it than the prism is to the cylinder
        // (whose area differs by a sixth of the facets' angle squared).
        let tessellation = turn * turn / 6.0;
        let mut points = 0usize;
        for depth in [0, 1, 2, 3, 4] {
            let (sum, count) = monomial(&surface, lo, h, 2, [2, 0, 0], depth, false);
            let error = (sum - exact).abs() / exact;
            println!("depth {depth}: {count} points, error {error:.2e}");
            assert!(count > points, "depth {depth} refines: {count} points after {points}");
            assert!(error < 0.1 * tessellation, "depth {depth}: error {error:.2e}");
            points = count;
        }
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

    /// A prism of `sides` sides about the z axis, its side one smooth face (tag 2) between two
    /// caps.
    fn prism(sides: usize, radius: f64, height: f64) -> BoundarySurface {
        let mut points = vec![[0.0, 0.0, 0.0], [0.0, 0.0, height]];
        for i in 0..sides {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / sides as f64;
            points.push([radius * angle.cos(), radius * angle.sin(), 0.0]);
            points.push([radius * angle.cos(), radius * angle.sin(), height]);
        }
        let (mut triangles, mut tags) = (Vec::new(), Vec::new());
        for i in 0..sides as u32 {
            let (a, b) = (2 + 2 * i, 2 + 2 * ((i + 1) % sides as u32));
            triangles.extend([[0, b, a], [1, a + 1, b + 1], [a, b, b + 1], [a, b + 1, a + 1]]);
            tags.extend([0, 1, 2, 2]);
        }
        BoundarySurface::new(&points, &triangles, &tags, &fieldcraft_mesh::surface::SurfaceOptions::default()).unwrap()
    }

    /// The corners and unit normals of a surface's triangles.
    fn facets(surface: &BoundarySurface) -> (Vec<[[f64; 3]; 3]>, Vec<[f64; 3]>) {
        let corners: Vec<[[f64; 3]; 3]> = surface.triangles().iter().map(|t| t.map(|v| surface.points()[v as usize])).collect();
        let normals = corners
            .iter()
            .map(|t| {
                let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
                let length = dot(n, n).sqrt();
                n.map(|x| x / length)
            })
            .collect();
        (corners, normals)
    }

    /// The facets of a cylinder have its curvature, one over its radius, however finely it is
    /// tessellated; its flat caps and a block, all of whose edges are sharp, have none.
    #[test]
    fn the_curvature_of_a_tessellated_cylinder_is_one_over_its_radius() {
        for (sides, radius) in [(12, 0.8), (48, 0.8), (512, 3.0)] {
            let surface = prism(sides, radius, 0.5);
            let (corners, normals) = facets(&surface);
            let feature = feature_flags(&surface);
            let curvature = curvatures(&Triangles { corners: &corners, normals: &normals, across: surface.neighbors(), feature: &feature });
            assert_eq!(curvature.len(), corners.len());
            for (found, tag) in curvature.iter().zip(surface.tri_face()) {
                let exact = if *tag == 2 { 1.0 / radius } else { 0.0 };
                assert!((found - exact).abs() < 1e-9 * (1.0 / radius), "{sides} sides, face {tag}: {found} against {exact}");
            }
        }
        let surface = corner([0.7, 0.6, 0.5]);
        let (corners, normals) = facets(&surface);
        let feature = feature_flags(&surface);
        let curvature = curvatures(&Triangles { corners: &corners, normals: &normals, across: surface.neighbors(), feature: &feature });
        assert!(curvature.iter().all(|c| *c == 0.0), "{curvature:?}");
    }

    /// The cells of a refined level: their planes are the grid's where they coincide, a vertex
    /// near one is moved onto it, and the eight parts of a cell own the pieces the cell owns.
    #[test]
    fn the_parts_of_a_cell_own_its_pieces() {
        let grid = Grid::over([0.0; 3], [2.0, 2.0, 1.0], 1.0, [0.0; 3], 100).unwrap();
        assert_eq!((grid.step(0), grid.step(2)), (1.0, 0.25));
        for i in 0..=2 {
            assert_eq!(grid.fine_plane(0, 0, i).to_bits(), grid.plane(0, i).to_bits());
            assert_eq!(grid.fine_plane(1, 3, 8 * i).to_bits(), grid.plane(1, i).to_bits());
        }
        assert_eq!((grid.fine_lo(1, [1, 2, 0]), grid.fine_hi(1, [1, 2, 0])), ([0.5, 1.0, 0.0], [1.0, 1.5, 0.5]));
        // Onto a plane of the level, and not onto one of a finer level.
        assert_eq!(grid.snap_to([0.5 + 1e-11, 0.25 + 1e-11, 1.0], 1), [0.5, 0.25 + 1e-11, 1.0]);
        assert_eq!(grid.snap_to([0.5 + 1e-11, 0.3, 2.5], 0), grid.snap([0.5 + 1e-11, 0.3, 2.5]));

        let area = |pieces: &[Piece]| -> f64 {
            pieces
                .iter()
                .map(|p| 0.5 * dot(cross(sub(p.corners[1], p.corners[0]), sub(p.corners[2], p.corners[0])), [1.0, 1.0, 1.0]).abs() / 3f64.sqrt())
                .sum()
        };
        // An inclined triangle through the first cell, and one in the plane x = 0.5 between its
        // parts, with the material at lower x.
        let triangles = [[[0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.9]], [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0]]];
        let sorted = sort_surface(&grid, &triangles);
        let cell = grid.index([0, 0, 0]);
        // As a box, a cell of the grid owns what the grid gives it.
        let (touching, pieces) = sort_into_box(&triangles, &[0, 1], grid.lo([0, 0, 0]), grid.hi([0, 0, 0]), 1.0);
        assert_eq!((touching, pieces), (sorted.touching[cell].clone(), sorted.pieces[cell].clone()));
        let (mut inclined, mut between) = (0.0, [0.0; 2]);
        for child in 0..8usize {
            let at = [child & 1, (child >> 1) & 1, (child >> 2) & 1];
            let (touching, pieces) = sort_into_box(&triangles, &sorted.touching[cell], grid.fine_lo(1, at), grid.fine_hi(1, at), 0.5);
            assert!(pieces.iter().all(|piece| touching.contains(&piece.triangle)));
            inclined += area(&pieces.iter().filter(|piece| piece.triangle == 0).copied().collect::<Vec<_>>());
            // The triangle in the plane: the doubled areas of its pieces along x.
            between[at[0]] += pieces
                .iter()
                .filter(|p| p.triangle == 1)
                .map(|p| cross(sub(p.corners[1], p.corners[0]), sub(p.corners[2], p.corners[0]))[0].abs())
                .sum::<f64>();
        }
        let whole = area(&sorted.pieces[cell].iter().filter(|piece| piece.triangle == 0).copied().collect::<Vec<_>>());
        assert!((inclined - whole).abs() < 1e-14 && whole > 0.5, "{inclined} of {whole}");
        // The parts at lower x own all of it: its normal points along +x.
        assert!((between[0] - 1.0).abs() < 1e-14 && between[1] == 0.0, "{between:?}");
    }
}
