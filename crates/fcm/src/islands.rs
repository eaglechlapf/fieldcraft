//! The separate pieces of material in a cell.
//!
//! A cell carries one polynomial, and a polynomial cannot tell the two sides of a narrow gap
//! apart: where one cell holds the two flanges of a sheet metal corner, the two jaws of a slit
//! clamp or two fins of a heat sink, its unknowns would join them, and the part would be solved
//! as if the gap were welded shut. So the material of a cut cell is sorted into its *islands*,
//! the pieces that are not connected inside the cell, and every island gets shape functions of
//! its own on the cell: the cell is laid over itself once per island, each copy integrated over
//! its island alone (A. Hansbo, P. Hansbo, *A finite element method for the simulation of strong
//! and weak discontinuities in solid mechanics*, Comput. Methods Appl. Mech. Engrg.
//! 193(33–35):3523–3540, 2004; the same space as the phantom nodes of J.-H. Song, P. M. A.
//! Areias, T. Belytschko, *A method for dynamic crack and shear band propagation with phantom
//! nodes*, Int. J. Numer. Methods Engrg. 67(6):868–893, 2006). Two islands of neighbouring cells
//! share the functions of the face between them where the material goes on from the one into
//! the other ([`links`]), and only there ([`crate::space::Space::separate`]).
//!
//! # Islands
//! The surface in the cell bounds its material, so the islands are found from the surface:
//! 1. The triangles that have a piece in the cell are grouped by the edges they share inside
//!    the cell: one group is one connected sheet of surface, with material behind all of it.
//!    One group is one island, and most cut cells stop here.
//! 2. Two groups bound the same island when material joins them. The quadrature of the cell
//!    ([`crate::grid`]) says so: a segment of a line that lies inside the part and ends on two
//!    triangles joins their groups (the two faces of a plate, the inside and the outside of a
//!    bend). A segment that ends on no triangle belongs to the group of the nearest piece of
//!    surface: no surface lies between a point and the surface nearest to it.
//! 3. Groups that are still apart are joined when a straight line between a point of the one's
//!    material and a point of the other's crosses no triangle.
//!
//! Every joining is a proof that the two groups bound one piece of material; none is assumed.
//! What is left apart is separated by the surface on every line that was tried: a gap. A piece
//! of material that is connected only around a corner no line was tried on would be taken for
//! two islands; its functions then still join in every neighbouring cell its material goes on
//! into.

use std::collections::BTreeSet;

use crate::grid::{self, NO_TRIANGLE, Piece, Triangles, cross, dot, sub};
use crate::inside::Solid;
use crate::moments::{Moments, Tables};
use crate::refine::Tree;
use crate::{Error, Result};

/// Barycentric coordinates and positions along a segment within this much of a triangle's edge
/// or of the segment's ends count as crossing: a line through an edge two triangles share must
/// not slip between them by rounding, and a line that is taken to cross is the safe mistake.
const NEAR: f64 = 1e-9; // unit-factors: barycentric, relative to the triangle and the segment
/// How far from the face between two cells, in cell sizes, its two sides are looked at
/// ([`links`]).
const OFF_FACE: f64 = 1e-6; // unit-factors: relative to the cell size
/// How far behind a corner of a piece of surface that lies in the face between two cells, along
/// the face and in cell sizes, the material beside it is looked at ([`links`]); also the least
/// part of a piece's normal that lies along the face, for the piece to have a direction there.
const BESIDE: f64 = 1e-5;
/// Corners of a cell's pieces in a face that are tried for material beside them, at most.
const MOST_BESIDE: usize = 8;
/// Leaves whose gaps' sides one task finds ([`sides`]).
const LEAVES: usize = 16;
/// Points kept of each group's material for [`integrate`]'s third step.
const MOST_SAMPLES: usize = 16;

/// Disjoint sets of the indices `0..len`.
#[derive(Debug, Clone)]
pub(crate) struct Sets {
    parent: Vec<u32>,
}

impl Sets {
    pub fn new(len: usize) -> Sets {
        Sets { parent: (0..len as u32).collect() }
    }

    /// The representative of the set `i` is in: the lowest index of the set.
    pub fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        // Bounded by the number of indices: the parents lead strictly down.
        for _ in 0..self.parent.len() {
            let Some(&up) = self.parent.get(root) else { return i };
            if up as usize == root {
                break;
            }
            root = up as usize;
        }
        let mut at = i;
        while at != root {
            let Some(up) = self.parent.get_mut(at) else { break };
            at = std::mem::replace(up, root as u32) as usize;
        }
        root
    }

    /// Make the sets of `a` and `b` one; `(kept, absorbed)` when they were two.
    pub fn join(&mut self, a: usize, b: usize) -> Option<(usize, usize)> {
        let (a, b) = (self.find(a), self.find(b));
        if a == b {
            return None;
        }
        let (kept, absorbed) = (a.min(b), a.max(b));
        if let Some(up) = self.parent.get_mut(absorbed) {
            *up = kept as u32;
        }
        Some((kept, absorbed))
    }
}

/// Whether the segment from `p` to `q` reaches the box `lo`–`hi` (touching counts).
fn reaches(p: [f64; 3], q: [f64; 3], lo: [f64; 3], hi: [f64; 3]) -> bool {
    let (mut from, mut to) = (0.0f64, 1.0f64);
    for a in 0..3 {
        let d = q[a] - p[a];
        if d == 0.0 {
            if p[a] < lo[a] || p[a] > hi[a] {
                return false;
            }
            continue;
        }
        let (s, t) = ((lo[a] - p[a]) / d, (hi[a] - p[a]) / d);
        from = from.max(s.min(t));
        to = to.min(s.max(t));
    }
    from <= to
}

/// Whether the segment from `p` to `q` crosses the triangle (T. Möller, B. Trumbore, *Fast,
/// minimum storage ray-triangle intersection*, J. Graphics Tools 2(1):21–28, 1997), within
/// [`NEAR`]. A segment in the triangle's plane does not cross it.
fn crosses(p: [f64; 3], q: [f64; 3], triangle: &[[f64; 3]; 3]) -> bool {
    let [a, b, c] = *triangle;
    let (along, first, second) = (sub(q, p), sub(b, a), sub(c, a));
    let h = cross(along, second);
    let determinant = dot(first, h);
    if determinant == 0.0 || !determinant.is_finite() {
        return false;
    }
    let s = sub(p, a);
    let u = dot(s, h) / determinant;
    let g = cross(s, first);
    let v = dot(along, g) / determinant;
    let t = dot(second, g) / determinant;
    u >= -NEAR && v >= -NEAR && u + v <= 1.0 + NEAR && (-NEAR..=1.0 + NEAR).contains(&t)
}

/// Whether the straight line from `p` to `q` crosses none of the triangles `reach`.
fn clear(p: [f64; 3], q: [f64; 3], reach: &[u32], surface: &Triangles) -> bool {
    !reach.iter().any(|t| surface.corners.get(*t as usize).is_some_and(|triangle| crosses(p, q, triangle)))
}

/// The square of the distance from `p` to a triangle.
pub(crate) fn distance_squared(p: [f64; 3], triangle: &[[f64; 3]; 3]) -> f64 {
    let d = sub(p, closest_point(p, triangle));
    dot(d, d)
}

/// The point of a triangle closest to `p` (C. Ericson, *Real-Time Collision Detection*, Morgan
/// Kaufmann 2005, section 5.1.5: by the Voronoi region of the triangle the point is in).
pub(crate) fn closest_point(p: [f64; 3], triangle: &[[f64; 3]; 3]) -> [f64; 3] {
    let [a, b, c] = *triangle;
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let away = |x: [f64; 3]| {
        let d = sub(p, x);
        dot(d, d)
    };
    let on = |from: [f64; 3], along: [f64; 3], s: f64| [from[0] + s * along[0], from[1] + s * along[1], from[2] + s * along[2]];
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 && d1 - d3 > 0.0 {
        return on(a, ab, d1 / (d1 - d3));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 && d2 - d6 > 0.0 {
        return on(a, ac, d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 && (d4 - d3) + (d5 - d6) > 0.0 {
        return on(b, sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let sum = va + vb + vc;
    if !(sum != 0.0 && sum.is_finite()) {
        // A triangle of no area: the nearest of its corners.
        let mut nearest = a;
        for corner in [b, c] {
            if away(corner) < away(nearest) {
                nearest = corner;
            }
        }
        return nearest;
    }
    let (v, w) = (vb / sum, vc / sum);
    [a[0] + v * ab[0] + w * ac[0], a[1] + v * ab[1] + w * ac[1], a[2] + v * ab[2] + w * ac[2]]
}

/// The piece nearest to `x` (the first of several as near).
fn nearest(pieces: &[Piece], x: [f64; 3]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (index, piece) in pieces.iter().enumerate() {
        let distance = distance_squared(x, &piece.corners);
        if best.is_none_or(|(_, least)| distance < least) {
            best = Some((index, distance));
        }
    }
    best.map(|(index, _)| index)
}

/// The island of a cell that holds the point `x` of the cell's material: the island of the
/// piece of surface nearest to it (no surface lies between a point and the surface nearest to
/// it, so that piece bounds the material the point is in). `of_piece` is the island of each of
/// the cell's `pieces` ([`CutLeaf::of_piece`]); a cell of one island has none, and the island
/// is 0.
pub(crate) fn island_at(pieces: &[Piece], of_piece: &[u16], x: [f64; 3]) -> usize {
    if of_piece.is_empty() {
        return 0;
    }
    nearest(pieces, x).and_then(|piece| of_piece.get(piece)).map_or(0, |island| usize::from(*island))
}

/// The sheets of surface in the cell from `lo` to `hi`: the triangles that have a piece in it,
/// in increasing order, their sets by the edges they share inside the cell (the module's first
/// step), and the number of those sets.
pub(crate) fn sheets(lo: [f64; 3], hi: [f64; 3], pieces: &[Piece], surface: &Triangles) -> (Vec<u32>, Sets, usize) {
    let mut owned: Vec<u32> = pieces.iter().map(|piece| piece.triangle).collect();
    owned.sort_unstable();
    owned.dedup();
    let mut sets = Sets::new(owned.len());
    let mut groups = owned.len();
    for (i, t) in owned.iter().enumerate() {
        let (Some(corners), Some(across)) = (surface.corners.get(*t as usize), surface.across.get(*t as usize)) else { continue };
        for edge in 0..3 {
            if let Ok(j) = owned.binary_search(&across[edge])
                && j > i
                && reaches(corners[edge], corners[(edge + 1) % 3], lo, hi)
                && sets.join(i, j).is_some()
            {
                groups -= 1;
            }
        }
    }
    (owned, sets, groups)
}

/// What the quadrature of a cut cell leaves behind.
#[derive(Debug, Clone)]
pub(crate) struct CutLeaf {
    /// The Legendre moments of the material of each island of the cell, in the cell's natural
    /// coordinates: at least one.
    pub inside: Vec<Moments>,
    /// Those of the cell's volume outside the part.
    pub outside: Moments,
    /// The quadrature points.
    pub points: usize,
    /// The island of each piece of surface the cell owns, in their order; empty when the cell
    /// has one island.
    pub of_piece: Vec<u16>,
    /// Whether the moments are exact, from the triangles in the cell ([`crate::exact`]), and
    /// not a quadrature's.
    pub exact: bool,
}

impl CutLeaf {
    pub fn islands(&self) -> usize {
        self.inside.len().max(1)
    }
}

/// The groups of a cell's triangles while its quadrature runs, each with the moments of the
/// material found behind it and a few points of that material.
struct Groups {
    sets: Sets,
    inside: Vec<Option<Moments>>,
    samples: Vec<Vec<[f64; 3]>>,
    /// The runs of points each group was given.
    runs: Vec<usize>,
}

impl Groups {
    fn join(&mut self, a: usize, b: usize) {
        let Some((kept, absorbed)) = self.sets.join(a, b) else { return };
        let moved = self.inside.get_mut(absorbed).and_then(Option::take);
        if let (Some(moved), Some(slot)) = (moved, self.inside.get_mut(kept)) {
            match slot {
                Some(total) => total.absorb(&moved),
                None => *slot = Some(moved),
            }
        }
        let moved = self.samples.get_mut(absorbed).map(std::mem::take).unwrap_or_default();
        if let Some(samples) = self.samples.get_mut(kept) {
            samples.extend(moved);
        }
        let moved = self.runs.get_mut(absorbed).map(std::mem::take).unwrap_or(0);
        if let Some(runs) = self.runs.get_mut(kept) {
            *runs += moved;
        }
    }

    fn weight(&self, root: usize) -> f64 {
        self.inside.get(root).and_then(Option::as_ref).map_or(0.0, Moments::total)
    }
}

/// Integrate the cut cell from `lo` to `hi`, of size `h`, and sort its material into islands.
/// With `exact` the moments come from the triangles in the cell ([`crate::exact`]) wherever
/// that decides them; otherwise, and in the cells it leaves, from the quadrature
/// ([`grid::visit_spans`]) and the module's three steps. With `separate` off the cell is one
/// island whatever it holds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate(
    tables: &Tables,
    lo: [f64; 3],
    hi: [f64; 3],
    h: f64,
    reach: &[u32],
    pieces: &[Piece],
    surface: &Triangles,
    solid: &Solid,
    rules: [&[(f64, f64)]; 3],
    depth: usize,
    max_points: usize,
    separate: bool,
    exact: bool,
) -> Result<CutLeaf> {
    if exact && let Some(leaf) = crate::exact::integrate(tables, lo, hi, reach, pieces, surface, solid, [rules[1], rules[2]], separate)? {
        return Ok(leaf);
    }
    // The triangles that have a piece here, and their groups by the edges they share here.
    let (owned, sets, groups) = if separate { sheets(lo, hi, pieces, surface) } else { (Vec::new(), Sets::new(0), 1) };
    let slot = |t: u32| if t == NO_TRIANGLE { None } else { owned.binary_search(&t).ok() };
    if !separate || groups <= 1 {
        let (mut inside, mut outside) = (Moments::new(tables), Moments::new(tables));
        let points = grid::visit_cut_quadrature(lo, h, reach, surface, solid, rules, depth, max_points, |point| {
            if point.inside { inside.add(point.xi, point.weight) } else { outside.add(point.xi, point.weight) }
        })?;
        return Ok(CutLeaf { inside: vec![inside], outside, points, of_piece: Vec::new(), exact: false });
    }

    // Several sheets of surface: the material behind each, joined where a line through the
    // material ends on two of them.
    let mut found = Groups { sets, inside: vec![None; owned.len()], samples: vec![Vec::new(); owned.len()], runs: vec![0; owned.len()] };
    let mut outside = Moments::new(tables);
    let physical = |xi: [f64; 3]| [lo[0] + 0.5 * (xi[0] + 1.0) * h, lo[1] + 0.5 * (xi[1] + 1.0) * h, lo[2] + 0.5 * (xi[2] + 1.0) * h];
    let points = grid::visit_spans(lo, h, reach, surface, solid, rules, depth, max_points, |span| {
        if !span.inside {
            span.points.iter().for_each(|point| outside.add(point.xi, point.weight));
            return;
        }
        if span.points.is_empty() {
            return;
        }
        let mut middle = [0.0; 3];
        for point in span.points {
            for a in 0..3 {
                middle[a] += point.xi[a] / span.points.len() as f64;
            }
        }
        let middle = physical(middle);
        let group = match (slot(span.ends[0]), slot(span.ends[1])) {
            (Some(lower), Some(upper)) => {
                found.join(lower, upper);
                Some(lower)
            }
            (Some(end), None) | (None, Some(end)) => Some(end),
            (None, None) => nearest(pieces, middle).and_then(|piece| pieces.get(piece)).and_then(|piece| slot(piece.triangle)),
        };
        let Some(group) = group.map(|group| found.sets.find(group)) else { return };
        if let Some(moments) = found.inside.get_mut(group) {
            let moments = moments.get_or_insert_with(|| Moments::new(tables));
            span.points.iter().for_each(|point| moments.add(point.xi, point.weight));
        }
        // A few points of the group's material, spread over the runs: the first, the second, the
        // fourth, the eighth…
        if let (Some(runs), Some(samples)) = (found.runs.get_mut(group), found.samples.get_mut(group)) {
            *runs += 1;
            if runs.is_power_of_two() && samples.len() < MOST_SAMPLES {
                samples.push(middle);
            }
        }
    })?;

    // Groups still apart whose material a straight line joins. Every joining leaves one group
    // fewer, so the passes end.
    let roots = |found: &mut Groups| -> Vec<usize> {
        let all: BTreeSet<usize> = (0..owned.len()).map(|i| found.sets.find(i)).collect();
        all.into_iter().collect()
    };
    for _ in 0..owned.len() {
        let apart: Vec<usize> = roots(&mut found).into_iter().filter(|root| found.weight(*root) > 0.0).collect();
        let mut joined = None;
        'pairs: for (i, first) in apart.iter().enumerate() {
            for second in apart.iter().skip(i + 1) {
                let (Some(from), Some(to)) = (found.samples.get(*first), found.samples.get(*second)) else { continue };
                if from.iter().any(|p| to.iter().any(|q| clear(*p, *q, reach, surface))) {
                    joined = Some((*first, *second));
                    break 'pairs;
                }
            }
        }
        let Some((first, second)) = joined else { break };
        found.join(first, second);
    }
    // A group no material was found behind (a sliver between the lines) goes with the largest.
    let all = roots(&mut found);
    let largest = all.iter().copied().fold(None, |best: Option<usize>, root| match best {
        Some(best) if found.weight(best) >= found.weight(root) => Some(best),
        _ => Some(root),
    });
    if let Some(largest) = largest {
        for root in all {
            if !(found.weight(root) > 0.0) {
                found.join(largest, root);
            }
        }
    }
    // The islands, in the order of their lowest triangle.
    let islands = roots(&mut found);
    let mut inside = Vec::with_capacity(islands.len());
    for root in &islands {
        inside.push(found.inside.get_mut(*root).and_then(Option::take).unwrap_or_else(|| Moments::new(tables)));
    }
    if islands.len() <= 1 {
        return Ok(CutLeaf { inside, outside, points, of_piece: Vec::new(), exact: false });
    }
    if islands.len() > usize::from(u16::MAX) {
        return Err(Error::Limit(format!("a cell holds {} separate pieces of material (use a smaller cell size)", islands.len())));
    }
    let of_piece = pieces
        .iter()
        .map(|piece| {
            let root = slot(piece.triangle).map(|group| found.sets.find(group));
            root.and_then(|root| islands.binary_search(&root).ok()).unwrap_or(0) as u16
        })
        .collect();
    Ok(CutLeaf { inside, outside, points, of_piece, exact: false })
}

/// A cell that holds material, as [`links`] reads it: its box, the pieces of surface it owns and
/// the island of each ([`CutLeaf::of_piece`]; empty for one island).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Held<'a> {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
    pub pieces: &'a [Piece],
    pub of_piece: &'a [u16],
}

/// Where the material of one cell goes on into a neighbour: `[node, island, node, island]` for
/// every two islands of two leaves that share a face (or a part of one, between cells of
/// different levels) and touch through it. `held` gives a leaf by its node.
///
/// Material crosses the face between two cells where the surface does, and then a triangle with
/// pieces in both cells has an edge of a piece in the face: the islands of that triangle in the
/// two cells touch. Or the surface does not cross where the material does. Then either the
/// middle of the face or of one of its quarters is inside the part, or the surface meets the
/// face in an edge of the part that lies in its plane, and a point of the face just behind
/// that edge is: the islands those points belong to touch. A point counts when the part holds
/// it on both sides of the face.
pub(crate) fn links<'a>(tree: &Tree, solid: &Solid, held: &(dyn Fn(usize) -> Option<Held<'a>> + Sync)) -> Result<Vec<[u32; 4]>> {
    let leaves = tree.leaves();
    let found = fieldcraft_parallel::map(leaves.len(), |index| -> Result<Vec<[u32; 4]>> {
        let mut out = Vec::new();
        let Some(&node) = leaves.get(index) else { return Ok(out) };
        let Some(cell) = tree.nodes.get(node as usize).filter(|cell| cell.material()) else { return Ok(out) };
        let Some(own) = held(node as usize) else { return Ok(out) };
        let level = usize::from(cell.level);
        for axis in 0..3 {
            for (upper, step) in [(false, -1i64), (true, 1)] {
                let mut beside = cell.ijk.map(i64::from);
                beside[axis] += step;
                let Some(other_node) = tree.covering(level, beside) else { continue };
                let Some(other_cell) = tree.nodes.get(other_node).filter(|other| other.is_leaf() && other.material()) else { continue };
                // Every pair once: from the finer of the two, or from below.
                if usize::from(other_cell.level) == level && !upper {
                    continue;
                }
                let Some(other) = held(other_node) else { continue };
                let plane = if upper { own.hi[axis] } else { own.lo[axis] };
                let mut touching: BTreeSet<(u16, u16)> = BTreeSet::new();
                // The triangles with an edge of a piece in the face.
                let mut on_plane: Vec<(u32, [f64; 3])> = Vec::new();
                for piece in own.pieces {
                    for corner in piece.corners {
                        if corner[axis] == plane {
                            on_plane.push((piece.triangle, corner));
                        }
                    }
                }
                on_plane
                    .sort_by(|a, b| a.0.cmp(&b.0).then(a.1[0].total_cmp(&b.1[0])).then(a.1[1].total_cmp(&b.1[1])).then(a.1[2].total_cmp(&b.1[2])));
                on_plane.dedup();
                for run in on_plane.chunk_by(|a, b| a.0 == b.0) {
                    let Some(&(triangle, _)) = run.first() else { continue };
                    if run.len() < 2 {
                        continue;
                    }
                    let island = |side: &Held| {
                        side.pieces.iter().position(|piece| piece.triangle == triangle).map(|piece| side.of_piece.get(piece).copied().unwrap_or(0))
                    };
                    if let (Some(first), Some(second)) = (island(&own), island(&other)) {
                        touching.insert((first, second));
                    }
                }
                // The middle of the face and of its quarters.
                let one_island_each = own.of_piece.is_empty() && other.of_piece.is_empty();
                if !(one_island_each && !touching.is_empty()) {
                    let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
                    for (s, t) in [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                        let mut x = [0.0; 3];
                        x[axis] = plane;
                        x[b] = own.lo[b] + s * (own.hi[b] - own.lo[b]);
                        x[c] = own.lo[c] + t * (own.hi[c] - own.lo[c]);
                        // Material on both sides of the face: a face of the part that lies in
                        // the plane has it on one.
                        let (mut below, mut above) = (x, x);
                        below[axis] -= OFF_FACE * (own.hi[axis] - own.lo[axis]);
                        above[axis] += OFF_FACE * (own.hi[axis] - own.lo[axis]);
                        if solid.contains(below)? && solid.contains(above)? {
                            touching.insert((island_at(own.pieces, own.of_piece, x) as u16, island_at(other.pieces, other.of_piece, x) as u16));
                            if one_island_each {
                                break;
                            }
                        }
                    }
                }
                // Where the surface meets the face without crossing it (an edge of the part that
                // lies in the plane: the triangles on its two sides only touch the plane), the
                // material beside the surface there: a point just behind a corner of a piece
                // that lies in the face, along the face.
                if touching.is_empty() || !one_island_each {
                    let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
                    let size = own.hi[axis] - own.lo[axis];
                    for (side, is_own) in [(&own, true), (&other, false)] {
                        let mut beside: Vec<([f64; 3], u16)> = Vec::new();
                        for (index, piece) in side.pieces.iter().enumerate() {
                            let normal = cross(sub(piece.corners[1], piece.corners[0]), sub(piece.corners[2], piece.corners[0]));
                            let mut along = normal;
                            along[axis] = 0.0;
                            let (length, whole) = (dot(along, along).sqrt(), dot(normal, normal).sqrt());
                            // A piece in the plane has no direction along it.
                            if !(length > BESIDE * whole) {
                                continue;
                            }
                            for corner in piece.corners {
                                // In the face the two cells share: all of the finer one's.
                                let shared = corner[b] >= own.lo[b] && corner[b] <= own.hi[b] && corner[c] >= own.lo[c] && corner[c] <= own.hi[c];
                                if corner[axis] == plane && shared {
                                    let x = [0, 1, 2].map(|i| corner[i] - BESIDE * size * along[i] / length);
                                    beside.push((x, side.of_piece.get(index).copied().unwrap_or(0)));
                                }
                            }
                        }
                        // A few of them, spread over the pieces.
                        let step = beside.len().div_ceil(MOST_BESIDE).max(1);
                        for (x, island) in beside.into_iter().step_by(step) {
                            let (mut below, mut above) = (x, x);
                            below[axis] -= OFF_FACE * size;
                            above[axis] += OFF_FACE * size;
                            if solid.contains(below)? && solid.contains(above)? {
                                let far = if is_own { &other } else { &own };
                                let across = island_at(far.pieces, far.of_piece, x) as u16;
                                touching.insert(if is_own { (island, across) } else { (across, island) });
                            }
                        }
                    }
                }
                out.extend(touching.into_iter().map(|(first, second)| [node, u32::from(first), other_node as u32, u32::from(second)]));
            }
        }
        Ok(out)
    });
    let mut all = Vec::new();
    for list in found {
        all.extend(list?);
    }
    Ok(all)
}

/// The least distance between the pieces of surface of two islands, from the corners and the
/// middle of each piece of the one to the pieces of the other and back; infinite when one has
/// none.
fn width(first: &Held, first_island: u16, second: &Held, second_island: u16) -> f64 {
    let of = |held: &Held, island: u16| -> Vec<Piece> {
        held.pieces
            .iter()
            .enumerate()
            .filter(|(index, _)| held.of_piece.get(*index).copied().unwrap_or(0) == island)
            .map(|(_, piece)| *piece)
            .collect()
    };
    let (a, b) = (of(first, first_island), of(second, second_island));
    // What the distances may round by: far below any gap, far above the last digits of the
    // coordinates.
    let size = a.iter().chain(&b).flat_map(|piece| piece.corners).flatten().fold(0.0f64, |most, x| most.max(x.abs()));
    let slack = (ROUNDING * size) * (ROUNDING * size);
    let least = nearer(&b, &a, nearer(&a, &b, f64::INFINITY, slack), slack);
    least.sqrt()
}

/// What a distance is taken to round by at most, as a fraction of the coordinates.
const ROUNDING: f64 = 1e-9; // unit-factors: a ratio of two lengths
/// How many times the least squared distance found a bound has to be for a piece to be left
/// out: a little more than once, for the rounding of the two.
const BEYOND: f64 = 1.000_001; // unit-factors: a ratio of two squared lengths

/// The least squared distance from the corners and the middle of each piece of `from` to the
/// pieces of `to`, or `least` when none is below it. Every distance that could be the least is
/// worked out as it always was; the pieces whose box lies farther from a point than the least
/// found so far are left out, which finds them along the axis the pieces spread farthest on.
/// That is what makes a dense surface affordable: two islands of a thousand pieces each are
/// four million distances from every point to every piece.
fn nearer(from: &[Piece], to: &[Piece], mut least: f64, slack: f64) -> f64 {
    let boxes: Vec<([f64; 3], [f64; 3])> = to
        .iter()
        .map(|piece| {
            let [p, q, r] = piece.corners;
            ([0, 1, 2].map(|i| p[i].min(q[i]).min(r[i])), [0, 1, 2].map(|i| p[i].max(q[i]).max(r[i])))
        })
        .collect();
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for (low, high) in &boxes {
        for i in 0..3 {
            lo[i] = lo[i].min(low[i]);
            hi[i] = hi[i].max(high[i]);
        }
    }
    let axis = if hi[0] - lo[0] >= hi[1] - lo[1] && hi[0] - lo[0] >= hi[2] - lo[2] {
        0
    } else if hi[1] - lo[1] >= hi[2] - lo[2] {
        1
    } else {
        2
    };
    // The pieces by the middle of their box along the axis, and how far a box reaches from it.
    let middle = |piece: usize| boxes.get(piece).map_or(0.0, |(low, high)| 0.5 * (low[axis] + high[axis]));
    let mut order: Vec<usize> = (0..to.len()).collect();
    order.sort_by(|a, b| middle(*a).total_cmp(&middle(*b)).then(a.cmp(b)));
    let middles: Vec<f64> = order.iter().map(|piece| middle(*piece)).collect();
    let reach = boxes.iter().fold(0.0f64, |most, (low, high)| most.max(0.5 * (high[axis] - low[axis])));
    let beyond = |bound: f64, least: f64| bound > BEYOND * least + slack;
    for piece in from {
        let [p, q, r] = piece.corners;
        let centre = [0, 1, 2].map(|i| (p[i] + q[i] + r[i]) / 3.0);
        for x in [p, q, r, centre] {
            let measure = |at: usize, least: &mut f64| {
                let Some((other, (low, high))) = order.get(at).and_then(|piece| Some((to.get(*piece)?, boxes.get(*piece)?))) else { return };
                let mut bound = 0.0;
                for i in 0..3 {
                    let apart = (low[i] - x[i]).max(x[i] - high[i]).max(0.0);
                    bound += apart * apart;
                }
                if !beyond(bound, *least) {
                    *least = least.min(distance_squared(x, &other.corners));
                }
            };
            // From the pieces beside the point outwards, each way until the boxes lie too far.
            let start = middles.partition_point(|middle| *middle < x[axis]);
            for (at, middle) in middles.iter().enumerate().skip(start) {
                let apart = middle - x[axis] - reach;
                if apart > 0.0 && beyond(apart * apart, least) {
                    break;
                }
                measure(at, &mut least);
            }
            for (at, middle) in middles.iter().enumerate().take(start).rev() {
                let apart = x[axis] - middle - reach;
                if apart > 0.0 && beyond(apart * apart, least) {
                    break;
                }
                measure(at, &mut least);
            }
        }
    }
    least
}

/// The two sides of a gap: two islands (`[node, island, node, island]`) that lie in one cell or
/// in two cells that share a face and that no link joins, and how far their pieces of surface
/// are apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Sides {
    pub islands: [u32; 4],
    pub width: f64,
}

/// Every two sides of a gap ([`Sides`]) among the leaves of `tree`, given the `links` between
/// their islands ([`links`]).
pub(crate) fn sides<'a>(tree: &Tree, links: &[[u32; 4]], held: &(dyn Fn(usize) -> Option<Held<'a>> + Sync)) -> Vec<Sides> {
    let mut joined: BTreeSet<[u32; 4]> = BTreeSet::new();
    for link in links {
        joined.insert(*link);
        joined.insert([link[2], link[3], link[0], link[1]]);
    }
    let islands_of = |side: &Held| side.of_piece.iter().copied().max().map_or(1, |most| most + 1);
    // Leaf by leaf, a few to a task: the widths are distances between all the pieces of two
    // islands, and the sides come out in the leaves' order whichever thread finds them.
    let leaves = tree.leaves();
    let of_leaf = |index: usize| -> Vec<Sides> {
        let mut found = Vec::new();
        let mut between = |islands: [u32; 4], apart: f64| {
            if apart.is_finite() && apart > 0.0 {
                found.push(Sides { islands, width: apart });
            }
        };
        let Some(&node) = leaves.get(index) else { return found };
        let Some(cell) = tree.nodes.get(node as usize).filter(|cell| cell.material()) else { return found };
        let Some(own) = held(node as usize) else { return found };
        // Two islands of one cell.
        let islands = islands_of(&own);
        for first in 0..islands {
            for second in first + 1..islands {
                between([node, u32::from(first), node, u32::from(second)], width(&own, first, &own, second));
            }
        }
        // Two islands of two cells that share a face, with no link between them.
        let level = usize::from(cell.level);
        for axis in 0..3 {
            for (upper, step) in [(false, -1i64), (true, 1)] {
                let mut beside = cell.ijk.map(i64::from);
                beside[axis] += step;
                let Some(other_node) = tree.covering(level, beside) else { continue };
                let Some(other_cell) = tree.nodes.get(other_node).filter(|other| other.is_leaf() && other.material()) else { continue };
                if usize::from(other_cell.level) == level && !upper {
                    continue;
                }
                let Some(other) = held(other_node) else { continue };
                // A cell wholly inside the part is joined to all that touches it.
                if own.pieces.is_empty() || other.pieces.is_empty() {
                    continue;
                }
                for first in 0..islands {
                    for second in 0..islands_of(&other) {
                        let pair = [node, u32::from(first), other_node as u32, u32::from(second)];
                        if !joined.contains(&pair) {
                            between(pair, width(&own, first, &other, second));
                        }
                    }
                }
            }
        }
        found
    };
    fieldcraft_parallel::map_grained(leaves.len(), LEAVES, of_leaf).into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_width_of_a_gap_is_the_least_of_all_the_distances() {
        // Two sheets of triangles a little apart, tilted and shifted against each other, in
        // several sizes; every distance worked out for comparison.
        let sheet = |n: usize, at: f64, tilt: f64, shift: f64, seed: usize| -> Vec<Piece> {
            let mut pieces = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    let point = |di: usize, dj: usize| {
                        let (x, y) = ((i + di) as f64 / n as f64 + shift, (j + dj) as f64 / n as f64);
                        let wave = 0.01 * (((i + di) * 7 + (j + dj) * 13 + seed) as f64).sin();
                        [x, y, at + tilt * x + wave]
                    };
                    pieces.push(Piece { corners: [point(0, 0), point(1, 0), point(0, 1)], triangle: 0 });
                    pieces.push(Piece { corners: [point(1, 0), point(1, 1), point(0, 1)], triangle: 0 });
                }
            }
            pieces
        };
        let every = |a: &[Piece], b: &[Piece]| {
            let mut least = f64::INFINITY;
            for (from, to) in [(a, b), (b, a)] {
                for piece in from {
                    let [p, q, r] = piece.corners;
                    for x in [p, q, r, [0, 1, 2].map(|i| (p[i] + q[i] + r[i]) / 3.0)] {
                        for other in to {
                            least = least.min(distance_squared(x, &other.corners));
                        }
                    }
                }
            }
            least.sqrt()
        };
        for (n, m, apart, tilt, shift) in [
            (1, 1, 0.3, 0.0, 0.0),
            (6, 9, 0.05, 0.02, 0.0),
            (12, 5, 0.2, -0.3, 0.4),
            (9, 9, 0.0, 0.0, 0.0),
            (7, 11, 0.5, 0.0, 3.0),
            (10, 3, 1e-7, 0.0, 0.1),
        ] {
            let (a, b) = (sheet(n, 0.0, 0.0, 0.0, 1), sheet(m, apart, tilt, shift, 5));
            // As islands 0 and 1 of one cell, and as the one island of each of two cells.
            let both: Vec<Piece> = a.iter().chain(&b).copied().collect();
            let of_piece: Vec<u16> = (0..both.len()).map(|piece| u16::from(piece >= a.len())).collect();
            let held = |pieces, of_piece| Held { lo: [0.0; 3], hi: [1.0; 3], pieces, of_piece };
            let expected = every(&a, &b);
            let one = held(&both[..], &of_piece[..]);
            assert_eq!(width(&one, 0, &one, 1).to_bits(), expected.to_bits(), "{n} by {m}, {apart} apart");
            assert_eq!(width(&held(&b[..], &[]), 0, &held(&a[..], &[]), 0).to_bits(), expected.to_bits(), "{n} by {m}, two cells");
        }
        // An island with no pieces is infinitely far.
        let a = sheet(2, 0.0, 0.0, 0.0, 1);
        let held = Held { lo: [0.0; 3], hi: [1.0; 3], pieces: &a, of_piece: &[] };
        assert_eq!(width(&held, 0, &held, 1), f64::INFINITY);
    }

    #[test]
    fn sets_join_and_find_the_lowest() {
        let mut sets = Sets::new(5);
        assert_eq!(sets.join(3, 4), Some((3, 4)));
        assert_eq!(sets.join(1, 4), Some((1, 3)));
        assert_eq!(sets.join(3, 1), None);
        assert_eq!([0, 1, 2, 3, 4].map(|i| sets.find(i)), [0, 1, 2, 1, 1]);
    }

    #[test]
    fn a_segment_crosses_a_triangle_it_passes_through() {
        let triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(crosses([0.2, 0.2, -1.0], [0.2, 0.2, 1.0], &triangle));
        // Short of it, beside it, and in its plane.
        assert!(!crosses([0.2, 0.2, -1.0], [0.2, 0.2, -0.1], &triangle));
        assert!(!crosses([0.8, 0.8, -1.0], [0.8, 0.8, 1.0], &triangle));
        assert!(!crosses([-1.0, 0.2, 0.0], [2.0, 0.2, 0.0], &triangle));
        // Through an edge counts.
        assert!(crosses([0.5, 0.0, -1.0], [0.5, 0.0, 1.0], &triangle));
    }

    #[test]
    fn the_distance_to_a_triangle_is_to_its_face_edge_or_corner() {
        let triangle = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(close(distance_squared([0.5, 0.5, 3.0], &triangle), 9.0));
        assert!(close(distance_squared([1.0, -2.0, 0.0], &triangle), 4.0));
        assert!(close(distance_squared([-1.0, -1.0, 0.0], &triangle), 2.0));
        assert!(close(distance_squared([2.0, 2.0, 0.0], &triangle), 2.0));
        assert!(close(distance_squared([3.0, 0.0, 4.0], &triangle), 17.0));
    }

    #[test]
    fn a_segment_reaches_a_box_it_passes_through_or_touches() {
        let (lo, hi) = ([0.0; 3], [1.0; 3]);
        assert!(reaches([-1.0, 0.5, 0.5], [2.0, 0.5, 0.5], lo, hi));
        assert!(reaches([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], lo, hi));
        assert!(!reaches([1.5, 0.0, 0.0], [2.0, 0.0, 0.0], lo, hi));
        assert!(!reaches([-1.0, 2.0, 0.5], [2.0, 2.0, 0.5], lo, hi));
    }
}
