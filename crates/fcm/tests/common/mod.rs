//! Analytic test parts, generated in code as face-tagged triangle surfaces.
#![allow(dead_code)]

use fieldcraft_mesh::surface::{BoundarySurface, SurfaceOptions};

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Quads with a face tag each, turned so their normals point out of the part.
#[derive(Default)]
struct Quads {
    positions: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    tags: Vec<u32>,
}

impl Quads {
    /// A quad by its corners in order around it; `outward` is any vector out of the part there.
    fn add(&mut self, corners: [[f64; 3]; 4], outward: [f64; 3], tag: u32) {
        let (u, v) = (sub(corners[1], corners[0]), sub(corners[2], corners[0]));
        let normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let along = normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2];
        let corners = if along < 0.0 { [corners[0], corners[3], corners[2], corners[1]] } else { corners };
        let base = self.positions.len() as u32;
        self.positions.extend(corners);
        self.triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]]);
        self.tags.extend([tag, tag]);
    }

    /// A triangle, turned like a quad.
    fn triangle(&mut self, corners: [[f64; 3]; 3], outward: [f64; 3], tag: u32) {
        let (u, v) = (sub(corners[1], corners[0]), sub(corners[2], corners[0]));
        let normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let along = normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2];
        let corners = if along < 0.0 { [corners[0], corners[2], corners[1]] } else { corners };
        let base = self.positions.len() as u32;
        self.positions.extend(corners);
        self.triangles.push([base, base + 1, base + 2]);
        self.tags.push(tag);
    }

    fn surface(self) -> BoundarySurface {
        BoundarySurface::new(&self.positions, &self.triangles, &self.tags, &SurfaceOptions::default()).unwrap()
    }
}

/// The faces of [`block`]: the one at the lower and the upper end of each axis.
pub const X_LO: u32 = 0;
pub const X_HI: u32 = 1;
pub const Y_LO: u32 = 2;
pub const Y_HI: u32 = 3;
pub const Z_LO: u32 = 4;
pub const Z_HI: u32 = 5;

/// A box from `lo` to `hi`.
pub fn block(lo: [f64; 3], hi: [f64; 3]) -> BoundarySurface {
    let mut quads = Quads::default();
    for axis in 0..3 {
        let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
        for (side, end) in [lo[axis], hi[axis]].into_iter().enumerate() {
            let corner = |tb: f64, tc: f64| {
                let mut p = [0.0; 3];
                p[axis] = end;
                p[b] = tb;
                p[c] = tc;
                p
            };
            let mut outward = [0.0; 3];
            outward[axis] = if side == 0 { -1.0 } else { 1.0 };
            quads.add([corner(lo[b], lo[c]), corner(hi[b], lo[c]), corner(hi[b], hi[c]), corner(lo[b], hi[c])], outward, (2 * axis + side) as u32);
        }
    }
    quads.surface()
}

/// The faces of [`quarter_tube`].
pub const INNER: u32 = 0;
pub const OUTER: u32 = 1;
/// The cut in the plane x = 0.
pub const AT_X0: u32 = 2;
/// The cut in the plane y = 0.
pub const AT_Y0: u32 = 3;
pub const AT_Z0: u32 = 4;
pub const AT_Z1: u32 = 5;

/// A quarter of a tube about the z axis, in x ≥ 0 and y ≥ 0, from z = 0 to `length`: its curved
/// faces are `segments` flat strips each.
pub fn quarter_tube(inner: f64, outer: f64, length: f64, segments: usize) -> BoundarySurface {
    // The same expression for a point wherever it is used, so shared vertices are identical;
    // the two ends lie exactly in their planes.
    let point = |r: f64, i: usize, z: f64| {
        if i == 0 {
            [r, 0.0, z]
        } else if i == segments {
            [0.0, r, z]
        } else {
            let angle = std::f64::consts::FRAC_PI_2 * i as f64 / segments as f64;
            [r * angle.cos(), r * angle.sin(), z]
        }
    };
    let mut quads = Quads::default();
    for i in 0..segments {
        let angle = std::f64::consts::FRAC_PI_2 * (i as f64 + 0.5) / segments as f64;
        let radial = [angle.cos(), angle.sin(), 0.0];
        quads.add(
            [point(inner, i, 0.0), point(inner, i + 1, 0.0), point(inner, i + 1, length), point(inner, i, length)],
            [-radial[0], -radial[1], 0.0],
            INNER,
        );
        quads.add([point(outer, i, 0.0), point(outer, i + 1, 0.0), point(outer, i + 1, length), point(outer, i, length)], radial, OUTER);
        for (z, outward, tag) in [(0.0, -1.0, AT_Z0), (length, 1.0, AT_Z1)] {
            quads.add([point(inner, i, z), point(outer, i, z), point(outer, i + 1, z), point(inner, i + 1, z)], [0.0, 0.0, outward], tag);
        }
    }
    for (i, outward, tag) in [(0, [0.0, -1.0, 0.0], AT_Y0), (segments, [-1.0, 0.0, 0.0], AT_X0)] {
        quads.add([point(inner, i, 0.0), point(outer, i, 0.0), point(outer, i, length), point(inner, i, length)], outward, tag);
    }
    quads.surface()
}

/// The inclined face of [`corner`]; its faces in the planes x = 0, y = 0 and z = 0 are
/// [`X_LO`], [`Y_LO`] and [`Z_LO`].
pub const INCLINED: u32 = 6;

/// The tetrahedron between the three coordinate planes and the plane through the points
/// `reach[0]`, `reach[1]` and `reach[2]` on the x, y and z axes.
pub fn corner(reach: [f64; 3]) -> BoundarySurface {
    let (o, x, y, z) = ([0.0; 3], [reach[0], 0.0, 0.0], [0.0, reach[1], 0.0], [0.0, 0.0, reach[2]]);
    let mut faces = Quads::default();
    faces.triangle([o, y, z], [-1.0, 0.0, 0.0], X_LO);
    faces.triangle([o, x, z], [0.0, -1.0, 0.0], Y_LO);
    faces.triangle([o, x, y], [0.0, 0.0, -1.0], Z_LO);
    faces.triangle([x, y, z], [1.0, 1.0, 1.0], INCLINED);
    faces.surface()
}

/// The faces of [`strip`] on its outer curve are tagged by the caller; these are free for it.
pub const OUTER_A: u32 = 6;
pub const OUTER_B: u32 = 7;

/// A plate between two curves in the xy plane, from z = 0 to `thickness`: `curves(i)` gives the
/// points of the inner and the outer curve for i = 0..=segments, which the caller makes exact
/// where they must lie in a plane. The faces are [`INNER`], `outer(i)` for the outer segment
/// from point i, [`AT_Y0`] at i = 0, [`AT_X0`] at i = segments, [`AT_Z0`] and [`AT_Z1`].
pub fn strip(segments: usize, thickness: f64, curves: impl Fn(usize) -> ([f64; 2], [f64; 2]), outer: impl Fn(usize) -> u32) -> BoundarySurface {
    let at = |p: [f64; 2], z: f64| [p[0], p[1], z];
    let mut quads = Quads::default();
    for i in 0..segments {
        let ((a0, b0), (a1, b1)) = (curves(i), curves(i + 1));
        // Out of the plate: away from the inner curve's chord towards the outer one's.
        let across = [0.5 * (b0[0] + b1[0] - a0[0] - a1[0]), 0.5 * (b0[1] + b1[1] - a0[1] - a1[1]), 0.0];
        quads.add([at(a0, 0.0), at(a1, 0.0), at(a1, thickness), at(a0, thickness)], [-across[0], -across[1], 0.0], INNER);
        quads.add([at(b0, 0.0), at(b1, 0.0), at(b1, thickness), at(b0, thickness)], across, outer(i));
        for (z, outward, tag) in [(0.0, -1.0, AT_Z0), (thickness, 1.0, AT_Z1)] {
            quads.add([at(a0, z), at(b0, z), at(b1, z), at(a1, z)], [0.0, 0.0, outward], tag);
        }
    }
    for (i, tag) in [(0, AT_Y0), (segments, AT_X0)] {
        let (a, b) = curves(i);
        let outward = if i == 0 { [0.0, -1.0, 0.0] } else { [-1.0, 0.0, 0.0] };
        quads.add([at(a, 0.0), at(b, 0.0), at(b, thickness), at(a, thickness)], outward, tag);
    }
    quads.surface()
}

/// A plate with a hole: [`strip`] closed on itself, the curves running once round. `outer` tags
/// each outer segment from its two points.
pub fn ring(
    segments: usize,
    thickness: f64,
    curves: impl Fn(usize) -> ([f64; 2], [f64; 2]),
    outer: impl Fn([f64; 2], [f64; 2]) -> u32,
) -> BoundarySurface {
    let at = |p: [f64; 2], z: f64| [p[0], p[1], z];
    let mut quads = Quads::default();
    for i in 0..segments {
        let ((a0, b0), (a1, b1)) = (curves(i), curves((i + 1) % segments));
        let across = [0.5 * (b0[0] + b1[0] - a0[0] - a1[0]), 0.5 * (b0[1] + b1[1] - a0[1] - a1[1]), 0.0];
        quads.add([at(a0, 0.0), at(a1, 0.0), at(a1, thickness), at(a0, thickness)], [-across[0], -across[1], 0.0], INNER);
        quads.add([at(b0, 0.0), at(b1, 0.0), at(b1, thickness), at(b0, thickness)], across, outer(b0, b1));
        for (z, outward, tag) in [(0.0, -1.0, AT_Z0), (thickness, 1.0, AT_Z1)] {
            quads.add([at(a0, z), at(b0, z), at(b1, z), at(a1, z)], [0.0, 0.0, outward], tag);
        }
    }
    quads.surface()
}

/// The faces of [`fork`]: the end of its root, the tips of its lower and upper prongs, and the
/// rest.
pub const ROOT: u32 = 0;
pub const TIP_LOWER: u32 = 1;
pub const TIP_UPPER: u32 = 2;
pub const SKIN: u32 = 3;

/// A fork: a root block from x = 0 to `root` and two prongs of `thickness` from there to
/// `root + length`, the lower from y = 0 and the upper `gap` above it, all from z = 0 to `depth`.
pub fn fork(root: f64, length: f64, thickness: f64, gap: f64, depth: f64) -> BoundarySurface {
    let (tip, top) = (root + length, 2.0 * thickness + gap);
    // The spans along y: the lower prong, the gap, the upper prong.
    let spans = [(0.0, thickness), (thickness, thickness + gap), (thickness + gap, top)];
    let mut quads = Quads::default();
    // A quad in a plane across `axis` at `at`, over [u0, u1] × [v0, v1] of the next two axes.
    let mut face = |axis: usize, at: f64, u: (f64, f64), v: (f64, f64), outward: f64, tag: u32| {
        let corner = |s: f64, t: f64| {
            let mut p = [0.0; 3];
            p[axis] = at;
            p[(axis + 1) % 3] = s;
            p[(axis + 2) % 3] = t;
            p
        };
        let mut normal = [0.0; 3];
        normal[axis] = outward;
        quads.add([corner(u.0, v.0), corner(u.1, v.0), corner(u.1, v.1), corner(u.0, v.1)], normal, tag);
    };
    for (index, span) in spans.into_iter().enumerate() {
        // Across x: the root's end, the end of the gap and the tips.
        face(0, 0.0, span, (0.0, depth), -1.0, ROOT);
        face(0, if index == 1 { root } else { tip }, span, (0.0, depth), 1.0, [TIP_LOWER, SKIN, TIP_UPPER][index]);
        // Across z: the root, and the prongs.
        for (z, outward) in [(0.0, -1.0), (depth, 1.0)] {
            face(2, z, (0.0, root), span, outward, SKIN);
            if index != 1 {
                face(2, z, (root, tip), span, outward, SKIN);
            }
        }
    }
    // Across y: under and over the fork, and the two faces of the gap.
    for (y, outward) in [(0.0, -1.0), (top, 1.0)] {
        face(1, y, (0.0, depth), (0.0, root), outward, SKIN);
        face(1, y, (0.0, depth), (root, tip), outward, SKIN);
    }
    face(1, thickness, (0.0, depth), (root, tip), 1.0, SKIN);
    face(1, thickness + gap, (0.0, depth), (root, tip), -1.0, SKIN);
    quads.surface()
}
