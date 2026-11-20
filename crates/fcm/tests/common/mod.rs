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
