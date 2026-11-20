//! Whether a point lies inside a closed, outward-oriented triangle surface, decided with exact
//! predicates.
//!
//! This is the test of the lattice Boltzmann voxeliser (`fieldcraft_lbm::voxel`) for one point at
//! a time: quadrature points do not sit on grid lines. A point `p` is inside when the winding
//! number of the surface around it is positive, counted along the ray from `p` in +x: every
//! triangle the ray crosses adds the sign of its normal's x component. Whether the ray's line
//! meets a triangle is the sign of three [`orient2d`] tests in the (y, z) projection; whether the
//! point lies before the triangle's plane is the sign of [`orient3d`].
//!
//! Degenerate positions (a line through a projected edge or vertex, a point on a triangle's
//! plane) are resolved by one symbolic perturbation of the point, `p + (ε, δ, δ²)` with
//! infinitesimals `1 ≫ ε ≫ δ` (H. Edelsbrunner, E. P. Mücke, *Simulation of Simplicity*, ACM
//! Trans. Graph. 9(1):66–104, 1990), so a watertight surface gives a consistent count and the
//! answer never depends on rounding. Triangles parallel to x are met by no perturbed line.
//!
//! The triangles are sorted into a uniform grid of buckets over their (y, z) bounding box, so a
//! query visits only the triangles whose projection can hold the point.

use fieldcraft_mesh::predicates::{Point2, Point3, Sign, orient2d, orient3d, snap_to_domain};
use fieldcraft_mesh::surface::BoundarySurface;

use crate::{Error, Result};

/// Buckets along each of y and z, at most.
const MAX_BUCKETS: usize = 256;
/// Triangle references over all buckets, at most (a surface of a few huge triangles and many
/// small ones stays far below this).
const MAX_REFERENCES: usize = 200_000_000;

struct Triangle {
    corners: [Point3; 3],
    projected: [Point2; 3],
    /// The orientation of the projection: the sign of the normal's x component.
    sign: Sign,
    lo: [f64; 2],
    hi: [f64; 2],
}

/// A closed surface prepared for inside tests.
pub struct Solid {
    triangles: Vec<Triangle>,
    origin: [f64; 2],
    step: [f64; 2],
    count: [usize; 2],
    buckets: Vec<Vec<u32>>,
}

/// The sign of `orient2d(a, b, p)` for `p` perturbed by `(δ, δ²)` in its two coordinates.
/// `orient2d` is affine in `p` with gradient `(a_v − b_v, b_u − a_u)`, so when it is zero the
/// sign is that of the first nonzero gradient component; both are zero only when `a = b`.
#[inline]
fn edge_sign(a: &Point2, b: &Point2, p: &Point2) -> Sign {
    let s = orient2d(a, b, p);
    if !s.is_zero() {
        return s;
    }
    let ([au, av], [bu, bv]) = (a.coords(), b.coords());
    if av != bv { Sign::of(av - bv) } else { Sign::of(bu - au) }
}

impl Solid {
    pub fn new(surface: &BoundarySurface) -> Result<Solid> {
        let mut triangles = Vec::with_capacity(surface.triangles().len());
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for t in surface.triangles() {
            let corner = |v: u32| surface.point3(v).ok_or_else(|| Error::Invalid(format!("surface vertex {v} is missing")));
            let corners = [corner(t[0])?, corner(t[1])?, corner(t[2])?];
            let project = |p: &Point3| -> Result<Point2> {
                let [_, y, z] = p.coords();
                Ok(Point2::new(y, z)?)
            };
            let projected = [project(&corners[0])?, project(&corners[1])?, project(&corners[2])?];
            let sign = orient2d(&projected[0], &projected[1], &projected[2]);
            if sign.is_zero() {
                continue;
            }
            let (mut tlo, mut thi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for q in &projected {
                let c = q.coords();
                for a in 0..2 {
                    tlo[a] = tlo[a].min(c[a]);
                    thi[a] = thi[a].max(c[a]);
                    lo[a] = lo[a].min(c[a]);
                    hi[a] = hi[a].max(c[a]);
                }
            }
            triangles.push(Triangle { corners, projected, sign, lo: tlo, hi: thi });
        }
        let per_axis = ((triangles.len() as f64).sqrt().ceil() as usize).clamp(1, MAX_BUCKETS);
        let mut solid = Solid { triangles, origin: [0.0; 2], step: [1.0; 2], count: [1; 2], buckets: Vec::new() };
        for a in 0..2 {
            let extent = hi[a] - lo[a];
            if extent > 0.0 && extent.is_finite() {
                solid.origin[a] = lo[a];
                solid.count[a] = per_axis;
                solid.step[a] = extent / per_axis as f64;
            }
        }
        let mut buckets = vec![Vec::new(); solid.count[0] * solid.count[1]];
        let mut references = 0usize;
        for (t, triangle) in solid.triangles.iter().enumerate() {
            let (j0, j1) = (solid.bucket(0, triangle.lo[0]), solid.bucket(0, triangle.hi[0]));
            let (k0, k1) = (solid.bucket(1, triangle.lo[1]), solid.bucket(1, triangle.hi[1]));
            references = references.saturating_add((j1 + 1 - j0) * (k1 + 1 - k0));
            if references > MAX_REFERENCES {
                return Err(Error::Limit(format!(
                    "the surface's triangles overlap too much in projection for the inside test ({MAX_REFERENCES} references)"
                )));
            }
            for k in k0..=k1 {
                for j in j0..=j1 {
                    if let Some(bucket) = buckets.get_mut(j + solid.count[0] * k) {
                        bucket.push(t as u32);
                    }
                }
            }
        }
        solid.buckets = buckets;
        Ok(solid)
    }

    /// The bucket index along axis `a` (0: y, 1: z) of a coordinate, clamped into the grid.
    fn bucket(&self, a: usize, x: f64) -> usize {
        let t = (x - self.origin[a]) / self.step[a];
        if !(t > 0.0) { 0 } else { (t as usize).min(self.count[a] - 1) }
    }

    /// Whether `p` is inside the surface. Fails only when a coordinate is outside the domain of
    /// the exact predicates (not finite, or of a magnitude no model has).
    pub fn contains(&self, p: [f64; 3]) -> Result<bool> {
        let [x, y, z] = [snap_to_domain(p[0]), snap_to_domain(p[1]), snap_to_domain(p[2])];
        let line = Point2::new(y, z)?;
        let point = Point3::new(x, y, z)?;
        let Some(bucket) = self.buckets.get(self.bucket(0, y) + self.count[0] * self.bucket(1, z)) else {
            return Ok(false);
        };
        let mut winding = 0i32;
        for &t in bucket {
            let Some(triangle) = self.triangles.get(t as usize) else { continue };
            if y < triangle.lo[0] || y > triangle.hi[0] || z < triangle.lo[1] || z > triangle.hi[1] {
                continue;
            }
            let [a, b, c] = &triangle.projected;
            let s = triangle.sign;
            if edge_sign(a, b, &line) != s || edge_sign(b, c, &line) != s || edge_sign(c, a, &line) != s {
                continue;
            }
            // Before the plane, the +x ray crosses the triangle: orient3d(a, b, c, p) has the
            // sign s there. A point on the plane is perturbed by +ε in x and counts as beyond it.
            let [a, b, c] = &triangle.corners;
            if orient3d(a, b, c, &point) == s {
                winding += s as i32;
            }
        }
        Ok(winding > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fieldcraft_mesh::surface::SurfaceOptions;

    /// The unit cube with outward triangles.
    fn cube() -> BoundarySurface {
        let p =
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]];
        let quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]];
        let triangles: Vec<[u32; 3]> = quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect();
        BoundarySurface::new(&p, &triangles, &[], &SurfaceOptions::default()).unwrap()
    }

    #[test]
    fn points_of_a_cube() {
        let solid = Solid::new(&cube()).unwrap();
        for p in [[0.5, 0.5, 0.5], [0.01, 0.99, 0.5], [0.999, 0.001, 0.001]] {
            assert!(solid.contains(p).unwrap(), "{p:?} is inside");
        }
        for p in [[1.5, 0.5, 0.5], [-0.1, 0.5, 0.5], [0.5, 1.01, 0.5], [0.5, 0.5, -0.01], [2.0, 2.0, 2.0]] {
            assert!(!solid.contains(p).unwrap(), "{p:?} is outside");
        }
    }

    /// Rays through the cube's diagonal edges and its corners, and points on its faces: every
    /// answer follows the perturbation, and no crossing is counted twice or lost.
    #[test]
    fn degenerate_rays_count_once() {
        let solid = Solid::new(&cube()).unwrap();
        // The line y = z runs through the diagonal edge of the faces across x.
        assert!(solid.contains([0.5, 0.5, 0.5]).unwrap());
        assert!(solid.contains([0.5, 0.25, 0.25]).unwrap());
        assert!(!solid.contains([1.5, 0.25, 0.25]).unwrap());
        assert!(!solid.contains([-0.5, 0.25, 0.25]).unwrap());
        // On a face: the perturbation moves the point by +ε in x, then (δ, δ²) in (y, z).
        assert!(solid.contains([0.0, 0.5, 0.5]).unwrap(), "the x = 0 face moves inside");
        assert!(!solid.contains([1.0, 0.5, 0.5]).unwrap(), "the x = 1 face moves outside");
        assert!(solid.contains([0.5, 0.0, 0.5]).unwrap(), "the y = 0 face moves inside");
        assert!(!solid.contains([0.5, 1.0, 0.5]).unwrap(), "the y = 1 face moves outside");
        // A corner of the bounding box and a point far away never panic.
        assert!(solid.contains([0.0, 0.0, 0.0]).unwrap());
        assert!(!solid.contains([1e30, -1e30, 1e30]).unwrap());
        assert!(solid.contains([f64::NAN, 0.5, 0.5]).is_err());
    }
}
