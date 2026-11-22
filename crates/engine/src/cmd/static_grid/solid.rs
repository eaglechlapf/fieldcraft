//! A body the grid solves: one of the open design, or a surface mesh that was imported
//! (`geometry.import.mesh`: a closed surface whose triangles carry face tags, from any CAD
//! system). The solver takes a body's surface as triangles with a tag for every face, which is
//! what both are; a design's body is triangulated as finely as is asked, an imported mesh is as
//! fine as its file, and its faces are named by their tags (`"0"`, `"1"`, …), as its sets name
//! them.

use std::borrow::Cow;

use fieldcraft_geometry::tagged_mesh::TaggedSurface;
use fieldcraft_geometry::{FaceSurface, Measure, Shape, SurfaceMesh};

use crate::{EngineError, Result, Session};

/// Where a body comes from.
#[derive(Clone, Copy)]
enum Source<'a> {
    Design(&'a Shape),
    Mesh(&'a TaggedSurface),
}

/// A body by its name, with what the grid solve asks of it.
#[derive(Clone, Copy)]
pub(in crate::cmd) struct Solid<'a> {
    pub name: &'a str,
    source: Source<'a>,
}

impl<'a> Solid<'a> {
    pub fn of_design(shape: &'a Shape) -> Solid<'a> {
        Solid { name: &shape.name, source: Source::Design(shape) }
    }

    pub fn of_mesh(surface: &'a TaggedSurface) -> Solid<'a> {
        Solid { name: &surface.name, source: Source::Mesh(surface) }
    }

    /// Every body of the session: the open design's, then the imported surfaces.
    pub fn all(s: &'a Session) -> Vec<Solid<'a>> {
        let design = s.design.iter().flat_map(|design| &design.bodies).map(Solid::of_design);
        design.chain(s.surfaces.iter().map(|surface| Solid::of_mesh(surface))).collect()
    }

    /// Whether the body is an imported surface mesh.
    pub fn imported(&self) -> bool {
        matches!(self.source, Source::Mesh(_))
    }

    /// The body's box, volume and area, in the geometry's unit.
    pub fn measure(&self) -> Result<Measure> {
        match self.source {
            Source::Design(shape) => Ok(shape.measure()?),
            Source::Mesh(surface) => {
                let mesh = &surface.mesh;
                let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
                for p in mesh.positions.iter().filter(|p| p.iter().all(|x| x.is_finite())) {
                    for a in 0..3 {
                        lo[a] = lo[a].min(p[a]);
                        hi[a] = hi[a].max(p[a]);
                    }
                }
                if !(0..3).all(|a| lo[a] <= hi[a]) {
                    return Err(EngineError::Params(format!("body `{}` has no points", self.name.chars().take(100).collect::<String>())));
                }
                // About the box's middle, so that the sums stay small beside their terms.
                let middle = [0, 1, 2].map(|a| 0.5 * (lo[a] + hi[a]));
                let (mut volume, mut area) = (0.0, 0.0);
                for triangle in &mesh.triangles {
                    let corner = |i: u32| mesh.positions.get(i as usize).map(|p| [0, 1, 2].map(|a| p[a] - middle[a]));
                    let (Some(a), Some(b), Some(c)) = (corner(triangle[0]), corner(triangle[1]), corner(triangle[2])) else { continue };
                    let (u, v) = ([0, 1, 2].map(|k| b[k] - a[k]), [0, 1, 2].map(|k| c[k] - a[k]));
                    let cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
                    area += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                    volume += (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0;
                }
                Ok(Measure { volume: volume.abs(), area, centroid: middle, bbox_min: lo, bbox_max: hi, faces: surface.faces as usize })
            }
        }
    }

    /// The body's surface as triangles, within `tolerance` (a length of the geometry) of it:
    /// an imported mesh as it is.
    pub fn tessellate(&self, tolerance: f64) -> Result<Cow<'a, SurfaceMesh>> {
        match self.source {
            Source::Design(shape) => Ok(Cow::Owned(shape.tessellate(tolerance)?)),
            Source::Mesh(surface) => Ok(Cow::Borrowed(&surface.mesh)),
        }
    }

    /// The names of the body's faces, by the tag its triangles carry.
    pub fn face_names(&self) -> Vec<String> {
        match self.source {
            Source::Design(shape) => shape.faces().iter().map(|face| face.name.clone()).collect(),
            Source::Mesh(surface) => (0..surface.faces).map(|face| face.to_string()).collect(),
        }
    }

    /// The name of the face with tag `face`.
    pub fn face_name(&self, face: u32) -> Option<String> {
        match self.source {
            Source::Design(shape) => shape.faces().get(face as usize).map(|face| face.name.clone()),
            Source::Mesh(surface) => (face < surface.faces).then(|| face.to_string()),
        }
    }

    /// The tag of the face called `name`.
    pub fn face_index(&self, name: &str) -> Option<u32> {
        match self.source {
            Source::Design(shape) => shape.face_index(name),
            // A tag as it is written, and no other spelling of the number.
            Source::Mesh(surface) => name.parse::<u32>().ok().filter(|face| *face < surface.faces && face.to_string() == name),
        }
    }

    /// The surfaces the faces lie on, by tag, where they are known: an imported mesh's are not.
    pub fn face_surfaces(&self) -> Result<Vec<Option<FaceSurface>>> {
        match self.source {
            Source::Design(shape) => Ok(shape.face_surfaces()?),
            Source::Mesh(_) => Ok(Vec::new()),
        }
    }
}

#[cfg(test)]
pub(in crate::cmd) mod tests {
    use super::*;

    /// A box from `lo` to `hi`, two triangles a face, each face with its tag: 0 and 1 the faces
    /// across x at `lo` and at `hi`, 2 and 3 across y, 4 and 5 across z.
    pub(in crate::cmd) fn block_between(name: &str, lo: [f64; 3], hi: [f64; 3]) -> TaggedSurface {
        let at = |i: usize| [0, 1, 2].map(|a| if i & (1 << a) == 0 { lo[a] } else { hi[a] });
        let quads: [[u32; 4]; 6] = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];
        let (mut triangles, mut tri_face) = (Vec::new(), Vec::new());
        for (tag, q) in quads.iter().enumerate() {
            triangles.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
            tri_face.extend([tag as u32, tag as u32]);
        }
        let mesh = SurfaceMesh { positions: (0..8).map(at).collect(), triangles, tri_face, ..SurfaceMesh::default() };
        TaggedSurface { name: name.into(), mesh, faces: 6, original: None }
    }

    /// A box from the origin to (2, 3, 4).
    fn block(name: &str) -> TaggedSurface {
        block_between(name, [0.0; 3], [2.0, 3.0, 4.0])
    }

    #[test]
    fn an_imported_surface_is_a_body_with_its_tags_for_face_names() {
        let surface = block("Block");
        let solid = Solid::of_mesh(&surface);
        assert!(solid.imported() && solid.name == "Block");
        let measure = solid.measure().unwrap();
        assert_eq!((measure.bbox_min, measure.bbox_max, measure.faces), ([0.0; 3], [2.0, 3.0, 4.0], 6));
        assert!((measure.volume - 24.0).abs() < 1e-12 && (measure.area - 52.0).abs() < 1e-12, "{} and {}", measure.volume, measure.area);
        assert_eq!(solid.face_names(), ["0", "1", "2", "3", "4", "5"]);
        assert_eq!((solid.face_name(5).as_deref(), solid.face_name(6)), (Some("5"), None));
        assert_eq!((solid.face_index("3"), solid.face_index("6"), solid.face_index("03"), solid.face_index("top")), (Some(3), None, None, None));
        assert!(solid.face_surfaces().unwrap().is_empty());
        // As it is, whatever tolerance is asked for.
        assert_eq!(solid.tessellate(1e-9).unwrap().triangles.len(), 12);
        // A surface with no point has no size; indices out of range are passed over.
        let empty = TaggedSurface { name: "None".into(), ..TaggedSurface::default() };
        assert!(Solid::of_mesh(&empty).measure().is_err());
        let mut broken = block("Broken");
        broken.mesh.triangles.push([0, 1, 99]);
        assert!((Solid::of_mesh(&broken).measure().unwrap().volume - 24.0).abs() < 1e-12);
    }
}
