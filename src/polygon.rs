use std::cmp::Ordering;

use float_cmp::approx_eq;
use nalgebra::point;
use nalgebra::vector;

use crate::Facet;
use crate::HalfSpace;
use crate::SurfaceInfo;
use crate::map::MAP_EPSILON;

#[derive(Clone, Debug)]
pub struct Polygon {
    pub facets: Vec<Facet>,
}

impl Polygon {
    pub fn new(facets: Vec<Facet>) -> Self {
        Self { facets }
    }

    pub fn cube(size: f32) -> Self {
        Self {
            facets: vec![
                Facet {
                    vertices: vec![
                        point![-size,  size, -size],
                        point![ size,  size, -size],
                        point![ size, -size, -size],
                        point![-size, -size, -size],
                    ],
                    surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, 0.0, -1.0]),
                },
                Facet {
                    vertices: vec![
                        point![-size,  size,  size],
                        point![ size,  size,  size],
                        point![ size, -size,  size],
                        point![-size, -size,  size],
                    ],
                    surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, 0.0, 1.0]),
                },
                Facet {
                    vertices: vec![
                        point![-size,  size,  size],
                        point![ size,  size,  size],
                        point![ size,  size, -size],
                        point![-size,  size, -size],
                    ],
                    surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, 1.0, 0.0]),
                },
                Facet {
                    vertices: vec![
                        point![-size, -size,  size],
                        point![ size, -size,  size],
                        point![ size, -size, -size],
                        point![-size, -size, -size],
                    ],
                    surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, -1.0, 0.0]),
                },
                Facet {
                    vertices: vec![
                        point![ size,  size,  size],
                        point![ size,  size, -size],
                        point![ size, -size, -size],
                        point![ size, -size,  size],
                    ],
                    surface_info: SurfaceInfo::default_normal_unchecked(vector![1.0, 0.0, 0.0]),
                },
                Facet {
                    vertices: vec![
                        point![-size,  size,  size],
                        point![-size,  size, -size],
                        point![-size, -size, -size],
                        point![-size, -size,  size],
                    ],
                    surface_info: SurfaceInfo::default_normal_unchecked(vector![-1.0, 0.0, 0.0]),
                },
            ],
        }
    }

    /// returns the resulting geometry
    pub fn clip(self, half_space: HalfSpace) -> Option<Self> {
        let mut facets = vec![];
        let mut generated = vec![];
        for facet in self.facets {
            if let Some((mut new, facet)) = facet.clip(half_space) {
                facets.push(facet);
                generated.append(&mut new);
            }
        }

        // remove duplicate vertices
        // TODO: use total_cmp when it stabilizes
        generated.sort_by(|a, b| {
            if !approx_eq!(f32, a.x, b.x, epsilon = MAP_EPSILON) { return a.x.partial_cmp(&b.x).unwrap(); }
            if !approx_eq!(f32, a.y, b.y, epsilon = MAP_EPSILON) { return a.y.partial_cmp(&b.y).unwrap(); }
            if !approx_eq!(f32, a.z, b.z, epsilon = MAP_EPSILON) { return a.z.partial_cmp(&b.z).unwrap(); }
            Ordering::Equal
        });

        generated.dedup_by(|a, b| {
            let x = approx_eq!(f32, a.x, b.x, epsilon = MAP_EPSILON);
            let y = approx_eq!(f32, a.y, b.y, epsilon = MAP_EPSILON);
            let z = approx_eq!(f32, a.z, b.z, epsilon = MAP_EPSILON);
            x && y && z
        });

        // fill the hole left by the cutting plane
        if let Some(facet) = Facet::from_points_unchecked(&generated, half_space) {
            facets.push(facet);
        }

        match facets.is_empty() {
            true => None,
            false => Some(Polygon::new(facets)),
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::point;
    use nalgebra::vector;

    use crate::HalfSpace;
    use crate::SurfaceInfo;

    use super::Polygon;

    #[test]
    fn clip_all() {
        let cube = Polygon::cube(100.0);
        let cube = cube.clip(HalfSpace::positive_y());
        assert!(cube.is_some());
        let cube = cube.unwrap();
        let cube = cube.clip(HalfSpace {
            point: point![0.0, 0.01, 0.0],
            surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, -1.0, 0.0]),
        });
        assert!(cube.is_none());

        let cube = Polygon::cube(100.0);
        let cube = cube.clip(HalfSpace::positive_z());
        assert!(cube.is_some());
        let cube = cube.unwrap();
        let cube = cube.clip(HalfSpace {
            point: point![0.0, 0.0, 0.1],
            surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, 1.0, -1.0]),
            ..Default::default()
        });
        assert!(cube.is_none());
    }
}
