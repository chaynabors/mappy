use std::collections::VecDeque;

use nalgebra::Point3;
use nalgebra::UnitVector3;
use nalgebra::point;

use crate::HalfSpace;
use crate::Spatial;
use crate::SurfaceInfo;

#[derive(Clone, Debug)]
pub struct Facet {
    pub vertices: Vec<Point3<f32>>,
    pub surface_info: SurfaceInfo,
}

impl Facet {
    /// produces a planar convex hull from a set of exterior points
    pub fn from_points_unchecked(
        points: &[Point3<f32>],
        half_space: HalfSpace,
    ) -> Option<Self> {
        if points.len() < 3 { return None; }
        if points.len() == 3 {
            return Some(Self {
                vertices: points.to_owned(),
                surface_info: half_space.surface_info,
            });
        }

        let mut points = VecDeque::from(points.to_owned());
        let mut vertices: Vec<Point3<f32>> = vec![];
        let mut p1 = points.pop_front().unwrap();
        let mut p2 = p1 + half_space.surface_info.normal.as_ref();
        let mut p3 = points.pop_front().unwrap();
        while points.len() != 0 {
            let normal = UnitVector3::new_normalize((p3 - p1).cross(&(p2 - p1)));
            let hsp = HalfSpace { point: p1, surface_info: SurfaceInfo::default_normal_unchecked(*normal) };

            let mut count = vertices.iter().fold(0, |acc, p| acc + p.inside_halfspace(hsp) as u32);
            count += points.iter().fold(0, |acc, p| acc + p.inside_halfspace(hsp) as u32);
            if count == 0 || count == (vertices.len() + points.len()) as u32 {
                // we have an exterior edge
                vertices.push(p1);
                p1 = p3;
                p2 = p1 + half_space.surface_info.normal.as_ref();
                p3 = points.pop_front().unwrap();
                continue;
            }

            points.push_back(p3);
            p3 = points.pop_front().unwrap();
        }

        vertices.push(p1);
        vertices.push(p3);

        Some(Facet {
            vertices,
            surface_info: half_space.surface_info,
        })
    }

    /// returns the resulting facet and any generated vertices
    pub fn clip(mut self, half_space: HalfSpace) -> Option<(Vec<Point3<f32>>, Self)> {
        // handle the facet being entirely inside or outside the clipping plane
        let inner_count = self.vertices.iter().fold(0, |cnt, vertex| cnt + vertex.inside_halfspace(half_space) as u32);
        match inner_count {
            i if i == 0 => return None,
            i if i == self.vertices.len() as u32 => return Some((vec![], self)),
            _ => ()
        };

        // clip the facet
        let mut generated = vec![];
        for i in (1..=self.vertices.len()).rev() {
            let p0 = self.vertices[i - 1];
            let p1 = self.vertices[i % self.vertices.len()];
            let dot = half_space.surface_info.normal.dot(&(p1 - p0));
            if dot != 0.0 {
                let f = -half_space.surface_info.normal.dot(&(p0 - half_space.point)) / dot;
                let vertex = p0 + (p1 - p0) * f;
                if f >= 0.0 && f <= 1.0 {
                    generated.push(vertex);
                    if f > 0.0 && f < 1.0 { self.vertices.insert(i, vertex); }
                }
            }
        }

        // remove vertices outside the halfspace
        self.vertices.retain(|vertex| vertex.inside_halfspace(half_space));

        match self.vertices.len() < 3 {
            true => None,
            false => Some((generated, self)),
        }
    }
}

impl Default for Facet {
    fn default() -> Self {
        Self {
            vertices: vec![
                point![-1.0,  1.0, 0.0],
                point![ 1.0,  1.0, 0.0],
                point![ 1.0, -1.0, 0.0],
                point![-1.0, -1.0, 0.0],
            ],
            surface_info: SurfaceInfo::default(),
        }
    }
}

impl PartialEq for Facet {
    fn eq(&self, other: &Self) -> bool {
        self.vertices == other.vertices
    }
}

#[cfg(test)]
mod test {
    use nalgebra::point;

    use crate::HalfSpace;

    use super::Facet;

    #[test]
    fn clip() {
        let hsp = HalfSpace::positive_y();
        let (generated, facet) = Facet::default().clip(hsp).unwrap();

        assert_eq!(
            generated,
            vec![
                point![-1.0,  0.0, 0.0],
                point![ 1.0,  0.0, 0.0],
            ],
        );

        assert_eq!(
            facet,
            Facet {
                vertices: vec![
                    point![ 1.0,  0.0, 0.0],
                    point![ 1.0, -1.0, 0.0],
                    point![-1.0, -1.0, 0.0],
                    point![-1.0,  0.0, 0.0],
                ],
                ..Default::default()
            },
        );

        let mut hsp = HalfSpace::positive_y();
        hsp.point = point![0.0, -f32::MAX, 0.0];
        assert_eq!(Facet::default().clip(hsp), None);
    }
}
