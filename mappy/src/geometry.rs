use glam::DVec2;
use glam::DVec3;
use glam::DVec4;
use mappy_parser::HalfSpace;

const MAP_SIZE: f64 = 32768.0;
const HALF_SIZE: f64 = MAP_SIZE * 0.5;
const MAP_EPSILON: f64 = 0.015625;

/// Mesh is an intermediate struct
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Mesh {
    pub(crate) vertices: Vec<Vertex>,
    pub(crate) edges: Vec<Edge>,
    pub(crate) facets: Vec<Facet>,
}

impl Mesh {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn clip(mut self, half_space: HalfSpace) -> Option<Self> {
        let mut clip_count = 0;
        for vertex in &mut self.vertices {
            if vertex.visible {
                vertex.distance = half_space.normal.dot(vertex.point - half_space.point);

                // The point is not within the halfspace
                if vertex.distance >= MAP_EPSILON {
                    clip_count += 1;
                    vertex.visible = false;
                } else if vertex.distance > -MAP_EPSILON {
                    vertex.distance = 0.0;
                }
            }
        }

        match clip_count {
            c if c == 0 => return Some(self),
            c if c == self.vertices.len() => return None,
            _ => (),
        };

        for i in 0..self.edges.len() {
            if self.edges[i].visible {
                let d0 = self.vertices[self.edges[i].vertices[0]].distance;
                let d1 = self.vertices[self.edges[i].vertices[1]].distance;

                if d0 >= 0.0 && d1 >= 0.0 {
                    for facet in self.edges[i].facets {
                        self.facets[facet].edges.retain(|edge| *edge != i);
                        if self.facets[facet].edges.is_empty() {
                            self.facets[facet].visible = false;
                        }
                    }

                    self.edges[i].visible = false;
                    continue;
                }

                if d0 <= 0.0 && d1 <= 0.0 {
                    continue;
                }

                let t = d0 / (d0 - d1);
                let t1 = (1.0 - t) * self.vertices[self.edges[i].vertices[0]].point;
                let t2 = t * self.vertices[self.edges[i].vertices[1]].point;
                let intersection = t1 + t2;

                let index = self.vertices.len();
                self.vertices.push(intersection.into());

                match d0 < 0.0 {
                    true => self.edges[i].vertices[1] = index,
                    false => self.edges[i].vertices[0] = index,
                }
            }
        }

        let closing_index = self.facets.len();
        self.facets.push(Facet::from(half_space));

        for i in 0..self.facets.len() {
            if self.facets[i].visible {
                for &edge in &self.facets[i].edges {
                    self.vertices[self.edges[edge].vertices[0]].occurs = 0;
                    self.vertices[self.edges[edge].vertices[1]].occurs = 0;
                }

                for &edge in &self.facets[i].edges {
                    self.vertices[self.edges[edge].vertices[0]].occurs += 1;
                    self.vertices[self.edges[edge].vertices[1]].occurs += 1;
                }

                let mut start = usize::MAX;
                let mut finish = usize::MAX;
                for &edge in &self.facets[i].edges {
                    let i0 = self.edges[edge].vertices[0];
                    let i1 = self.edges[edge].vertices[1];

                    if self.vertices[i0].occurs == 1 {
                        if start == usize::MAX {
                            start = i0;
                        } else if finish == usize::MAX {
                            finish = i0;
                        }
                    }

                    if self.vertices[i1].occurs == 1 {
                        if start == usize::MAX {
                            start = i1;
                        } else if finish == usize::MAX {
                            finish = i1;
                        }
                    }
                }

                if start != usize::MAX {
                    let edge_index = self.edges.len();
                    self.edges
                        .push(Edge::new([start, finish], [i, closing_index]));

                    self.facets[i].edges.push(edge_index);
                    self.facets[closing_index].edges.push(edge_index);
                }
            }
        }

        Some(self)
    }

    /// The facets of the mesh with ordered vertices
    ///
    /// Facet order will remain the same
    pub(crate) fn ordered_facets(&self) -> Vec<usize> {
        let mut facets = vec![];
        for facet in &self.facets {
            if facet.visible {
                let vertices = self.ordered_vertices(facet);
                facets.push(vertices.len() - 1);

                if facet.normal.dot(self.normal_from_vertices(&vertices)) < 0.0 {
                    for i in (0..=vertices.len() - 2).rev() {
                        facets.push(vertices[i]);
                    }
                } else {
                    for i in 0..=vertices.len() - 2 {
                        facets.push(vertices[i])
                    }
                }
            }
        }

        facets
    }

    fn ordered_vertices(&self, facet: &Facet) -> Vec<usize> {
        let mut edges = facet.edges.clone();

        let mut choice = 1;
        for i0 in 0..edges.len() - 2 {
            let i1 = i0 + 1;
            let current = self.edges[edges[i0]].vertices[choice];

            for j in i1..edges.len() {
                if self.edges[edges[j]].vertices[0] == current {
                    edges.swap(i1, j);
                    choice = 1;
                    break;
                }

                if self.edges[edges[j]].vertices[1] == current {
                    edges.swap(i1, j);
                    choice = 0;
                    break;
                }
            }
        }

        let mut vertices = Vec::with_capacity(edges.len() + 1);
        vertices.push(self.edges[edges[0]].vertices[0]);
        vertices.push(self.edges[edges[0]].vertices[1]);

        for i in 1..edges.len() {
            if self.edges[edges[i]].vertices[0] == vertices[i] {
                vertices.push(self.edges[edges[i]].vertices[1]);
            } else {
                vertices.push(self.edges[edges[i]].vertices[0]);
            }
        }

        vertices
    }

    /// NOTE: the first and last vertices must be the same
    fn normal_from_vertices(&self, vertices: &[usize]) -> DVec3 {
        let mut normal = DVec3::ZERO;
        for i in 0..vertices.len() - 2 {
            normal += self.vertices[vertices[i]]
                .point
                .cross(self.vertices[vertices[i + 1]].point)
        }

        normal.normalize()
    }
}

impl Default for Mesh {
    #[rustfmt::skip]
    fn default() -> Self {
        Self {
            vertices: vec![
                Vertex::new( HALF_SIZE,  HALF_SIZE,  HALF_SIZE), // 0
                Vertex::new( HALF_SIZE,  HALF_SIZE, -HALF_SIZE), // 1
                Vertex::new( HALF_SIZE, -HALF_SIZE,  HALF_SIZE), // 2
                Vertex::new( HALF_SIZE, -HALF_SIZE, -HALF_SIZE), // 3
                Vertex::new(-HALF_SIZE,  HALF_SIZE,  HALF_SIZE), // 4
                Vertex::new(-HALF_SIZE,  HALF_SIZE, -HALF_SIZE), // 5
                Vertex::new(-HALF_SIZE, -HALF_SIZE,  HALF_SIZE), // 6
                Vertex::new(-HALF_SIZE, -HALF_SIZE, -HALF_SIZE), // 7
            ],
            edges: vec![
                Edge::new([0, 1], [0, 1]), //  0
                Edge::new([0, 2], [0, 2]), //  1
                Edge::new([0, 4], [1, 2]), //  2
                Edge::new([1, 3], [0, 5]), //  3
                Edge::new([1, 5], [1, 5]), //  4
                Edge::new([2, 3], [0, 4]), //  5
                Edge::new([2, 6], [2, 4]), //  6
                Edge::new([3, 7], [4, 5]), //  7
                Edge::new([4, 5], [1, 3]), //  8
                Edge::new([4, 6], [2, 3]), //  9
                Edge::new([5, 7], [3, 5]), // 10
                Edge::new([6, 7], [3, 4]), // 11
            ],
            facets: vec![
                Facet::new(vec![ 0,  3,  5,  1], DVec3::X),     // 0
                Facet::new(vec![ 0,  4,  8,  2], DVec3::Y),     // 1
                Facet::new(vec![ 1,  2,  9,  6], DVec3::Z),     // 2
                Facet::new(vec![ 8, 10, 11,  9], DVec3::NEG_X), // 3
                Facet::new(vec![ 5,  6, 11,  7], DVec3::NEG_Y), // 4
                Facet::new(vec![ 3,  7, 10,  4], DVec3::NEG_Z), // 5
            ],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Vertex {
    pub(crate) point: DVec3,
    pub(crate) distance: f64,
    pub(crate) occurs: usize,
    pub(crate) visible: bool,
}

impl Vertex {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            point: DVec3::new(x, y, z),
            distance: 0.0,
            occurs: 0,
            visible: true,
        }
    }
}

impl From<DVec3> for Vertex {
    fn from(point: DVec3) -> Self {
        Self {
            point,
            distance: 0.0,
            occurs: 0,
            visible: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Edge {
    pub(crate) vertices: [usize; 2],
    pub(crate) facets: [usize; 2],
    pub(crate) visible: bool,
}

impl Edge {
    fn new(vertices: [usize; 2], facets: [usize; 2]) -> Self {
        Self {
            vertices,
            facets,
            visible: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Facet {
    pub(crate) edges: Vec<usize>,
    pub(crate) normal: DVec3,
    pub(crate) texture: String,
    pub(crate) uv_axes: [DVec4; 2],
    pub(crate) scale: DVec2,
    pub(crate) visible: bool,
}

impl Facet {
    fn new(edges: Vec<usize>, normal: DVec3) -> Self {
        Self {
            edges,
            normal,
            visible: true,
            ..Default::default()
        }
    }
}

impl From<HalfSpace> for Facet {
    fn from(half_space: HalfSpace) -> Self {
        Self {
            edges: vec![],
            normal: half_space.normal,
            texture: half_space.texture,
            uv_axes: half_space.uv_axes,
            scale: half_space.scale,
            visible: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_all() {
        let mesh = Mesh::new();
        assert_eq!(
            mesh.clip(HalfSpace {
                point: DVec3::NEG_Z * MAP_SIZE,
                normal: DVec3::Z,
                ..Default::default()
            }),
            None
        );
    }

    #[test]
    fn clip_none() {
        let poly1 = Mesh::new()
            .clip(HalfSpace {
                point: DVec3::Z * MAP_SIZE,
                normal: DVec3::Z,
                ..Default::default()
            })
            .unwrap();

        let poly2 = Mesh::new();

        for v in 0..poly1.vertices.len() {
            assert_eq!(poly1.vertices[v].point, poly2.vertices[v].point);
            assert_eq!(poly1.vertices[v].visible, poly2.vertices[v].visible);
        }

        for e in 0..poly1.edges.len() {
            assert_eq!(poly1.edges[e], poly2.edges[e]);
        }

        for f in 0..poly1.facets.len() {
            assert_eq!(poly1.facets[f], poly2.facets[f]);
        }
    }

    #[test]
    fn clip() {
        let mesh = Mesh::new().clip(HalfSpace {
            point: DVec3::Z,
            normal: DVec3::Z,
            ..Default::default()
        });

        assert_eq!(
            mesh,
            Some(Mesh {
                vertices: vec![
                    Vertex {
                        point: DVec3::new(HALF_SIZE, HALF_SIZE, HALF_SIZE),
                        distance: HALF_SIZE - 1.0,
                        occurs: 0,
                        visible: false,
                    },
                    Vertex {
                        point: DVec3::new(HALF_SIZE, HALF_SIZE, -HALF_SIZE),
                        distance: -HALF_SIZE - 1.0,
                        occurs: 2,
                        visible: true,
                    },
                    Vertex {
                        point: DVec3::new(HALF_SIZE, -HALF_SIZE, HALF_SIZE),
                        distance: HALF_SIZE - 1.0,
                        occurs: 0,
                        visible: false,
                    },
                    Vertex {
                        point: DVec3::new(HALF_SIZE, -HALF_SIZE, -HALF_SIZE),
                        distance: -HALF_SIZE - 1.0,
                        occurs: 2,
                        visible: true,
                    },
                    Vertex {
                        point: DVec3::new(-HALF_SIZE, HALF_SIZE, HALF_SIZE),
                        distance: HALF_SIZE - 1.0,
                        occurs: 0,
                        visible: false,
                    },
                    Vertex {
                        point: DVec3::new(-HALF_SIZE, HALF_SIZE, -HALF_SIZE),
                        distance: -HALF_SIZE - 1.0,
                        occurs: 2,
                        visible: true,
                    },
                    Vertex {
                        point: DVec3::new(-HALF_SIZE, -HALF_SIZE, HALF_SIZE),
                        distance: HALF_SIZE - 1.0,
                        occurs: 0,
                        visible: false,
                    },
                    Vertex {
                        point: DVec3::new(-HALF_SIZE, -HALF_SIZE, -HALF_SIZE),
                        distance: -HALF_SIZE - 1.0,
                        occurs: 2,
                        visible: true,
                    },
                    Vertex {
                        point: DVec3::new(HALF_SIZE, HALF_SIZE, 1.0),
                        distance: 0.0,
                        occurs: 2,
                        visible: true,
                    },
                    Vertex {
                        point: DVec3::new(HALF_SIZE, -HALF_SIZE, 1.0),
                        distance: 0.0,
                        occurs: 2,
                        visible: true,
                    },
                    Vertex {
                        point: DVec3::new(-HALF_SIZE, HALF_SIZE, 1.0),
                        distance: 0.0,
                        occurs: 2,
                        visible: true,
                    },
                    Vertex {
                        point: DVec3::new(-HALF_SIZE, -HALF_SIZE, 1.0),
                        distance: 0.0,
                        occurs: 2,
                        visible: true,
                    },
                ],
                edges: vec![
                    Edge {
                        vertices: [8, 1],
                        facets: [0, 1],
                        visible: true,
                    },
                    Edge {
                        vertices: [0, 2],
                        facets: [0, 2],
                        visible: false,
                    },
                    Edge {
                        vertices: [0, 4],
                        facets: [1, 2],
                        visible: false,
                    },
                    Edge {
                        vertices: [1, 3],
                        facets: [0, 5],
                        visible: true,
                    },
                    Edge {
                        vertices: [1, 5],
                        facets: [1, 5],
                        visible: true,
                    },
                    Edge {
                        vertices: [9, 3],
                        facets: [0, 4],
                        visible: true,
                    },
                    Edge {
                        vertices: [2, 6],
                        facets: [2, 4],
                        visible: false,
                    },
                    Edge {
                        vertices: [3, 7],
                        facets: [4, 5],
                        visible: true,
                    },
                    Edge {
                        vertices: [10, 5],
                        facets: [1, 3],
                        visible: true,
                    },
                    Edge {
                        vertices: [4, 6],
                        facets: [2, 3],
                        visible: false,
                    },
                    Edge {
                        vertices: [5, 7],
                        facets: [3, 5],
                        visible: true,
                    },
                    Edge {
                        vertices: [11, 7],
                        facets: [3, 4],
                        visible: true,
                    },
                    Edge {
                        vertices: [8, 9],
                        facets: [0, 6],
                        visible: true,
                    },
                    Edge {
                        vertices: [8, 10],
                        facets: [1, 6],
                        visible: true,
                    },
                    Edge {
                        vertices: [10, 11],
                        facets: [3, 6],
                        visible: true,
                    },
                    Edge {
                        vertices: [9, 11],
                        facets: [4, 6],
                        visible: true,
                    },
                ],
                facets: vec![
                    Facet {
                        edges: vec![0, 3, 5, 12],
                        normal: DVec3::new(1.0, 0.0, 0.0),
                        texture: "".to_owned(),
                        uv_axes: [
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                        ],
                        scale: DVec2::new(0.0, 0.0),
                        visible: true,
                    },
                    Facet {
                        edges: vec![0, 4, 8, 13],
                        normal: DVec3::new(0.0, 1.0, 0.0),
                        texture: "".to_owned(),
                        uv_axes: [
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                        ],
                        scale: DVec2::new(0.0, 0.0),
                        visible: true,
                    },
                    Facet {
                        edges: vec![],
                        normal: DVec3::new(0.0, 0.0, 1.0),
                        texture: "".to_owned(),
                        uv_axes: [
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                        ],
                        scale: DVec2::new(0.0, 0.0),
                        visible: false,
                    },
                    Facet {
                        edges: vec![8, 10, 11, 14],
                        normal: DVec3::new(-1.0, 0.0, 0.0),
                        texture: "".to_owned(),
                        uv_axes: [
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                        ],
                        scale: DVec2::new(0.0, 0.0),
                        visible: true,
                    },
                    Facet {
                        edges: vec![5, 11, 7, 15],
                        normal: DVec3::new(0.0, -1.0, 0.0),
                        texture: "".to_owned(),
                        uv_axes: [
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                        ],
                        scale: DVec2::new(0.0, 0.0),
                        visible: true,
                    },
                    Facet {
                        edges: vec![3, 7, 10, 4],
                        normal: DVec3::new(0.0, 0.0, -1.0),
                        texture: "".to_owned(),
                        uv_axes: [
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                        ],
                        scale: DVec2::new(0.0, 0.0),
                        visible: true,
                    },
                    Facet {
                        edges: vec![12, 13, 14, 15],
                        normal: DVec3::new(0.0, 0.0, 1.0),
                        texture: "".to_owned(),
                        uv_axes: [
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                            DVec4::new(0.0, 0.0, 0.0, 0.0),
                        ],
                        scale: DVec2::new(0.0, 0.0),
                        visible: true,
                    },
                ],
            }),
        );
    }

    #[test]
    fn clip_twice() {
        // sanity test
        Mesh::new()
            .clip(HalfSpace {
                point: DVec3::ZERO,
                normal: DVec3::Z,
                ..Default::default()
            })
            .unwrap()
            .clip(HalfSpace {
                point: DVec3::ZERO,
                normal: DVec3::Y,
                ..Default::default()
            });
    }

    #[test]
    fn ordered() {
        Mesh::new()
            .clip(HalfSpace {
                point: DVec3::ZERO,
                normal: DVec3::Z,
                ..Default::default()
            })
            .unwrap();
    }
}
