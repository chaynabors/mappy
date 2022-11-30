mod geometry;

use std::collections::HashMap;

use glam::Vec2;
use glam::Vec3;
use glam::Vec4;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to parse map: '{0}'")]
    FailedToParse(mappy_parser::Error),
    #[error("an entity in the map is missing a classname")]
    EntityMissingClassName,
}

/// A mappy map, containing all of the geometry, textures, and properties associated with each entity
#[derive(Debug, Default)]
pub struct Map {
    pub entities: Vec<Entity>,
    pub textures: Vec<[u32; 256 * 256]>,
    pub light_maps: Vec<[u32; 4096 * 4096]>,
}

impl Map {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let map_raw =
            mappy_parser::Map::from_bytes(bytes).map_err(|err| Error::FailedToParse(err))?;

        let mut entities = vec![];
        for mut entity in map_raw.entities {
            let name = match entity.properties.remove("classname") {
                Some(name) => name,
                None => return Err(Error::EntityMissingClassName),
            };

            let properties = entity.properties;

            let mut meshes = vec![];
            for brush in entity.brushes {
                let mut mesh = Some(geometry::Mesh::new());
                for half_space in brush.half_spaces {
                    match mesh {
                        Some(unclipped) => mesh = unclipped.clip(half_space),
                        None => break,
                    }
                }

                if let Some(mesh) = mesh {
                    meshes.push(Mesh::from(mesh));
                }
            }

            entities.push(Entity {
                name,
                properties,
                meshes,
            })
        }

        Ok(Self {
            entities,
            textures: vec![],
            light_maps: vec![],
        })
    }
}

#[derive(Debug)]
pub struct Entity {
    pub name: String,
    pub properties: HashMap<String, String>,
    pub meshes: Vec<Mesh>,
}

#[derive(Debug)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub facets: Vec<Facet>,
}

impl From<geometry::Mesh> for Mesh {
    fn from(mesh: geometry::Mesh) -> Self {
        let mut vertices = Vec::with_capacity(mesh.vertices.len());
        let mut map = vec![usize::MAX; mesh.vertices.len()];

        for i in 0..mesh.vertices.len() {
            if mesh.vertices[i].visible {
                map[i] = vertices.len();

                let mut point = mesh.vertices[i].point.as_vec3();
                point = Vec3::new(point.x, point.z, -point.y) / 64.0;
                vertices.push(point.to_array());
            }
        }

        let mut facets = Vec::with_capacity(mesh.facets.len());
        let ordered_facets = mesh.ordered_facets();

        let mut i = 0;
        let mut k = 0;
        while i < ordered_facets.len() {
            let index_count = ordered_facets[i];
            let facet = &mesh.facets[k];

            i += 1;
            k += 1;

            let mut indices = Vec::with_capacity(index_count);
            for _ in 0..index_count {
                indices.push(u16::try_from(map[ordered_facets[i]]).unwrap());
                i += 1;
            }

            facets.push(Facet {
                indices,
                normal: facet.normal.as_vec3().to_array(),
                texture: facet.texture.to_owned(),
                uv_axes: [facet.uv_axes[0].as_vec4(), facet.uv_axes[1].as_vec4()],
                scale: facet.scale.as_vec2(),
            });
        }

        Self { vertices, facets }
    }
}

#[derive(Debug)]
pub struct Facet {
    pub indices: Vec<u16>,
    pub normal: [f32; 3],
    pub texture: String,
    pub uv_axes: [Vec4; 2],
    pub scale: Vec2,
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;

    #[test]
    fn test_unclipped() {
        dbg!(Mesh::from(geometry::Mesh::new()));
    }

    #[test]
    fn test_clipped() {
        dbg!(Mesh::from(
            geometry::Mesh::new()
                .clip(mappy_parser::HalfSpace {
                    point: DVec3::X,
                    normal: DVec3::X,
                    ..Default::default()
                })
                .unwrap()
                .clip(mappy_parser::HalfSpace {
                    point: DVec3::Y,
                    normal: DVec3::Y,
                    ..Default::default()
                })
                .unwrap()
                .clip(mappy_parser::HalfSpace {
                    point: DVec3::Z,
                    normal: DVec3::Z,
                    ..Default::default()
                })
                .unwrap()
                .clip(mappy_parser::HalfSpace {
                    point: DVec3::NEG_X,
                    normal: DVec3::NEG_X,
                    ..Default::default()
                })
                .unwrap()
                .clip(mappy_parser::HalfSpace {
                    point: DVec3::NEG_Y,
                    normal: DVec3::NEG_Y,
                    ..Default::default()
                })
                .unwrap()
                .clip(mappy_parser::HalfSpace {
                    point: DVec3::NEG_Z,
                    normal: DVec3::NEG_Z,
                    ..Default::default()
                })
                .unwrap()
        ));
    }

    #[test]
    fn test_cube() {
        Map::from_bytes(include_bytes!("../../maps/cube.map")).unwrap();
    }
}
