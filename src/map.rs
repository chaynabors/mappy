use nalgebra::Point3;
use nom::Finish;
use nom::error::VerboseError;

use crate::Error;
use crate::SurfaceInfo;

pub const MAP_EPSILON: f32 = 0.125;

#[derive(Debug, Default)]
pub struct Map<'a> {
    /// The number of entities in the map
    pub entity_count: u32,
    /// The number of properties an individual entity has
    pub property_counts: Vec<u32>,
    /// A contiguous chunk of memory representing all of the properties for every entity
    ///
    /// To get all the properties for a single entity, see `property_counts`
    pub properties: Vec<(&'a str, &'a str)>,
    /// The number of facets in each entity
    pub facet_counts: Vec<u32>,
    /// The number of vertices in each facet
    pub vertex_counts: Vec<u32>,
    /// A contiguous chunk of memory representing all of the vertices in the map3
    ///
    /// Vertices are ordered in such a way to allow CCW triangle strip rasterization
    pub vertices: Vec<Point3<f32>>,
    /// Surface info for each facet, including normal and texture data
    ///
    /// Individual surfaces directly correspond with each facet defined by the vertex_counts field
    pub surface_info: Vec<SurfaceInfo>,
    /// The file name of each texture needed to display the map
    ///
    /// This name does not include the path or extension, and it's expected
    /// that these are in a known location.
    pub textures: Vec<&'a str>,
}

impl<'a> Map<'a> {
    pub fn from_str(s: &'a str) -> Result<Map, Error> {
        let (_, map) = parser::map::<VerboseError<&str>>(s).finish()?;
        Ok(map)
    }
}

mod parser {
    use nalgebra::UnitVector3;
    use nalgebra::Vector3;
    use nom::IResult;
    use nom::branch::alt;
    use nom::bytes::complete::tag;
    use nom::bytes::complete::take_until;
    use nom::bytes::complete::take_while1;
    use nom::character::complete::char;
    use nom::character::complete::multispace1;
    use nom::character::complete::not_line_ending;
    use nom::combinator::map as mapp;
    use nom::error::ContextError;
    use nom::error::ParseError;
    use nom::multi::many0;
    use nom::number::complete::float;
    use nom::sequence::delimited;
    use nom::sequence::preceded;
    use nom::sequence::separated_pair;
    use nom::sequence::terminated;
    use nom::sequence::tuple;

    use crate::HalfSpace;
    use crate::Property;
    use crate::SurfaceInfo;
    use crate::polygon::Polygon;

    use super::Map;

    fn comment<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str,
    ) -> IResult<&'a str, &'a str, E> {
        preceded(tag("//"), not_line_ending)(i)
    }

    fn ws<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str,
    ) -> IResult<&'a str, Vec<&'a str>, E> {
        many0(alt((multispace1, comment)))(i)
    }

    fn string<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str
    ) -> IResult<&'a str, &'a str, E> {
        delimited(char('"'), take_until("\""), char('"'))(i)
    }

    fn property<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str
    ) -> IResult<&'a str, (&'a str, &'a str), E> {
        separated_pair(string, ws, string)(i)
    }

    fn point<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str
    ) -> IResult<&'a str, Vector3<f32>, E> {
        mapp(
            tuple((char('('), ws, float, ws, float, ws, float, ws, char(')'))),
            |(_, _, x, _, y, _, z, _, _)| Vector3::new(x, z, y)
        )(i)
    }

    fn texture<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str
    ) -> IResult<&'a str, &'a str, E> {
        take_while1(|c: char| c == '_' || c.is_alphabetic())(i)
    }

    fn half_space<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str
    ) -> IResult<&'a str, (HalfSpace, &'a str), E> {
        mapp(
            tuple((point, ws, point, ws, point, ws, texture, ws,
                float, ws, float, ws, float, ws, float, ws, float)),
            |(a, _, b, _, c, _, tex, _, x, _, y, _, r, _, sx, _, sy)| {
                let half_space = HalfSpace {
                    point: a.into(),
                    surface_info: SurfaceInfo {
                        normal: UnitVector3::new_normalize((b - a).cross(&(c - a))),
                        x_offset: x,
                        y_offset: y,
                        rotation: r,
                        x_scale: sx,
                        y_scale: sy,
                        _pad: [0; 32],
                        _pad2: [0; 24],
                    },
                };

                (half_space, tex)
            },
        )(i)
    }

    fn polygon<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str
    ) -> IResult<&'a str, Option<(Polygon, Vec<&'a str>)>, E> {
        let (i, _) = terminated(char('{'), ws)(i)?;
        let (i, (half_spaces, textures)): (&'a str, (Vec<HalfSpace>, Vec<&'a str>)) = mapp(
            many0(terminated(half_space, ws)),
            |hst| hst.into_iter().unzip()
        )(i)?;
        let (i, _) = char('}')(i)?;

        let size = half_spaces.iter().fold(0.0, |mut size: f32, half_space| {
            size = size.max(half_space.point.x.abs());
            size = size.max(half_space.point.y.abs());
            size.max(half_space.point.z.abs())
        });

        let mut polygon = match size {
            size if size > f32::EPSILON => Some(Polygon::cube(size)),
            _ => None,
        };

        for half_space in half_spaces {
            polygon = match polygon {
                Some(polygon) => polygon.clip(half_space),
                None => None,
            }
        }

        Ok((i, match polygon {
            Some(polygon) => Some((polygon, textures)),
            None => None,
        }))
    }

    fn entity<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str
    ) -> IResult<&'a str, (Vec<Property>, Vec<Polygon>, Vec<&'a str>), E> {
        let mut properties = vec![];
        let mut polygons = vec![];
        let mut textures = Vec::new();
        let (i, _) = delimited(
            terminated(char('{'), ws),
            many0(terminated(
                alt((
                    mapp(property, |(key, value)| { properties.push((key, value)); }),
                    mapp(polygon, |polygon| if let Some((polygon, new_textures)) = polygon {
                        polygons.push(polygon);
                        textures.extend(new_textures);
                    }),
                )),
                ws,
            )),
            char('}'),
        )(i)?;
        Ok((i, (properties, polygons, textures)))
    }

    pub fn map<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
        i: &'a str
    ) -> IResult<&'a str, Map, E> {
        let mut map = Map::default();
        let (i, _) = ws(i)?;
        let (i, _) = many0(mapp(terminated(entity, ws), |(properties, polygons, textures)| {
            map.entity_count += 1;
            map.property_counts.push(properties.len() as _);
            map.properties.extend(properties);
            for polygon in &polygons {
                map.facet_counts.push(polygon.facets.len() as u32);
                for facet in &polygon.facets {
                    map.vertex_counts.push(facet.vertices.len() as u32);
                    // Order the vertices to be drawn as a triangle strip
                    map.vertices.push(facet.vertices[1]);
                    map.vertices.push(facet.vertices[0]);
                    map.vertices.push(facet.vertices[2]);
                    // Push any additional vertices
                    for i in 0..facet.vertices.len() - 3 {
                        let even_offset = (i / 2 + 3) * (i % 2);
                        let odd_offset = (facet.vertices.len() - 1 - i / 2) * ((i + 1) % 2);
                        map.vertices.push(facet.vertices[even_offset + odd_offset]);
                    }

                    map.surface_info.push(facet.surface_info);
                }
            }

            map.textures.extend(textures);
        }))(i)?;
        let (i, _) = ws(i)?;
        Ok((i, map))
    }
}

#[cfg(test)]
mod tests {
    use core::panic;

    use crate::Map;

    #[test]
    fn map_empty() {
        test_map(&String::default());
    }

    #[test]
    fn map_entity() {
        test_map(&"{}");
    }

    #[test]
    fn map_property() {
        test_map(&"{\"hello\" \"world\"}");
    }

    #[test]
    fn map_brush() {
        test_map(&"{{ ( 0 0 0 ) ( 0 0 1 ) ( 0 -1 0 ) __TB_empty 0 0 90 1 1 }}");
    }

    #[test]
    fn map_cube() {
        test_map(include_str!("maps/cube.map"));
    }

    #[test]
    fn map_pyramid() {
        test_map(include_str!("maps/pyramid.map"));
    }

    #[test]
    fn map_cylinder() {
        test_map(include_str!("maps/cylinder.map"));
    }

    /// This is nothing more than a sanity test and only catches panics
    #[test]
    fn map_cabin() {
        test_map(include_str!("maps/cabin.map"));
    }

    fn test_map(map: &str) {
        match Map::from_str(map) {
            Ok(map) => println!("{map:?}"),
            Err(e) => panic!("{e}"),
        }
    }
}
