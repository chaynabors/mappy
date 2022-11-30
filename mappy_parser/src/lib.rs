use std::collections::HashMap;

use glam::DQuat;
use glam::DVec2;
use glam::DVec3;
use glam::DVec4;
use glam::Vec3Swizzles;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_until;
use nom::bytes::complete::take_while1;
use nom::character::complete::char;
use nom::character::complete::multispace0;
use nom::character::complete::multispace1;
use nom::character::complete::not_line_ending;
use nom::combinator::map;
use nom::error::ContextError;
use nom::error::ParseError;
use nom::error::VerboseError;
use nom::multi::many0;
use nom::multi::many_m_n;
use nom::number::complete::double;
use nom::sequence::delimited;
use nom::sequence::preceded;
use nom::sequence::separated_pair;
use nom::sequence::terminated;
use nom::sequence::tuple;
use nom::Finish;
use nom::IResult;

pub type Error = String;

const BASE_AXES: [DVec3; 18] = [
    DVec3::new(0.0, 0.0, 1.0),
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(0.0, -1.0, 0.0),
    DVec3::new(0.0, 0.0, -1.0),
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(0.0, -1.0, 0.0),
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(0.0, 0.0, -1.0),
    DVec3::new(-1.0, 0.0, 0.0),
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(0.0, 0.0, -1.0),
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(0.0, 0.0, -1.0),
    DVec3::new(0.0, -1.0, 0.0),
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(0.0, 0.0, -1.0),
];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Map {
    pub entities: Vec<Entity>,
}

impl Map {
    pub fn from_bytes(bytes: &[u8]) -> Result<Map, Error> {
        let (_, map) = parse_map::<VerboseError<&str>>(&String::from_utf8_lossy(bytes))
            .finish()
            .map_err(|err| err.to_string())?;
        Ok(map)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Entity {
    pub properties: HashMap<String, String>,
    pub brushes: Vec<Brush>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Brush {
    pub half_spaces: Vec<HalfSpace>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HalfSpace {
    pub point: DVec3,
    pub normal: DVec3,
    pub texture: String,
    pub uv_axes: [DVec4; 2],
    pub scale: DVec2,
}

fn parse_map<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, Map, E> {
    delimited(ws, map(many0(entity), |entities| Map { entities }), ws)(i)
}

fn entity<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, Entity, E> {
    let mut properties = HashMap::new();
    let mut brushes = vec![];

    let (i, _) = delimited(
        terminated(char('{'), ws),
        many0(terminated(
            alt((
                map(brush, |brush| brushes.push(brush)),
                map(separated_pair(string, multispace1, string), |(k, v)| {
                    properties.insert(k, v);
                }),
            )),
            ws,
        )),
        char('}'),
    )(i)?;

    Ok((
        i,
        Entity {
            properties,
            brushes,
        },
    ))
}

fn brush<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, Brush, E> {
    delimited(
        terminated(char('{'), multispace0),
        map(
            many_m_n(
                4,
                usize::MAX,
                terminated(alt((standard_half_space, valve_half_space)), multispace1),
            ),
            |half_spaces| Brush { half_spaces },
        ),
        char('}'),
    )(i)
}

fn standard_half_space<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, HalfSpace, E> {
    map(
        tuple((
            point,
            multispace1,
            point,
            multispace1,
            point,
            multispace1,
            texture,
            multispace1,
            double,
            multispace1,
            double,
            multispace1,
            double,
            multispace1,
            double,
            multispace1,
            double,
        )),
        |(a, _, b, _, c, _, tex, _, x, _, y, _, r, _, sx, _, sy)| {
            let normal = -(b - a).cross(c - a).normalize();

            let mut best_index = 0;
            let mut best_dot = 0.0;
            for i in 0..6 {
                let dot = normal.dot(BASE_AXES[i * 3]);
                if dot > best_dot {
                    best_dot = dot;
                    best_index = i;
                }
            }

            let projection_axis = BASE_AXES[(best_index / 2) * 6];
            let rotation = DQuat::from_axis_angle(projection_axis, r);

            let mut x_axis = (rotation * BASE_AXES[best_index * 3 + 1]).xyzx();
            x_axis.w = x;

            let mut y_axis = (rotation * BASE_AXES[best_index * 3 + 2]).xyzx();
            y_axis.w = y;

            HalfSpace {
                point: a,
                normal,
                texture: tex.to_owned(),
                uv_axes: [x_axis, y_axis],
                scale: DVec2::new(sx, sy),
            }
        },
    )(i)
}

fn valve_half_space<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, HalfSpace, E> {
    map(
        tuple((
            point,
            multispace1,
            point,
            multispace1,
            point,
            multispace1,
            texture,
            multispace1,
            uv_axis,
            multispace1,
            uv_axis,
            multispace1,
            double,
            multispace1,
            double,
            multispace1,
            double,
        )),
        |(a, _, b, _, c, _, tex, _, t1, _, t2, _, _, _, sx, _, sy)| HalfSpace {
            point: a,
            normal: (c - a).cross(b - a).normalize(),
            texture: tex.to_owned(),
            uv_axes: [t1, t2],
            scale: DVec2::new(sx, sy),
        },
    )(i)
}

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
    i: &'a str,
) -> IResult<&'a str, String, E> {
    map(
        delimited(char('"'), take_until("\""), char('"')),
        |str: &str| str.to_owned(),
    )(i)
}

fn point<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, DVec3, E> {
    map(
        tuple((
            char('('),
            multispace0,
            double,
            multispace1,
            double,
            multispace1,
            double,
            multispace0,
            char(')'),
        )),
        |(_, _, x, _, y, _, z, _, _)| DVec3::new(x, y, z),
    )(i)
}

fn texture<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, &'a str, E> {
    take_while1(|c: char| c == '_' || c.is_alphanumeric())(i)
}

fn uv_axis<'a, E: ParseError<&'a str> + ContextError<&'a str>>(
    i: &'a str,
) -> IResult<&'a str, DVec4, E> {
    map(
        tuple((
            char('['),
            multispace0,
            double,
            multispace1,
            double,
            multispace1,
            double,
            multispace1,
            double,
            multispace0,
            char(']'),
        )),
        |(_, _, tx, _, ty, _, tz, _, toffset, _, _)| DVec4::new(tx, ty, tz, toffset),
    )(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_empty() {
        assert_eq!(Map::from_bytes(&[]).unwrap(), Map::default());
    }

    #[test]
    fn map_entity() {
        assert_eq!(
            Map::from_bytes("{}".as_bytes()).unwrap(),
            Map {
                entities: vec![Entity::default()]
            }
        );
    }

    #[test]
    fn map_property() {
        assert_eq!(
            Map::from_bytes("{\"hello\" \"world\"}".as_bytes()).unwrap(),
            Map {
                entities: vec![Entity {
                    properties: HashMap::from([("hello".to_owned(), "world".to_owned())]),
                    brushes: vec![],
                }]
            }
        );
    }

    #[test]
    fn map_brush() {
        let half_space = HalfSpace {
            point: DVec3::new(0.0, 0.0, 0.0),
            normal: DVec3::Z,
            texture: "__TB_empty".to_owned(),
            uv_axes: [
                DVec4::new(1.0, 0.0, 0.0, 0.0),
                DVec4::new(0.0, -1.0, 0.0, 0.0),
            ],
            scale: DVec2::new(1.0, 1.0),
        };

        assert_eq!(
            Map::from_bytes(
                "{{
                    ( 0 0 0 ) ( 1 0 0 ) ( 0 1 0 ) __TB_empty 0 0 0 1 1
                    ( 0 0 0 ) ( 1 0 0 ) ( 0 1 0 ) __TB_empty 0 0 0 1 1
                    ( 0 0 0 ) ( 1 0 0 ) ( 0 1 0 ) __TB_empty 0 0 0 1 1
                    ( 0 0 0 ) ( 1 0 0 ) ( 0 1 0 ) __TB_empty 0 0 0 1 1
                }}"
                .as_bytes(),
            )
            .unwrap(),
            Map {
                entities: vec![Entity {
                    properties: HashMap::default(),
                    brushes: vec![Brush {
                        half_spaces: vec![
                            half_space.clone(),
                            half_space.clone(),
                            half_space.clone(),
                            half_space
                        ]
                    }],
                }]
            }
        );
    }

    #[test]
    fn map_cube() {
        let map = Map::from_bytes(include_bytes!("../../maps/cube.map")).unwrap();
        assert!(map.entities.len() > 0);
        assert!(map.entities[0].brushes.len() > 0);
        assert!(map.entities[0].brushes[0].half_spaces.len() > 3);
    }

    #[test]
    fn map_pyramid() {
        Map::from_bytes(include_bytes!("../../maps/pyramid.map")).unwrap();
    }

    #[test]
    fn map_cylinder() {
        Map::from_bytes(include_bytes!("../../maps/cylinder.map")).unwrap();
    }
}
