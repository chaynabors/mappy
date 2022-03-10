pub extern crate bytemuck as bm;
pub extern crate nalgebra as na;

mod error;
mod facet;
mod half_space;
mod map;
mod polygon;
mod spatial;
mod surface_info;

pub use error::Error;
pub use map::Map;
pub use surface_info::SurfaceInfo;

use facet::Facet;
use half_space::HalfSpace;
use spatial::Spatial;

pub type Property<'a> = (&'a str, &'a str);
