use nalgebra::Point3;

use crate::HalfSpace;
use crate::map::MAP_EPSILON;

pub trait Spatial {
    fn inside_halfspace(&self, half_space: HalfSpace) -> bool;
}

impl Spatial for Point3<f32> {
    fn inside_halfspace(&self, half_space: HalfSpace) -> bool {
        (self - half_space.point).dot(&half_space.surface_info.normal) < MAP_EPSILON
    }
}

#[cfg(test)]
mod test {
    use nalgebra::point;

    use crate::HalfSpace;

    use super::Spatial;

    #[test]
    fn inside_halfspace() {
        let hsp = HalfSpace::positive_y();

        // test inside
        assert_eq!(
            point![0.0, -1.0, 0.0].inside_halfspace(hsp),
            true,
        );

        // test outside
        assert_eq!(
            point![0.0, 1.0, 0.0].inside_halfspace(hsp),
            false,
        );

        // test on
        assert_eq!(
            point![0.0, 0.0, 0.0].inside_halfspace(hsp),
            true,
        )
    }
}
