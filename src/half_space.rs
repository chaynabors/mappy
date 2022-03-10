use nalgebra::Point3;
use nalgebra::vector;

use crate::SurfaceInfo;

#[derive(Clone, Copy, Debug)]
pub struct HalfSpace {
    pub point: Point3<f32>,
    pub surface_info: SurfaceInfo,
}

impl HalfSpace {
    /// A half space centered about the origin pointing toward positive y
    ///
    /// All points with a negative y value are considered within the half space
    #[allow(dead_code)]
    pub(crate) fn positive_y() -> Self {
        Self {
            point: Point3::default(),
            surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, 1.0, 0.0]),
        }
    }

    /// A half space centered about the origin pointing toward negative y
    ///
    /// All points with a positive y value are considered within the half space
    #[allow(dead_code)]
    pub(crate) fn negative_y() -> Self {
        Self {
            point: Point3::default(),
            surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, -1.0, 0.0]),
        }
    }

    /// A half space centered about the origin pointing toward positive z
    ///
    /// All points with a negative z value are considered within the half space
    #[allow(dead_code)]
    pub(crate) fn positive_z() -> Self {
        Self {
            point: Point3::default(),
            surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, 0.0, 1.0]),
        }
    }

    /// A half space centered about the origin pointing toward negative z
    ///
    /// All points with a positive z value are considered within the half space
    #[allow(dead_code)]
    pub(crate) fn negative_z() -> Self {
        Self {
            point: Point3::default(),
            surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, 0.0, -1.0]),
        }
    }
}

impl Default for HalfSpace {
    fn default() -> Self {
        Self {
            point: Point3::origin(),
            surface_info: SurfaceInfo::default_normal_unchecked(vector![0.0, 0.0, 1.0]),
        }
    }
}
