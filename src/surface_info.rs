use bytemuck::Pod;
use bytemuck::Zeroable;
use nalgebra::UnitVector3;
use nalgebra::Vector3;
use nalgebra::vector;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SurfaceInfo {
    pub normal: UnitVector3<f32>,
    pub x_offset: f32,
    pub y_offset: f32,
    pub rotation: f32,
    pub x_scale: f32,
    pub y_scale: f32,
    pub _pad: [u32; 32],
    pub _pad2: [u32; 24],
}

impl SurfaceInfo {
    pub(crate) fn default_normal_unchecked(normal: Vector3<f32>) -> Self {
        Self {
            normal: UnitVector3::new_unchecked(normal),
            ..Default::default()
        }
    }
}

impl Default for SurfaceInfo {
    fn default() -> Self {
        Self {
            normal: UnitVector3::new_unchecked(vector![0.0, 0.0, 1.0]),
            x_offset: 0.0,
            y_offset: 0.0,
            rotation: 0.0,
            x_scale: 1.0,
            y_scale: 1.0,
            _pad: [0; 32],
            _pad2: [0; 24],
        }
    }
}
