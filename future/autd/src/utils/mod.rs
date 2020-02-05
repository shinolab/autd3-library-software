/*
 * File: mod.rs
 * Project: autd
 * Created Date: 02/09/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

extern crate na;

pub mod directivity;

pub type Vector3f = na::Vec3<f32>;
pub type Quaternionf = na::Quaternion<f32>;
pub type Matrix4x4f = na::Matrix4<f32>;

pub(crate) fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if max < value {
        max
    } else {
        value
    }
}

pub fn is_missing_transducer(x: usize, y: usize) -> bool {
    y == 1 && (x == 1 || x == 2 || x == 16)
}

pub(crate) fn tr(trans: Vector3f, rot: Quaternionf) -> Matrix4x4f {
    let coord = rot.coords;
    let x = coord.x;
    let y = coord.y;
    let z = coord.z;
    let w = coord.w;
    Matrix4x4f::new(
        1.0 - 2.0 * y * y - 2.0 * z * z,
        2.0 * x * y - 2.0 * w * z,
        2.0 * x * z + 2.0 * w * y,
        trans.x,
        2.0 * x * y + 2.0 * w * z,
        1.0 - 2.0 * x * x - 2.0 * z * z,
        2.0 * y * z - 2.0 * w * x,
        trans.y,
        2.0 * x * z - 2.0 * w * y,
        2.0 * y * z + 2.0 * w * x,
        1.0 - 2.0 * x * x - 2.0 * y * y,
        trans.z,
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

pub(crate) fn angle_axis(angle: f32, dir: Vector3f) -> Quaternionf {
    Quaternionf::new(
        (angle / 2.0).cos(),
        dir.x * (angle / 2.0).sin(),
        dir.y * (angle / 2.0).sin(),
        dir.z * (angle / 2.0).sin(),
    )
}

pub(crate) fn convert_to_vec3(vec4: na::Vec4<f32>) -> Vector3f {
    Vector3f::new(vec4.x, vec4.y, vec4.z)
}
