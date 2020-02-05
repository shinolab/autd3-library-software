/*
 * File: mod.rs
 * Project: gain
 * Created Date: 15/11/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 09/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

pub mod bessel_beam_gain;
pub mod csv_gain;
pub mod focal_point_gain;
pub mod grouped_gain;
pub mod holo_gain;
pub mod null_gain;
pub mod plane_wave_gain;

pub use bessel_beam_gain::BesselBeamGain;
pub use csv_gain::CSVGain;
pub use focal_point_gain::FocalPointGain;
pub use grouped_gain::GroupedGain;
pub use holo_gain::HoloGain;
pub use null_gain::NullGain;
pub use plane_wave_gain::PlaneWaveGain;

use crate::geometry::Geometry;
use std::f32::consts::PI;

pub trait Gain: Send {
    fn build(&mut self, geometry: &Geometry);
    fn get_data(&self) -> &Vec<u8>;
}

pub fn convert_to_pwm_params(amp: u8, phase: u8) -> (u8, u8) {
    let d = (amp as f32 / 255.0).asin() / PI; // duty (0 ~ 0.5)
    let s = ((phase as i32 + 64 - (127.5 * d) as i32) % 256) as u8;
    let d = (510.0 * d) as u8;
    (d, s)
}
