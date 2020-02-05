/*
 * File: focal_point_gain.rs
 * Project: gain
 * Created Date: 15/11/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 26/11/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::gain::convert_to_pwm_params;
use crate::gain::Gain;
use crate::geometry::Geometry;
use crate::utils::*;

use crate::consts::{NUM_TRANS_IN_UNIT, ULTRASOUND_WAVELENGTH};

pub struct FocalPointGain {
    point: Vector3f,
    amp: u8,
    data: Option<Vec<u8>>,
}

impl FocalPointGain {
    pub fn create(point: Vector3f) -> Box<FocalPointGain> {
        FocalPointGain::create_with_amp(point, 0xff)
    }

    pub fn create_with_amp(point: Vector3f, amp: u8) -> Box<FocalPointGain> {
        Box::new(FocalPointGain {
            point,
            amp,
            data: None,
        })
    }
}

impl Gain for FocalPointGain {
    fn get_data(&self) -> &Vec<u8> {
        assert!(self.data.is_some());
        match &self.data {
            Some(data) => data,
            None => panic!(),
        }
    }

    fn build(&mut self, geometry: &Geometry) {
        if self.data.is_some() {
            return;
        }

        let ndevice = geometry.num_devices();
        let ntrans = NUM_TRANS_IN_UNIT * ndevice;
        let mut data = Vec::with_capacity(ntrans * 2);

        let point = self.point;
        let amp = self.amp;

        for i in 0..ntrans {
            let trp = geometry.position(i);
            let dist = (trp - point).norm();
            let fphase = (dist % ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
            let amp: u8 = amp;
            let phase = (255.0 * (1.0 - fphase)) as u8;
            let (d, s) = convert_to_pwm_params(amp, phase);
            data.push(s);
            data.push(d);
        }
        self.data = Some(data);
    }
}
