/*
 * File: bessel_beam_gain.rs
 * Project: gain
 * Created Date: 22/11/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 29/11/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * -----
 * The following algorithm is originally developed by Keisuke Hasegawa et al.
 * K. Hasegawa, et al. "Electronically Steerable Ultrasound-Driven Long Narrow Air Stream," Applied Physics Letters, 111, 064104 (2017).
 *
 */

use crate::gain::convert_to_pwm_params;
use crate::gain::Gain;
use crate::geometry::Geometry;
use crate::utils::*;

use crate::consts::{NUM_TRANS_IN_UNIT, ULTRASOUND_WAVELENGTH};

pub struct BesselBeamGain {
    point: Vector3f,
    dir: Vector3f,
    theta_z: f32,
    amp: u8,
    data: Option<Vec<u8>>,
}

impl BesselBeamGain {
    pub fn create(point: Vector3f, dir: Vector3f, theta_z: f32) -> Box<BesselBeamGain> {
        BesselBeamGain::create_with_amp(point, dir, theta_z, 0xff)
    }

    pub fn create_with_amp(
        point: Vector3f,
        dir: Vector3f,
        theta_z: f32,
        amp: u8,
    ) -> Box<BesselBeamGain> {
        Box::new(BesselBeamGain {
            point,
            dir,
            theta_z,
            amp,
            data: None,
        })
    }
}

impl Gain for BesselBeamGain {
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

        let dir = self.dir.normalize();
        let v = Vector3f::new(dir.y, -dir.x, 0.);
        let theta_w = v.norm().asin();
        let point = self.point;
        let theta_z = self.theta_z;
        for i in 0..ntrans {
            let trp = geometry.position(i);
            let r = trp - point;
            let xr = r.cross(&v);
            let r = r * theta_w.cos() + xr * theta_w.sin() + v * (v.dot(&r) * (1. - theta_w.cos()));
            let dist = theta_z.sin() * (r.x * r.x + r.y * r.y).sqrt() - theta_z.cos() * r.z;
            let fphase = (dist % ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
            let amp: u8 = self.amp;
            let phase = (255.0 * (1.0 - fphase)) as u8;
            let (d, s) = convert_to_pwm_params(amp, phase);
            data.push(s);
            data.push(d);
        }
        self.data = Some(data);
    }
}
