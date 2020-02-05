/*
 * File: null_gain.rs
 * Project: gain
 * Created Date: 19/11/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 02/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::gain::Gain;
use crate::geometry::Geometry;

use crate::consts::NUM_TRANS_IN_UNIT;

pub struct NullGain {
    data: Option<Vec<u8>>,
}

impl NullGain {
    pub fn create() -> Box<NullGain> {
        Box::new(NullGain { data: None })
    }
}

impl Gain for NullGain {
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
        self.data = Some(vec![0x00; ntrans * 2]);
    }
}
