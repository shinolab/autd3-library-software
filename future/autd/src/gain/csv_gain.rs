/*
 * File: csv_gain.rs
 * Project: gain
 * Created Date: 02/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::gain::convert_to_pwm_params;
use crate::gain::Gain;
use crate::geometry::Geometry;
use failure::{Error, Fail};
use std::ffi::OsString;
use std::fs::File;

use crate::consts::NUM_TRANS_IN_UNIT;

#[derive(Fail, Debug)]
pub enum CVSGainError {
    #[fail(
        display = "The file must consist of two columns, normalized amp and phase, with delimiter ','."
    )]
    ParseError,
}

pub struct CSVGain {
    data: Option<Vec<u8>>,
}

impl CSVGain {
    pub fn create(file_path: &OsString) -> Result<Box<CSVGain>, Error> {
        let mut data = Vec::new();
        let file = File::open(file_path)?;
        let mut rdr = csv::Reader::from_reader(file);
        for result in rdr.records() {
            let record = result?;
            if record.len() != 2 {
                return Err(CVSGainError::ParseError.into());
            }
            let amp: f32 = record[0].parse()?;
            let phase: f32 = record[1].parse()?;
            let amp = (amp * 255.0) as u8;
            let phase = (phase * 255.0) as u8;
            let (d, s) = convert_to_pwm_params(amp, phase);
            data.push(s);
            data.push(d);
        }
        Ok(Box::new(CSVGain { data: Some(data) }))
    }
}

impl Gain for CSVGain {
    fn get_data(&self) -> &Vec<u8> {
        assert!(self.data.is_some());
        match &self.data {
            Some(data) => data,
            None => panic!(),
        }
    }

    fn build(&mut self, geometry: &Geometry) {
        let ndevice = geometry.num_devices();
        let ntrans = NUM_TRANS_IN_UNIT * ndevice;

        if let Some(data) = &mut self.data {
            data.resize(ntrans * 2, 0);
        }
    }
}
