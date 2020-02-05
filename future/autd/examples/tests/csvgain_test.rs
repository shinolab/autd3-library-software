/*
 * File: csvgain_test.rs
 * Project: example
 * Created Date: 12/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use std::ffi::OsString;

use autd::consts::{NUM_TRANS_X, NUM_TRANS_Y, TRANS_SIZE, ULTRASOUND_WAVELENGTH};
use autd::gain::*;
use autd::modulation::*;
use autd::prelude::*;

pub fn csvgain_test(ifname: &str, link_type: LinkType) -> Result<AUTD, failure::Error> {
    let mut autd = AUTD::create();
    autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    autd.open(ifname, link_type)?;

    let path = OsString::from("csv_gain_focal.csv");
    //write
    {
        let mut wtr = csv::Writer::from_path(&path).unwrap();
        let x = 90.;
        let y = 70.;
        let z = 150.;
        for ty_idx in 0..NUM_TRANS_Y {
            for tx_idx in 0..NUM_TRANS_X {
                if !autd::utils::is_missing_transducer(tx_idx, ty_idx) {
                    let tx = tx_idx as f32 * TRANS_SIZE;
                    let ty = ty_idx as f32 * TRANS_SIZE;
                    let dist = ((tx - x) * (tx - x) + (ty - y) * (ty - y) + z * z).sqrt();
                    let phase = 1.0 - (dist % ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
                    let amp = 1.0;
                    wtr.serialize([amp, phase])?; // The file must consist of two columns, normalized amp and phase, with delimiter ','.
                }
            }
        }
        wtr.flush()?;
    }
    let g = CSVGain::create(&path)?;
    autd.append_gain_sync(g);

    let m = SineModulation::create(150);
    autd.append_modulation_sync(m);
    Ok(autd)
}
