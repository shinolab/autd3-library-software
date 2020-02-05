/*
 * File: two_autds_test.rs
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

use autd::consts::AUTD_WIDTH;
use autd::gain::*;
use autd::modulation::*;
use autd::prelude::*;

pub fn two_autds_test(ifname: &str, link_type: LinkType) -> Result<AUTD, failure::Error> {
    let mut autd = AUTD::create();
    autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    autd.add_device(Vector3f::x() * AUTD_WIDTH, Vector3f::zeros()); // if second AUTD is placed next to first AUTD
    autd.open(ifname, link_type)?;

    println!("Start synchronizing AUTDs...");
    autd.calibrate(); // Calibration is required if you use more than one AUTD.
                      // AUTD will move without calibration, but it may cause noise and weak tactile presentation.
    println!("Finish synchronizing AUTDs.");

    let g = FocalPointGain::create(Vector3f::new(AUTD_WIDTH, 70., 200.));
    autd.append_gain_sync(g);

    let m = SineModulation::create(150);
    autd.append_modulation_sync(m);

    Ok(autd)
}
