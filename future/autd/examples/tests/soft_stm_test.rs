/*
 * File: soft_stm_test.rs
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

use std::f32::consts::PI;

use autd::gain::*;
use autd::modulation::*;
use autd::prelude::*;

pub fn software_stm_test(ifname: &str, link_type: LinkType) -> Result<AUTD, failure::Error> {
    let mut autd = AUTD::create();
    autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    autd.open(ifname, link_type)?;

    let m = SineModulation::create(150);
    autd.append_modulation_sync(m);

    let mut gains: Vec<Box<dyn Gain>> = Vec::new();
    let center = Vector3f::new(90., 70., 150.);
    let r = 30.;
    for i in 0..200 {
        let theta = 2. * PI * i as f32 / 200.;
        let p = Vector3f::new(r * theta.cos(), r * theta.sin(), 0.);
        gains.push(FocalPointGain::create(center + p));
    }
    autd.append_stm_gains(gains);
    autd.start_stm(1.);

    Ok(autd)
}
