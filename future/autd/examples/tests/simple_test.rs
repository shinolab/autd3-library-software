/*
 * File: simple_test.rs
 * Project: example
 * Created Date: 12/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 06/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use autd::gain::*;
use autd::modulation::*;
use autd::prelude::*;

pub fn simple_test(ifname: &str, link_type: LinkType) -> Result<AUTD, failure::Error> {
    let mut autd = AUTD::create();

    autd.add_device(Vector3f::zeros(), Vector3f::zeros());

    autd.open(ifname, link_type)?;

    let g = FocalPointGain::create(Vector3f::new(90., 70., 150.));
    autd.append_gain_sync(g);

    let m = SineModulation::create(150);
    autd.append_modulation_sync(m);

    Ok(autd)
}
