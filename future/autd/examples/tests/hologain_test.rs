/*
 * File: hologain_test.rs
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

use autd::gain::*;
use autd::modulation::*;
use autd::prelude::*;

pub fn hologain_test(ifname: &str, link_type: LinkType) -> Result<AUTD, failure::Error> {
    let mut autd = AUTD::create();
    autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    autd.open(ifname, link_type)?;

    let g = HoloGain::create(
        vec![
            Vector3f::new(70., 70., 150.),
            Vector3f::new(110., 70., 150.),
        ],
        vec![1., 1.],
    );
    autd.append_gain_sync(g);

    let m = SineModulation::create(150);
    autd.append_modulation_sync(m);
    Ok(autd)
}
