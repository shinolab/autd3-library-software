/*
 * File: groupedgain_test.rs
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

use std::collections::HashMap;

use autd::gain::*;
use autd::modulation::*;
use autd::prelude::*;

pub fn grouped_gain_test(ifname: &str, link_type: LinkType) -> Result<AUTD, failure::Error> {
    let mut autd = AUTD::create();
    autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    autd.open(ifname, link_type)?;

    let g1 = FocalPointGain::create(Vector3f::new(90., 70., 200.));
    let g2 = HoloGain::create(
        vec![
            Vector3f::new(70., 70., 150.),
            Vector3f::new(110., 70., 150.),
        ],
        vec![1., 1.],
    );
    let mut ids = HashMap::new();
    // Any type of key which implements "Sized + Send + Hash + Eq" can be used
    ids.insert("A", vec![0]); // Group "A" consists of devices with id: 0,...
    ids.insert("B", vec![1]); // Group "B" consists of devices with id: 1,...
    let mut gains: HashMap<_, Box<dyn Gain>> = HashMap::new();
    gains.insert("A", g1);
    gains.insert("B", g2);
    let g = GroupedGain::create(ids, gains);
    autd.append_gain_sync(g);

    let m = SineModulation::create(150);
    autd.append_modulation_sync(m);
    Ok(autd)
}
