/*
 * File: grouped_gain.rs
 * Project: gain
 * Created Date: 02/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 03/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::gain::Gain;
use crate::geometry::Geometry;
use std::collections::{HashMap, HashSet};

use crate::consts::NUM_TRANS_IN_UNIT;
use std::hash::Hash;

pub struct GroupedGain<T: Sized + Send + Hash + Eq> {
    id_map: HashMap<T, Vec<usize>>,
    gain_map: HashMap<T, Box<dyn Gain>>,
    data: Option<Vec<u8>>,
}

impl<T: Sized + Send + Hash + Eq> GroupedGain<T> {
    pub fn create(
        id_map: HashMap<T, Vec<usize>>,
        gain_map: HashMap<T, Box<dyn Gain>>,
    ) -> Box<GroupedGain<T>> {
        let gids: HashSet<&T> = id_map.keys().collect();
        let gain_gids: HashSet<&T> = gain_map.keys().collect();

        assert!(gain_gids.is_subset(&gids));

        Box::new(GroupedGain {
            id_map,
            gain_map,
            data: None,
        })
    }
}

impl<T: Sized + Send + Hash + Eq> Gain for GroupedGain<T> {
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
        let mut data = vec![0x00; ntrans * 2];

        for gain in self.gain_map.values_mut() {
            gain.build(geometry);
        }

        for (group_id, device_ids) in &self.id_map {
            if let Some(gain) = &self.gain_map.get(group_id) {
                let d = gain.get_data();
                for device_id in device_ids {
                    if *device_id >= ndevice {
                        panic!(
                        "You specified device id ({}) in GroupedGain, but only {} AUTDs are connected.",
                        *device_id, ndevice
                    );
                    }
                    unsafe {
                        let src = d.as_ptr() as *const u8;
                        let dst = data.as_mut_ptr().add(device_id * NUM_TRANS_IN_UNIT * 2);
                        std::ptr::copy_nonoverlapping(src, dst, NUM_TRANS_IN_UNIT * 2);
                    }
                }
            }
        }

        self.data = Some(data);
    }
}
