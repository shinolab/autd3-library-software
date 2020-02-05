/*
 * File: mod.rs
 * Project: modulation
 * Created Date: 16/11/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

pub mod raw_pcm_modulation;
pub mod sine_modulation;

pub use raw_pcm_modulation::RawPCMModulation;
pub use sine_modulation::SineModulation;

use crate::consts::MOD_FRAME_SIZE;

pub struct Modulation {
    pub buffer: Vec<u8>,
    pub sent: usize,
}

impl Modulation {
    pub fn create(amp: u8) -> Modulation {
        Modulation {
            buffer: vec![amp; MOD_FRAME_SIZE],
            sent: 0,
        }
    }
}
