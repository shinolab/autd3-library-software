/*
 * File: sine_modulation.rs
 * Project: modulation
 * Created Date: 16/11/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 26/11/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

extern crate num;

use crate::consts::MOD_BUF_SIZE_FPGA;
use crate::consts::MOD_SAMPLING_FREQUENCY;
use crate::modulation::Modulation;
use crate::utils::*;

use num::integer::gcd;

pub struct SineModulation {}

impl SineModulation {
    pub fn create_with_amp(freq: i32, amp: f32, offset: f32) -> Modulation {
        assert!(offset + 0.5 * amp <= 1.0 && offset - 0.5 * amp >= 0.0);
        let sf = MOD_SAMPLING_FREQUENCY as i32;
        let freq = clamp(freq, 1, sf / 2);
        let d = gcd(sf, freq);
        let n = MOD_BUF_SIZE_FPGA / d;
        let rep = freq / d;
        let mut buffer = Vec::with_capacity(n as usize);

        for i in 0..n {
            let tamp = ((2 * rep * i) as f32 / n as f32) % 2.0;
            let tamp = if tamp > 1.0 { 2.0 - tamp } else { tamp };
            let tamp = offset + (tamp - 0.5) * amp;
            buffer.push((tamp * 255.0) as u8);
        }

        Modulation { buffer, sent: 0 }
    }
    pub fn create(freq: i32) -> Modulation {
        SineModulation::create_with_amp(freq, 1.0, 0.5)
    }
}
