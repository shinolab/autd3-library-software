/*
 * File: rx_global_header.rs
 * Project: src
 * Created Date: 21/11/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::consts::*;
use std::fmt;
use std::sync::atomic::{self, AtomicU8};

static MSG_ID: AtomicU8 = AtomicU8::new(1);

#[allow(dead_code)]
#[repr(packed)]
pub struct RxGlobalHeader {
    pub(crate) msg_id: u8,
    pub ctrl_flag: RxGlobalControlFlags,
    frequency_shift: i8,
    mod_size: u8,
    mod_data: [u8; MOD_FRAME_SIZE],
}

bitflags! {
pub struct RxGlobalControlFlags : u8 {
    const NONE = 0;
    const LOOP_BEGIN = 1;
    const LOOP_END = 1 << 1;
    //
    const SILENT = 1 << 3;
    const FORCE_FAN = 1 << 4;
    const IS_SYNC_FIRST_SYNC_N = 1 << 5; // Reserved: Never use
}
}

impl RxGlobalHeader {
    pub fn new(ctrl_flag: RxGlobalControlFlags, data: &[u8]) -> RxGlobalHeader {
        MSG_ID.fetch_add(1, atomic::Ordering::SeqCst);
        MSG_ID.compare_and_swap(0xff, 1, atomic::Ordering::SeqCst);

        let mut data_array = [0x00; MOD_FRAME_SIZE];
        data_array[..data.len()].clone_from_slice(&data[..]);

        RxGlobalHeader {
            msg_id: MSG_ID.load(atomic::Ordering::SeqCst),
            ctrl_flag,
            frequency_shift: -3,
            mod_size: data.len() as u8,
            mod_data: data_array,
        }
    }
}

impl fmt::Debug for RxGlobalHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r"RxGlobalHeader {{
    msg_id: {},
    ctrl_flag: {:?},
    frequency_shift: {},
    mod_size: {},
    mod_data: {:?},
}}",
            self.msg_id,
            self.ctrl_flag,
            self.frequency_shift,
            self.mod_size,
            &self.mod_data[..],
        )
    }
}
