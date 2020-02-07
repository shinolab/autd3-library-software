/*
 * File: local_ethercat.rs
 * Project: ruads
 * Created Date: 16/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use libc::c_void;

use crate::ads_error::ADSError;
use crate::native_methods::*;

const INDEX_GROUP: u64 = 0x304_0030;
const INDEX_OFFSET_BASE: u64 = 0x8100_0000;
const PORT: u16 = 301;

pub struct LocalADSLink {
    port: i64,
    send_addr: AmsAddr,
}

impl LocalADSLink {
    pub fn open() -> Result<Self, failure::Error> {
        unsafe {
            let port = (TC_ADS.tc_ads_port_open)();
            if port == 0 {
                return Err(ADSError::FailedOpenPort.into());
            }

            let mut ams_addr: AmsAddr = std::mem::zeroed();
            let n_err = (TC_ADS.tc_ads_get_local_address)(port, &mut ams_addr as *mut _);

            if n_err != 0 {
                return Err(ADSError::FailedGetLocalAddress { n_err }.into());
            }

            Ok(LocalADSLink {
                port,
                send_addr: AmsAddr {
                    net_id: ams_addr.net_id,
                    port: PORT,
                },
            })
        }
    }

    pub fn is_open(&self) -> bool {
        self.port > 0
    }

    pub fn close(&mut self) -> i64 {
        self.port = 0;
        unsafe { (TC_ADS.tc_ads_port_close)(0) }
    }

    pub fn send(&mut self, data: Vec<u8>) -> Result<(), failure::Error> {
        unsafe {
            let n_err = (TC_ADS.tc_ads_sync_write_req)(
                self.port,
                &self.send_addr as *const _,
                INDEX_GROUP,
                INDEX_OFFSET_BASE,
                data.len() as u64,
                data.as_ptr() as *const c_void,
            );

            if n_err > 0 {
                Err(ADSError::FailedSendData { n_err }.into())
            } else {
                Ok(())
            }
        }
    }
}
