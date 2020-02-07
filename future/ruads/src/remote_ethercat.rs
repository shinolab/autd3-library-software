/*
 * File: remote_ethercat.rs
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

const INDEX_GROUP: u32 = 0x304_0030;
const INDEX_OFFSET_BASE: u32 = 0x8100_0000;
const PORT: u16 = 301;

pub struct RemoteADSLink {
    port: i64,
    send_addr: AmsAddr,
}

impl RemoteADSLink {
    pub fn open(location: &str) -> Result<Self, failure::Error> {
        let sep: Vec<&str> = location.split(':').collect();

        match sep.len() {
            1 => Self::open_impl(sep[0], ""),
            2 => Self::open_impl(sep[1], sep[0]),
            _ => Err(ADSError::InvalidAddress.into()),
        }
    }

    fn open_impl(ams_net_id: &str, ipv4addr: &str) -> Result<Self, failure::Error> {
        unsafe {
            let octets: Vec<&str> = ams_net_id.split('.').collect();
            if octets.len() != 6 {
                return Err(ADSError::AmsNetIdParseError.into());
            }

            let addr = if ipv4addr == "" {
                octets[0..4].join(".")
            } else {
                ipv4addr.to_string()
            };

            let net_id = AmsNetId {
                b: [
                    octets[0].parse().unwrap(),
                    octets[1].parse().unwrap(),
                    octets[2].parse().unwrap(),
                    octets[3].parse().unwrap(),
                    octets[4].parse().unwrap(),
                    octets[5].parse().unwrap(),
                ],
            };

            let caddr = std::ffi::CString::new(addr).unwrap();
            if AdsAddRoute(net_id, caddr.as_ptr()) != 0 {
                return Err(ADSError::FailedConnetRemote.into());
            }
            let port = AdsPortOpenEx();
            if port == 0 {
                return Err(ADSError::FailedOpenPort.into());
            }

            Ok(RemoteADSLink {
                port,
                send_addr: AmsAddr { net_id, port: PORT },
            })
        }
    }

    pub fn is_open(&self) -> bool {
        self.port > 0
    }

    pub fn close(&mut self) -> i64 {
        self.port = 0;
        unsafe { AdsPortCloseEx(0) }
    }

    pub fn send(&mut self, data: Vec<u8>) -> Result<(), failure::Error> {
        unsafe {
            let n_err = AdsSyncWriteReqEx(
                self.port,
                &self.send_addr as *const _,
                INDEX_GROUP,
                INDEX_OFFSET_BASE,
                data.len() as u32,
                data.as_ptr() as *const c_void,
            );

            if n_err > 0 {
                if n_err == ADSERR_DEVICE_INVALIDSIZE {
                    Err(ADSError::ErrorDeviceInvalidSize.into())
                } else {
                    Err(ADSError::FailedSendData { n_err }.into())
                }
            } else {
                Ok(())
            }
        }
    }
}
