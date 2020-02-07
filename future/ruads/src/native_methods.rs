/*
 * File: native_methods.rs
 * Project: ruads
 * Created Date: 16/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 18/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use libc::{c_char, c_void};
use libloading as lib;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct AmsNetId {
    pub b: [u8; 6],
}

#[repr(C)]
pub struct AmsAddr {
    pub net_id: AmsNetId,
    pub port: u16,
}

const ERR_ADSERRS: i64 = 0x0700;
pub const ADSERR_DEVICE_INVALIDSIZE: i64 = 0x05 + ERR_ADSERRS;

#[link(name = "ads", kind = "static")]
extern "C" {
    pub fn AdsAddRoute(ams: AmsNetId, ip: *const c_char) -> i64;
    pub fn AdsPortOpenEx() -> i64;
    pub fn AdsPortCloseEx(port: i64) -> i64;
    pub fn AdsSyncWriteReqEx(
        port: i64,
        pAddr: *const AmsAddr,
        indexGroup: u32,
        indexOffset: u32,
        bufferLength: u32,
        buffer: *const c_void,
    ) -> i64;
}

pub struct TcAds {
    pub tc_ads_port_open: lib::Symbol<'static, unsafe extern "C" fn() -> i64>,
    pub tc_ads_port_close: lib::Symbol<'static, unsafe extern "C" fn(i64) -> i64>,
    pub tc_ads_get_local_address:
        lib::Symbol<'static, unsafe extern "C" fn(i64, *mut AmsAddr) -> i64>,
    pub tc_ads_sync_write_req: lib::Symbol<
        'static,
        unsafe extern "C" fn(i64, *const AmsAddr, u64, u64, u64, *const c_void) -> i64,
    >,
}

impl TcAds {
    fn new() -> TcAds {
        unsafe {
            TcAds {
                tc_ads_port_open: DLL.get(b"AdsPortOpenEx").unwrap(),
                tc_ads_port_close: DLL.get(b"AdsPortCloseEx").unwrap(),
                tc_ads_get_local_address: DLL.get(b"AdsGetLocalAddressEx").unwrap(),
                tc_ads_sync_write_req: DLL.get(b"AdsSyncWriteReqEx").unwrap(),
            }
        }
    }
}

lazy_static! {
    static ref DLL: lib::Library = lib::Library::new("TcAdsDll").unwrap();
    pub static ref TC_ADS: TcAds = TcAds::new();
}
