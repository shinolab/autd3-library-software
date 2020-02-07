/*
 * File: ads_error.rs
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

use failure::Fail;

#[derive(Fail, Debug)]
pub enum ADSError {
    #[fail(display = "Failed to open a new ADS port")]
    FailedOpenPort,
    #[fail(display = "AdsGetLocalAddress (error code: {})", n_err)]
    FailedGetLocalAddress { n_err: i64 },
    #[fail(display = "Failed to send data (error code: {})", n_err)]
    // https://infosys.beckhoff.com/english.php?content=../content/1033/tcadscommon/html/tcadscommon_intro.htm&id=
    FailedSendData { n_err: i64 },
    #[fail(display = "Ams net id must have 6 octets")]
    AmsNetIdParseError,
    #[fail(display = "Could not connect to remote")]
    FailedConnetRemote,
    #[fail(display = "The number of devices is invalid")]
    ErrorDeviceInvalidSize,
    #[fail(display = "Invalid address")]
    InvalidAddress,
}
