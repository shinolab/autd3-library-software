/*
 * File: soem_error.rs
 * Project: rusoem
 * Created Date: 21/11/2019
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
pub enum SOEMError {
    #[fail(display = "No socket connection on {}", ifname)]
    NoSocketConnection { ifname: String },
    #[fail(
        display = "The number of slaves you specified: {}, but found: {}",
        num, wc
    )]
    SlaveNotFound { wc: u16, num: u16 },
    #[fail(display = "One ore more slaves are not responding")]
    NotResponding,
    #[fail(display = "Create Timer failed")]
    CreateTimerError,
    #[fail(display = "Delete Timer failed")]
    DeleteTimerError,
}
