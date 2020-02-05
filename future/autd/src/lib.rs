/*
 * File: lib.rs
 * Project: autd
 * Created Date: 29/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#[macro_use]
extern crate bitflags;
extern crate chrono;
extern crate failure;
extern crate hound;
#[cfg(all(windows, feature = "twincat"))]
extern crate ruads;
extern crate rusoem;
extern crate timer;
#[cfg(windows)]
extern crate winapi;

pub mod autd;
pub mod consts;
pub mod gain;
pub mod geometry;
mod link;
pub mod modulation;
pub mod prelude;
pub mod rx_global_header;
pub mod utils;

pub use autd::AUTD;
pub use hound::SampleFormat;
pub use rusoem::EthernetAdapters;
