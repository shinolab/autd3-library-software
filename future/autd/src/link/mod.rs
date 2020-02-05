/*
 * File: mod.rs
 * Project: autd
 * Created Date: 02/09/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

extern crate num;
use num::traits::FromPrimitive;

#[cfg(feature = "dbg_link")]
mod dbg_link;
#[cfg(all(windows, feature = "twincat"))]
mod ethercat_link;
#[cfg(all(windows, feature = "twincat"))]
mod local_ethercat_link;
mod soem_link;

#[cfg(feature = "dbg_link")]
pub use dbg_link::DbgLink;
#[cfg(all(windows, feature = "twincat"))]
pub use ethercat_link::EtherCATLink;
#[cfg(all(windows, feature = "twincat"))]
pub use local_ethercat_link::LocalEtherCATLink;
pub use soem_link::SoemLink;

#[derive(Copy, Clone)]
pub enum LinkType {
    SOEM,
    #[cfg(feature = "dbg_link")]
    DBG,
    #[cfg(all(windows, feature = "twincat"))]
    TwinCAT,
}

pub trait Link: Send {
    fn send(&mut self, data: Vec<u8>);
    fn close(&mut self);
    fn is_open(&self) -> bool;
    fn calibrate(&mut self) -> bool;
}

impl FromPrimitive for LinkType {
    fn from_i64(n: i64) -> Option<LinkType> {
        match n {
            0 => Some(LinkType::SOEM),
            _ => None,
        }
    }
    fn from_u64(n: u64) -> Option<LinkType> {
        match n {
            0 => Some(LinkType::SOEM),
            _ => None,
        }
    }
}
