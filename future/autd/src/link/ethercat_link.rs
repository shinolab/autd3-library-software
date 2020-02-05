/*
 * File: ethercat_link.rs
 * Project: link
 * Created Date: 16/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::link::Link;

use failure;
use ruads::RemoteADSLink;

pub struct EtherCATLink {
    handler: RemoteADSLink,
}

impl EtherCATLink {
    pub fn open(location: &str) -> Result<Box<Self>, failure::Error> {
        let handler = RemoteADSLink::open(location)?;
        Ok(Box::new(EtherCATLink { handler }))
    }
}

impl Link for EtherCATLink {
    fn is_open(&self) -> bool {
        self.handler.is_open()
    }

    fn send(&mut self, data: Vec<u8>) {
        if let Err(e) = self.handler.send(data) {
            eprintln!("{}", e);
        }
    }

    fn close(&mut self) {
        self.handler.close();
    }

    fn calibrate(&mut self) -> bool {
        true
    }
}
