/*
 * File: local_ethercat_link.rs
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
use ruads::LocalADSLink;

pub struct LocalEtherCATLink {
    handler: LocalADSLink,
}

impl LocalEtherCATLink {
    pub fn open() -> Result<Box<Self>, failure::Error> {
        let handler = LocalADSLink::open()?;
        Ok(Box::new(LocalEtherCATLink { handler }))
    }
}

impl Link for LocalEtherCATLink {
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
