/*
 * File: soem_link.rs
 * Project: autd
 * Created Date: 02/09/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::consts::{BODY_SIZE, HEADER_SIZE, INPUT_FRAME_SIZE};
use crate::consts::{EC_SM2_CYCTIME_NS, EC_SYNC0_CYCTIME_NS, SYNC0_STEP};
use crate::link::Link;
use crate::rx_global_header::{RxGlobalControlFlags, RxGlobalHeader};
use rusoem::{ECConfig, RuSOEM};

use failure;
use std::mem;

pub struct SoemLink {
    handler: RuSOEM,
    dev_num: u16,
}

impl SoemLink {
    pub fn open(name: &str, dev_num: u16) -> Result<Box<Self>, failure::Error> {
        let config = ECConfig {
            header_size: HEADER_SIZE,
            body_size: BODY_SIZE,
            input_size: INPUT_FRAME_SIZE,
        };
        let mut handler = RuSOEM::new(name, config);
        handler.start(dev_num, EC_SM2_CYCTIME_NS, EC_SYNC0_CYCTIME_NS)?;

        Ok(Box::new(SoemLink { handler, dev_num }))
    }
}

impl Link for SoemLink {
    fn is_open(&self) -> bool {
        self.handler.is_open()
    }

    fn send(&mut self, data: Vec<u8>) {
        self.handler.send(data);
    }

    fn close(&mut self) {
        self.handler.close();
    }

    fn calibrate(&mut self) -> bool {
        let succeed_calib = |v: &Vec<u16>| {
            let min = v.iter().fold(0xFFFFu16, |x, &y| y.min(x));
            for b in v {
                let h = (b & 0xC000) >> 14;
                let base = b & 0x3FFF;
                if h != 1 || (base - min) % SYNC0_STEP != 0 {
                    return false;
                }
            }
            true
        };

        let mut success = false;
        let mut v: Vec<u16> = vec![];
        for _ in 0..10 {
            self.handler.close();
            self.handler
                .start(self.dev_num, EC_SM2_CYCTIME_NS, 1000 * 1000 * 1000)
                .unwrap();
            unsafe {
                let size = mem::size_of::<RxGlobalHeader>();
                let mut body = vec![0x00; size];
                let header = body.as_mut_ptr() as *mut RxGlobalHeader;
                (*header).msg_id = 0xFF;
                (*header).ctrl_flag = RxGlobalControlFlags::IS_SYNC_FIRST_SYNC_N;
                self.send(body);
            }
            std::thread::sleep(std::time::Duration::from_secs(self.dev_num as u64 / 5 + 2));
            self.handler.close();
            self.handler
                .start(self.dev_num, EC_SM2_CYCTIME_NS, EC_SYNC0_CYCTIME_NS)
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(500));
            let input = self.handler.read::<u16>();
            if let Some(input) = input {
                v = input;
                if succeed_calib(&v) {
                    success = true;
                    break;
                }
            }
        }

        if !success {
            eprintln!("Failed to calibrate");
            eprintln!("======== Log ========");
            eprintln!("#Dev\tHeader\tBase");
            for (i, b) in v.iter().enumerate() {
                let h = (b & 0xC000) >> 14;
                let base = b & 0x3FFF;
                eprintln!("{}\t{}\t{}", i, h, base);
            }
            eprintln!("=====================");
        }

        success
    }
}
