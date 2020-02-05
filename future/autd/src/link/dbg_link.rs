/*
 * File: dbg_link.rs
 * Project: link
 * Created Date: 11/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use crate::link::Link;

use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};

use chrono::Local;
use failure;
use std::path::Path;

pub struct DbgLink {
    dir_path: String,
    writer: Option<BufWriter<File>>,
    is_open: bool,
}

impl DbgLink {
    pub fn open(dir_path: &str) -> Result<Box<Self>, failure::Error> {
        fs::create_dir_all(dir_path)?;
        Ok(Box::new(DbgLink {
            dir_path: dir_path.to_string(),
            writer: None,
            is_open: true,
        }))
    }
}

impl Link for DbgLink {
    fn is_open(&self) -> bool {
        self.is_open
    }

    fn send(&mut self, data: Vec<u8>) {
        let now = Local::now().format("%Y%m%d%H%M%S_%f").to_string();
        let file = File::create(Path::new(&self.dir_path).join(format!("log{}.dat", now))).unwrap();
        let mut w = BufWriter::new(file);
        w.write_all(&data).unwrap();
    }

    fn close(&mut self) {
        if let Some(w) = &mut self.writer {
            w.flush().unwrap();
        }
        self.is_open = false;
    }

    fn calibrate(&mut self) -> bool {
        true
    }
}
