/*
 * File: dbg.rs
 * Project: examples
 * Created Date: 16/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

extern crate colored;

mod tests;

use autd::prelude::*;
use colored::*;
use std::io;
use std::io::Write;
use tests::*;

fn main() {
    print!("{}", "enter log directory: ".green().bold());
    io::stdout().flush().unwrap();

    let mut s = String::new();
    io::stdin().read_line(&mut s).unwrap();

    let x: &[_] = &['\r', '\n'];
    run(s.trim_matches(x), LinkType::DBG);
}
