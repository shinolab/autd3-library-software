/*
 * File: twincat.rs
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

mod tests;

use autd::prelude::*;
use tests::*;

fn main() {
    run("", LinkType::TwinCAT);
}
