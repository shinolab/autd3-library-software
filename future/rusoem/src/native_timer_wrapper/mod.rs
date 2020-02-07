/*
 * File: mod.rs
 * Project: native_timer_wrapper
 * Created Date: 12/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 15/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
pub use self::windows::*;

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "linux")]
pub use self::linux::*;

#[cfg(target_os = "macos")]
pub mod macosx;
#[cfg(target_os = "macos")]
pub use self::macosx::*;
