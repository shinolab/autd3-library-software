/*
 * File: build.rs
 * Project: ruads
 * Created Date: 16/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 18/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

fn main() {
    println!("cargo:rustc-link-lib=ws2_32");
    cc::Build::new()
        .warnings(true)
        .cpp(true)
        .static_flag(true)
        .file("deps/ADS/AdsLib/AdsDef.cpp")
        .file("deps/ADS/AdsLib/AdsLib.cpp")
        .file("deps/ADS/AdsLib/AmsConnection.cpp")
        .file("deps/ADS/AdsLib/AmsPort.cpp")
        .file("deps/ADS/AdsLib/AmsRouter.cpp")
        .file("deps/ADS/AdsLib/Frame.cpp")
        .file("deps/ADS/AdsLib/Log.cpp")
        .file("deps/ADS/AdsLib/NotificationDispatcher.cpp")
        .file("deps/ADS/AdsLib/Sockets.cpp")
        .include("deps/ADS/AdsLib")
        .compile("libads.a");
}
