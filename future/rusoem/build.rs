/*
 * File: build.rs
 * Project: rusoem
 * Created Date: 29/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#[cfg(target_os = "windows")]
use std::env;

#[cfg(target_os = "windows")]
fn main() {
    let home_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-lib=winmm");
    println!("cargo:rustc-link-lib=ws2_32");
    println!(
        "cargo:rustc-link-search={}\\deps\\SOEM\\oshw\\win32\\wpcap\\Lib\\x64",
        home_dir
    );
    println!("cargo:rustc-link-lib=static=Packet");
    println!("cargo:rustc-link-lib=static=wpcap");
    cc::Build::new()
        .warnings(true)
        .flag("/DWIN32")
        .flag("/wd26451")
        .flag("/wd6385")
        .flag("/wd6386")
        .flag("/wd6011")
        .flag("/wd26495")
        .flag("/wd4996")
        .flag("/wd6001")
        .flag("/wd4200")
        .flag("/wd4201")
        .flag("/wd4701")
        .flag("/wd4244")
        .flag("/wd4214")
        .static_flag(true)
        .file("deps/SOEM/soem/ethercatbase.c")
        .file("deps/SOEM/soem/ethercatcoe.c")
        .file("deps/SOEM/soem/ethercatconfig.c")
        .file("deps/SOEM/soem/ethercatdc.c")
        .file("deps/SOEM/soem/ethercateoe.c")
        .file("deps/SOEM/soem/ethercatmain.c")
        .file("deps/SOEM/soem/ethercatprint.c")
        .file("deps/SOEM/soem/ethercatsoe.c")
        .file("deps/SOEM/osal/win32/osal.c")
        .file("deps/SOEM/oshw/win32/nicdrv.c")
        .file("deps/SOEM/oshw/win32/oshw.c")
        .include("deps/SOEM/soem")
        .include("deps/SOEM/osal")
        .include("deps/SOEM/osal/win32")
        .include("deps/SOEM/oshw/win32")
        .include("deps/SOEM/oshw/win32/wpcap/Include")
        .include("deps/SOEM/oshw/win32/wpcap/Include/pcap")
        .compile("libsoem.a");
}

#[cfg(target_os = "macos")]
fn main() {
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=pcap");
    cc::Build::new()
        .warnings(true)
        .static_flag(true)
        .file("deps/SOEM/soem/ethercatbase.c")
        .file("deps/SOEM/soem/ethercatcoe.c")
        .file("deps/SOEM/soem/ethercatconfig.c")
        .file("deps/SOEM/soem/ethercatdc.c")
        .file("deps/SOEM/soem/ethercateoe.c")
        .file("deps/SOEM/soem/ethercatmain.c")
        .file("deps/SOEM/soem/ethercatprint.c")
        .file("deps/SOEM/soem/ethercatsoe.c")
        .file("deps/SOEM/osal/macosx/osal.c")
        .file("deps/SOEM/oshw/macosx/nicdrv.c")
        .file("deps/SOEM/oshw/macosx/oshw.c")
        .include("deps/SOEM/soem")
        .include("deps/SOEM/osal")
        .include("deps/SOEM/osal/macosx")
        .include("deps/SOEM/oshw/macosx")
        .compile("libsoem.a");
}

#[cfg(target_os = "linux")]
fn main() {
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=rt");
    cc::Build::new()
        .warnings(true)
        .static_flag(true)
        .file("deps/SOEM/soem/ethercatbase.c")
        .file("deps/SOEM/soem/ethercatcoe.c")
        .file("deps/SOEM/soem/ethercatconfig.c")
        .file("deps/SOEM/soem/ethercatdc.c")
        .file("deps/SOEM/soem/ethercateoe.c")
        .file("deps/SOEM/soem/ethercatmain.c")
        .file("deps/SOEM/soem/ethercatprint.c")
        .file("deps/SOEM/soem/ethercatsoe.c")
        .file("deps/SOEM/osal/linux/osal.c")
        .file("deps/SOEM/oshw/linux/nicdrv.c")
        .file("deps/SOEM/oshw/linux/oshw.c")
        .include("deps/SOEM/soem")
        .include("deps/SOEM/osal")
        .include("deps/SOEM/osal/linux")
        .include("deps/SOEM/oshw/linux")
        .compile("libsoem.a");
}
