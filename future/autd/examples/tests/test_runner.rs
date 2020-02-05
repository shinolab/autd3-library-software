/*
 * File: test_runner.rs
 * Project: autd
 * Created Date: 29/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

extern crate colored;

use std::io;
use std::io::Write;

use crate::tests::*;
use autd::prelude::*;
use colored::*;

pub fn run(ifname: &str, link_type: LinkType) {
    let examples: Vec<(fn(&str, LinkType) -> Result<AUTD, failure::Error>, _)> = vec![
        (simple_test, "Simple Test"),
        (bessel_test, "BesselBeam Test"),
        (hologain_test, "HoloGain Test (2 focal points)"),
        (csvgain_test, "CSVGain Test"),
        (raw_pcm_modulation_test, "RawPCMModulation Test"),
        (
            software_stm_test,
            "Software Spatio-Temporal Modulation Test",
        ),
        (two_autds_test, "2 AUTDs Test (simple)"),
        (grouped_gain_test, "2 AUTDs Test (grouped gain)"),
    ];

    loop {
        for (i, (_, desc)) in examples.iter().enumerate() {
            println!("[{}]: {}", i, desc);
        }
        println!("[Others]: Finish");
        println!(
            "{}",
            "Make sure you connected ONLY appropriate numbers of AUTD."
                .yellow()
                .bold()
        );

        print!("{}", "Choose number: ".green().bold());
        io::stdout().flush().unwrap();

        let mut s = String::new();
        io::stdin().read_line(&mut s).unwrap();
        let i: usize = match s.trim().parse() {
            Ok(num) if num < examples.len() => num,
            _ => break,
        };

        let (f, _) = examples[i];

        match f(&ifname, link_type) {
            Ok(autd) => {
                println!("press any key to finish...");
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                autd.close();
                println!("finish");
            }
            Err(e) => {
                eprintln!("{}", e.to_string().red().bold());
            }
        }
    }
}
