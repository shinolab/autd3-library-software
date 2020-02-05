/*
 * File: rawpcmmodulation_test.rs
 * Project: example
 * Created Date: 12/12/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/12/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

use std::f32::consts::PI;
use std::ffi::OsString;

use autd::gain::*;
use autd::modulation::*;
use autd::prelude::*;

pub fn raw_pcm_modulation_test(ifname: &str, link_type: LinkType) -> Result<AUTD, failure::Error> {
    let mut autd = AUTD::create();
    autd.add_device(Vector3f::zeros(), Vector3f::zeros());
    autd.open(ifname, link_type)?;

    let g = FocalPointGain::create(Vector3f::new(90., 70., 150.));
    autd.append_gain_sync(g);

    let path = OsString::from("sine.wav");
    // write 150 Hz sine wave
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 4000,
            bits_per_sample: 8,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for t in (0..80).map(|x| x as f32 / 4000.0) {
            let sample = (t * 150. * 2.0 * PI).sin();
            let amplitude = std::i8::MAX as f32;
            let p = (sample * amplitude) as i8;
            writer.write_sample(p)?;
        }
    }

    let m = RawPCMModulation::create(&path, 8, hound::SampleFormat::Int).unwrap();
    autd.append_modulation_sync(m);

    Ok(autd)
}
