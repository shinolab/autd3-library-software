/*
 * File: constants.rs
 * Project: src
 * Created Date: 21/11/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

pub const AUTD_WIDTH: f32 = 192.0;
pub const AUTD_HEIGHT: f32 = 151.4;

pub const TRANS_SIZE: f32 = 10.18;
pub const NUM_TRANS_X: usize = 18;
pub const NUM_TRANS_Y: usize = 14;
pub const NUM_TRANS_IN_UNIT: usize = NUM_TRANS_X * NUM_TRANS_Y - 3;
pub const ULTRASOUND_WAVELENGTH: f32 = 8.5;
pub const ULTRASOUND_FREQUENCY: f32 = 40000.0;

pub const MOD_SAMPLING_FREQUENCY: f32 = 4000.0;
pub const MOD_BUF_SIZE_FPGA: i32 = 4000;

pub const LM_BASE_FREQUENCY: u32 = 2_560_000;
pub const LM_SHIFT_MIN: u16 = 6;
pub const LM_SHIFT_MAX: u16 = 12;
pub const LM_BUF_SIZE_FPGA: u32 = 2000;

pub const EC_SM2_CYCTIME_NS: u32 = 1_000_000;
pub const EC_SYNC0_BASE_TIME: u32 = 500_000;
pub const EC_SYNC0_CYCTIME_NS: u32 = 1_000_000;

pub(crate) const SYNC0_STEP: u16 = (EC_SYNC0_CYCTIME_NS / EC_SYNC0_BASE_TIME) as u16;

pub(crate) const HEADER_SIZE: usize = 128;
pub(crate) const BODY_SIZE: usize = NUM_TRANS_IN_UNIT * 2;
pub(crate) const MOD_FRAME_SIZE: usize = 124;
pub(crate) const INPUT_FRAME_SIZE: usize = 2;
