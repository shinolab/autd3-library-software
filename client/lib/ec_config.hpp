// File: ec_config.hpp
// Project: lib
// Created Date: 21/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 29/03/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "privdef.hpp"

constexpr auto HEADER_SIZE = sizeof(RxGlobalHeader);

constexpr size_t EC_OUTPUT_FRAME_SIZE = NUM_TRANS_IN_UNIT * 2 + HEADER_SIZE;
constexpr size_t EC_INPUT_FRAME_SIZE = 2;

constexpr uint32_t EC_SM3_CYCLE_TIME_MICRO_SEC = 1000;
constexpr uint32_t EC_SYNC0_CYCLE_TIME_MICRO_SEC = 2000;

constexpr uint32_t EC_SM3_CYCLE_TIME_NANO_SEC = EC_SM3_CYCLE_TIME_MICRO_SEC * 1000;
constexpr uint32_t EC_SYNC0_CYCLE_TIME_NANO_SEC = EC_SYNC0_CYCLE_TIME_MICRO_SEC * 1000;

constexpr uint16_t SYNC0_STEP = EC_SYNC0_CYCLE_TIME_MICRO_SEC * MOD_SAMPLING_FREQ / (1000 * 1000);
constexpr uint32_t MOD_PERIOD_MS = static_cast<uint32_t>((MOD_BUF_SIZE / MOD_SAMPLING_FREQ) * 1000);

constexpr uint8_t CALIBRATION_HEADER = 0xff;

constexpr uint16_t SYNC_HEADER_MASK = 0x00C0;
constexpr uint16_t SYNC_HEADER_SUCCES = 0x0040;
constexpr uint16_t SYNC_BASE_MASK = 0x003F;
