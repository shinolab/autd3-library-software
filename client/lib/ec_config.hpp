// File: ec_config.hpp
// Project: lib
// Created Date: 21/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 31/03/2020
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

constexpr uint8_t READ_MOD_IDX_BASE_HEADER = 0xC0;
constexpr uint8_t READ_MOD_IDX_BASE_HEADER_AFTER_SHIFT = 0xE0;
constexpr uint8_t READ_MOD_IDX_BASE_HEADER_MASK = 0xE0;

constexpr uint8_t CMD_OP = 0x00;
constexpr uint8_t CMD_BRAM_WRITE = 0x01;
constexpr uint8_t CMD_READ_CPU_VER_LSB = 0x02;
constexpr uint8_t CMD_READ_CPU_VER_MSB = 0x03;
constexpr uint8_t CMD_READ_FPGA_VER_LSB = 0x04;
constexpr uint8_t CMD_READ_FPGA_VER_MSB = 0x05;
constexpr uint8_t CMD_READ_MOD_SYNC_BASE = 0x06;
constexpr uint8_t CMD_SHIFT_MOD_SYNC_BASE = 0x07;
constexpr uint8_t CMD_NEG_SYNC_FIRST_SYNC0 = 0x08;

constexpr uint8_t OP_MODE_MSG_ID_MIN = 0x20;
constexpr uint8_t OP_MODE_MSG_ID_MAX = 0xBF;

constexpr size_t EC_DEVICE_PER_FRAME = 2;
constexpr size_t EC_FRAME_LENGTH = 14 + 2 + (10 + EC_OUTPUT_FRAME_SIZE + EC_INPUT_FRAME_SIZE + 2) * EC_DEVICE_PER_FRAME + 10;
constexpr double EC_SPEED_BPS = 100.0 * 1000 * 1000;
constexpr double EC_TRAFFIC_DELAY = EC_FRAME_LENGTH * 8 / EC_SPEED_BPS;
