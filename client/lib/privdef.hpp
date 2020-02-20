// File: privdef.hpp
// Project: lib
// Created Date: 07/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 20/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#ifndef CLIENT_LIB_PRIVDEF_HPP_
#define CLIENT_LIB_PRIVDEF_HPP_

#include <stdint.h>

#include <array>

constexpr auto NUM_TRANS_IN_UNIT = 249;
constexpr auto NUM_TRANS_X = 18;
constexpr auto NUM_TRANS_Y = 14;
constexpr auto TRANS_SIZE_MM = 10.18f;
template <typename T>
constexpr auto IS_MISSING_TRANSDUCER(T X, T Y) {
  return (Y == 1 && (X == 1 || X == 2 || X == 16));
}

constexpr auto FPGA_CLOCK = 25600000;

constexpr auto ULTRASOUND_FREQUENCY = 40000;
constexpr auto ULTRASOUND_WAVELENGTH = 8.5f;

constexpr auto MOD_SAMPLING_FREQ = 4000;
constexpr auto MOD_BUF_SIZE = 4000;
constexpr auto MOD_FRAME_SIZE = 124;
constexpr auto HEADER_SIZE = MOD_FRAME_SIZE + 4;

constexpr size_t EC_OUTPUT_FRAME_SIZE = NUM_TRANS_IN_UNIT * 2 + HEADER_SIZE;
constexpr size_t EC_INPUT_FRAME_SIZE = 2;

constexpr uint32_t EC_SM3_CYCLE_TIME_MICRO_SEC = 1000;
constexpr uint32_t EC_SYNC0_CYCLE_TIME_MICRO_SEC = 1000;

constexpr uint32_t EC_SM3_CYCLE_TIME_NANO_SEC = EC_SM3_CYCLE_TIME_MICRO_SEC * 1000;
constexpr uint32_t EC_SYNC0_CYCLE_TIME_NANO_SEC = EC_SYNC0_CYCLE_TIME_MICRO_SEC * 1000;

enum RxGlobalControlFlags {
  LOOP_BEGIN = 1 << 0,
  LOOP_END = 1 << 1,
  SILENT = 1 << 3,
  IS_SYNC_FIRST_SYNC0 = 1 << 5,  //  reserved, do not use
};

struct RxGlobalHeader {
  uint8_t msg_id;
  uint8_t control_flags;
  int8_t _frequency_shift;  // NO USE
  uint8_t mod_size;
  uint8_t mod[MOD_FRAME_SIZE];
};

#endif  // CLIENT_LIB_PRIVDEF_HPP_
