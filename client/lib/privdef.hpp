// File: privdef.hpp
// Project: lib
// Created Date: 07/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 12/10/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <stdint.h>

#include <array>

namespace autd {
constexpr auto MOD_FRAME_SIZE_V_0_1 = 124;
constexpr auto MOD_FRAME_SIZE_V_0_6 = 120;

enum RxGlobalControlFlags {
  LOOP_BEGIN = 1 << 0,
  LOOP_END = 1 << 1,
  MOD_BEGIN = 1 << 2,
  SILENT = 1 << 3,
  FORCE_FAN = 1 << 4,
  SEQ_MODE = 1 << 5,
  SEQ_BEGIN = 1 << 6,
  SEQ_END = 1 << 7
};

struct RxGlobalHeaderV_0_1 {
  uint8_t msg_id;
  uint8_t control_flags;
  int8_t frequency_shift;
  uint8_t mod_size;
  uint8_t mod[MOD_FRAME_SIZE_V_0_1];
};

struct RxGlobalHeaderV_0_6 {
  uint8_t msg_id;
  uint8_t control_flags;
  uint8_t command;
  uint8_t mod_size;
  uint16_t seq_size;
  uint16_t seq_div;
  uint8_t mod[MOD_FRAME_SIZE_V_0_6];
};
}  // namespace autd
