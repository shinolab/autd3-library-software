// File: privdef.hpp
// Project: lib
// Created Date: 07/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <cstdint>

namespace autd {

constexpr uint32_t MOD_SAMPLING_FREQ_BASE = 8000;

constexpr size_t MOD_FRAME_SIZE = 120;

enum RX_GLOBAL_CONTROL_FLAGS {
  LOOP_BEGIN = 1 << 0,
  LOOP_END = 1 << 1,
  MOD_BEGIN = 1 << 2,
  SILENT = 1 << 3,
  FORCE_FAN = 1 << 4,
  SEQ_MODE = 1 << 5,
  SEQ_BEGIN = 1 << 6,
  SEQ_END = 1 << 7
};

struct RxGlobalHeader {
  uint8_t msg_id;
  uint8_t control_flags;
  uint8_t command;
  uint8_t mod_size;
  uint16_t seq_size;
  uint16_t seq_div;
  uint8_t mod[MOD_FRAME_SIZE];
};
}  // namespace autd
