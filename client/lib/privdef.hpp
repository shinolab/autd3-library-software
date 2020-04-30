// File: privdef.hpp
// Project: lib
// Created Date: 07/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 30/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <stdint.h>

#include <array>

namespace autd {
constexpr auto MOD_FRAME_SIZE = 124;

enum RxGlobalControlFlags {
  LOOP_BEGIN = 1 << 0,
  LOOP_END = 1 << 1,
  SILENT = 1 << 3,
  FORCE_FAN = 1 << 4,
  IS_SYNC_FIRST_SYNC0 = 1 << 5,  //  reserved, do not use
  MOD_SYNC_BASE_SHIFT = 1 << 6,
};

struct RxGlobalHeader {
  uint8_t msg_id;
  uint8_t control_flags;
  uint8_t command;
  uint8_t mod_size;
  uint8_t mod[MOD_FRAME_SIZE];
};
}  // namespace autd
