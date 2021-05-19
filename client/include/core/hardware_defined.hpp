// File: hardware_defined.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <array>
#include <cstdint>

namespace autd::core {
constexpr size_t NUM_TRANS_IN_UNIT = 249;
constexpr size_t NUM_TRANS_X = 18;
constexpr size_t NUM_TRANS_Y = 14;
constexpr double TRANS_SPACING_MM = 10.16;
constexpr double AUTD_WIDTH = 192.0;
constexpr double AUTD_HEIGHT = 151.4;

template <typename T>
constexpr auto IsMissingTransducer(T x, T y) {
  return y == 1 && (x == 1 || x == 2 || x == 16);
}

constexpr size_t FPGA_CLOCK = 20400000;
constexpr size_t ULTRASOUND_FREQUENCY = 40000;

constexpr uint16_t MOD_BUF_SIZE_MAX = 32000;
constexpr size_t MOD_SAMPLING_FREQ_BASE = 40000;
constexpr size_t MOD_FRAME_SIZE = 124;

constexpr size_t POINT_SEQ_BUFFER_SIZE_MAX = 40000;
constexpr size_t POINT_SEQ_CLK_IDX_MAX = 40000;
constexpr size_t POINT_SEQ_BASE_FREQ = 40000;

using DataArray = std::array<uint16_t, NUM_TRANS_IN_UNIT>;

enum RX_GLOBAL_CONTROL_FLAGS {
  MOD_BEGIN = 1 << 0,
  MOD_END = 1 << 1,
  READ_FPGA_INFO = 1 << 2,
  SILENT = 1 << 3,
  FORCE_FAN = 1 << 4,
  SEQ_MODE = 1 << 5,
  SEQ_BEGIN = 1 << 6,
  SEQ_END = 1 << 7
};

enum class COMMAND : uint8_t {
  OP = 0x00,
  READ_CPU_VER_LSB = 0x02,
  READ_CPU_VER_MSB = 0x03,
  READ_FPGA_VER_LSB = 0x04,
  READ_FPGA_VER_MSB = 0x05,
  SEQ_MODE = 0x06,
  INIT_MOD_CLOCK = 0x07,
  CLEAR = 0x09,
  SET_DELAY = 0x0A
};

/**
 * \brief Data header common to all devices
 */
struct RxGlobalHeader {
  uint8_t msg_id;
  uint8_t control_flags;
  COMMAND command;
  uint8_t mod_size;
  uint8_t mod[MOD_FRAME_SIZE];
};

/**
 * \brief Focus struct used in sequence mode
 */
struct SeqFocus {
  uint8_t buf[10];
  void set(const int32_t offset, const int32_t v) {
    uint8_t tmp = v & 0xFF;
    buf[offset] = tmp;
    tmp = v >> 8 & 0xFF;
    buf[offset + 1] = tmp;
    tmp = v >> 16 & 0x7F;
    tmp |= v >> 24 & 0x80;
    buf[offset + 2] = tmp;
  }
  void set(const int32_t x, const int32_t y, const int32_t z, const uint8_t duty) {
    set(0, x);
    set(3, y);
    set(6, z);
    buf[9] = duty;
  }
};

}  // namespace autd::core
