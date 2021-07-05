// File: hardware_defined.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/07/2021
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

constexpr size_t FPGA_CLOCK = 20480000;
constexpr size_t ULTRASOUND_FREQUENCY = 40000;

constexpr size_t MOD_BUF_SIZE_MAX = 65536;
constexpr size_t MOD_SAMPLING_FREQ_BASE = 40000;
constexpr size_t MOD_FRAME_SIZE = 124;

constexpr size_t POINT_SEQ_BUFFER_SIZE_MAX = 65536;
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
  CLEAR = 0x09,
  SET_DELAY_OFFSET = 0x0A,
  PAUSE = 0x0B,
  RESUME = 0x0C,
  EMULATOR_SET_GEOMETRY = 0xFF
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
  SeqFocus() = default;

  void set(const int32_t x, const int32_t y, const int32_t z, const uint8_t duty) {
    _buf[0] = x & 0xFFFF;             // x 0-15 bit
    uint16_t tmp = x >> 16 & 0x0001;  // x 16th bit
    tmp |= x >> 30 & 0x0002;          // x sign bit
    tmp |= y << 2 & 0xFFFC;           // y 0-13 bit
    _buf[1] = tmp;
    tmp = y >> 14 & 0x0007;   // y 14-16 bit
    tmp |= y >> 28 & 0x0008;  // y sign bit
    tmp |= z << 4 & 0xFFF0;   // z 0-11 bit
    _buf[2] = tmp;
    tmp = z >> 12 & 0x001F;     // z 12-16 bit
    tmp |= z >> 26 & 0x0020;    // z sign bit
    tmp |= duty << 6 & 0x3FC0;  // duty
    _buf[3] = tmp;
  }

 private:
  uint16_t _buf[4];
};

}  // namespace autd::core
