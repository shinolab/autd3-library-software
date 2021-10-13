// File: hardware_defined.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/10/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <array>
#include <cstdint>

namespace autd {
namespace core {
constexpr size_t NUM_TRANS_IN_UNIT = 249;
constexpr size_t NUM_TRANS_X = 18;
constexpr size_t NUM_TRANS_Y = 14;
constexpr double TRANS_SPACING_MM = 10.16;
constexpr double DEVICE_WIDTH = 192.0;
constexpr double DEVICE_HEIGHT = 151.4;

template <typename T>
constexpr auto is_missing_transducer(T x, T y) {
  return y == 1 && (x == 1 || x == 2 || x == 16);
}

constexpr size_t FPGA_CLOCK = 20480000;
constexpr size_t ULTRASOUND_FREQUENCY = 40000;

constexpr size_t MOD_BUF_SIZE_MAX = 65536;
constexpr size_t MOD_SAMPLING_FREQ_BASE = 40000;
constexpr size_t MOD_FRAME_SIZE = 124;

constexpr size_t POINT_SEQ_BUFFER_SIZE_MAX = 65536;
constexpr size_t GAIN_SEQ_BUFFER_SIZE_MAX = 2048;
constexpr size_t SEQ_BASE_FREQ = 40000;

constexpr bool PHASE_INVERTED = true;

using DataArray = std::array<uint16_t, NUM_TRANS_IN_UNIT>;

enum FPGA_CONTROL_FLAGS {
  OUTPUT_ENABLE = 1 << 0,
  OUTPUT_BALANCE = 1 << 1,
  SILENT = 1 << 3,
  FORCE_FAN = 1 << 4,
  OP_MODE = 1 << 5,
  SEQ_MODE = 1 << 6,
};

constexpr bool OP_MODE_NORMAL = false;
constexpr bool OP_MODE_SEQ = true;
constexpr bool SEQ_MODE_POINT = false;
constexpr bool SEQ_MODE_GAIN = true;

enum CPU_CONTROL_FLAGS : uint8_t {
  MOD_BEGIN = 1 << 0,
  MOD_END = 1 << 1,
  SEQ_BEGIN = 1 << 2,
  SEQ_END = 1 << 3,
  READS_FPGA_INFO = 1 << 4,
  DELAY_OFFSET = 1 << 5,
};

constexpr uint8_t MSG_CLEAR = 0x00;
constexpr uint8_t MSG_RD_CPU_V_LSB = 0x01;
constexpr uint8_t MSG_RD_CPU_V_MSB = 0x02;
constexpr uint8_t MSG_RD_FPGA_V_LSB = 0x03;
constexpr uint8_t MSG_RD_FPGA_V_MSB = 0x04;
constexpr uint8_t MSG_EMU_GEOMETRY_SET = 0x05;
constexpr uint8_t MSG_NORMAL_BASE = 0x06;

/**
 * \brief Data header common to all devices
 */
struct GlobalHeader {
  uint8_t msg_id;
  uint8_t fpga_ctrl_flags;
  uint8_t cpu_ctrl_flags;
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

enum class GAIN_MODE : uint16_t {
  DUTY_PHASE_FULL = 0x0001,
  PHASE_FULL = 0x0002,
  PHASE_HALF = 0x0004,
};

}  // namespace core
}  // namespace autd
