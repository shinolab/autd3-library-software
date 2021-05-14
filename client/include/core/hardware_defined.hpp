// File: hardware_defined.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/05/2021
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

constexpr auto FPGA_CLOCK = 20400000;
constexpr auto ULTRASOUND_FREQUENCY = 40000;

constexpr uint32_t MOD_SAMPLING_FREQ_BASE = 8000;
constexpr size_t MOD_FRAME_SIZE = 124;

constexpr auto POINT_SEQ_BUFFER_SIZE_MAX = 40000;
constexpr auto POINT_SEQ_CLK_IDX_MAX = 40000;
constexpr auto POINT_SEQ_BASE_FREQ = 40000;

using AUTDDataArray = std::array<uint16_t, NUM_TRANS_IN_UNIT>;

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
  INIT_FPGA_REF_CLOCK = 0x07,
  CLEAR = 0x09,
};

struct RxGlobalHeader {
  uint8_t msg_id;
  uint8_t control_flags;
  COMMAND command;
  uint8_t mod_size;
  uint8_t mod[MOD_FRAME_SIZE];
};

struct SeqFocus {
  uint16_t x15_0;
  uint16_t y7_0_x23_16;
  uint16_t y23_8;
  uint16_t z15_0;
  uint16_t duty_z23_16;
};

enum class MOD_SAMPLING_FREQ {
  SMPL_125_HZ = 125,
  SMPL_250_HZ = 250,
  SMPL_500_HZ = 500,
  SMPL_1_KHZ = 1000,
  SMPL_2_KHZ = 2000,
  SMPL_4_KHZ = 4000,
  SMPL_8_KHZ = 8000,
};

enum class MOD_BUF_SIZE {
  BUF_125 = 125,
  BUF_250 = 250,
  BUF_500 = 500,
  BUF_1000 = 1000,
  BUF_2000 = 2000,
  BUF_4000 = 4000,
  BUF_8000 = 8000,
  BUF_16000 = 16000,
  BUF_32000 = 32000,
};

}  // namespace autd::core
