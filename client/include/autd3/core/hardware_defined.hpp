// File: hardware_defined.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
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
constexpr size_t MOD_SAMPLING_FREQ_DIV_MAX = 65536;
constexpr size_t MOD_FRAME_SIZE = 124;

constexpr size_t POINT_SEQ_BUFFER_SIZE_MAX = 65536;
constexpr size_t GAIN_SEQ_BUFFER_SIZE_MAX = 2048;
constexpr size_t SEQ_BASE_FREQ = 40000;
constexpr size_t SEQ_SAMPLING_FREQ_DIV_MAX = 65536;

constexpr bool PHASE_INVERTED = true;

enum FPGA_CONTROL_FLAGS {
  NONE = 0,
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
  WRITE_BODY = 1 << 6,
  WAIT_ON_SYNC = 1 << 7,
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

struct Drive final {
  Drive() : Drive(0x00, 0x00) {}
  explicit Drive(const uint8_t duty, const uint8_t phase) : phase(phase), duty(duty) {}

  uint8_t phase;
  uint8_t duty;
};

struct DelayOffset {
  DelayOffset() : DelayOffset(0x00, 0x01) {}
  explicit DelayOffset(const uint8_t delay, const uint8_t offset) : delay(delay), offset(offset) {}

  uint8_t delay;
  uint8_t offset;
};

union Body {
  Drive drives[NUM_TRANS_IN_UNIT];
  DelayOffset delay_offsets[NUM_TRANS_IN_UNIT];
};

struct RxMessage final {
  uint8_t ack;
  uint8_t msg_id;
};

}  // namespace core
}  // namespace autd
