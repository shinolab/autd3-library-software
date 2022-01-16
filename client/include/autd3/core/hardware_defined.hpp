// File: hardware_defined.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 15/01/2022
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

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
  READS_FPGA_INFO = 1 << 2,
  SILENT = 1 << 3,
  FORCE_FAN = 1 << 4,
  SEQ_MODE = 1 << 5,
  SEQ_GAIN_MODE = 1 << 6,
};

enum CPU_CONTROL_FLAGS : uint8_t {
  MOD_BEGIN = 1 << 0,
  MOD_END = 1 << 1,
  SEQ_BEGIN = 1 << 2,
  SEQ_END = 1 << 3,
  SET_SILENT_STEP = 1 << 4,
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

/**
 * \brief Drive contains phase and duty ratio data of a transducer
 */
struct Drive final {
  Drive() : Drive(0x00, 0x00) {}
  explicit Drive(const uint8_t duty, const uint8_t phase) : phase(phase), duty(duty) {}

  uint8_t phase;
  uint8_t duty;
};

/**
 * \brief DelayOffset contains delay and duty offset data of a transducer
 */
struct DelayOffset {
  DelayOffset() : DelayOffset(0x00, 0x01) {}
  explicit DelayOffset(const uint8_t delay, const uint8_t offset) : delay(delay), offset(offset) {}

  uint8_t delay;
  uint8_t offset;
};

/**
 * \brief Body represents the data per device.
 */
union Body {
  Drive drives[NUM_TRANS_IN_UNIT];
  DelayOffset delay_offsets[NUM_TRANS_IN_UNIT];
};

/**
 * \brief RxMessage contains a message id and ack which send from each device
 */
struct RxMessage final {
  uint8_t ack;
  uint8_t msg_id;
};

/**
 * \brief FPGAInfo contains a FPGA information of a device
 */
class FPGAInfo final {
 public:
  FPGAInfo() : _info(0) {}

  void set(const RxMessage& rx) { _info = rx.ack; }

  [[nodiscard]] bool is_running_fan() const { return (_info & 0x01) != 0; }

 private:
  uint8_t _info;
};

/**
 * \brief TxDatagram represents the data sent to devices.
 */
class TxDatagram final {
 public:
  TxDatagram() : _data(nullptr), _header_size(0), _body_size(0), _num_bodies(0) {}

  explicit TxDatagram(const size_t device_num) : _header_size(sizeof(GlobalHeader)), _body_size(sizeof(Body)), _num_bodies(device_num) {
    _data = std::make_unique<uint8_t[]>(_header_size + _num_bodies * _body_size);
  }

  [[nodiscard]] const uint8_t* data() const { return _data.get(); }
  [[nodiscard]] const GlobalHeader* header() const { return reinterpret_cast<const GlobalHeader*>(_data.get()); }
  [[nodiscard]] const Body* body(const size_t i) const { return reinterpret_cast<const Body*>(&_data[_header_size + _body_size * i]); }

  uint8_t* data() { return _data.get(); }
  GlobalHeader* header() { return reinterpret_cast<GlobalHeader*>(_data.get()); }
  Body* body(const size_t i) { return reinterpret_cast<Body*>(&_data[_header_size + _body_size * i]); }

  [[nodiscard]] size_t header_size() const { return _header_size; }
  [[nodiscard]] size_t body_size() const { return _body_size; }
  [[nodiscard]] size_t num_bodies() const { return _num_bodies; }
  size_t& num_bodies() { return _num_bodies; }

  [[nodiscard]] size_t size() const { return _header_size + _num_bodies * _body_size; }

  void copy_from(const TxDatagram& other) {
    _header_size = other.header_size();
    _body_size = other.body_size();
    _num_bodies = other.num_bodies();
    std::memcpy(_data.get(), other.data(), _header_size + _num_bodies * _body_size);
  }

 private:
  std::unique_ptr<uint8_t[]> _data;
  size_t _header_size;
  size_t _body_size;
  size_t _num_bodies;
};

/**
 * \brief RxDatagram represents the data received from devices.
 */
class RxDatagram final {
 public:
  RxDatagram() = default;
  explicit RxDatagram(const size_t device_num) { _data.resize(device_num); }

  uint8_t* data() { return reinterpret_cast<uint8_t*>(_data.data()); }
  [[nodiscard]] const uint8_t* data() const { return reinterpret_cast<const uint8_t*>(_data.data()); }

  RxMessage const& operator[](const size_t i) const { return _data[i]; }
  RxMessage& operator[](const size_t i) { return _data[i]; }

  [[nodiscard]] size_t size() const { return _data.size() * sizeof(RxMessage); }
  [[nodiscard]] size_t num_messages() const { return _data.size(); }

  [[nodiscard]] std::vector<RxMessage>::const_iterator begin() const { return _data.begin(); }
  [[nodiscard]] std::vector<RxMessage>::const_iterator end() const { return _data.end(); }
  [[nodiscard]] std::vector<RxMessage>::iterator begin() { return _data.begin(); }
  [[nodiscard]] std::vector<RxMessage>::iterator end() { return _data.end(); }

  void copy_from(const RxMessage* const rx) { std::memcpy(data(), rx, _data.size() * sizeof(RxMessage)); }

 private:
  std::vector<RxMessage> _data;
};

/**
 * \brief check if the data which have msg_id have been processed in the devices.
 * \param msg_id message id
 * \param rx pointer to received data
 * \return whether the data have been processed
 */
static bool is_msg_processed(const uint8_t msg_id, const RxDatagram& rx) {
  size_t processed = 0;
  for (auto& msg : rx)
    if (msg.msg_id == msg_id) processed++;
  return processed == rx.num_messages();
}

}  // namespace core
}  // namespace autd
