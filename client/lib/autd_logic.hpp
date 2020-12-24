// File: autd_logic.hpp
// Project: lib
// Created Date: 22/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 24/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "consts.hpp"
#include "ec_config.hpp"
#include "firmware_version.hpp"
#include "gain.hpp"
#include "link.hpp"
#include "modulation.hpp"
#include "privdef.hpp"
#include "sequence.hpp"

namespace autd::_internal {

using std::unique_ptr;

constexpr uint8_t CMD_OP = 0x00;
constexpr uint8_t CMD_BRAM_WRITE = 0x01;
constexpr uint8_t CMD_READ_CPU_VER_LSB = 0x02;
constexpr uint8_t CMD_READ_CPU_VER_MSB = 0x03;
constexpr uint8_t CMD_READ_FPGA_VER_LSB = 0x04;
constexpr uint8_t CMD_READ_FPGA_VER_MSB = 0x05;
constexpr uint8_t CMD_SEQ_MODE = 0x06;
constexpr uint8_t CMD_INIT_REF_CLOCK = 0x07;
constexpr uint8_t CMD_CALIB_SEQ_CLOCK = 0x08;
constexpr uint8_t CMD_CLEAR = 0x09;
constexpr uint8_t CMD_SET_DELAY = 0x0A;

constexpr uint8_t OP_MODE_MSG_ID_MIN = 0x20;
constexpr uint8_t OP_MODE_MSG_ID_MAX = 0xBF;

class AUTDLogic {
 public:
  AUTDLogic();

  bool is_open();
  GeometryPtr geometry();
  bool &silent_mode();

  void OpenWith(LinkPtr link);

  void BuildGain(GainPtr gain);
  void BuildModulation(ModulationPtr mod);

  void Send(GainPtr gain, ModulationPtr mod);
  void SendBlocking(GainPtr gain, ModulationPtr mod);
  void SendBlocking(SequencePtr seq);
  bool SendBlocking(size_t size, unique_ptr<uint8_t[]> data, size_t trial);
  void SendData(size_t size, unique_ptr<uint8_t[]> data);

  bool WaitMsgProcessed(uint8_t msg_id, size_t max_trial = 200, uint8_t mask = 0xFF);
  bool Calibrate(Configuration config);
  void CalibrateSeq();
  bool Clear();
  void Close();
  void SetDelay(std::vector<std::array<uint16_t, NUM_TRANS_IN_UNIT>> &delay);
  FirmwareInfoList firmware_info_list();

  unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id);
  unique_ptr<uint8_t[]> MakeBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id);
  unique_ptr<uint8_t[]> MakeCalibBody(Configuration config, size_t *const size);
  unique_ptr<uint8_t[]> MakeCalibSeqBody(std::vector<uint16_t> comps, size_t *const size);

 private:
  static uint8_t get_id() {
    static std::atomic<uint8_t> id{OP_MODE_MSG_ID_MIN - 1};

    id.fetch_add(0x01);
    uint8_t expected = OP_MODE_MSG_ID_MAX + 1;
    id.compare_exchange_weak(expected, OP_MODE_MSG_ID_MIN);

    return id.load();
  }

  static inline uint16_t log2u(const uint32_t x) {
#ifdef _MSC_VER
    unsigned long n;  // NOLINT
    _BitScanReverse(&n, x);
#else
    uint32_t n;
    n = 31 - __builtin_clz(x);
#endif
    return static_cast<uint16_t>(n);
  }

  GeometryPtr _geometry;
  LinkPtr _link;

  std::vector<uint8_t> _rx_data;

  bool _seq_mode;
  bool _silent_mode;
  Configuration _config;
};
}  // namespace autd::_internal
