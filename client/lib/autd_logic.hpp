// File: autd_logic.hpp
// Project: lib
// Created Date: 22/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "firmware_version.hpp"
#include "gain.hpp"
#include "link.hpp"
#include "modulation.hpp"
#include "result.hpp"
#include "sequence.hpp"

namespace autd::internal {

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

constexpr uint8_t OP_MODE_MSG_ID_MIN = 0x20;
constexpr uint8_t OP_MODE_MSG_ID_MAX = 0xBF;

class AUTDLogic {
 public:
  AUTDLogic();

  [[nodiscard]] bool is_open() const;
  [[nodiscard]] GeometryPtr geometry() const noexcept;
  bool& silent_mode() noexcept;

  Result<bool, std::string> OpenWith(LinkPtr link);

  void BuildGain(const GainPtr& gain);
  void BuildModulation(const ModulationPtr& mod) const;

  Result<bool, std::string> Send(const GainPtr& gain, const ModulationPtr& mod);
  Result<bool, std::string> SendBlocking(const GainPtr& gain, const ModulationPtr& mod);
  Result<bool, std::string> SendBlocking(const SequencePtr& seq);
  Result<bool, std::string> SendBlocking(size_t size, unique_ptr<uint8_t[]> data, size_t trial);
  [[nodiscard]] Result<bool, std::string> SendData(size_t size, unique_ptr<uint8_t[]> data) const;

  Result<bool, std::string> WaitMsgProcessed(uint8_t msg_id, size_t max_trial = 200, uint8_t mask = 0xFF);
  Result<bool, std::string> Synchronize(Configuration config);
  Result<bool, std::string> SynchronizeSeq();
  Result<bool, std::string> Clear();
  Result<bool, std::string> Close();
  std::vector<FirmwareInfo> firmware_info_list();

  unique_ptr<uint8_t[]> MakeBody(const GainPtr& gain, const ModulationPtr& mod, size_t* size, uint8_t* send_msg_id) const;
  unique_ptr<uint8_t[]> MakeBody(const SequencePtr& seq, size_t* size, uint8_t* send_msg_id) const;
  Result<unique_ptr<uint8_t[]>, std::string> MakeCalibBody(Configuration config, size_t* size);
  unique_ptr<uint8_t[]> MakeCalibSeqBody(const std::vector<uint16_t>& comps, size_t* size) const;

 private:
  static uint8_t get_id() {
    static std::atomic<uint8_t> id{OP_MODE_MSG_ID_MIN - 1};

    id.fetch_add(0x01);
    uint8_t expected = OP_MODE_MSG_ID_MAX + 1;
    id.compare_exchange_weak(expected, OP_MODE_MSG_ID_MIN);

    return id.load();
  }

  static uint16_t Log2U(const uint32_t x) {
#ifdef _MSC_VER
    unsigned long n;         // NOLINT
    _BitScanReverse(&n, x);  // NOLINT
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
}  // namespace autd::internal
