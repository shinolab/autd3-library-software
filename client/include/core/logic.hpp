// File: logic.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "firmware_version.hpp"
#include "gain.hpp"
#include "link.hpp"
#include "modulation.hpp"
#include "result.hpp"

namespace autd::core {

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

  [[nodiscard]] Result<bool, std::string> OpenWith(LinkPtr link);

  [[nodiscard]] Result<bool, std::string> BuildGain(const GainPtr& gain);
  [[nodiscard]] Result<bool, std::string> BuildModulation(const ModulationPtr& mod) const;

  [[nodiscard]] Result<bool, std::string> Send(const GainPtr& gain, const ModulationPtr& mod);
  [[nodiscard]] Result<bool, std::string> SendBlocking(const GainPtr& gain, const ModulationPtr& mod);
  [[nodiscard]] Result<bool, std::string> SendBlocking(size_t size, const uint8_t* data, size_t trial);
  [[nodiscard]] Result<bool, std::string> SendData(size_t size, const uint8_t* data) const;

  [[nodiscard]] Result<bool, std::string> WaitMsgProcessed(uint8_t msg_id, size_t max_trial = 200, uint8_t mask = 0xFF);

  [[nodiscard]] Result<bool, std::string> Synchronize(Configuration config);

  [[nodiscard]] Result<bool, std::string> Clear();
  [[nodiscard]] Result<bool, std::string> Close();

  [[nodiscard]] Result<std::vector<FirmwareInfo>, std::string> firmware_info_list() {
    const auto size = this->_geometry->num_devices();

    std::vector<FirmwareInfo> infos;
    auto make_header = [](const uint8_t command) {
      auto header_bytes = std::make_unique<uint8_t[]>(sizeof(RxGlobalHeader));
      auto* header = reinterpret_cast<RxGlobalHeader*>(&header_bytes[0]);
      header->msg_id = command;
      header->command = command;
      return header_bytes;
    };

    auto concat_byte = [](const uint8_t high, const uint16_t low) { return static_cast<uint16_t>(static_cast<uint16_t>(high) << 8 | low); }

    std::vector<uint16_t>
        cpu_versions(size);
    std::vector<uint16_t> fpga_versions(size);

    const auto send_size = sizeof(RxGlobalHeader);
    auto header = make_header(CMD_READ_CPU_VER_LSB);
    auto res = this->SendData(send_size, &header[0]);
    if (res.is_err()) return Err(res.unwrap_err());

    res = WaitMsgProcessed(CMD_READ_CPU_VER_LSB, 50);
    if (res.is_err()) return Err(res.unwrap_err());

    for (size_t i = 0; i < size; i++) cpu_versions[i] = _rx_data[2 * i];

    header = make_header(CMD_READ_CPU_VER_MSB);
    res = this->SendData(send_size, &header[0]);
    if (res.is_err()) return Err(res.unwrap_err());
    res = WaitMsgProcessed(CMD_READ_CPU_VER_MSB, 50);
    if (res.is_err()) return Err(res.unwrap_err());

    for (size_t i = 0; i < size; i++) cpu_versions[i] = ConcatByte(_rx_data[2 * i], cpu_versions[i]);

    header = make_header(CMD_READ_FPGA_VER_LSB);
    res = this->SendData(send_size, &header[0]);
    if (res.is_err()) return Err(res.unwrap_err());

    res = WaitMsgProcessed(CMD_READ_FPGA_VER_LSB, 50);
    if (res.is_err()) return Err(res.unwrap_err());

    for (size_t i = 0; i < size; i++) fpga_versions[i] = _rx_data[2 * i];

    header = make_header(CMD_READ_FPGA_VER_MSB);
    res = this->SendData(send_size, &header[0]);

    if (res.is_err()) return Err(res.unwrap_err());
    res = WaitMsgProcessed(CMD_READ_FPGA_VER_MSB, 50);
    if (res.is_err()) return Err(res.unwrap_err());

    for (size_t i = 0; i < size; i++) fpga_versions[i] = ConcatByte(_rx_data[2 * i], fpga_versions[i]);

    for (size_t i = 0; i < size; i++) {
      auto info = FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]);
      infos.emplace_back(info);
    }
    return Ok(std::move(infos));
  }

  unique_ptr<uint8_t[]> AUTDLogic::MakeBody(const GainPtr& gain, const ModulationPtr& mod, size_t* const size, uint8_t* const send_msg_id) const {
    const auto num_devices = gain != nullptr ? gain->geometry()->num_devices() : 0;

    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    auto body = std::make_unique<uint8_t[]>(*size);

    auto* header = reinterpret_cast<RxGlobalHeader*>(&body[0]);
    *send_msg_id = get_id();
    header->msg_id = *send_msg_id;
    header->control_flags = 0;
    header->mod_size = 0;
    header->command = CMD_OP;

    if (this->_seq_mode) header->control_flags |= SEQ_MODE;
    if (this->_silent_mode) header->control_flags |= SILENT;

    if (mod != nullptr) {
      const auto mod_size = static_cast<uint8_t>(std::clamp(mod->buffer.size() - mod->sent(), size_t{0}, MOD_FRAME_SIZE));
      header->mod_size = mod_size;
      if (mod->sent() == 0) header->control_flags |= MOD_BEGIN;
      if (mod->sent() + mod_size >= mod->buffer.size()) header->control_flags |= MOD_END;

      std::memcpy(header->mod, &mod->buffer[mod->sent()], mod_size);
      mod->sent() += mod_size;
    }

    auto* cursor = &body[0] + sizeof(RxGlobalHeader);
    const auto byte_size = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
    if (gain != nullptr) {
      for (size_t i = 0; i < gain->geometry()->num_devices(); i++) {
        std::memcpy(cursor, &gain->data()[i].at(0), byte_size);
        cursor += byte_size;
      }
    }
    return body;
  }

  Result<unique_ptr<uint8_t[]>, std::string> MakeCalibBody(const Configuration config, size_t* const size) {
    this->_config = config;

    const auto num_devices = this->_geometry->num_devices();
    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    auto body = std::make_unique<uint8_t[]>(*size);

    auto* header = reinterpret_cast<RxGlobalHeader*>(&body[0]);
    header->msg_id = CMD_INIT_REF_CLOCK;
    header->command = CMD_INIT_REF_CLOCK;

    const auto mod_sampling_freq = static_cast<uint32_t>(_config.mod_sampling_freq());
    const auto mod_buf_size = static_cast<uint32_t>(_config.mod_buf_size());

    if (mod_buf_size < mod_sampling_freq) return Err(std::string("Modulation buffer size must be not less than sampling frequency"));

    const auto mod_idx_shift = Log2U(MOD_SAMPLING_FREQ_BASE / mod_sampling_freq);
    const auto ref_clk_cyc_shift = Log2U(mod_buf_size / mod_sampling_freq);

    auto* cursor = reinterpret_cast<uint16_t*>(&body[0] + sizeof(RxGlobalHeader));
    for (size_t i = 0; i < num_devices; i++) {
      cursor[0] = mod_idx_shift;
      cursor[1] = ref_clk_cyc_shift;
      cursor += NUM_TRANS_IN_UNIT;
    }

    return Ok(std::move(body));
  }

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
}  // namespace autd::core
