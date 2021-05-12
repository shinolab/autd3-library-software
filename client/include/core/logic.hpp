// File: logic.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ec_config.hpp"
#include "firmware_version.hpp"
#include "gain.hpp"
#include "link.hpp"
#include "modulation.hpp"
#include "result.hpp"

namespace autd::core {

using std::unique_ptr;

constexpr uint8_t CMD_OP = 0x00;
constexpr uint8_t CMD_READ_CPU_VER_LSB = 0x02;
constexpr uint8_t CMD_READ_CPU_VER_MSB = 0x03;
constexpr uint8_t CMD_READ_FPGA_VER_LSB = 0x04;
constexpr uint8_t CMD_READ_FPGA_VER_MSB = 0x05;
constexpr uint8_t CMD_SEQ_MODE = 0x06;
constexpr uint8_t CMD_INIT_FPGA_REF_CLOCK = 0x07;
constexpr uint8_t CMD_CLEAR = 0x09;

class Logic {
 public:
  static Result<bool, std::string> Clear(const LinkPtr link, const size_t num_devices, uint8_t* tx, uint8_t* rx) {
    return SendHeader(link, num_devices, CMD_CLEAR, tx, rx);
  }

  static Result<bool, std::string> Synchronize(const LinkPtr link, const size_t num_devices, uint8_t* tx, uint8_t* rx, Configuration config) {
    uint8_t msg_id = 0;
    PackHeader(CMD_INIT_FPGA_REF_CLOCK, false, false, tx, &msg_id);
    size_t size = 0;
    auto res = PackSyncBody(config, num_devices, tx, &size);
    if (res.is_err()) return res;

    return WaitMsgProcessed(link, msg_id, num_devices, rx, 5000);
  }

  static Result<bool, std::string> SendHeader(const LinkPtr link, const size_t num_devices, const uint8_t cmd, uint8_t* tx, uint8_t* rx) {
    const auto send_size = sizeof(RxGlobalHeader);
    uint8_t msg_id = 0;
    PackHeader(cmd, false, false, tx, &msg_id);
    if (auto res = link->Send(send_size, tx); res.is_err()) return res;
    return WaitMsgProcessed(link, msg_id, num_devices, rx, 50);
  }

  [[nodiscard]] static Result<bool, std::string> WaitMsgProcessed(LinkPtr link, const uint8_t msg_id, const size_t num_devices, uint8_t* rx,
                                                                  const size_t max_trial = 200) {
    if (link == nullptr || !link->is_open()) return Ok(false);

    const auto buffer_len = num_devices * EC_INPUT_FRAME_SIZE;
    for (size_t i = 0; i < max_trial; i++) {
      if (auto res = link->Read(&rx[0], buffer_len); res.is_err()) return res;

      size_t processed = 0;
      for (size_t dev = 0; dev < num_devices; dev++)
        if (const uint8_t proc_id = rx[dev * 2 + 1]; proc_id == msg_id) processed++;

      if (processed == num_devices) return Ok(true);

      auto wait =
          static_cast<size_t>(std::ceil(static_cast<double>(EC_TRAFFIC_DELAY) * 1000 / EC_DEVICE_PER_FRAME * static_cast<double>(num_devices)));
      std::this_thread::sleep_for(std::chrono::milliseconds(wait));
    }

    return Ok(false);
  }

  [[nodiscard]] static Result<std::vector<FirmwareInfo>, std::string> firmware_info_list(const LinkPtr link, const size_t num_devices, uint8_t* tx,
                                                                                         uint8_t* rx) {
    auto concat_byte = [](const uint8_t high, const uint16_t low) { return static_cast<uint16_t>(static_cast<uint16_t>(high) << 8 | low); };

    Result<bool, std::string> res = Ok(true);

    std::vector<uint16_t> cpu_versions(num_devices);
    res = SendHeader(link, num_devices, CMD_READ_CPU_VER_LSB, &tx[0], &rx[0]);
    if (res.is_err()) return Err(res.unwrap_err());
    for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = rx[2 * i];
    res = SendHeader(link, num_devices, CMD_READ_CPU_VER_MSB, &tx[0], &rx[0]);
    if (res.is_err()) return Err(res.unwrap_err());
    for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = concat_byte(rx[2 * i], cpu_versions[i]);

    std::vector<uint16_t> fpga_versions(num_devices);
    res = SendHeader(link, num_devices, CMD_READ_FPGA_VER_LSB, &tx[0], &rx[0]);
    if (res.is_err()) return Err(res.unwrap_err());
    for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = rx[2 * i];
    res = SendHeader(link, num_devices, CMD_READ_FPGA_VER_LSB, &tx[0], &rx[0]);
    if (res.is_err()) return Err(res.unwrap_err());
    for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = concat_byte(rx[2 * i], fpga_versions[i]);

    std::vector<FirmwareInfo> infos;
    for (size_t i = 0; i < num_devices; i++) infos.emplace_back(FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]));
    return Ok(std::move(infos));
  }

  static uint8_t get_id() {
    static std::atomic<uint8_t> id{0};

    id.fetch_add(0x01);
    uint8_t expected = 0xff;
    id.compare_exchange_weak(expected, 0);

    return id.load();
  }

  static void PackHeader(const uint8_t cmd, const bool silent_mode, const bool seq_mode, uint8_t* data, uint8_t* const msg_id) {
    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    *msg_id = get_id();
    header->msg_id = *msg_id;
    header->control_flags = 0;
    header->mod_size = 0;
    header->command = cmd;

    if (seq_mode) header->control_flags |= SEQ_MODE;
    if (silent_mode) header->control_flags |= SILENT;
  }

  static void PackHeader(const ModulationPtr& mod, const bool silent_mode, const bool seq_mode, uint8_t* data, uint8_t* const msg_id) {
    PackHeader(CMD_OP, silent_mode, seq_mode, data, msg_id);
    if (mod == nullptr) return;
    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    const auto mod_size = static_cast<uint8_t>(std::clamp(mod->buffer().size() - mod->sent(), size_t{0}, MOD_FRAME_SIZE));
    header->mod_size = mod_size;
    if (mod->sent() == 0) header->control_flags |= MOD_BEGIN;
    if (mod->sent() + mod_size >= mod->buffer().size()) header->control_flags |= MOD_END;

    std::memcpy(header->mod, &mod->buffer()[mod->sent()], mod_size);
    mod->sent() += mod_size;
  }

  static void PackBody(const GainPtr& gain, uint8_t* data, size_t* size) {
    const auto num_devices = gain != nullptr ? gain->data().size() : 0;

    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    if (gain == nullptr) return;

    auto* cursor = data + sizeof(RxGlobalHeader);
    const auto byte_size = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
    for (size_t i = 0; i < num_devices; i++) {
      std::memcpy(cursor, &gain->data()[i].at(0), byte_size);
      cursor += byte_size;
    }
  }

 private:
  [[nodiscard]] static Result<bool, std::string> PackSyncBody(const Configuration config, const size_t num_devices, uint8_t* data,
                                                              size_t* const size) {
    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;

    const auto mod_sampling_freq = static_cast<uint32_t>(config.mod_sampling_freq());
    const auto mod_buf_size = static_cast<uint32_t>(config.mod_buf_size());

    if (mod_buf_size < mod_sampling_freq) return Err(std::string("Modulation buffer size must be not less than sampling frequency"));

    const auto mod_idx_shift = Log2U(MOD_SAMPLING_FREQ_BASE / mod_sampling_freq);
    const auto ref_clk_cyc_shift = Log2U(mod_buf_size / mod_sampling_freq);

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
    for (size_t i = 0; i < num_devices; i++) {
      cursor[0] = mod_idx_shift;
      cursor[1] = ref_clk_cyc_shift;
      cursor += NUM_TRANS_IN_UNIT;
    }

    return Ok(true);
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
};
}  // namespace autd::core
