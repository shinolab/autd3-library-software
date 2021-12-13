// File: logic.hpp
// Project: core
// Created Date: 13/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>

#include "interface.hpp"

namespace autd::core {
/**
 * \brief Get unique message id
 * \return message id
 */
static uint8_t get_id() {
  static std::atomic id{MSG_NORMAL_BASE};
  uint8_t expected = 0xff;
  if (!id.compare_exchange_weak(expected, MSG_NORMAL_BASE)) id.fetch_add(0x01);
  return id.load();
}

/**
 * \brief check if the data which have msg_id have been processed in the devices.
 * \param num_devices number of devices
 * \param msg_id message id
 * \param rx pointer to received data
 * \return whether the data have been processed
 */
static bool is_msg_processed(const size_t num_devices, const uint8_t msg_id, const RxDatagram& rx) {
  size_t processed = 0;
  for (auto& [ack, rx_msg_id] : rx)
    if (rx_msg_id == msg_id) processed++;
  return processed == num_devices;
}

class CommonHeader final : public IDatagramHeader {
 public:
  void init() override {}

  uint8_t pack(TxDatagram& tx, uint8_t& fpga_ctrl_flag, uint8_t& cpu_ctrl_flag) override {
    const auto msg_id = get_id();
    auto* header = reinterpret_cast<GlobalHeader*>(tx.data());
    header->msg_id = msg_id;
    header->fpga_ctrl_flags = fpga_ctrl_flag;
    header->cpu_ctrl_flags = cpu_ctrl_flag;
    header->mod_size = 0;
    tx.num_bodies() = 0;
    return msg_id;
  }

  [[nodiscard]] bool is_finished() const override { return true; }

  CommonHeader() noexcept = default;
  ~CommonHeader() override = default;
  CommonHeader(const CommonHeader& v) noexcept = delete;
  CommonHeader& operator=(const CommonHeader& obj) = delete;
  CommonHeader(CommonHeader&& obj) = default;
  CommonHeader& operator=(CommonHeader&& obj) = default;
};

class SpecialMessageIdHeader final : public IDatagramHeader {
 public:
  void init() override {}

  uint8_t pack(TxDatagram& tx, uint8_t& fpga_ctrl_flag, uint8_t& cpu_ctrl_flag) override {
    auto* header = reinterpret_cast<GlobalHeader*>(tx.data());
    header->msg_id = _msg_id;
    header->fpga_ctrl_flags = fpga_ctrl_flag;
    header->cpu_ctrl_flags = cpu_ctrl_flag;
    header->mod_size = 0;
    tx.num_bodies() = 0;
    return _msg_id;
  }

  [[nodiscard]] bool is_finished() const override { return true; }

  explicit SpecialMessageIdHeader(const uint8_t msg_id) noexcept : _msg_id(msg_id) {}
  ~SpecialMessageIdHeader() override = default;
  SpecialMessageIdHeader(const SpecialMessageIdHeader& v) noexcept = delete;
  SpecialMessageIdHeader& operator=(const SpecialMessageIdHeader& obj) = delete;
  SpecialMessageIdHeader(SpecialMessageIdHeader&& obj) = default;
  SpecialMessageIdHeader& operator=(SpecialMessageIdHeader&& obj) = default;

 private:
  uint8_t _msg_id;
};

class NullBody final : public IDatagramBody {
 public:
  void init() override {}

  void pack(const Geometry&, TxDatagram&, uint8_t&, uint8_t&) override {}

  [[nodiscard]] bool is_finished() const override { return true; }

  NullBody() noexcept = default;
  ~NullBody() override = default;
  NullBody(const NullBody& v) noexcept = delete;
  NullBody& operator=(const NullBody& obj) = delete;
  NullBody(NullBody&& obj) = default;
  NullBody& operator=(NullBody&& obj) = default;
};

}  // namespace autd::core
