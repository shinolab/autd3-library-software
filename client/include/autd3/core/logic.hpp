// File: logic.hpp
// Project: core
// Created Date: 13/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "interface.hpp"

namespace autd::core {

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

  void pack(const uint8_t msg_id, TxDatagram& tx, const uint8_t fpga_ctrl_flag, const uint8_t cpu_ctrl_flag) override {
    auto* header = reinterpret_cast<GlobalHeader*>(tx.data());
    header->msg_id = msg_id;
    header->fpga_ctrl_flags = (header->fpga_ctrl_flags & ~_fpga_flag_mask) | (fpga_ctrl_flag & _fpga_flag_mask);
    header->cpu_ctrl_flags = cpu_ctrl_flag;
    header->mod_size = 0;
    tx.num_bodies() = 0;
  }

  [[nodiscard]] bool is_finished() const override { return true; }

  explicit CommonHeader(const uint8_t fpga_flag_mask) noexcept : _fpga_flag_mask(fpga_flag_mask) {}
  ~CommonHeader() override = default;
  CommonHeader(const CommonHeader& v) noexcept = delete;
  CommonHeader& operator=(const CommonHeader& obj) = delete;
  CommonHeader(CommonHeader&& obj) = default;
  CommonHeader& operator=(CommonHeader&& obj) = default;

 private:
  uint8_t _fpga_flag_mask;
};

class SpecialMessageIdHeader final : public IDatagramHeader {
 public:
  void init() override {}

  void pack(uint8_t, TxDatagram& tx, const uint8_t fpga_ctrl_flag, const uint8_t cpu_ctrl_flag) override {
    auto* header = reinterpret_cast<GlobalHeader*>(tx.data());
    header->msg_id = _msg_id;
    header->fpga_ctrl_flags = (header->fpga_ctrl_flags & ~_fpga_flag_mask) | (fpga_ctrl_flag & _fpga_flag_mask);
    header->cpu_ctrl_flags = cpu_ctrl_flag;
    header->mod_size = 0;
    tx.num_bodies() = 0;
  }

  [[nodiscard]] bool is_finished() const override { return true; }

  explicit SpecialMessageIdHeader(const uint8_t msg_id, const uint8_t fpga_flag_mask) noexcept : _msg_id(msg_id), _fpga_flag_mask(fpga_flag_mask) {}
  ~SpecialMessageIdHeader() override = default;
  SpecialMessageIdHeader(const SpecialMessageIdHeader& v) noexcept = delete;
  SpecialMessageIdHeader& operator=(const SpecialMessageIdHeader& obj) = delete;
  SpecialMessageIdHeader(SpecialMessageIdHeader&& obj) = default;
  SpecialMessageIdHeader& operator=(SpecialMessageIdHeader&& obj) = default;

 private:
  uint8_t _msg_id;
  uint8_t _fpga_flag_mask;
};

class NullBody final : public IDatagramBody {
 public:
  void init() override {}

  void pack(const Geometry&, TxDatagram& tx) override { tx.num_bodies() = 0; }

  [[nodiscard]] bool is_finished() const override { return true; }

  NullBody() noexcept = default;
  ~NullBody() override = default;
  NullBody(const NullBody& v) noexcept = delete;
  NullBody& operator=(const NullBody& obj) = delete;
  NullBody(NullBody&& obj) = default;
  NullBody& operator=(NullBody&& obj) = default;
};

}  // namespace autd::core
