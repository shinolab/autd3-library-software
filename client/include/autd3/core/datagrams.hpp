// File: datagrams.hpp
// Project: core
// Created Date: 13/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 15/01/2022
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "interface.hpp"

namespace autd::core::datagram {

/**
 * @brief IDatagramHeader with common data
 */
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

/**
 * @brief IDatagramHeader with special message id
 */
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

/**
 * @brief IDatagramHeader with silent step data
 */
class SilentStepHeader final : public IDatagramHeader {
 public:
  void init() override {}

  void pack(const uint8_t msg_id, TxDatagram& tx, const uint8_t fpga_ctrl_flag, const uint8_t cpu_ctrl_flag) override {
    auto* header = reinterpret_cast<GlobalHeader*>(tx.data());
    header->msg_id = msg_id;
    header->fpga_ctrl_flags = (header->fpga_ctrl_flags & ~_fpga_flag_mask) | (fpga_ctrl_flag & _fpga_flag_mask);
    header->cpu_ctrl_flags = cpu_ctrl_flag | SET_SILENT_STEP;
    header->mod_size = _silent_step;
    tx.num_bodies() = 0;
  }

  [[nodiscard]] bool is_finished() const override { return true; }

  explicit SilentStepHeader(const uint8_t silent_step, const uint8_t fpga_flag_mask) noexcept
      : _silent_step(silent_step), _fpga_flag_mask(fpga_flag_mask) {}
  ~SilentStepHeader() override = default;
  SilentStepHeader(const SilentStepHeader& v) noexcept = delete;
  SilentStepHeader& operator=(const SilentStepHeader& obj) = delete;
  SilentStepHeader(SilentStepHeader&& obj) = default;
  SilentStepHeader& operator=(SilentStepHeader&& obj) = default;

 private:
  uint8_t _silent_step;
  uint8_t _fpga_flag_mask;
};

/**
 * @brief NullBody is used for TxDatagram without Body data
 */
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

}  // namespace autd::core::datagram
