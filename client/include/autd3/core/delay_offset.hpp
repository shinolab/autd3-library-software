// File: delay_offset.hpp
// Project: core
// Created Date: 12/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "hardware_defined.hpp"
#include "interface.hpp"

namespace autd::core {
/**
 * @brief DelayOffsets controls the duty offset and delay of each transducer in AUTD devices.
 */
class DelayOffsets final : public IDatagramBody {
 public:
  void init() override {}

  uint8_t pack(const Geometry&, TxDatagram& tx, uint8_t&& fpga_ctrl_flag, uint8_t&& cpu_ctrl_flag) override {
    const auto msg_id = get_id();
    cpu_ctrl_flag |= WRITE_BODY;
    std::memcpy(tx.body(0), _data.data(), _data.size() * sizeof(DelayOffset));
    return msg_id;
  }

  [[nodiscard]] bool is_finished() const override { return true; }

  DelayOffsets() noexcept = default;
  ~DelayOffsets() override = default;
  DelayOffsets(const DelayOffsets& v) noexcept = delete;
  DelayOffsets& operator=(const DelayOffsets& obj) = delete;
  DelayOffsets(DelayOffsets&& obj) = default;
  DelayOffsets& operator=(DelayOffsets&& obj) = default;

 private:
  std::vector<DelayOffset> _data;
};
}  // namespace autd::core
