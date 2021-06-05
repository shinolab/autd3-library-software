// File: twincat_link.hpp
// Project: include
// Created Date: 10/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "core/link.hpp"

namespace autd::link {

/**
 * @brief Link using TwinCAT
 */
class TwinCAT : public core::Link {
 public:
  /**
   * @brief Create TwinCAT
   */
  static core::LinkPtr create();

  TwinCAT() = default;
  ~TwinCAT() override = default;
  TwinCAT(const TwinCAT& v) noexcept = delete;
  TwinCAT& operator=(const TwinCAT& obj) = delete;
  TwinCAT(TwinCAT&& obj) = delete;
  TwinCAT& operator=(TwinCAT&& obj) = delete;

  Error open() override = 0;
  Error close() override = 0;
  Error send(const uint8_t* buf, size_t size) override = 0;
  Error read(uint8_t* rx, size_t buffer_len) override = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
