// File: twincat_link.hpp
// Project: include
// Created Date: 10/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
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
class TwinCATLink : public core::Link {
 public:
  /**
   * @brief Create TwinCATLink
   */
  static core::LinkPtr create();

  TwinCATLink() = default;
  ~TwinCATLink() override = default;
  TwinCATLink(const TwinCATLink& v) noexcept = delete;
  TwinCATLink& operator=(const TwinCATLink& obj) = delete;
  TwinCATLink(TwinCATLink&& obj) = delete;
  TwinCATLink& operator=(TwinCATLink&& obj) = delete;

  Error open() override = 0;
  Error close() override = 0;
  Error send(size_t size, const uint8_t* buf) override = 0;
  Error read(uint8_t* rx, size_t buffer_len) override = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
