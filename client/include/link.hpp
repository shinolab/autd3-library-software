// File: link.hpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "core.hpp"

namespace autd::link {
/**
 * @brief Link is the interface to the AUTD device
 */
class Link {
 public:
  virtual ~Link() {}
  virtual void Open() = 0;
  virtual void Close() = 0;
  virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  virtual std::vector<uint8_t> Read(uint32_t buffer_len) = 0;
  virtual bool is_open() = 0;
};
}  // namespace autd::link
