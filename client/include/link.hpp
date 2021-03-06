﻿// File: link.hpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 27/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <optional>

namespace autd {
namespace link {
/**
 * @brief Link is the interface to the AUTD device
 */
class Link {
 public:
  Link() = default;
  virtual ~Link() = default;
  Link(const Link& v) = delete;
  Link& operator=(const Link& obj) = delete;
  Link(Link&& obj) = delete;
  Link& operator=(Link&& obj) = delete;

  virtual void Open() = 0;
  virtual void Close() = 0;
  /**
   * @brief  Send data to devices
   * @return return nullopt if no error, otherwise return error code.
   */
  virtual std::optional<int32_t> Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  /**
   * @brief  Read data from devices
   * @return return nullopt if no error, otherwise return error code.
   */
  virtual std::optional<int32_t> Read(uint8_t* rx, uint32_t buffer_len) = 0;
  virtual bool is_open() = 0;
};
}  // namespace link
using LinkPtr = std::unique_ptr<link::Link>;
}  // namespace autd
