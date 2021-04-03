// File: link.hpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "result.hpp"

namespace autd {
namespace link {
/**
 * @brief Link is the interface to the AUTD device
 */
class Link {
 public:
  Link() {}
  virtual ~Link() noexcept(false) {}
  Link(const Link& v) = delete;
  Link& operator=(const Link& obj) = delete;
  Link(Link&& obj) = delete;
  Link& operator=(Link&& obj) = delete;

  virtual Result<bool, std::string> Open() = 0;
  virtual Result<bool, std::string> Close() = 0;
  /**
   * @brief  Send data to devices
   * @return return whether success to send data, or Err with error message if some unrecoverable error ocurred
   */
  virtual Result<bool, std::string> Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  /**
   * @brief  Read data from devices
   * @return return whether success to read data, or Err with error message if some unrecoverable error ocurred
   */
  virtual Result<bool, std::string> Read(uint8_t* rx, uint32_t buffer_len) = 0;
  virtual bool is_open() = 0;
};
}  // namespace link
using LinkPtr = std::unique_ptr<link::Link>;
}  // namespace autd
