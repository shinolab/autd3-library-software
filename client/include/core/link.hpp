// File: link.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "result.hpp"

namespace autd::core {

class Link;
using LinkPtr = std::shared_ptr<Link>;

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

  /**
   * @brief Open link
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] virtual Error open() = 0;
  /**
   * @brief Close link
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] virtual Error close() = 0;
  /**
   * @brief  Send data to devices
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] virtual Error send(size_t size, const uint8_t* buf) = 0;
  /**
   * @brief  Read data from devices
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] virtual Error read(uint8_t* rx, size_t buffer_len) = 0;

  [[nodiscard]] virtual bool is_open() = 0;
};
}  // namespace autd::core
