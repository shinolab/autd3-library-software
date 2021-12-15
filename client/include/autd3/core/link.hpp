// File: link.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "hardware_defined.hpp"

namespace autd::core {

/**
 * @brief Link is the interface to the AUTD device
 */
class Link {
 public:
  Link() = default;
  virtual ~Link() = default;
  Link(const Link& v) = delete;
  Link& operator=(const Link& obj) = delete;
  Link(Link&& obj) = default;
  Link& operator=(Link&& obj) = default;

  /**
   * @brief Open link
   */
  virtual void open() = 0;

  /**
   * @brief Close link
   */
  virtual void close() = 0;

  /**
   * @brief  Send data to devices
   */
  virtual void send(const TxDatagram& tx) = 0;

  /**
   * @brief  Read data from devices
   */
  virtual void receive(RxDatagram& rx) = 0;

  [[nodiscard]] virtual bool is_open() = 0;
};

using LinkPtr = std::unique_ptr<Link>;

}  // namespace autd::core
