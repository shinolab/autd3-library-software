// File: link_soem.hpp
// Project: include
// Created Date: 10/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/link.hpp"

namespace autd::link {
using EtherCATAdapter = std::pair<std::string, std::string>;
using EtherCATAdapters = std::vector<EtherCATAdapter>;

/**
 * @brief Link using [SOEM](https://github.com/OpenEtherCATsociety/SOEM)
 */
class SOEMLink : virtual public core::Link {
 public:
  /**
   * @brief Create SOEM link.
   * @param[in] ifname Network interface name. (e.g. eth0)
   * @param[in] device_num The number of AUTD you connected.
   *
   * @details Available Network interface names are obtained by EnumerateAdapters().
   *          The numbers of connected devices is obtained by Geometry::num_devices().
   */
  static core::LinkPtr Create(const std::string& ifname, size_t device_num);

  /**
   * @brief Enumerate Ethernet adapters of the computer.
   */
  static EtherCATAdapters EnumerateAdapters(size_t* size);
  SOEMLink() = default;
  ~SOEMLink() override = default;
  SOEMLink(const SOEMLink& v) noexcept = delete;
  SOEMLink& operator=(const SOEMLink& obj) = delete;
  SOEMLink(SOEMLink&& obj) = delete;
  SOEMLink& operator=(SOEMLink&& obj) = delete;

  Result<bool, std::string> Open() override = 0;
  Result<bool, std::string> Close() override = 0;
  Result<bool, std::string> Send(size_t size, const uint8_t* buf) override = 0;
  Result<bool, std::string> Read(uint8_t* rx, size_t buffer_len) override = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
