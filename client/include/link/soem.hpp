// File: soem_link.hpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 01/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#pragma once

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS  // NOLINT
#endif

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "link.hpp"

namespace autd::link {
using EtherCATAdapter = std::pair<std::string, std::string>;
using EtherCATAdapters = std::vector<EtherCATAdapter>;

/**
 * @brief Link using [SOEM](https://github.com/OpenEtherCATsociety/SOEM)
 */
class SOEMLink : public Link {
 public:
  /**
   * @brief Create SOEM link.
   * @param[in] ifname Network interface name. (e.g. eth0)
   * @param[in] device_num The number of AUTD you connected.
   *
   * @details Available Network interface names are obtained by EnumerateAdapters().
   *          The numbers of connected devices is obtained by Geometry::num_devices().
   */
  static LinkPtr Create(const std::string& ifname, size_t device_num);

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

  bool Open() override = 0;
  bool Close() override = 0;
  std::optional<std::string> Send(size_t size, std::unique_ptr<uint8_t[]> buf) override = 0;
  std::optional<std::string> Read(uint8_t* rx, uint32_t buffer_len) override = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
