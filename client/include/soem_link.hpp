// File: soem_link.hpp
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

#include <string>
#include <vector>

#include "core/link.hpp"

namespace autd::link {

struct EtherCATAdapter final {
  EtherCATAdapter(const std::string& desc, const std::string& name) {
    this->desc = desc;
    this->name = name;
  }

  std::string desc;
  std::string name;
};

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
  static std::vector<EtherCATAdapter> EnumerateAdapters();
  SOEMLink() = default;
  ~SOEMLink() override = default;
  SOEMLink(const SOEMLink& v) noexcept = delete;
  SOEMLink& operator=(const SOEMLink& obj) = delete;
  SOEMLink(SOEMLink&& obj) = delete;
  SOEMLink& operator=(SOEMLink&& obj) = delete;

  Error Open() override = 0;
  Error Close() override = 0;
  Error Send(size_t size, const uint8_t* buf) override = 0;
  Error Read(uint8_t* rx, size_t buffer_len) override = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
