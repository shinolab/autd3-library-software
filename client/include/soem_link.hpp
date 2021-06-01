// File: soem_link.hpp
// Project: include
// Created Date: 10/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <string>
#include <vector>

#include "core/link.hpp"

namespace autd::link {

/**
 * \brief EtherCAT adapter information to SOEMLink
 */
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
   * @param ifname Network interface name. (e.g. eth0)
   * @param device_num The number of AUTD you connected.
   * @param cycle_ticks cycle time in ticks
   * @param bucket_size output buffer bucket size
   * @details Available Network interface names are obtained by EnumerateAdapters().
   *          The numbers of connected devices is obtained by Geometry::num_devices().
   */
  static core::LinkPtr create(const std::string& ifname, size_t device_num, uint32_t cycle_ticks = 1, size_t bucket_size = 32);

  /**
   * @brief Enumerate Ethernet adapters of the computer.
   */
  static std::vector<EtherCATAdapter> enumerate_adapters();
  SOEMLink() = default;
  ~SOEMLink() override = default;
  SOEMLink(const SOEMLink& v) noexcept = delete;
  SOEMLink& operator=(const SOEMLink& obj) = delete;
  SOEMLink(SOEMLink&& obj) = delete;
  SOEMLink& operator=(SOEMLink&& obj) = delete;

  Error open() override = 0;
  Error close() override = 0;
  Error send(const uint8_t* buf, size_t size) override = 0;
  Error read(uint8_t* rx, size_t buffer_len) override = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
