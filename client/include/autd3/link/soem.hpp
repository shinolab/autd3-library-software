// File: soem.hpp
// Project: include
// Created Date: 10/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "autd3/core/link.hpp"

namespace autd::link {

/**
 * \brief EtherCAT adapter information to SOEM
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
class SOEM : virtual public core::Link {
 public:
  /**
   * @brief Create SOEM link.
   * @param ifname Network interface name. (e.g. eth0)
   * @param device_num The number of AUTD you connected.
   * @param cycle_ticks cycle time in ticks
   * @details Available Network interface names are obtained by EnumerateAdapters().
   *          The numbers of connected devices is obtained by Geometry::num_devices().
   */
  static std::unique_ptr<SOEM> create(const std::string& ifname, size_t device_num, uint32_t cycle_ticks = 1);

  /**
   * @brief Enumerate Ethernet adapters of the computer.
   */
  static std::vector<EtherCATAdapter> enumerate_adapters();
  SOEM() = default;
  ~SOEM() override = default;
  SOEM(const SOEM& v) noexcept = delete;
  SOEM& operator=(const SOEM& obj) = delete;
  SOEM(SOEM&& obj) = delete;
  SOEM& operator=(SOEM&& obj) = delete;

  void open() override = 0;
  void close() override = 0;
  void send(const uint8_t* buf, size_t size) override = 0;
  void read(uint8_t* rx, size_t buffer_len) override = 0;
  virtual void set_lost_handler(std::function<void(std::string)> handler) = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
