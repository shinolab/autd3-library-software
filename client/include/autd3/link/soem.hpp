// File: soem.hpp
// Project: include
// Created Date: 10/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "autd3/core/link.hpp"

namespace autd::link {

/**
 * \brief EtherCAT adapter information to SOEM
 */
struct EtherCATAdapter final {
  EtherCATAdapter(std::string desc, std::string name) : desc(std::move(desc)), name(std::move(name)) {}

  std::string desc;
  std::string name;
};

/**
 * @brief Link using [SOEM](https://github.com/OpenEtherCATSociety/SOEM)
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
  void send(const core::TxDatagram& tx) override = 0;
  void receive(core::RxDatagram& rx) override = 0;
  virtual void on_lost(std::function<void(std::string)> callback) = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
