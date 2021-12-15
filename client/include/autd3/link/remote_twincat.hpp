// File: twincat.hpp
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

#include <memory>
#include <string>

#include "autd3/core/link.hpp"

namespace autd::link {

/**
 * @brief Link using TwinCAT via Beckhoff ADS
 */
class RemoteTwinCAT : public core::Link {
 public:
  /**
   * @brief Create RemoteTwinCAT
   */
  static core::LinkPtr create(const std::string& ipv4_addr, const std::string& remote_ams_net_id, const std::string& local_ams_net_id = "");

  RemoteTwinCAT() = default;
  ~RemoteTwinCAT() override = default;
  RemoteTwinCAT(const RemoteTwinCAT& v) noexcept = delete;
  RemoteTwinCAT& operator=(const RemoteTwinCAT& obj) = delete;
  RemoteTwinCAT(RemoteTwinCAT&& obj) = delete;
  RemoteTwinCAT& operator=(RemoteTwinCAT&& obj) = delete;

  void open() override = 0;
  void close() override = 0;
  void send(const core::TxDatagram& tx) override = 0;
  void receive(core::RxDatagram& rx) override = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
