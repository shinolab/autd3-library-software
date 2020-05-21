// File: ethercat_link.hpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 21/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <stdio.h>

#include <memory>
#include <string>
#include <vector>

#include "core.hpp"
#include "link.hpp"

namespace autd {

/**
 * @brief EtherCAT Link using remote TwinCAT server
 */
class EthercatLink : public Link {
 public:
  /**
   * @brief Create EtherCAT link.
   * @param[in] ams_net_id AMS Net Id
   *
   * @details The ipv4 addr will be extracted from leading 4 octets of ams net id.
   */
  static LinkPtr Create(std::string ams_net_id);
  /**
   * @brief Create EtherCAT link.
   * @param[in] ipv4addr IPv4 address
   * @param[in] ams_net_id AMS Net Id
   *
   * @details The ipv4 addr will be extracted from leading 4 octets of ams net id if not specified.
   */
  static LinkPtr Create(std::string ipv4addr, std::string ams_net_id);

  ~EthercatLink() override {}

 protected:
  void Open() override = 0;
  void Close() override = 0;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) override = 0;
  std::vector<uint8_t> Read(uint32_t buffer_len) override = 0;
  bool is_open() override = 0;
  bool CalibrateModulation() override = 0;
};

/**
 * @brief EtherCAT Link using local TwinCAT server
 */
class LocalEthercatLink : public Link {
 public:
  /**
   * @brief Create LocalEtherCAT link.
   */
  static LinkPtr Create();
  ~LocalEthercatLink() override {}

 protected:
  void Open() override = 0;
  void Close() override = 0;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) override = 0;
  std::vector<uint8_t> Read(uint32_t buffer_len) override = 0;
  bool is_open() override = 0;
  bool CalibrateModulation() override = 0;
};
}  // namespace autd
