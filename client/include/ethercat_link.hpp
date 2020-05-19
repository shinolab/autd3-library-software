// File: ethercat_link.hpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 19/05/2020
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
class EthercatLink : public Link {
 public:
  static LinkPtr Create(std::string ipv4addr);
  static LinkPtr Create(std::string ipv4addr, std::string ams_net_id);

  ~EthercatLink() override {}

 protected:
  void Open() = 0;
  void Close() = 0;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  std::vector<uint8_t> Read(uint32_t buffer_len) = 0;
  bool is_open() = 0;
  bool CalibrateModulation() = 0;
};

class LocalEthercatLink : public Link {
 public:
  static LinkPtr Create();
  ~LocalEthercatLink() override {}

 protected:
  void Open() = 0;
  void Close() = 0;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  std::vector<uint8_t> Read(uint32_t buffer_len) = 0;
  bool is_open() = 0;
  bool CalibrateModulation() = 0;
};
}  // namespace autd
