// File: soem_link.hpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core.hpp"
#include "link.hpp"

namespace autd {

class SOEMLink : public Link {
 public:
  static LinkPtr Create(std::string ifname, int device_num);
  static EtherCATAdapters EnumerateAdapters(int *const size);
  ~SOEMLink() override {}

 protected:
  void Open() = 0;
  void Close() = 0;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  std::vector<uint8_t> Read(uint32_t buffer_len) = 0;
  bool is_open() = 0;
  bool CalibrateModulation() = 0;
};
}  // namespace autd
