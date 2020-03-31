// File: autdsoem.hpp
// Project: include
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 31/03/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace autdsoem {

struct ECConfig {
  uint32_t ec_sm3_cyctime_ns;
  uint32_t ec_sync0_cyctime_ns;
  size_t header_size;
  size_t body_size;
  size_t input_frame_size;
};

class ISOEMController {
 public:
  static std::unique_ptr<ISOEMController> Create();

  virtual void Open(const char *ifname, size_t dev_num, ECConfig config) = 0;
  virtual void Close() = 0;

  virtual bool is_open() = 0;

  virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  virtual std::vector<uint8_t> Read() = 0;
  virtual int64_t ec_dc_time() = 0;
};

struct EtherCATAdapterInfo {
 public:
  EtherCATAdapterInfo() {}
  EtherCATAdapterInfo(const EtherCATAdapterInfo &info) {
    desc = info.desc;
    name = info.name;
  }
  static std::vector<EtherCATAdapterInfo> EnumerateAdapters();

  std::string desc;
  std::string name;
};
}  // namespace autdsoem
