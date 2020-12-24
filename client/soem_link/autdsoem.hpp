// File: autdsoem.hpp
// Project: include
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 24/12/2020
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
  uint32_t ec_sm3_cycle_time_ns;
  uint32_t ec_sync0_cycle_time_ns;
  size_t header_size;
  size_t body_size;
  size_t input_frame_size;
};

class SOEMController {
 public:
  static std::unique_ptr<SOEMController> Create();
  SOEMController() = default;
  virtual ~SOEMController() = default;
  SOEMController(const SOEMController& v) noexcept = delete;
  SOEMController& operator=(const SOEMController& obj) = delete;
  SOEMController(SOEMController&& obj) = default;
  SOEMController& operator=(SOEMController&& obj) = default;

  virtual void Open(const char* ifname, size_t dev_num, ECConfig config) = 0;
  virtual void Close() = 0;

  virtual bool is_open() = 0;

  virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  virtual std::vector<uint8_t> Read() = 0;
};

struct EtherCATAdapterInfo final {
  EtherCATAdapterInfo() = default;
  ~EtherCATAdapterInfo() = default;
  EtherCATAdapterInfo& operator=(const EtherCATAdapterInfo& obj) = delete;
  EtherCATAdapterInfo(EtherCATAdapterInfo&& obj) = default;
  EtherCATAdapterInfo& operator=(EtherCATAdapterInfo&& obj) = default;

  EtherCATAdapterInfo(const EtherCATAdapterInfo& info) {
    desc = info.desc;
    name = info.name;
  }
  static std::vector<EtherCATAdapterInfo> EnumerateAdapters();

  std::string desc;
  std::string name;
};
}  // namespace autdsoem
