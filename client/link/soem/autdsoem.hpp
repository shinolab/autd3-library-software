// File: autdsoem.hpp
// Project: soem
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "autd3/core/osal_timer.hpp"

namespace autd::autdsoem {

struct SOEMCallback final : core::CallbackHandler {
  virtual ~SOEMCallback() = default;
  SOEMCallback(const SOEMCallback& v) noexcept = delete;
  SOEMCallback& operator=(const SOEMCallback& obj) = delete;
  SOEMCallback(SOEMCallback&& obj) = delete;
  SOEMCallback& operator=(SOEMCallback&& obj) = delete;

  SOEMCallback() : _autd3_rt_lock(false) {}

  void callback() override;

 private:
  std::atomic<bool> _autd3_rt_lock;
};

struct ECConfig {
  uint32_t ec_sm3_cycle_time_ns;
  uint32_t ec_sync0_cycle_time_ns;
  size_t header_size;
  size_t body_size;
  size_t input_frame_size;
};

class SOEMController {
 public:
  SOEMController();
  ~SOEMController();
  SOEMController(const SOEMController& v) noexcept = delete;
  SOEMController& operator=(const SOEMController& obj) = delete;
  SOEMController(SOEMController&& obj) = delete;
  SOEMController& operator=(SOEMController&& obj) = delete;

  void open(const char* ifname, size_t dev_num, ECConfig config);
  void close();

  [[nodiscard]] bool is_open() const;

  void send(const uint8_t* buf, size_t size) const;
  void read(uint8_t* rx) const;

 private:
  void setup_sync0(bool activate, uint32_t cycle_time_ns) const;

  std::unique_ptr<uint8_t[]> _io_map;
  size_t _io_map_size;
  size_t _output_size;
  size_t _dev_num;
  ECConfig _config;
  bool _is_open;

  std::unique_ptr<core::Timer<SOEMCallback>> _timer;
};

struct EtherCATAdapterInfo final {
  EtherCATAdapterInfo(const std::string& desc, const std::string& name) {
    this->desc = desc;
    this->name = name;
  }
  static std::vector<EtherCATAdapterInfo> enumerate_adapters();

  std::string desc;
  std::string name;
};
}  // namespace autd::autdsoem
