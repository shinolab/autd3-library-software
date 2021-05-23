// File: autdsoem.hpp
// Project: soem
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "core/osal_timer.hpp"
#include "core/result.hpp"

namespace autd::autdsoem {

struct ECConfig {
  uint32_t ec_sm3_cycle_time_ns;
  uint32_t ec_sync0_cycle_time_ns;
  size_t header_size;
  size_t body_size;
  size_t input_frame_size;
  size_t bucket_size;
};

class SOEMController {
 public:
  SOEMController();
  ~SOEMController();
  SOEMController(const SOEMController& v) noexcept = delete;
  SOEMController& operator=(const SOEMController& obj) = delete;
  SOEMController(SOEMController&& obj) = delete;
  SOEMController& operator=(SOEMController&& obj) = delete;

  [[nodiscard]] Error open(const char* ifname, size_t dev_num, ECConfig config);
  [[nodiscard]] Error close();

  [[nodiscard]] bool is_open() const;

  [[nodiscard]] Error send(size_t size, const uint8_t* buf);
  [[nodiscard]] Error read(uint8_t* rx) const;

 private:
  void setup_sync0(bool activate, uint32_t cycle_time_ns) const;

  std::unique_ptr<uint8_t[]> _io_map;
  size_t _io_map_size;
  size_t _output_size;
  size_t _dev_num;
  ECConfig _config;
  bool _is_open;

  std::vector<std::pair<std::unique_ptr<uint8_t[]>, size_t>> _send_bucket;
  size_t _send_bucket_ptr, _send_bucket_size;
  std::thread _send_thread;
  std::condition_variable _send_cond;
  std::mutex _send_mtx;

  Timer _timer;
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
