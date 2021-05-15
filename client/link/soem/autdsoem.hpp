// File: autdsoem.hpp
// Project: soem
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
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
};

class SOEMController {
 public:
  SOEMController();
  ~SOEMController();
  SOEMController(const SOEMController& v) noexcept = delete;
  SOEMController& operator=(const SOEMController& obj) = delete;
  SOEMController(SOEMController&& obj) = delete;
  SOEMController& operator=(SOEMController&& obj) = delete;

  [[nodiscard]] Result<bool, std::string> Open(const char* ifname, size_t dev_num, ECConfig config);
  [[nodiscard]] Result<bool, std::string> Close();

  [[nodiscard]] bool is_open() const;

  [[nodiscard]] Result<bool, std::string> Send(size_t size, const uint8_t* buf);
  [[nodiscard]] Result<bool, std::string> Read(uint8_t* rx) const;

 private:
  void CreateSendThread(size_t header_size, size_t body_size);
  void SetupSync0(bool activate, uint32_t cycle_time_ns) const;

  uint8_t* _io_map;
  size_t _io_map_size = 0;
  size_t _output_frame_size = 0;
  uint32_t _sync0_cyc_time = 0;
  size_t _dev_num = 0;
  ECConfig _config;
  bool _is_open = false;

  std::queue<std::pair<std::unique_ptr<uint8_t[]>, size_t>> _send_q;
  std::thread _send_thread;
  std::condition_variable _send_cond;
  std::mutex _send_mtx;

  Timer _timer;
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
}  // namespace autd::autdsoem
