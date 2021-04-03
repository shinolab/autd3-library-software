// File: autdsoem.hpp
// Project: include
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "timer.hpp"

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
  SOEMController();
  ~SOEMController();
  SOEMController(const SOEMController& v) noexcept = delete;
  SOEMController& operator=(const SOEMController& obj) = delete;
  SOEMController(SOEMController&& obj) = delete;
  SOEMController& operator=(SOEMController&& obj) = delete;

  bool Open(const char* ifname, size_t dev_num, ECConfig config);
  bool Close();

  bool is_open() const;

  void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
  void Read(uint8_t* rx) const;

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

  autd::Timer _timer;
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
