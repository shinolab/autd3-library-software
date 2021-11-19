// File: autdsoem.hpp
// Project: soem
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "autd3/core/osal_timer.hpp"

namespace autd::autdsoem {

struct SOEMCallback final : core::CallbackHandler {
  virtual ~SOEMCallback() = default;
  SOEMCallback(const SOEMCallback& v) noexcept = delete;
  SOEMCallback& operator=(const SOEMCallback& obj) = delete;
  SOEMCallback(SOEMCallback&& obj) = delete;
  SOEMCallback& operator=(SOEMCallback&& obj) = delete;

  explicit SOEMCallback(const int expected_wkc, std::function<bool()> error_handler, std::atomic<bool>* sent)
      : _rt_lock(false), _expected_wkc(expected_wkc), _error_handler(std::move(error_handler)), _sent(sent) {}

  void callback() override;

 private:
  std::atomic<bool> _rt_lock;
  int _expected_wkc;
  std::function<bool()> _error_handler;
  std::atomic<bool>* _sent;
};

struct ECConfig {
  uint32_t ec_sm3_cycle_time_ns;
  uint32_t ec_sync0_cycle_time_ns;
  size_t header_size;
  size_t body_size;
  size_t input_frame_size;
};

constexpr size_t SEND_BUF_SIZE = 32;

class SOEMController {
 public:
  SOEMController();
  ~SOEMController();
  SOEMController(const SOEMController& v) noexcept = delete;
  SOEMController& operator=(const SOEMController& obj) = delete;
  SOEMController(SOEMController&& obj) = delete;
  SOEMController& operator=(SOEMController&& obj) = delete;

  void open(const char* ifname, size_t dev_num, ECConfig config);
  void on_lost(std::function<void(std::string)> callback);
  void close();

  [[nodiscard]] bool is_open() const;

  void send(const uint8_t* buf, size_t size);
  void read(uint8_t* rx) const;

 private:
  void setup_sync0(bool activate, uint32_t cycle_time_ns) const;

  bool error_handle();
  std::function<void(std::string)> _on_lost = nullptr;

  std::unique_ptr<uint8_t[]> _io_map;
  size_t _io_map_size;
  size_t _output_size;
  size_t _dev_num;
  ECConfig _config;
  bool _is_open;

  std::unique_ptr<std::pair<std::unique_ptr<uint8_t[]>, size_t>[]> _send_buf;
  size_t _send_buf_cursor;
  size_t _send_buf_size;
  std::mutex _send_mtx;
  std::condition_variable _send_cond;
  std::thread _send_thread;
  std::atomic<bool> _sent;

  std::unique_ptr<core::Timer<SOEMCallback>> _timer;
};

struct EtherCATAdapterInfo final {
  EtherCATAdapterInfo(std::string desc, std::string name) : desc(std::move(desc)), name(std::move(name)) {}
  static std::vector<EtherCATAdapterInfo> enumerate_adapters();

  std::string desc;
  std::string name;
};
}  // namespace autd::autdsoem
