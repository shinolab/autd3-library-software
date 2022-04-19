// File: autdsoem.hpp
// Project: soem
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/03/2022
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "autd3/core/osal_timer.hpp"

namespace autd::autdsoem {

struct SOEMCallback final : core::timer::CallbackHandler {
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

struct IOMap {
  IOMap() : _size(0), _buf(nullptr), _config(), _device_num(0) {}

  explicit IOMap(const size_t device_num, const ECConfig& config)
      : _size(device_num * (config.header_size + config.body_size + config.input_frame_size)),
        _buf(std::make_unique<uint8_t[]>(_size)),
        _config(config),
        _device_num(device_num) {}

  void resize(const size_t device_num, const ECConfig& config) {
    if (const auto size = device_num * (config.header_size + config.body_size + config.input_frame_size); _size != size) {
      _device_num = device_num;
      _size = size;
      _config = config;
      _buf = std::make_unique<uint8_t[]>(_size);
    }
  }

  [[nodiscard]] size_t size() const { return _size; }

  core::GlobalHeader* header(const size_t i) {
    return reinterpret_cast<core::GlobalHeader*>(&_buf[(_config.body_size + _config.header_size) * i + _config.body_size]);
  }

  core::Body* body(const size_t i) { return reinterpret_cast<core::Body*>(&_buf[(_config.body_size + _config.header_size) * i]); }

  [[nodiscard]] const core::RxMessage* input() const {
    return reinterpret_cast<const core::RxMessage*>(&_buf[(_config.body_size + _config.header_size) * _device_num]);
  }

  void copy_from(core::TxDatagram& tx) {
    for (size_t i = 0; i < tx.num_bodies(); i++) std::memcpy(body(i), tx.body(i), tx.body_size());
    if (tx.header_size() > 0)
      for (size_t i = 0; i < _device_num; i++) std::memcpy(header(i), tx.header(), tx.header_size());
  }

  [[nodiscard]] uint8_t* get() const { return _buf.get(); }

 private:
  size_t _size;
  std::unique_ptr<uint8_t[]> _buf;
  ECConfig _config;
  size_t _device_num;
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

  void send(const core::TxDatagram& tx);
  void receive(core::RxDatagram& rx) const;

 private:
  bool error_handle();
  std::function<void(std::string)> _on_lost = nullptr;

  IOMap _io_map;
  size_t _dev_num;
  uint32_t _sm3_cycle_time_ms;
  bool _is_open;
  std::unique_ptr<uint32_t[]> _user_data;

  std::vector<core::TxDatagram> _send_buf;
  size_t _send_buf_cursor;
  size_t _send_buf_size;
  std::mutex _send_mtx;
  std::condition_variable _send_cond;
  std::thread _send_thread;
  std::atomic<bool> _sent;

  std::unique_ptr<core::timer::Timer<SOEMCallback>> _timer;
};

struct EtherCATAdapterInfo final {
  EtherCATAdapterInfo(std::string desc, std::string name) : desc(std::move(desc)), name(std::move(name)) {}
  static std::vector<EtherCATAdapterInfo> enumerate_adapters();

  std::string desc;
  std::string name;
};
}  // namespace autd::autdsoem
