// File: autdsoem.cpp
// Project: autdsoem
// Created Date: 23/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 22/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "./ethercat.h"

// "ethercat.h" must be included before followings
#include "autdsoem.hpp"

namespace autd::autdsoem {

static std::atomic autd3_rt_lock(false);
static std::atomic autd3_send_cond(false);

bool SOEMController::is_open() const { return _is_open; }

Error SOEMController::send(const size_t size, const uint8_t* buf) {
  if (!_is_open) return Err(std::string("link is closed"));
  if (this->_send_bucket_size == _config.bucket_size) return Ok(false);

  {
    std::unique_lock lk(_send_mtx);
    auto& [buf_, size_] = this->_send_bucket[(this->_send_bucket_ptr + this->_send_bucket_size) % _config.bucket_size];
    std::memcpy(&buf_[0], buf, size);
    size_ = size;
    this->_send_bucket_size++;
  }
  _send_cond.notify_all();
  return Ok(true);
}

Error SOEMController::read(uint8_t* rx) const {
  if (!_is_open) return Err(std::string("link is closed"));
  std::memcpy(rx, &_io_map[this->_output_size], this->_dev_num * this->_config.input_frame_size);
  return Ok(true);
}

void SOEMController::setup_sync0(const bool activate, const uint32_t cycle_time_ns) const {
  for (size_t slave = 1; slave <= _dev_num; slave++) ec_dcsync0(static_cast<uint16_t>(slave), activate, cycle_time_ns, 0);
}

Error SOEMController::open(const char* ifname, const size_t dev_num, const ECConfig config) {
  _dev_num = dev_num;
  _config = config;
  const auto header_size = config.header_size;
  const auto body_size = config.body_size;
  const auto output_size = (header_size + body_size) * _dev_num;
  _output_size = output_size;

  if (const auto size = _output_size + config.input_frame_size * _dev_num; size != _io_map_size) {
    _io_map_size = size;

    _io_map = std::make_unique<uint8_t[]>(size);
    std::memset(&_io_map[0], 0x00, _io_map_size);

    std::vector<std::pair<std::unique_ptr<uint8_t[]>, size_t>>().swap(this->_send_bucket);
    this->_send_bucket.resize(config.bucket_size);
    for (size_t i = 0; i < config.bucket_size; i++) this->_send_bucket[i].first = std::make_unique<uint8_t[]>(header_size + body_size * _dev_num);
  }

  if (ec_init(ifname) <= 0) return Err(std::string("No socket connection on ") + std::string(ifname));

  const auto wc = ec_config(0, &_io_map[0]);
  if (wc <= 0) return Err(std::string("No slaves found!"));

  if (static_cast<size_t>(wc) != dev_num) {
    std::stringstream ss;
    ss << "The number of slaves you added: " << dev_num << ", but found: " << wc;
    return Err(ss.str());
  }

  ec_configdc();

  ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);

  ec_slave[0].state = EC_STATE_OPERATIONAL;
  ec_send_processdata();
  ec_receive_processdata(EC_TIMEOUTRET);

  ec_writestate(0);

  auto chk = 200;
  do {
    ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
  } while (chk-- && ec_slave[0].state != EC_STATE_OPERATIONAL);

  if (ec_slave[0].state != EC_STATE_OPERATIONAL) return Err(std::string("One ore more slaves are not responding"));

  setup_sync0(true, config.ec_sync0_cycle_time_ns);

  auto interval_us = config.ec_sm3_cycle_time_ns / 1000;
  this->_timer.SetInterval(interval_us);
  if (auto res = this->_timer.start([]() {
        if (auto expected = false; autd3_rt_lock.compare_exchange_weak(expected, true)) {
          const auto pre = autd3_send_cond.load(std::memory_order_acquire);
          ec_send_processdata();
          if (!pre) autd3_send_cond.store(true, std::memory_order_release);
          ec_receive_processdata(EC_TIMEOUTRET);
          autd3_rt_lock.store(false, std::memory_order_release);
        }
      });
      res.is_err())
    return res;

  _is_open = true;
  _send_thread = std::thread([this, header_size, body_size]() {
    while (this->is_open()) {
      {
        std::unique_lock lk(_send_mtx);
        _send_cond.wait(lk, [&] { return this->_send_bucket_size != 0 || !this->is_open(); });
        if (this->_send_bucket_size == 0 || !this->is_open()) continue;

        auto& [buf, size] = _send_bucket[this->_send_bucket_ptr];
        if (size > header_size)
          for (size_t i = 0; i < _dev_num; i++) std::memcpy(&_io_map[(header_size + body_size) * i], &buf[header_size + body_size * i], body_size);
        for (size_t i = 0; i < _dev_num; i++) std::memcpy(&_io_map[(header_size + body_size) * i + body_size], &buf[0], header_size);
        this->_send_bucket_size--;
        this->_send_bucket_ptr = (this->_send_bucket_ptr + 1) % this->_config.bucket_size;
      }

      autd3_send_cond.store(false, std::memory_order_release);
      while (!autd3_send_cond.load(std::memory_order_acquire) && this->is_open()) {
      }
    }
  });

  return Ok(true);
}

Error SOEMController::close() {
  if (!is_open()) return Ok(true);
  this->_is_open = false;

  _send_cond.notify_all();
  if (std::this_thread::get_id() != _send_thread.get_id() && this->_send_thread.joinable()) this->_send_thread.join();

  {
    std::unique_lock lk(_send_mtx);
    std::vector<std::pair<std::unique_ptr<uint8_t[]>, size_t>>().swap(this->_send_bucket);
  }

  std::memset(&_io_map[0], 0x00, _output_size);

  if (auto res = this->_timer.stop(); res.is_err()) return res;

  setup_sync0(false, _config.ec_sync0_cycle_time_ns);

  ec_slave[0].state = EC_STATE_INIT;
  ec_writestate(0);
  ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);
  ec_close();

  _io_map = nullptr;
  _io_map_size = 0;

  return Ok(true);
}

SOEMController::SOEMController()
    : _io_map(nullptr), _io_map_size(0), _output_size(0), _dev_num(0), _config(), _is_open(false), _send_bucket_ptr(0), _send_bucket_size(0) {}

SOEMController::~SOEMController() { (void)this->close(); }

std::vector<EtherCATAdapterInfo> EtherCATAdapterInfo::enumerate_adapters() {
  auto* adapter = ec_find_adapters();
  std::vector<EtherCATAdapterInfo> adapters;
  while (adapter != nullptr) {
    EtherCATAdapterInfo info(std::string(adapter->desc), std::string(adapter->name));
    adapters.emplace_back(info);
    adapter = adapter->next;
  }
  return adapters;
}
}  // namespace autd::autdsoem
