// File: autdsoem.cpp
// Project: autdsoem
// Created Date: 23/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
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

bool SOEMController::is_open() const { return _is_open; }

Error SOEMController::send(const size_t size, const uint8_t* buf) const {
  if (!_is_open) return Err(std::string("link is closed"));

  const auto header_size = this->_config.header_size;
  const auto body_size = this->_config.body_size;
  const auto output_frame_size = header_size + body_size;
  if (size > header_size)
    for (size_t i = 0; i < _dev_num; i++) std::memcpy(&_io_map[output_frame_size * i], &buf[header_size + body_size * i], body_size);
  for (size_t i = 0; i < _dev_num; i++) std::memcpy(&_io_map[output_frame_size * i + body_size], &buf[0], header_size);
  return Ok(true);
}

Error SOEMController::read(uint8_t* rx) const {
  if (!_is_open) return Err(std::string("link is closed"));
  std::memcpy(rx, &_io_map[this->_output_frame_size], this->_dev_num * this->_config.input_frame_size);
  return Ok(true);
}

void SOEMController::setup_sync0(const bool activate, const uint32_t cycle_time_ns) const {
  for (size_t slave = 1; slave <= _dev_num; slave++) ec_dcsync0(static_cast<uint16_t>(slave), activate, cycle_time_ns, 0);
}

Error SOEMController::open(const char* ifname, const size_t dev_num, const ECConfig config) {
  _dev_num = dev_num;
  _config = config;
  _output_frame_size = (config.header_size + config.body_size) * _dev_num;

  if (const auto size = _output_frame_size + config.input_frame_size * _dev_num; size != _io_map_size) {
    _io_map_size = size;

    delete[] _io_map;
    _io_map = new uint8_t[size];
    std::memset(_io_map, 0x00, _io_map_size);
  }

  _sync0_cyc_time = config.ec_sync0_cycle_time_ns;

  if (ec_init(ifname) <= 0) return Err(std::string("No socket connection on ") + std::string(ifname));

  const auto wc = ec_config(0, _io_map);
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

  setup_sync0(true, _sync0_cyc_time);

  auto interval_us = config.ec_sm3_cycle_time_ns / 1000;
  this->_timer.SetInterval(interval_us);
  if (auto res = this->_timer.start([]() {
        if (auto expected = false; autd3_rt_lock.compare_exchange_weak(expected, true)) {
          ec_send_processdata();
          autd3_rt_lock.store(false, std::memory_order_release);
          ec_receive_processdata(EC_TIMEOUTRET);
        }
      });
      res.is_err())
    return res;

  _is_open = true;

  return Ok(true);
}

Error SOEMController::close() {
  if (!_is_open) return Ok(true);
  _is_open = false;

  _send_cond.notify_all();
  if (std::this_thread::get_id() != _send_thread.get_id() && this->_send_thread.joinable()) this->_send_thread.join();

  {
    std::unique_lock lk(_send_mtx);
    std::queue<std::pair<std::unique_ptr<uint8_t[]>, size_t>>().swap(_send_q);
  }

  std::memset(_io_map, 0x00, _output_frame_size);

  if (auto res = this->_timer.stop(); res.is_err()) return res;

  setup_sync0(false, _sync0_cyc_time);

  ec_slave[0].state = EC_STATE_INIT;
  ec_writestate(0);
  ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);
  ec_close();

  return Ok(true);
}

SOEMController::SOEMController() : _config() {
  this->_is_open = false;
  this->_io_map = nullptr;
}

SOEMController::~SOEMController() {
  (void)this->close();
  delete[] _io_map;
  _io_map = nullptr;
}

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
