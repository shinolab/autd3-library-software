// File: autdsoem.cpp
// Project: autdsoem
// Created Date: 23/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 17/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#ifdef _WINDOWS
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#endif

#include <atomic>
#include <chrono>
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
static std::atomic autd3_lib_send_cond(false);

bool SOEMController::is_open() const { return _is_open; }

Error SOEMController::Send(size_t size, const uint8_t* buf) {
  if (buf == nullptr) return Err(std::string("data is null"));
  if (!_is_open) return Err(std::string("link is closed"));
  {
    auto buf_ = std::make_unique<uint8_t[]>(size);
    std::memcpy(&buf_[0], &buf[0], size);
    std::unique_lock lk(_send_mtx);
    _send_q.push(std::pair(std::move(buf_), size));
  }
  _send_cond.notify_all();
  return Ok();
}

Error SOEMController::Read(uint8_t* rx) const {
  if (!_is_open) return Err(std::string("link is closed"));

  const auto input_frame_idx = this->_output_frame_size;
  for (size_t i = 0; i < _dev_num; i++) {
    rx[2 * i] = _io_map[input_frame_idx + 2 * i];
    rx[2 * i + 1] = _io_map[input_frame_idx + 2 * i + 1];
  }
  return Ok();
}

void SOEMController::SetupSync0(const bool activate, const uint32_t cycle_time_ns) const {
  using std::chrono::high_resolution_clock, std::chrono::duration_cast, std::chrono::nanoseconds;
  const auto ref_time = high_resolution_clock::now();
  for (size_t slave = 1; slave <= _dev_num; slave++) {
    const auto elapsed = duration_cast<nanoseconds>(ref_time - high_resolution_clock::now()).count();
    ec_dcsync0(static_cast<uint16_t>(slave), activate, cycle_time_ns, static_cast<int32>(elapsed / cycle_time_ns * cycle_time_ns));
  }
}

Error SOEMController::Open(const char* ifname, const size_t dev_num, const ECConfig config) {
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

  SetupSync0(true, _sync0_cyc_time);

  auto interval_us = config.ec_sm3_cycle_time_ns / 1000;
  this->_timer.SetInterval(interval_us);

  if (auto res = this->_timer.Start([]() {
        if (auto expected = false; autd3_rt_lock.compare_exchange_weak(expected, true)) {
          const auto pre = autd3_lib_send_cond.load(std::memory_order_acquire);
          ec_send_processdata();
          if (!pre) autd3_lib_send_cond.store(true, std::memory_order_release);
          autd3_rt_lock.store(false, std::memory_order_release);
          ec_receive_processdata(EC_TIMEOUTRET);
        }
      });
      res.is_err())
    return res;

  _is_open = true;
  CreateSendThread(config.header_size, config.body_size);

  return Ok();
}

Error SOEMController::Close() {
  if (!_is_open) return Ok();
  _is_open = false;

  _send_cond.notify_all();
  if (std::this_thread::get_id() != _send_thread.get_id() && this->_send_thread.joinable()) this->_send_thread.join();
  {
    std::unique_lock lk(_send_mtx);
    std::queue<std::pair<std::unique_ptr<uint8_t[]>, size_t>>().swap(_send_q);
  }

  std::memset(_io_map, 0x00, _output_frame_size);

  if (auto res = this->_timer.Stop(); res.is_err()) return res;

  SetupSync0(false, _sync0_cyc_time);

  ec_slave[0].state = EC_STATE_INIT;
  ec_writestate(0);

  ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);

  ec_close();

  return Ok();
}

void SOEMController::CreateSendThread(size_t header_size, size_t body_size) {
  _send_thread = std::thread([this, header_size, body_size]() {
    while (_is_open) {
      std::unique_ptr<uint8_t[]> buf = nullptr;
      size_t size = 0;
      {
        std::unique_lock lk(_send_mtx);
        _send_cond.wait(lk, [&] { return !_send_q.empty() || !_is_open; });
        if (!_send_q.empty()) {
          auto [fst, snd] = move(_send_q.front());
          buf = std::move(fst);
          size = snd;
        }
      }

      if (buf != nullptr && _is_open) {
        const auto includes_gain = (size - header_size) / body_size > 0;
        const auto output_frame_size = header_size + body_size;

        for (size_t i = 0; i < _dev_num; i++) {
          if (includes_gain) std::memcpy(&_io_map[output_frame_size * i], &buf[header_size + body_size * i], body_size);
          std::memcpy(&_io_map[output_frame_size * i + body_size], &buf[0], header_size);
        }

        {
          autd3_lib_send_cond.store(false, std::memory_order_release);
          while (!autd3_lib_send_cond.load(std::memory_order_acquire) && _is_open) {
          }
        }

        _send_q.pop();
      }
    }
  });
}

SOEMController::SOEMController() : _config() {
  this->_is_open = false;
  this->_io_map = nullptr;
}

SOEMController::~SOEMController() {
  (void)this->Close();
  delete[] _io_map;
  _io_map = nullptr;
}

std::vector<EtherCATAdapterInfo> EtherCATAdapterInfo::EnumerateAdapters() {
  auto* adapter = ec_find_adapters();
  std::vector<EtherCATAdapterInfo> adapters;
  while (adapter != nullptr) {
    auto* info = new EtherCATAdapterInfo;
    info->desc = std::string(adapter->desc);
    info->name = std::string(adapter->name);
    adapters.emplace_back(*info);
    adapter = adapter->next;
  }
  return adapters;
}
}  // namespace autd::autdsoem
