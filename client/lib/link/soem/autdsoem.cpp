// File: autdsoem.cpp
// Project: autdsoem
// Created Date: 23/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "./ethercat.h"

// "ethercat.h" must be included before followings
#include "autd3/core/exception.hpp"
#include "autdsoem.hpp"

namespace autd::autdsoem {

void SOEMCallback::callback() {
  if (auto expected = false; _rt_lock.compare_exchange_weak(expected, true)) {
    ec_send_processdata();
    this->_sent->store(true, std::memory_order_release);
    if (const auto wkc = ec_receive_processdata(EC_TIMEOUTRET); wkc != _expected_wkc)
      if (!this->_error_handler()) return;
    _rt_lock.store(false, std::memory_order_release);
  }
}

bool SOEMController::is_open() const { return _is_open; }

void SOEMController::send(const core::TxDatagram& tx) {
  if (!_is_open) throw core::exception::LinkError("link is closed");

  while (_send_buf_size == SEND_BUF_SIZE) std::this_thread::sleep_for(std::chrono::milliseconds(this->_config.ec_sm3_cycle_time_ns / 1000 / 1000));
  {
    std::lock_guard lock(_send_mtx);
    _send_buf[_send_buf_cursor].copy_from(tx);
    _send_buf_size++;
    _send_buf_cursor = (_send_buf_cursor + 1) % SEND_BUF_SIZE;
  }

  _send_cond.notify_one();
}

void SOEMController::receive(core::RxDatagram& rx) const {
  if (!_is_open) throw core::exception::LinkError("link is closed");
  std::memcpy(rx.data(), &_io_map[this->_output_size], this->_dev_num * this->_config.input_frame_size);
}

void SOEMController::setup_sync0(const bool activate, const uint32_t cycle_time_ns) const {
  for (size_t slave = 1; slave <= _dev_num; slave++) ec_dcsync0(static_cast<uint16_t>(slave), activate, cycle_time_ns, 0);
}

void SOEMController::open(const char* ifname, const size_t dev_num, const ECConfig config) {
  _dev_num = dev_num;
  _config = config;

  const auto header_size = config.header_size;
  const auto body_size = config.body_size;
  const auto output_size = (header_size + body_size) * _dev_num;
  _output_size = output_size;

  this->_send_buf.reserve(SEND_BUF_SIZE);
  for (size_t i = 0; i < SEND_BUF_SIZE; i++) this->_send_buf.emplace_back(dev_num);

  if (const auto size = _output_size + config.input_frame_size * _dev_num; size != _io_map_size) {
    _io_map_size = size;

    _io_map = std::make_unique<uint8_t[]>(size);
    std::memset(&_io_map[0], 0x00, _io_map_size);
  }

  if (ec_init(ifname) <= 0) {
    std::stringstream ss;
    ss << "No socket connection on" << ifname;
    throw core::exception::LinkError(ss.str());
  }

  const auto wc = ec_config_init(0);
  if (wc <= 0) throw core::exception::LinkError("No slaves found!");

  if (static_cast<size_t>(wc) != dev_num) {
    std::stringstream ss;
    ss << "The number of slaves you added: " << dev_num << ", but found: " << wc;
    throw core::exception::LinkError(ss.str());
  }

  ec_config_map(&_io_map[0]);

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

  if (ec_slave[0].state != EC_STATE_OPERATIONAL) throw core::exception::LinkError("One ore more slaves are not responding");

  setup_sync0(true, config.ec_sync0_cycle_time_ns);

  const auto expected_wkc = ec_group[0].outputsWKC * 2 + ec_group[0].inputsWKC;
  const auto interval_us = config.ec_sm3_cycle_time_ns / 1000;
  this->_timer = core::Timer<SOEMCallback>::start(std::make_unique<SOEMCallback>(
                                                      expected_wkc, [this] { return this->error_handle(); }, &this->_sent),
                                                  interval_us);

  _is_open = true;
  this->_send_thread = std::thread([this] {
    while (this->_is_open) {
      {
        std::unique_lock lock(this->_send_mtx);
        if (this->_send_buf_size == 0) this->_send_cond.wait(lock, [this] { return !this->_is_open || this->_send_buf_size != 0; });
        if (!this->_is_open) return;
        const auto idx = this->_send_buf_cursor >= this->_send_buf_size ? this->_send_buf_cursor - this->_send_buf_size
                                                                        : this->_send_buf_cursor + SEND_BUF_SIZE - this->_send_buf_size;
        auto& buf = this->_send_buf[idx];

        if (buf.body_size() > 0)
          for (size_t i = 0; i < _dev_num; i++) std::memcpy(&_io_map[(buf.header_size() + buf.body_size()) * i], buf.body(i), buf.body_size());
        if (buf.header_size() > 0)
          for (size_t i = 0; i < _dev_num; i++)
            std::memcpy(&_io_map[(buf.header_size() + buf.num_bodies()) * i + buf.body_size()], buf.header(), buf.header_size());

        this->_send_buf_size--;
      }
      this->_sent.store(false, std::memory_order_release);
      while (this->_is_open && !this->_sent.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(this->_config.ec_sm3_cycle_time_ns / 1000 / 1000));
    }
  });
}

void SOEMController::on_lost(std::function<void(std::string)> callback) { _on_lost = std::move(callback); }

void SOEMController::close() {
  if (!is_open()) return;

  this->_send_cond.notify_one();
  while (_send_buf_size > 0) std::this_thread::sleep_for(std::chrono::milliseconds(this->_config.ec_sm3_cycle_time_ns / 1000 / 1000));

  this->_is_open = false;

  this->_send_cond.notify_one();
  if (this->_send_thread.joinable()) this->_send_thread.join();

  const auto _ = this->_timer->stop();

  setup_sync0(false, _config.ec_sync0_cycle_time_ns);

  ec_slave[0].state = EC_STATE_INIT;
  ec_writestate(0);
  ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);
  ec_close();

  _io_map = nullptr;
  _io_map_size = 0;
}

bool SOEMController::error_handle() {
  ec_group[0].docheckstate = 0;
  ec_readstate();
  std::stringstream ss;
  for (uint16_t slave = 1; slave <= static_cast<uint16_t>(ec_slavecount); slave++) {
    if (ec_slave[slave].state != EC_STATE_OPERATIONAL) {
      ec_group[0].docheckstate = 1;
      if (ec_slave[slave].state == EC_STATE_SAFE_OP + EC_STATE_ERROR) {
        ss << "ERROR : slave " << slave << " is in SAFE_OP + ERROR, attempting ack\n";
        ec_slave[slave].state = EC_STATE_SAFE_OP + EC_STATE_ACK;
        ec_writestate(slave);
      } else if (ec_slave[slave].state == EC_STATE_SAFE_OP) {
        ss << "WARNING : slave " << slave << " is in SAFE_OP, change to OPERATIONAL\n";
        ec_slave[slave].state = EC_STATE_OPERATIONAL;
        ec_writestate(slave);
      } else if (ec_slave[slave].state > EC_STATE_NONE) {
        if (ec_reconfig_slave(slave, 500)) {
          ec_slave[slave].islost = 0;
          ss << "MESSAGE : slave " << slave << " reconfigured\n";
        }
      } else if (!ec_slave[slave].islost) {
        ec_statecheck(slave, EC_STATE_OPERATIONAL, EC_TIMEOUTRET);
        if (ec_slave[slave].state == EC_STATE_NONE) {
          ec_slave[slave].islost = 1;
          ss << "ERROR : slave " << slave << " lost\n";
        }
      }
    }
    if (ec_slave[slave].islost) {
      if (ec_slave[slave].state == EC_STATE_NONE) {
        if (ec_recover_slave(slave, 500)) {
          ec_slave[slave].islost = 0;
          ss << "MESSAGE : slave " << slave << " recovered\n";
        }
      } else {
        ec_slave[slave].islost = 0;
        ss << "MESSAGE : slave " << slave << " found\n";
      }
    }
  }
  if (!ec_group[0].docheckstate) return true;

  this->_is_open = false;
  if (_on_lost != nullptr) _on_lost(ss.str());
  return false;
}

SOEMController::SOEMController()
    : _io_map(nullptr), _io_map_size(0), _output_size(0), _dev_num(0), _config(), _is_open(false), _send_buf_cursor(0), _send_buf_size(0) {}

SOEMController::~SOEMController() {
  try {
    this->close();
  } catch (std::exception&) {
  }
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
