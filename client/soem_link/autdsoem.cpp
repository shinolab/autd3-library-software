// File: autdsoem.cpp
// Project: autdsoem
// Created Date: 23/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#if (_WIN32 || _WIN64)
#ifndef WINDOWS
#define WINDOWS
#endif
#elif defined __APPLE__
#ifndef MACOSX
#define MACOSX
#endif
#elif defined __linux__
#ifndef LINUX
#define LINUX
#endif
#else
#error "Not supported."
#endif

#ifdef WINDOWS
#define __STDC_LIMIT_MACROS  // NOLINT
#include <winerror.h>
#else
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#endif

#ifdef MACOSX
#include <dispatch/dispatch.h>
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "./ethercat.h"
#include "autdsoem.hpp"

namespace autdsoem {

static std::atomic<bool> AUTD3_LIB_SEND_COND(false);
static std::atomic<bool> AUTD3_LIB_RT_THREAD_LOCK(false);

class SOEMControllerImpl final : public SOEMController {
 public:
  SOEMControllerImpl();
  ~SOEMControllerImpl() override;
  SOEMControllerImpl(const SOEMControllerImpl& obj) = delete;
  SOEMControllerImpl& operator=(const SOEMControllerImpl& obj) = delete;
  SOEMControllerImpl(SOEMControllerImpl&& obj) = delete;
  SOEMControllerImpl& operator=(SOEMControllerImpl&& obj) = delete;

  bool is_open() override;

  void Open(const char* ifname, size_t dev_num, ECConfig config) override;
  void Close() override;

  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) override;
  std::vector<uint8_t> Read() override;

 private:
#ifdef WINDOWS
  static void CALLBACK rt_thread(UINT u_timer_id, UINT u_msg, DWORD_PTR dw_user, DWORD_PTR dw1, DWORD_PTR dw2);
#elif defined MACOSX
  static void rt_thread();
#elif defined LINUX
  static void rt_thread(union sigval sv);
#endif

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

#ifdef WINDOWS
  uint32_t _timer_id = 0;
#elif defined MACOSX
  dispatch_queue_t _queue;
  dispatch_source_t _timer;
#elif defined LINUX
  timer_t _timer_id;
#endif
};

bool SOEMControllerImpl::is_open() { return _is_open; }

void SOEMControllerImpl::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  {
    std::unique_lock<std::mutex> lk(_send_mtx);
    _send_q.push(std::pair(std::move(buf), size));
  }
  _send_cond.notify_all();
}

std::vector<uint8_t> SOEMControllerImpl::Read() {
  std::vector<uint8_t> res;
  const auto input_frame_idx = this->_output_frame_size;
  for (size_t i = 0; i < _dev_num; i++) {
    res.push_back(_io_map[input_frame_idx + 2 * i]);
    res.push_back(_io_map[input_frame_idx + 2 * i + 1]);
  }
  return res;
}

#ifdef WINDOWS
void CALLBACK SOEMControllerImpl::rt_thread([[maybe_unused]] UINT u_timer_id, [[maybe_unused]] UINT u_msg, [[maybe_unused]] DWORD_PTR dw_user,
                                            [[maybe_unused]] DWORD_PTR dw1, [[maybe_unused]] DWORD_PTR dw2)
#elif defined MACOSX
void SOEMControllerImpl::rt_thread()
#elif defined LINUX
void SOEMControllerImpl::rt_thread(union sigval sv)
#endif
{
  auto expected = false;
  if (AUTD3_LIB_RT_THREAD_LOCK.compare_exchange_weak(expected, true)) {
    const auto pre = AUTD3_LIB_SEND_COND.load(std::memory_order_acquire);
    ec_send_processdata();
    ec_receive_processdata(EC_TIMEOUTRET);
    if (!pre) {
      AUTD3_LIB_SEND_COND.store(true, std::memory_order_release);
    }

    AUTD3_LIB_RT_THREAD_LOCK.store(false, std::memory_order_release);
  }
}

void SOEMControllerImpl::SetupSync0(const bool activate, const uint32_t cycle_time_ns) const {
  using std::chrono::system_clock, std::chrono::duration_cast, std::chrono::nanoseconds;
  const auto ref_time = system_clock::now();
  for (size_t slave = 1; slave <= _dev_num; slave++) {
    const auto elapsed = duration_cast<nanoseconds>(ref_time - system_clock::now()).count();
    ec_dcsync0(static_cast<uint16_t>(slave), activate, cycle_time_ns, static_cast<int32>((elapsed / cycle_time_ns) * cycle_time_ns));
  }
}

void SOEMControllerImpl::Open(const char* ifname, const size_t dev_num, const ECConfig config) {
  _dev_num = dev_num;
  _config = config;
  _output_frame_size = (config.header_size + config.body_size) * _dev_num;

  const auto size = _output_frame_size + (config.input_frame_size * _dev_num);
  if (size != _io_map_size) {
    _io_map_size = size;

    delete[] _io_map;
    _io_map = new uint8_t[size];
    memset(_io_map, 0x00, _io_map_size);
  }

  _sync0_cyc_time = config.ec_sync0_cycle_time_ns;

  if (ec_init(ifname) <= 0) {
    std::cerr << "No socket connection on " << ifname << std::endl;
    return;
  }

  const auto wc = ec_config(0, _io_map);
  if (wc <= 0) {
    std::cerr << "No slaves found!" << std::endl;
    return;
  }
  if (static_cast<size_t>(wc) != dev_num) {
    std::cerr << "The number of slaves you added:" << dev_num << ", but found: " << wc << std::endl;
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
  } while (chk-- && (ec_slave[0].state != EC_STATE_OPERATIONAL));

  if (ec_slave[0].state != EC_STATE_OPERATIONAL) {
    std::cerr << "One ore more slaves are not responding." << std::endl;
    return;
  }

  SetupSync0(true, _sync0_cyc_time);

#ifdef WINDOWS
  const uint32_t u_resolution = 1;
  timeBeginPeriod(u_resolution);
  _timer_id = timeSetEvent(config.ec_sm3_cycle_time_ns / 1000 / 1000, u_resolution, static_cast<LPTIMECALLBACK>(rt_thread), NULL, TIME_PERIODIC);

  if (_timer_id == 0) {
    std::cerr << "timeSetEvent failed." << std::endl;
    return;
  }

  auto* const h_process = GetCurrentProcess();
  if (!SetPriorityClass(h_process, REALTIME_PRIORITY_CLASS)) {
    std::cerr << "Failed to SetPriorityClass" << std::endl;
  }

#elif defined MACOSX
  _queue = dispatch_queue_create("timerQueue", 0);

  _timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
  dispatch_source_set_event_handler(_timer, ^{
    rt_thread();
  });

  dispatch_source_set_cancel_handler(_timer, ^{
    dispatch_release(_timer);
    dispatch_release(_queue);
  });

  dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, 0);
  dispatch_source_set_timer(_timer, start, config.ec_sm3_cycle_time_ns, 0);
  dispatch_resume(_timer);
#elif defined LINUX
  struct itimerspec itval;
  struct sigevent se;

  itval.it_value.tv_sec = 0;
  itval.it_value.tv_nsec = config.ec_sm3_cycle_time_ns;
  itval.it_interval.tv_sec = 0;
  itval.it_interval.tv_nsec = config.ec_sm3_cycle_time_ns;

  memset(&se, 0, sizeof(se));
  se.sigev_value.sival_ptr = NULL;
  se.sigev_notify = SIGEV_THREAD;
  se.sigev_notify_function = rt_thread;
  se.sigev_notify_attributes = NULL;

  if (timer_create(CLOCK_REALTIME, &se, &_timer_id) < 0) {
    std::cerr << "Error: timer_create." << std::endl;
    return;
  }

  if (timer_settime(_timer_id, 0, &itval, NULL) < 0) {
    std::cerr << "Error: timer_settime." << std::endl;
    return;
  }
#endif

  _is_open = true;
  CreateSendThread(config.header_size, config.body_size);
}

void SOEMControllerImpl::Close() {
  if (_is_open) {
    _is_open = false;
    _send_cond.notify_all();
    if (std::this_thread::get_id() != _send_thread.get_id() && this->_send_thread.joinable()) this->_send_thread.join();
    {
      std::unique_lock<std::mutex> lk(_send_mtx);
      std::queue<std::pair<std::unique_ptr<uint8_t[]>, size_t>>().swap(_send_q);
    }

    memset(_io_map, 0x00, _output_frame_size);

#ifdef WINDOWS
    if (_timer_id != 0) {
      const uint32_t u_resolution = 1;
      timeKillEvent(_timer_id);
      timeEndPeriod(u_resolution);
    }
#elif defined MACOSX
    dispatch_source_cancel(_timer);
#elif defined LINUX
    timer_delete(_timer_id);
#endif

    SetupSync0(false, _sync0_cyc_time);

    ec_slave[0].state = EC_STATE_INIT;
    ec_writestate(0);

    ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);

    ec_close();
  }
}

void SOEMControllerImpl::CreateSendThread(size_t header_size, size_t body_size) {
  _send_thread = std::thread([this, header_size, body_size]() {
    while (_is_open) {
      std::unique_ptr<uint8_t[]> buf = nullptr;
      size_t size = 0;
      {
        std::unique_lock<std::mutex> lk(_send_mtx);
        _send_cond.wait(lk, [&] { return !_send_q.empty() || !_is_open; });
        if (!_send_q.empty()) {
          auto [fst, snd] = move(_send_q.front());
          buf = move(fst);
          size = snd;
        }
      }

      if (buf != nullptr && _is_open) {
        const auto includes_gain = ((size - header_size) / body_size) > 0;
        const auto output_frame_size = header_size + body_size;

        for (size_t i = 0; i < _dev_num; i++) {
          if (includes_gain) memcpy(&_io_map[output_frame_size * i], &buf[header_size + body_size * i], body_size);
          memcpy(&_io_map[output_frame_size * i + body_size], &buf[0], header_size);
        }

        {
          AUTD3_LIB_SEND_COND.store(false, std::memory_order_release);
          while (!AUTD3_LIB_SEND_COND.load(std::memory_order_acquire) && _is_open) {
          }
        }

        _send_q.pop();
      }
    }
  });
}

SOEMControllerImpl::SOEMControllerImpl() : _config() {
  this->_is_open = false;
  this->_io_map = nullptr;
}

SOEMControllerImpl::~SOEMControllerImpl() {
  delete[] _io_map;
  _io_map = nullptr;
}

std::unique_ptr<SOEMController> SOEMController::Create() { return std::make_unique<SOEMControllerImpl>(); }

std::vector<EtherCATAdapterInfo> EtherCATAdapterInfo::EnumerateAdapters() {
  auto* adapter = ec_find_adapters();
  auto adapters = std::vector<EtherCATAdapterInfo>();
  while (adapter != nullptr) {
    auto* info = new EtherCATAdapterInfo;
    info->desc = std::string(adapter->desc);
    info->name = std::string(adapter->name);
    adapters.push_back(*info);
    adapter = adapter->next;
  }
  return adapters;
}
}  // namespace autdsoem
