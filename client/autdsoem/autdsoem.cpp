// File: autdsoem.cpp
// Project: autdsoem
// Created Date: 23/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 01/04/2020
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
#define __STDC_LIMIT_MACROS
#include <WinError.h>
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
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "../lib/autdsoem.hpp"
#include "./ethercat.h"

namespace autdsoem {

static std::atomic<bool> AUTD3_LIB_SEND_COND(false);
static std::atomic<bool> AUTD3_LIB_RTTHREAD_LOCK(false);

class SOEMController : public ISOEMController {
 public:
  SOEMController();
  ~SOEMController();

  bool is_open() final;

  void Open(const char *ifname, size_t dev_num, ECConfig config) final;
  void Close() final;

  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) final;
  std::vector<uint8_t> Read() final;
  int64_t ec_dc_time() final;

  bool _is_open = false;

 private:
#ifdef WINDOWS
  static void CALLBACK RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired);
#elif defined MACOSX
  static void RTthread(SOEMController *pimpl);
#elif defined LINUX
  static void RTthread(union sigval sv);
#endif

  void CreateSendThread(size_t header_size, size_t body_size);
  void SetupSync0(bool actiavte, uint32_t cycle_time_ns);

  uint8_t *_io_map;
  size_t _io_map_size = 0;
  size_t _output_frame_size = 0;
  uint32_t _sync0_cyctime = 0;
  size_t _dev_num = 0;
  ECConfig _config;

  std::queue<std::pair<std::unique_ptr<uint8_t[]>, size_t>> _send_q;
  std::thread _send_thread;
  std::condition_variable _send_cond;
  std::mutex _send_mtx;

#ifdef WINDOWS
  HANDLE _timerQueue = NULL;
  HANDLE _timer = NULL;
#elif defined MACOSX
  dispatch_queue_t _queue;
  dispatch_source_t _timer;
#elif defined LINUX
  timer_t _timer_id;
#endif
};

bool SOEMController::is_open() { return _is_open; }

void SOEMController::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  {
    std::unique_lock<std::mutex> lk(_send_mtx);
    _send_q.push(std::pair(std::move(buf), size));
  }
  _send_cond.notify_all();
}

std::vector<uint8_t> SOEMController::Read() {
  std::vector<uint8_t> res;
  const auto input_frame_idx = this->_output_frame_size;
  for (size_t i = 0; i < _dev_num; i++) {
    res.push_back(_io_map[input_frame_idx + 2 * i]);
    res.push_back(_io_map[input_frame_idx + 2 * i + 1]);
  }
  return res;
}

int64_t SOEMController::ec_dc_time() { return ec_DCtime % _sync0_cyctime; }

#ifdef WINDOWS
void CALLBACK SOEMController::RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired)
#elif defined MACOSX
void SOEMController::RTthread(SOEMController *cnt)
#elif defined LINUX
void SOEMController::RTthread(union sigval sv)
#endif
{
  bool expected = false;
  if (AUTD3_LIB_RTTHREAD_LOCK.compare_exchange_weak(expected, true)) {
    auto pre = AUTD3_LIB_SEND_COND.load(std::memory_order_acquire);
    ec_send_processdata();
    ec_receive_processdata(EC_TIMEOUTRET);
    if (!pre) {
      AUTD3_LIB_SEND_COND.store(true, std::memory_order_release);
    }

    AUTD3_LIB_RTTHREAD_LOCK.store(false, std::memory_order_release);
  }
}

void SOEMController::SetupSync0(bool actiavte, uint32_t cycle_time_ns) {
  auto exceed = cycle_time_ns > 100 * 1000 * 1000u;  // 100 ms, todo
  for (uint16 slave = 1; slave <= _dev_num; slave++) {
    if (exceed) {
      ec_dcsync0(slave, actiavte, cycle_time_ns, 0);
    } else {
      int shift = static_cast<int>(_dev_num) - slave;
      ec_dcsync0(slave, actiavte, cycle_time_ns, shift * cycle_time_ns);
    }
  }
}

void SOEMController::Open(const char *ifname, size_t dev_num, ECConfig config) {
  _dev_num = dev_num;
  _config = config;
  _output_frame_size = (config.header_size + config.body_size) * _dev_num;

  auto size = _output_frame_size + (config.input_frame_size * _dev_num);
  if (size != _io_map_size) {
    _io_map_size = size;

    if (_io_map != nullptr) {
      delete[] _io_map;
    }

    _io_map = new uint8_t[size];

    memset(_io_map, 0x00, _io_map_size);
  }

  _sync0_cyctime = config.ec_sync0_cyctime_ns;

  if (ec_init(ifname) <= 0) {
    std::cerr << "No socket connection on " << ifname << std::endl;
    return;
  }

  auto wc = ec_config(0, _io_map);
  if (wc <= 0) {
    std::cerr << "No slaves found!" << std::endl;
    return;
  } else if (wc != dev_num) {
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

  SetupSync0(true, _sync0_cyctime);

#ifdef WINDOWS
  _timerQueue = CreateTimerQueue();
  if (_timerQueue == NULL) {
    std::cerr << "CreateTimerQueue failed." << std::endl;
    return;
  }

  if (!CreateTimerQueueTimer(&_timer, _timerQueue, (WAITORTIMERCALLBACK)RTthread, reinterpret_cast<void *>(this), 0,
                             config.ec_sm3_cyctime_ns / 1000 / 1000, 0)) {
    std::cerr << "CreateTimerQueueTimer failed." << std::endl;
    return;
  }

  HANDLE hProcess = GetCurrentProcess();
  if (!SetPriorityClass(hProcess, REALTIME_PRIORITY_CLASS)) {
    std::cerr << "Failed to SetPriorityClass" << std::endl;
  }

#elif defined MACOSX
  _queue = dispatch_queue_create("timerQueue", 0);

  _timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
  dispatch_source_set_event_handler(_timer, ^{
    RTthread(this);
  });

  dispatch_source_set_cancel_handler(_timer, ^{
    dispatch_release(_timer);
    dispatch_release(_queue);
  });

  dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, 0);
  dispatch_source_set_timer(_timer, start, config.ec_sm3_cyctime_ns, 0);
  dispatch_resume(_timer);
#elif defined LINUX
  struct itimerspec itval;
  struct sigevent se;

  itval.it_value.tv_sec = 0;
  itval.it_value.tv_nsec = config.ec_sm3_cyctime_ns;
  itval.it_interval.tv_sec = 0;
  itval.it_interval.tv_nsec = config.ec_sm3_cyctime_ns;

  memset(&se, 0, sizeof(se));
  se.sigev_value.sival_ptr = this;
  se.sigev_notify = SIGEV_THREAD;
  se.sigev_notify_function = RTthread;
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

void SOEMController::Close() {
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
    if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0)) {
      if (GetLastError() != ERROR_IO_PENDING) std::cerr << "DeleteTimerQueue failed." << std::endl;
    }
#elif defined MACOSX
    dispatch_source_cancel(_timer);
#elif defined LINUX
    timer_delete(_timer_id);
#endif

    SetupSync0(false, _sync0_cyctime);

    ec_slave[0].state = EC_STATE_INIT;
    ec_writestate(0);

    ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);

    ec_close();
  }
}

void SOEMController::CreateSendThread(size_t header_size, size_t body_size) {
  _send_thread = std::thread(
      [this](size_t header_size, size_t body_size) {
        while (_is_open) {
          std::unique_ptr<uint8_t[]> buf = nullptr;
          size_t size = 0;
          {
            std::unique_lock<std::mutex> lk(_send_mtx);
            _send_cond.wait(lk, [&] { return _send_q.size() > 0 || !_is_open; });
            if (_send_q.size() > 0) {
              auto tmp = move(_send_q.front());
              buf = move(tmp.first);
              size = tmp.second;
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
              while (!AUTD3_LIB_SEND_COND.load(std::memory_order_acquire)) {
              }
            }

            _send_q.pop();
          }
        }
      },
      header_size, body_size);
}

SOEMController::SOEMController() {
  this->_is_open = false;
  this->_io_map = nullptr;
}

SOEMController::~SOEMController() {
  if (_io_map != nullptr) delete[] _io_map;
}

std::unique_ptr<ISOEMController> ISOEMController::Create() { return std::make_unique<SOEMController>(); }

std::vector<EtherCATAdapterInfo> EtherCATAdapterInfo::EnumerateAdapters() {
  auto adapter = ec_find_adapters();
  auto _adapters = std::vector<EtherCATAdapterInfo>();
  while (adapter != NULL) {
    auto *info = new EtherCATAdapterInfo;
    info->desc = std::string(adapter->desc);
    info->name = std::string(adapter->name);
    _adapters.push_back(*info);
    adapter = adapter->next;
  }
  return _adapters;
}
}  // namespace autdsoem
