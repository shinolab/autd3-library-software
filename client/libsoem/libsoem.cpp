// File: libsoem.cpp
// Project: libsoem
// Created Date: 23/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 19/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#if (_WIN32 || _WIN64)
#define WINDOWS
#elif defined __APPLE__
#define MACOSX
#elif defined __linux__
#define LINUX
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

#include "ethercat.h"
#include "libsoem.hpp"

namespace libsoem {

static std::atomic<bool> AUTD3_LIB_SEND_COND(false);
static std::atomic<bool> AUTD3_LIB_RTTHREAD_LOCK(false);

class SOEMController::impl {
 public:
  void Open(const char *ifname, size_t devNum, uint32_t ec_sm3_cyctime_ns, uint32_t ec_sync0_cyctime_ns, size_t header_size, size_t body_size,
            size_t input_frame_size);
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
  void SetWaitForProcessMsg(bool is_wait);
  std::vector<uint16_t> Read(size_t input_frame_idx);
  bool Close();
  ~impl();

  bool _isOpened = false;

 private:
#ifdef WINDOWS
  static void CALLBACK RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired);
#elif defined MACOSX
  static void RTthread(SOEMController::impl *pimpl);
#elif defined LINUX
  static void RTthread(union sigval sv);
#endif
  void SetupSync0(bool actiavte, uint32_t CycleTime);
  void CreateCopyThread(size_t header_size, size_t body_size);

  bool WaitForProcessMsg(uint8_t sendMsgId);

  uint8_t *_IOmap;
  size_t _iomap_size = 0;
  size_t _output_frame_size = 0;
  uint32_t _sync0_cyctime = 0;

  std::queue<size_t> _send_size_q;
  std::queue<std::unique_ptr<uint8_t[]>> _send_buf_q;
  std::thread _cpy_thread;
  std::condition_variable _cpy_cond;
  bool _sent = false;
  bool _isWaitProcessMsg = true;
  std::mutex _cpy_mtx;
  std::mutex _send_mtx;
  std::mutex _waitcond_mtx;

  size_t _devNum = 0;

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

void SOEMController::impl::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  {
    std::unique_lock<std::mutex> lk(_cpy_mtx);
    _send_size_q.push(size);
    _send_buf_q.push(std::move(buf));
  }
  _cpy_cond.notify_all();
}

#ifdef WINDOWS
void CALLBACK SOEMController::impl::RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired)
#elif defined MACOSX
void libsoem::SOEMController::impl::RTthread(SOEMController::impl *pimpl)
#elif defined LINUX
void libsoem::SOEMController::impl::RTthread(union sigval sv)
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

std::vector<uint16_t> SOEMController::impl::Read(size_t input_frame_idx) {
  std::vector<uint16_t> res;
  for (size_t i = 0; i < _devNum; i++) {
    uint16_t base = ((uint16_t)_IOmap[input_frame_idx + 2 * i + 1] << 8) | _IOmap[input_frame_idx + 2 * i];
    res.push_back(base);
  }
  return res;
}

void SOEMController::impl::SetupSync0(bool actiavte, uint32_t CycleTime) {
  auto exceed = CycleTime > 1000000u;
  for (uint16 slave = 1; slave <= _devNum; slave++) {
    if (exceed) {
      ec_dcsync0(slave, actiavte, CycleTime, 0);  // SYNC0
    } else {
      int shift = static_cast<int>(_devNum) - slave;
      ec_dcsync0(slave, actiavte, CycleTime, shift * CycleTime);  // SYNC0
    }
  }
}

void SOEMController::impl::SetWaitForProcessMsg(bool iswait) {
  std::unique_lock<std::mutex> lk(_waitcond_mtx);
  this->_isWaitProcessMsg = iswait;
}

bool SOEMController::impl::WaitForProcessMsg(uint8_t sendMsgId) {
  auto chk = 10;
  while (chk--) {
    int processed = 0;
    for (size_t i = 0; i < _devNum; i++) {
      uint8_t recv_id = _IOmap[_output_frame_size + 2 * i + 1];
      if (recv_id == sendMsgId) {
        processed++;
      }
    }
    if (processed == _devNum) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

void SOEMController::impl::CreateCopyThread(size_t header_size, size_t body_size) {
  _cpy_thread = std::thread(
      [this](size_t header_size, size_t body_size) {
        while (_isOpened) {
          std::unique_ptr<uint8_t[]> buf = nullptr;
          size_t size = 0;
          {
            std::unique_lock<std::mutex> lk(_cpy_mtx);
            _cpy_cond.wait(lk, [&] { return _send_buf_q.size() > 0 || !_isOpened; });
            if (_send_buf_q.size() > 0) {
              buf = move(_send_buf_q.front());
              size = _send_size_q.front();
            }
          }

          if (buf != nullptr && _isOpened) {
            const auto includes_gain = ((size - header_size) / body_size) > 0;
            const auto output_frame_size = header_size + body_size;

            for (size_t i = 0; i < _devNum; i++) {
              if (includes_gain) memcpy(&_IOmap[output_frame_size * i], &buf[header_size + body_size * i], body_size);
              memcpy(&_IOmap[output_frame_size * i + body_size], &buf[0], header_size);
            }

            {
              AUTD3_LIB_SEND_COND.store(false, std::memory_order_release);
              while (!AUTD3_LIB_SEND_COND.load(std::memory_order_acquire)) {
              }
            }

            bool isWait;
            {
              std::unique_lock<std::mutex> lk(_waitcond_mtx);
              isWait = _isWaitProcessMsg;
            }

            if (isWait) {
              auto sendMsgId = buf[0];
              auto sucess = WaitForProcessMsg(sendMsgId);
              if (!sucess) {
                std::cerr << "Some data may not have been transferred to AUTDs." << std::endl;
              }
            }

            _send_size_q.pop();
            _send_buf_q.pop();
          }
        }
      },
      header_size, body_size);
}

void SOEMController::impl::Open(const char *ifname, size_t devNum, uint32_t ec_sm3_cyctime_ns, uint32_t ec_sync0_cyctime_ns, size_t header_size,
                                size_t body_size, size_t input_frame_size) {
  _devNum = devNum;
  _output_frame_size = (header_size + body_size) * _devNum;

  auto size = (header_size + body_size + input_frame_size) * _devNum;
  if (size != _iomap_size) {
    _iomap_size = size;

    if (_IOmap != nullptr) {
      delete[] _IOmap;
    }

    _IOmap = new uint8_t[size];

    memset(_IOmap, 0x00, _iomap_size);
  }

  _sync0_cyctime = ec_sync0_cyctime_ns;

  if (ec_init(ifname)) {
    if (ec_config(0, _IOmap) > 0) {
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

      if (ec_slave[0].state == EC_STATE_OPERATIONAL) {
        _isOpened = true;

        SetupSync0(true, _sync0_cyctime);

#ifdef WINDOWS
        _timerQueue = CreateTimerQueue();
        if (_timerQueue == NULL) std::cerr << "CreateTimerQueue failed." << std::endl;

        if (!CreateTimerQueueTimer(&_timer, _timerQueue, (WAITORTIMERCALLBACK)RTthread, reinterpret_cast<void *>(this), 0,
                                   ec_sm3_cyctime_ns / 1000 / 1000, 0))
          std::cerr << "CreateTimerQueueTimer failed." << std::endl;

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
        dispatch_source_set_timer(_timer, start, ec_sm3_cyctime_ns, 0);
        dispatch_resume(_timer);
#elif defined LINUX
        struct itimerspec itval;
        struct sigevent se;

        itval.it_value.tv_sec = 0;
        itval.it_value.tv_nsec = ec_sm3_cyctime_ns;
        itval.it_interval.tv_sec = 0;
        itval.it_interval.tv_nsec = ec_sm3_cyctime_ns;

        memset(&se, 0, sizeof(se));
        se.sigev_value.sival_ptr = this;
        se.sigev_notify = SIGEV_THREAD;
        se.sigev_notify_function = RTthread;
        se.sigev_notify_attributes = NULL;

        if (timer_create(CLOCK_REALTIME, &se, &_timer_id) < 0) std::cerr << "Error: timer_create." << std::endl;

        if (timer_settime(_timer_id, 0, &itval, NULL) < 0) std::cerr << "Error: timer_settime." << std::endl;
#endif

        CreateCopyThread(header_size, body_size);
      } else {
        std::cerr << "One ore more slaves are not responding." << std::endl;
      }
    } else {
      std::cerr << "No slaves found!" << std::endl;
    }
  } else {
    std::cerr << "No socket connection on " << ifname << std::endl;
  }
}

bool SOEMController::impl::Close() {
  if (_isOpened) {
    _isOpened = false;
    _cpy_cond.notify_all();
    if (std::this_thread::get_id() != _cpy_thread.get_id() && this->_cpy_thread.joinable()) this->_cpy_thread.join();
    {
      std::unique_lock<std::mutex> lk(_cpy_mtx);
      std::queue<size_t>().swap(_send_size_q);
      std::queue<std::unique_ptr<uint8_t[]>>().swap(_send_buf_q);
    }

    memset(_IOmap, 0x00, _output_frame_size);
    AUTD3_LIB_SEND_COND.store(false, std::memory_order_release);
    do {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (!AUTD3_LIB_SEND_COND.load(std::memory_order_acquire));

#ifdef WINDOWS
    if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0)) {
      if (GetLastError() != ERROR_IO_PENDING) std::cerr << "DeleteTimerQueue failed." << std::endl;
    }
#elif defined MACOSX
    dispatch_source_cancel(_timer);
#elif defined LINUX
    timer_delete(_timer_id);
#endif
    auto chk = 200;
    auto clear = true;
    do {
#ifdef WINDOWS
      RTthread(NULL, FALSE);
#elif defined MACOSX
      RTthread(nullptr);
#elif defined LINUX
      RTthread(sigval{});
#endif
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

      auto r = Read(_output_frame_size);
      for (auto c : r) {
        uint8_t id = c & 0xff00;
        if (id != 0) {
          clear = false;
          break;
        } else {
          clear = true;
        }
      }
    } while (chk-- && !clear);

    SetupSync0(false, _sync0_cyctime);

    ec_slave[0].state = EC_STATE_INIT;
    ec_writestate(0);

    ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);

    ec_close();

    return clear;
  } else {
    return true;
  }
}

SOEMController::impl::~impl() {
  if (_IOmap != nullptr) delete[] _IOmap;
}

SOEMController::SOEMController() { this->_pimpl = std::make_unique<impl>(); }

SOEMController::~SOEMController() { this->_pimpl->Close(); }

void SOEMController::Open(const char *ifname, size_t devNum, uint32_t ec_sm3_cyctime_ns, uint32_t ec_sync0_cyctime_ns, size_t header_size,
                          size_t body_size, size_t input_frame_size) {
  this->_pimpl->Open(ifname, devNum, ec_sm3_cyctime_ns, ec_sync0_cyctime_ns, header_size, body_size, input_frame_size);
}

void SOEMController::Send(size_t size, std::unique_ptr<uint8_t[]> buf) { this->_pimpl->Send(size, std::move(buf)); }

void SOEMController::SetWaitForProcessMsg(bool isWait) { this->_pimpl->SetWaitForProcessMsg(isWait); }

std::vector<uint16_t> SOEMController::Read(size_t input_frame_idx) { return this->_pimpl->Read(input_frame_idx); }

bool SOEMController::Close() { return this->_pimpl->Close(); }

bool SOEMController::isOpen() { return this->_pimpl->_isOpened; }

std::vector<EtherCATAdapterInfo> EtherCATAdapterInfo::EnumerateAdapters() {
  auto adapter = ec_find_adapters();
  auto _adapters = std::vector<EtherCATAdapterInfo>();
  while (adapter != NULL) {
    auto *info = new EtherCATAdapterInfo;
    info->desc = std::make_shared<std::string>(adapter->desc);
    info->name = std::make_shared<std::string>(adapter->name);
    _adapters.push_back(*info);
    adapter = adapter->next;
  }
  return _adapters;
}
}  // namespace libsoem
