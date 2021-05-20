// File: timer.hpp
// Project: win32
// Created Date: 01/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <Windows.h>

#include <functional>
#include <string>
#include <thread>

#include "core/result.hpp"

namespace autd {

class Timer {
 public:
  Timer() noexcept : _interval_us(1) {}
  ~Timer() { (void)this->stop(); }
  bool SetInterval(uint32_t &interval_us) {
    auto result = true;
    if (interval_us % 1000 != 0) {
      interval_us = interval_us / 1000 * 1000;
      result = false;
    }
    this->_interval_us = interval_us;
    return result;
  }
  [[nodiscard]] Error start(const std::function<void()> &callback) {
    if (auto res = this->stop(); res.is_err()) return res;
    this->_cb = callback;

    const uint32_t u_resolution = 1;
    timeBeginPeriod(u_resolution);

    auto *const h_process = GetCurrentProcess();
    SetPriorityClass(h_process, REALTIME_PRIORITY_CLASS);

    _timer_id = timeSetEvent(this->_interval_us / 1000, u_resolution, timer_thread, reinterpret_cast<DWORD_PTR>(this), TIME_PERIODIC);
    if (_timer_id == 0) return Err(std::string("timeSetEvent failed"));

    this->_loop = true;
    return Ok(true);
  }

  [[nodiscard]] Error stop() {
    if (!this->_loop) return Ok(true);
    this->_loop = false;

    const uint32_t u_resolution = 1;
    timeEndPeriod(u_resolution);
    if (timeKillEvent(_timer_id) != TIMERR_NOERROR) return Err(std::string("timeKillEvent failed"));

    return Ok(true);
  }

  Timer(const Timer &) = delete;
  Timer(Timer &&) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(Timer &&) = delete;

 private:
  static void CALLBACK timer_thread([[maybe_unused]] UINT u_timer_id, [[maybe_unused]] UINT u_msg, DWORD_PTR dw_user, [[maybe_unused]] DWORD_PTR dw1,
                                    [[maybe_unused]] DWORD_PTR dw2) {
    auto *const timer = reinterpret_cast<Timer *>(dw_user);
    timer->_cb();
  }

  uint32_t _interval_us;
  std::function<void()> _cb;

  uint32_t _timer_id = 0;
  std::thread _main_thread;
  bool _loop = false;
};
}  // namespace autd
