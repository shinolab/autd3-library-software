// File: timer.hpp
// Project: win32
// Created Date: 01/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <Windows.h>

#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "../osal_callback.hpp"
#include "core/result.hpp"

namespace autd::core {

template <typename T>
class Timer {
 public:
  Timer(std::unique_ptr<T> handler, const uint32_t timer_id) : _handler(std::move(handler)), _timer_id(timer_id) {}
  ~Timer() { (void)this->stop(); }

  [[nodiscard]] static Result<std::unique_ptr<Timer>, std::string> start(std::unique_ptr<T> handler, uint32_t interval_us) {
    interval_us = interval_us / 1000 * 1000;

    const uint32_t u_resolution = 1;
    timeBeginPeriod(u_resolution);

    auto *const h_process = GetCurrentProcess();
    SetPriorityClass(h_process, REALTIME_PRIORITY_CLASS);

    const auto timer_id = timeSetEvent(interval_us / 1000, u_resolution, timer_thread, reinterpret_cast<DWORD_PTR>(handler.get()), TIME_PERIODIC);
    if (timer_id == 0) return Err(std::string("timeSetEvent failed"));

    return Ok(std::make_unique<Timer>(std::move(handler), timer_id));
  }

  [[nodiscard]] Result<std::unique_ptr<T>, std::string> stop() {
    const uint32_t u_resolution = 1;
    timeEndPeriod(u_resolution);
    if (timeKillEvent(_timer_id) != TIMERR_NOERROR) return Err(std::string("timeKillEvent failed"));
    return Ok(std::move(this->_handler));
  }

  Timer(const Timer &) = delete;
  Timer(Timer &&) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(Timer &&) = delete;

 private:
  static void CALLBACK timer_thread([[maybe_unused]] UINT u_timer_id, [[maybe_unused]] UINT u_msg, DWORD_PTR dw_user, [[maybe_unused]] DWORD_PTR dw1,
                                    [[maybe_unused]] DWORD_PTR dw2) {
    auto *const handler = reinterpret_cast<T *>(dw_user);
    handler->callback();
  }

  std::unique_ptr<T> _handler;
  uint32_t _timer_id = 0;
};
}  // namespace autd::core
