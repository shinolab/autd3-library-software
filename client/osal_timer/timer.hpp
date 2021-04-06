// File: timer.hpp
// Project: lib
// Created Date:02/07/2018
// Author: Shun Suzuki and Saya Mizutani
// -----
// Last Modified: 06/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#if WIN32
#include <Windows.h>
#elif __APPLE__
#include <dispatch/dispatch.h>
#else
#include <signal.h>
#include <time.h>
#endif

#include <functional>
#include <thread>

#include "result.hpp"

namespace autd {
class Timer {
 public:
  Timer() noexcept;
  explicit Timer(bool high_resolution) noexcept;
  ~Timer();
  bool SetInterval(uint32_t &interval_us);
  [[nodiscard]] Result<bool, std::string> Start(const std::function<void()> &callback);
  [[nodiscard]] Result<bool, std::string> Stop();

  Timer(const Timer &) = delete;
  Timer(Timer &&) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(Timer &&) = delete;

 private:
  uint32_t _interval_us;
  std::function<void()> _cb;

#if WIN32
  uint32_t _timer_id = 0;
#elif __APPLE__
  dispatch_queue_t _queue;
  dispatch_source_t _timer;
#else
  timer_t _timer_id;
#endif

  std::thread _main_thread;
  bool _loop = false;
#if WIN32
  bool _high_resolution;
  void MainLoop() const;
  static void CALLBACK TimerThread(UINT u_timer_id, UINT u_msg, DWORD_PTR dw_user, DWORD_PTR dw1, DWORD_PTR dw2);
#elif __APPLE__
  static void MainLoop(Timer *ptr);
#else
  static void MainLoop(int signum);
  static void Notify(union sigval sv);
#endif
  [[nodiscard]] Result<bool, std::string> InitTimer();
};
}  // namespace autd
