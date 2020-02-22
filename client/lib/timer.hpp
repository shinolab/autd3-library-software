// File: timer.hpp
// Project: lib
// Created Date:02/07/2018
// Author: Shun Suzuki and Saya Mizutani
// -----
// Last Modified: 22/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once
#include <future>

#if WIN32
#include <Windows.h>
#elif __APPLE__
#include <dispatch/dispatch.h>
#else
#include <signal.h>
#include <time.h>
#endif

namespace autd {
class Timer {
 public:
  Timer() noexcept;
  explicit Timer(bool high_resolusion) noexcept;
  ~Timer() noexcept(false);
  void SetInterval(int interval);
  void Start(const std::function<void()> &callback);
  void Stop();

  Timer(const Timer &) = delete;
  Timer(Timer &&) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(Timer &&) = delete;

 private:
  int _interval_us;
  std::function<void()> _cb;

#if WIN32
  HANDLE _timerQueue = NULL;
  HANDLE _timer = NULL;
#elif __APPLE__
  dispatch_queue_t _queue;
  dispatch_source_t _timer;
#else
  timer_t _timer_id;
#endif

  std::thread _mainThread;
  bool _loop = false;
#if WIN32
  bool _high_resolusion;
  void MainLoop();
  static void CALLBACK TimerThread(PVOID lpParam, BOOLEAN TimerOrWaitFired);
#elif __APPLE__
  static void MainLoop(Timer *ptr);
#else
  static void MainLoop(int signum);
  static void Notify(union sigval sv);
#endif
  void InitTimer();
};
}  // namespace autd
