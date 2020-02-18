// File: timer.cpp
// Project: win32
// Created Date: 02/07/2018
// Author: Shun Suzuki and Saya Mizutani
// -----
// Last Modified: 18/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../timer.hpp"

#include <Windows.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace autd {

static constexpr auto TIME_SCALE = 1000 * 1000L;  // us

static std::atomic<bool> AUTD3_LIB_TIMER_LOCK(false);

Timer::Timer() noexcept : Timer::Timer(false) {}

Timer::Timer(bool highResolusion) noexcept {
  this->_interval_us = 1;
  this->_highResolusion = highResolusion;
}

Timer::~Timer() noexcept(false) { this->Stop(); }

void Timer::SetInterval(int interval_us) {
  if (interval_us <= 0) throw new std::runtime_error("Interval must be positive integer.");

  if (!this->_highResolusion && ((interval_us % 1000) != 0)) {
    std::cerr << "The accuracy of the Windows timer is 1 ms. It may not run properly." << std::endl;
  }

  this->_interval_us = interval_us;
}

void Timer::Start(const std::function<void()> &callback) {
  this->Stop();
  this->cb = callback;
  this->_loop = true;
  if (this->_highResolusion) {
    this->InitTimer();
  } else {
    this->_timerQueue = CreateTimerQueue();
    if (this->_timerQueue == NULL) std::cerr << "CreateTimerQueue failed." << std::endl;

    if (!CreateTimerQueueTimer(&this->_timer, this->_timerQueue, (WAITORTIMERCALLBACK)TimerThread, reinterpret_cast<void *>(this), 0,
                               this->_interval_us / 1000, 0))
      std::cerr << "CreateTimerQueueTimer failed." << std::endl;
  }
}

void Timer::Stop() {
  if (this->_loop) {
    this->_loop = false;

    if (this->_highResolusion) {
      this->_mainThread.join();

    } else {
      if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0)) {
        if (GetLastError() != ERROR_IO_PENDING) std::cerr << "DeleteTimerQueue failed." << std::endl;
      }
    }
  }
}

void Timer::InitTimer() {
  this->_mainThread = std::thread([&] { Timer::MainLoop(); });
}

inline void MicroSleep(int micro_sec) noexcept {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);

  const auto sleep = micro_sec * (freq.QuadPart / TIME_SCALE);

  LARGE_INTEGER start;
  QueryPerformanceCounter(&start);
  while (true) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (now.QuadPart - start.QuadPart > sleep) break;
  }
}

void Timer::MainLoop() {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);

  LARGE_INTEGER start;
  QueryPerformanceCounter(&start);

  auto count = 0xffffffffL;
  int sleep_t = 0;
  while (this->_loop) {
    if (count > 0xfffffff0) {
      count = 0;
      QueryPerformanceCounter(&start);
    }

    this->cb();

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    const auto elasped = static_cast<double>(now.QuadPart - start.QuadPart) / freq.QuadPart * TIME_SCALE;

    sleep_t = static_cast<int>(this->_interval_us * ++count - elasped);
    if (sleep_t > 0) {
      MicroSleep(sleep_t);
    } else {
      continue;
    }
  }
}

void Timer::TimerThread(PVOID lpParam, BOOLEAN TimerOrWaitFired) {
  bool expected = false;
  if (AUTD3_LIB_TIMER_LOCK.compare_exchange_weak(expected, true)) {
    Timer *_ptimer = reinterpret_cast<Timer *>(lpParam);
    _ptimer->cb();
    AUTD3_LIB_TIMER_LOCK.store(false, std::memory_order_release);
  }
}
}  // namespace autd
