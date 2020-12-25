// File: timer.cpp
// Project: win32
// Created Date: 02/07/2018
// Author: Shun Suzuki and Saya Mizutani
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../timer.hpp"

#include <Windows.h>

#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>

namespace autd {

static constexpr auto TIME_SCALE = 1000 * 1000L;  // us

static std::atomic<bool> AUTD3_LIB_TIMER_LOCK(false);

Timer::Timer() noexcept : Timer(false) {}

Timer::Timer(const bool high_resolution) noexcept : _interval_us(1), _high_resolution(high_resolution) {}

Timer::~Timer() noexcept(false) { this->Stop(); }

void Timer::SetInterval(const int interval_us) {
  if (interval_us <= 0) throw std::runtime_error("Interval must be positive integer.");

  if (!this->_high_resolution && interval_us % 1000 != 0) {
    std::cerr << "The accuracy of the Windows timer is 1 ms. It may not run properly." << std::endl;
  }

  this->_interval_us = interval_us;
}

void Timer::Start(const std::function<void()> &callback) {
  this->Stop();
  this->_cb = callback;
  this->_loop = true;
  if (this->_high_resolution) {
    this->InitTimer();
  } else {
    const uint32_t u_resolution = 1;
    timeBeginPeriod(u_resolution);
    _timer_id = timeSetEvent(this->_interval_us / 1000, u_resolution, static_cast<LPTIMECALLBACK>(TimerThread), reinterpret_cast<DWORD_PTR>(this),
                             TIME_PERIODIC);
    if (_timer_id == 0) {
      std::cerr << "timeSetEvent failed." << std::endl;
    }
  }
}

void Timer::Stop() {
  if (this->_loop) {
    this->_loop = false;

    if (this->_high_resolution) {
      this->_mainThread.join();

    } else {
      if (_timer_id != 0) {
        const uint32_t u_resolution = 1;
        timeKillEvent(_timer_id);
        timeEndPeriod(u_resolution);
      }
    }
  }
}

void Timer::InitTimer() {
  this->_mainThread = std::thread([&] { Timer::MainLoop(); });
}

inline void MicroSleep(const int micro_sec) noexcept {
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

void Timer::MainLoop() const {
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);

  LARGE_INTEGER start;
  QueryPerformanceCounter(&start);

  auto count = 0xffffffffL;
  while (this->_loop) {
    if (count > 0xfffffff0) {
      count = 0;
      QueryPerformanceCounter(&start);
    }

    this->_cb();

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    const auto elapsed = static_cast<double>(now.QuadPart - start.QuadPart) / static_cast<double>(freq.QuadPart) * TIME_SCALE;

    const auto sleep_t = static_cast<int>(this->_interval_us * ++count - elapsed);
    if (sleep_t > 0) {
      MicroSleep(sleep_t);
    }
  }
}

void Timer::TimerThread([[maybe_unused]] UINT u_timer_id, [[maybe_unused]] UINT u_msg, const DWORD_PTR dw_user, [[maybe_unused]] DWORD_PTR dw1,
                        [[maybe_unused]] DWORD_PTR dw2) {
  auto expected = false;
  if (AUTD3_LIB_TIMER_LOCK.compare_exchange_weak(expected, true)) {
    auto *const timer = reinterpret_cast<Timer *>(dw_user);
    timer->_cb();
    AUTD3_LIB_TIMER_LOCK.store(false, std::memory_order_release);
  }
}
}  // namespace autd
