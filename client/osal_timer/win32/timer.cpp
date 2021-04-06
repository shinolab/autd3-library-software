// File: timer.cpp
// Project: win32
// Created Date: 02/07/2018
// Author: Shun Suzuki and Saya Mizutani
// -----
// Last Modified: 06/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../timer.hpp"

namespace autd {

static constexpr auto TIME_SCALE = 1000 * 1000L;  // us

Timer::Timer() noexcept : Timer(false) {}

Timer::Timer(const bool high_resolution) noexcept : _interval_us(1), _high_resolution(high_resolution) {}

Timer::~Timer() { (void)this->Stop(); }

bool Timer::SetInterval(uint32_t &interval_us) {
  auto result = true;
  if (!this->_high_resolution && interval_us % 1000 != 0) {
    interval_us = interval_us / 1000 * 1000;
    result = false;
  }

  this->_interval_us = interval_us;
  return result;
}

Result<bool, std::string> Timer::Start(const std::function<void()> &callback) {
  auto res = this->Stop();
  if (res.is_err()) return res;
  this->_cb = callback;
  this->_loop = true;

  if (this->_high_resolution) return this->InitTimer();

  const uint32_t u_resolution = 1;
  timeBeginPeriod(u_resolution);

  auto *const h_process = GetCurrentProcess();
  SetPriorityClass(h_process, REALTIME_PRIORITY_CLASS);

  _timer_id = timeSetEvent(this->_interval_us / 1000, u_resolution, static_cast<LPTIMECALLBACK>(TimerThread), reinterpret_cast<DWORD_PTR>(this),
                           TIME_PERIODIC);
  if (_timer_id == 0) {
    this->_loop = false;
    return Err(std::string("timeSetEvent failed"));
  }

  return Ok(true);
}

Result<bool, std::string> Timer::Stop() {
  if (!this->_loop) return Ok(false);
  this->_loop = false;

  if (this->_high_resolution) {
    this->_main_thread.join();
    return Ok(true);
  }

  const uint32_t u_resolution = 1;
  timeEndPeriod(u_resolution);
  if (timeKillEvent(_timer_id) != TIMERR_NOERROR) return Err(std::string("timeKillEvent failed"));

  return Ok(true);
}

Result<bool, std::string> Timer::InitTimer() {
  this->_main_thread = std::thread([&] { MainLoop(); });
  return Ok(true);
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
  auto *const timer = reinterpret_cast<Timer *>(dw_user);
  timer->_cb();
}
}  // namespace autd
