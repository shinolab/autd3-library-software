﻿// File: timer.cpp
// Project: win32
// Created Date: 02/07/2018
// Author: Shun Suzuki and Saya Mizutani
// -----
// Last Modified: 16/11/2020
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

Timer::Timer(bool high_resolusion) noexcept {
  this->_interval_us = 1;
  this->_high_resolusion = high_resolusion;
}

Timer::~Timer() noexcept(false) { this->Stop(); }

void Timer::SetInterval(int interval_us) {
  if (interval_us <= 0) throw new std::runtime_error("Interval must be positive integer.");

  if (!this->_high_resolusion && ((interval_us % 1000) != 0)) {
    std::cerr << "The accuracy of the Windows timer is 1 ms. It may not run properly." << std::endl;
  }

  this->_interval_us = interval_us;
}

void Timer::Start(const std::function<void()> &callback) {
  this->Stop();
  this->_cb = callback;
  this->_loop = true;
  if (this->_high_resolusion) {
    this->InitTimer();
  } else {
    uint32_t uResolution = 1;
    timeBeginPeriod(uResolution);
    _timer_id = timeSetEvent(this->_interval_us / 1000, uResolution, (LPTIMECALLBACK)TimerThread, reinterpret_cast<DWORD_PTR>(this), TIME_PERIODIC);
    if (_timer_id == 0) {
      std::cerr << "timeSetEvent failed." << std::endl;
    }
  }
}

void Timer::Stop() {
  if (this->_loop) {
    this->_loop = false;

    if (this->_high_resolusion) {
      this->_mainThread.join();

    } else {
      if (_timer_id != 0) {
        uint32_t uResolution = 1;
        timeKillEvent(_timer_id);
        timeEndPeriod(uResolution);
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

    this->_cb();

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

void Timer::TimerThread(UINT uTimerID, UINT uMsg, DWORD_PTR dwUser, DWORD_PTR dw1, DWORD_PTR dw2) {
  bool expected = false;
  if (AUTD3_LIB_TIMER_LOCK.compare_exchange_weak(expected, true)) {
    Timer *_ptimer = reinterpret_cast<Timer *>(dwUser);
    _ptimer->_cb();
    AUTD3_LIB_TIMER_LOCK.store(false, std::memory_order_release);
  }
}
}  // namespace autd
