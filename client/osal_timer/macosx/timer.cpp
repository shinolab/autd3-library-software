﻿// File: timer.cpp
// Project: macosx
// Created Date: 04/09/2019
// Author: Shun Suzuki
// -----
// Last Modified: 06/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#include "../timer.hpp"

#include <signal.h>
#include <string.h>
#include <time.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace autd {

static constexpr auto TIME_SCALE = 1000L;  // us

Timer::Timer() noexcept : Timer::Timer(false) {}

Timer::Timer(bool high_resolution) noexcept : _interval_us(1) { (void)high_resolution; }
Timer::~Timer() { (void)this->Stop(); }

bool Timer::SetInterval(uint32_t &interval_us) {
  this->_interval_us = interval_us;
  return true;
}

Result<bool, std::string> Timer::Start(const std::function<void()> &callback) {
  auto res = this->Stop();
  if (res.is_err()) return res;

  this->_cb = callback;
  this->_loop = true;
  return this->InitTimer();
}

Result<bool, std::string> Timer::Stop() {
  if (!this->_loop) return Ok(false);

  dispatch_source_cancel(_timer);
  this->_loop = false;

  return Ok(true);
}

Result<bool, std::string> Timer::InitTimer() {
  _queue = dispatch_queue_create("timerQueue", 0);

  _timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
  dispatch_source_set_event_handler(_timer, ^{
    MainLoop(this);
  });

  dispatch_source_set_cancel_handler(_timer, ^{
    dispatch_release(_timer);
    dispatch_release(_queue);
  });

  dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, 0);
  dispatch_source_set_timer(_timer, start, this->_interval_us * TIME_SCALE, 0);
  dispatch_resume(_timer);

  return Ok(true);
}

void Timer::MainLoop(Timer *ptr) { ptr->_cb(); }
}  // namespace autd
