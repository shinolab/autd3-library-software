// File: timer.hpp
// Project: macosx
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <dispatch/dispatch.h>

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
    this->_interval_us = interval_us;
    return true;
  }
  [[nodiscard]] Error start(const std::function<void()> &callback) {
    if (auto res = this->stop(); res.is_err()) return res;

    this->_cb = callback;
    this->_loop = true;
    return this->init_timer();
  }
  [[nodiscard]] Error stop() {
    if (!this->_loop) return Ok(true);

    dispatch_source_cancel(_timer);
    this->_loop = false;

    return Ok(true);
  }

  Timer(const Timer &) = delete;
  Timer(Timer &&) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(Timer &&) = delete;

 private:
  uint32_t _interval_us;
  std::function<void()> _cb;

  dispatch_queue_t _queue;
  dispatch_source_t _timer;

  bool _loop;

  void main_loop(Timer *ptr) { ptr->_cb(); }

  [[nodiscard]] Error init_timer() {
    _queue = dispatch_queue_create("timerQueue", 0);

    _timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
    dispatch_source_set_event_handler(_timer, ^{
      main_loop(this);
    });

    dispatch_source_set_cancel_handler(_timer, ^{
      dispatch_release(_timer);
      dispatch_release(_queue);
    });

    dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, 0);
    dispatch_source_set_timer(_timer, start, this->_interval_us * 1000L, 0);
    dispatch_resume(_timer);

    return Ok(true);
  }
};
}  // namespace autd
