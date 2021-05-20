// File: timer.hpp
// Project: linux
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <signal.h>
#include <time.h>

#include <cstring>
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

    const auto r = timer_delete(_timer_id);
    if (r < 0) return Err(std::string("timer_delete failed"));

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

  timer_t _timer_id;

  bool _loop = false;

  static void notify(union sigval sv) {
    auto *timer = reinterpret_cast<Timer *>(sv.sival_ptr);
    timer->_cb();
  }

  [[nodiscard]] Error init_timer() {
    struct itimerspec itval;
    struct sigevent se;

    itval.it_value.tv_sec = 0;
    itval.it_value.tv_nsec = this->_interval_us * 1000L;
    itval.it_interval.tv_sec = 0;
    itval.it_interval.tv_nsec = this->_interval_us * 1000L;

    memset(&se, 0, sizeof(se));
    se.sigev_value.sival_ptr = this;
    se.sigev_notify = SIGEV_THREAD;
    se.sigev_notify_function = notify;
    se.sigev_notify_attributes = NULL;

    if (timer_create(CLOCK_REALTIME, &se, &_timer_id) < 0) return Err(std::string("timer_create failed"));

    if (timer_settime(_timer_id, 0, &itval, NULL) < 0) return Err(std::string("timer_settime failed"));

    return Ok(true);
  }
};
}  // namespace autd
