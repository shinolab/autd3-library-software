// File: timer.hpp
// Project: linux
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <signal.h>
#include <time.h>

#include <functional>
#include <thread>

#include "result.hpp"

namespace autd {

namespace {
static constexpr auto TIME_SCALE = 1000L;  // us to ns
}

class Timer {
 public:
  Timer() noexcept : _interval_us(1) {}
  ~Timer() { (void)this->Stop(); }
  bool SetInterval(uint32_t &interval_us) {
    this->_interval_us = interval_us;
    return true;
  }
  [[nodiscard]] Result<bool, std::string> Start(const std::function<void()> &callback) {
    auto res = this->Stop();
    if (res.is_err()) return res;

    this->_cb = callback;
    this->_loop = true;
    return this->InitTimer();
  }
  [[nodiscard]] Result<bool, std::string> Stop() {
    if (!this->_loop) return Ok(false);

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

  std::thread _main_thread;
  bool _loop = false;

  static void Notify(union sigval sv) {
    auto *timer = reinterpret_cast<Timer *>(sv.sival_ptr);
    timer->_cb();
  }

  [[nodiscard]] Result<bool, std::string> InitTimer() {
    struct sigaction act;
    struct itimerspec itval;
    struct sigevent se;

    memset(&act, 0, sizeof(struct sigaction));
    act.sa_handler = MainLoop;
    act.sa_flags = SA_RESTART;
    if (sigaction(SIGALRM, &act, NULL) < 0) return Err(std::string("sigaction failed"));

    itval.it_value.tv_sec = 0;
    itval.it_value.tv_nsec = this->_interval_us * TIME_SCALE;
    itval.it_interval.tv_sec = 0;
    itval.it_interval.tv_nsec = this->_interval_us * TIME_SCALE;

    memset(&se, 0, sizeof(se));
    se.sigev_value.sival_ptr = this;
    se.sigev_notify = SIGEV_THREAD;
    se.sigev_notify_function = Notify;
    se.sigev_notify_attributes = NULL;

    if (timer_create(CLOCK_REALTIME, &se, &_timer_id) < 0) return Err(std::string("timer_create failed"));

    if (timer_settime(_timer_id, 0, &itval, NULL) < 0) return Err(std::string("timer_settime failed"));

    return Ok(true);
  }
};
}  // namespace autd
