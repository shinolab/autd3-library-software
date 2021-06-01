// File: timer.hpp
// Project: linux
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
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

namespace autd::core {

template <typename T>
class Timer {
 public:
  Timer(std::unique_ptr<T> handler, const timer_t timer_id) : _handler(std::move(handler)), _timer_id(timer_id) {}
  ~Timer() { (void)this->stop(); }
  [[nodiscard]] static Result<std::unique_ptr<Timer>, std::string> start(std::unique_ptr<T> handler, const uint32_t interval_us) {
    struct itimerspec itval;
    struct sigevent se;

    itval.it_value.tv_sec = 0;
    itval.it_value.tv_nsec = interval_us * 1000L;
    itval.it_interval.tv_sec = 0;
    itval.it_interval.tv_nsec = interval_us * 1000L;

    memset(&se, 0, sizeof(se));
    se.sigev_value.sival_ptr = handler.get();
    se.sigev_notify = SIGEV_THREAD;
    se.sigev_notify_function = notify;
    se.sigev_notify_attributes = NULL;

    timer_t timer_id;
    if (timer_create(CLOCK_REALTIME, &se, &timer_id) < 0) return Err(std::string("timer_create failed"));
    if (timer_settime(timer_id, 0, &itval, NULL) < 0) return Err(std::string("timer_settime failed"));

    return std::make_unique<Timer>(std::move(handler), timer_id);
  }
  [[nodiscard]] Result<std::unique_ptr<T>, std::string> stop() {
    if (timer_delete(_timer_id) < 0) return Err(std::string("timer_delete failed"));
    return Ok(std::move(this->_handler));
  }

  Timer(const Timer &) = delete;
  Timer(Timer &&) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(Timer &&) = delete;

 private:
  std::unique_ptr<T> _handler;
  timer_t _timer_id;

  static void notify(union sigval sv) {
    auto *timer = reinterpret_cast<T *>(sv.sival_ptr);
    timer->callback();
  }
};
}  // namespace autd::core
