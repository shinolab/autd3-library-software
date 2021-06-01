// File: timer.hpp
// Project: macosx
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
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

namespace autd::core {

template <typename T>
class Timer {
 public:
  Timer(std::unique_ptr<T> handler, dispatch_queue_t queue, dispatch_source_t timer) _handler(std::move(handler)), _queue(queue), _timer(timer) {}
  ~Timer() { (void)this->stop(); }
  [[nodiscard]] static Result<std::unique_ptr<Timer>, std::string> start(std::unique_ptr<T> handler, const uint32_t interval_us) {
    auto queue = dispatch_queue_create("timerQueue", 0);

    auto timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue);
    dispatch_source_set_event_handler(timer, ^{
      main_loop(handler.get());
    });

    dispatch_source_set_cancel_handler(timer, ^{
      dispatch_release(timer);
      dispatch_release(queue);
    });

    dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, 0);
    dispatch_source_set_timer(timer, start, interval_us * 1000L, 0);
    dispatch_resume(timer);

    return std::make_unique<Timer>(std::move(handler), queue, timer);
  }

  [[nodiscard]] Result<std::unique_ptr<T>, std::string> stop() {
    dispatch_source_cancel(_timer);
    return Ok(std::move(this->_handler));
  }

  Timer(const Timer &) = delete;
  Timer(Timer &&) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer &operator=(Timer &&) = delete;

 private:
  std::unique_ptr<T> _handler;

  dispatch_queue_t _queue;
  dispatch_source_t _timer;

  static void main_loop(T *ptr) { ptr->callback(); }
};
}  // namespace autd::core
