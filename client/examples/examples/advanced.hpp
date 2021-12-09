// File: advanced.hpp
// Project: examples
// Created Date: 19/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "autd3.hpp"

class BurstModulation final : public autd::core::Modulation {
 public:
  void calc() override {
    this->_buffer.resize(_buf_size, 0);
    this->_buffer[_buf_size - 1] = 0xFF;
  }

  BurstModulation(const size_t buf_size = 4000, const uint16_t mod_freq_div = 10) : Modulation(mod_freq_div), _buf_size(buf_size) {}

 private:
  size_t _buf_size;
};

class UniformGain final : public autd::core::Gain {
 public:
  UniformGain() : Gain() {}

  void calc(const autd::Geometry& geometry) override {
    for (const auto& dev : geometry)
      for (const auto& trans : dev) {
        this->_data[trans.id()].duty = 0xFF;
        this->_data[trans.id()].phase = 0x80;
      }
  }
};

inline void advanced_test(autd::Controller& autd) {
  autd.silent_mode() = false;

  autd.delay_offset()[0].delay = 4;  // 4 cycle = 100 us delay in 0-th transducer
  autd.set_delay_offset();           // apply change

  UniformGain g;
  BurstModulation m;
  autd.send(g, m);
}
