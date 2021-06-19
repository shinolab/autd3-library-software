// File: advanced.hpp
// Project: examples
// Created Date: 19/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"

class BurstModulation final : public autd::core::Modulation {
 public:
  static autd::ModulationPtr create(size_t buf_size = 4000, uint16_t mod_freq_div = 10) {
    return std::make_shared<BurstModulation>(buf_size, mod_freq_div);
  }
  autd::Error calc() override {
    this->buffer().resize(_buf_size, 0);
    this->buffer().at(_buf_size - 1) = 0xFF;
    return autd::Ok(true);
  }

  BurstModulation(const size_t buf_size, const uint16_t mod_freq_div) : Modulation(mod_freq_div), _buf_size(buf_size) {}

 private:
  size_t _buf_size;
};

class UniformGain final : public autd::core::Gain {
 public:
  static autd::GainPtr create() { return std::make_shared<UniformGain>(); }
  autd::Error calc(const autd::GeometryPtr& geometry) override {
    for (size_t i = 0; i < geometry->num_transducers(); i++)
      this->_data[geometry->device_idx_for_trans_idx(i)].at(i % autd::NUM_TRANS_IN_UNIT) = 0xFF80;
    this->_built = true;
    return autd::Ok(true);
  }
};

inline void advanced_test(autd::ControllerPtr& autd) {
  autd->silent_mode() = false;

  std::vector<std::array<uint8_t, autd::NUM_TRANS_IN_UNIT>> delays;
  delays.resize(autd->geometry()->num_devices());
  delays[0][0] = 4;  // 4 cycle = 100 us delay in 0-th transducer
  autd->set_output_delay(delays).unwrap();

  std::vector<std::array<bool, autd::NUM_TRANS_IN_UNIT>> enable;
  enable.resize(autd->geometry()->num_devices());
  for (size_t i = 0; i < autd->geometry()->num_devices(); i++) enable[i].fill(true);
  enable[0][17] = false;  // disable 17-th transducer
  autd->set_enable(enable).unwrap();

  const auto g = UniformGain::create();
  const auto m = BurstModulation::create();
  autd->send(g, m).unwrap();
}
