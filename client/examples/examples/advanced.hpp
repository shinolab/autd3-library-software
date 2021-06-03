// File: advanced.hpp
// Project: examples
// Created Date: 19/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 03/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"

class BurstModulation final : public autd::core::Modulation {
 public:
  static autd::ModulationPtr create() { return std::make_shared<BurstModulation>(); }
  autd::Error build(const autd::Configuration config) override {
    this->buffer().resize(config.mod_buf_size(), 0);
    this->buffer().at(0) = 0xFF;
    return autd::Ok(true);
  }
};

class UniformGain final : public autd::core::Gain {
 public:
  static autd::GainPtr create() { return std::make_shared<UniformGain>(); }
  autd::Error calc(const autd::GeometryPtr& geometry) override {
    for (size_t i = 0; i < geometry->num_transducers(); i++)
      this->_data[geometry->device_idx_for_trans_idx(i)].at(i % autd::NUM_TRANS_IN_UNIT) = 0xFF00;
    this->_built = true;
    return autd::Ok(true);
  }
};

inline void advanced_test(autd::ControllerPtr& autd) {
  autd->silent_mode() = false;

  std::vector<autd::DataArray> delays;
  autd::DataArray delay{};
  delay.fill(0);
  delay[0] = 4;  // 4 cycle = 100 us delay in 0-th transducer
  delays.emplace_back(delay);
  autd->set_output_delay(delays).unwrap();

  const auto g = UniformGain::create();
  const auto m = BurstModulation::create();
  autd->send(g, m).unwrap();
}
