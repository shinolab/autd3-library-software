// File: advanced.hpp
// Project: examples
// Created Date: 19/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"

class BurstModulation final : public autd::core::Modulation {
 public:
  static autd::ModulationPtr Create() { return std::make_shared<BurstModulation>(); }

  autd::Error build(const autd::Configuration config) override {
    const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());
    this->buffer().resize(mod_buf_size, 0);
    this->buffer().at(0) = 0xFF;
    return autd::Ok();
  }
};

class UniformGain final : public autd::core::Gain {
 public:
  static autd::GainPtr Create() { return std::make_shared<UniformGain>(); }
  autd::Error calc(const autd::GeometryPtr& geometry) override {
    for (size_t i = 0; i < geometry->num_transducers(); i++)
      this->_data[geometry->device_idx_for_trans_idx(i)].at(i % autd::NUM_TRANS_IN_UNIT) = 0xFF00;

    this->_built = true;
    return autd::Ok();
  }
};

inline void advanced_test(autd::Controller& autd) {
  autd.silent_mode() = false;

  const auto g = UniformGain::Create();
  const auto m = BurstModulation::Create();
  autd.send(g, m).unwrap();

  std::vector<autd::DataArray> delay;
  autd::DataArray ar{};
  ar.fill(0);
  ar[0] = 4;  // 4 cycle = 100 us delay in 0-th transducer
  delay.emplace_back(ar);

  autd.set_output_delay(delay).unwrap();
}
