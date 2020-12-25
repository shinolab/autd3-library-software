// File: delay.hpp
// Project: examples
// Created Date: 24/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"

using autd::NUM_TRANS_X;
using autd::NUM_TRANS_Y;
using autd::TRANS_SIZE_MM;

class BurstModulation final : public autd::modulation::Modulation {
 public:
  static autd::ModulationPtr Create() {
    autd::ModulationPtr mod = std::make_shared<BurstModulation>();
    return mod;
  }

  void Build(const autd::Configuration config) override {
    const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());
    this->buffer.resize(mod_buf_size, 0);
    this->buffer.at(0) = 0xFF;
  }
};

class ConstGain final : public autd::gain::Gain {
 public:
  static autd::GainPtr Create() { return std::make_shared<ConstGain>(); }
  void Build() override {
    if (this->built()) return;

    auto geometry = this->geometry();

    autd::gain::CheckAndInit(geometry, &this->_data);

    for (size_t i = 0; i < geometry->num_transducers(); i++) {
      this->_data[geometry->device_idx_for_trans_idx(i)].at(i % autd::NUM_TRANS_IN_UNIT) = 0xFF00;
    }

    this->_built = true;
  }
};

inline void DelayTest(const autd::ControllerPtr& autd) {
  autd->SetSilentMode(false);

  const auto m = BurstModulation::Create();
  autd->AppendModulationSync(m);

  const auto g = ConstGain::Create();
  autd->AppendGainSync(g);

  std::vector<std::array<uint16_t, autd::NUM_TRANS_IN_UNIT>> delay;
  std::array<uint16_t, autd::NUM_TRANS_IN_UNIT> ar{};
  ar.fill(0);
  ar[17] = 1;
  delay.push_back(ar);

  autd->SetDelay(delay);
}
