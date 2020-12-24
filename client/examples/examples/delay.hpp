// File: delay.hpp
// Project: examples
// Created Date: 24/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 24/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "autd3.hpp"

using autd::NUM_TRANS_X;
using autd::NUM_TRANS_Y;
using autd::TRANS_SIZE_MM;

class BurstModulation : public autd::modulation::Modulation {
 public:
  static autd::ModulationPtr Create() {
    auto mod = std::make_shared<BurstModulation>();
    return mod;
  }

  void Build(autd::Configuration config) override {
    const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());
    this->buffer.resize(mod_buf_size, 0);
    this->buffer.at(0) = 0xFF;
  }
};

class ConstGain : public autd::gain::Gain {
 public:
  static autd::GainPtr Create() { return std::make_shared<ConstGain>(); }
  void Build() override {
    if (this->built()) return;

    auto geometry = this->geometry();

    autd::gain::CheckAndInit(geometry, &this->_data);

    const auto ntrans = geometry->numTransducers();
    for (size_t i = 0; i < ntrans; i++) {
      this->_data[geometry->deviceIdxForTransIdx(i)].at(i % autd::NUM_TRANS_IN_UNIT) = 0xFF00;
    }

    this->_built = true;
  }
};

void delay_test(autd::ControllerPtr autd) {
  autd->SetSilentMode(false);

  auto m = BurstModulation::Create();
  autd->AppendModulationSync(m);

  auto g = ConstGain::Create();
  autd->AppendGainSync(g);

  std::vector<std::array<uint16_t, autd::NUM_TRANS_IN_UNIT>> delay;
  std::array<uint16_t, autd::NUM_TRANS_IN_UNIT> ar;
  ar.fill(0);
  ar[17] = 1;
  delay.push_back(ar);

  autd->SetDelay(delay);
}
