// File: stm.hpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SIZE_MM, autd::PI;

inline void STMTest(const autd::ControllerPtr& autd) {
  autd->SetSilentMode(true);

  const auto m = autd::modulation::Modulation::Create(255);
  autd->AppendModulationSync(m).unwrap();

  const auto center = autd::Vector3(TRANS_SIZE_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SIZE_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto point_num = 200;
  for (auto i = 0; i < point_num; i++) {
    const auto radius = 30.0;
    const auto theta = 2.0 * PI * static_cast<autd::Float>(i) / point_num;
    const auto pos = autd::Vector3(radius * cos(theta), radius * sin(theta), 0.0);
    const auto g = autd::gain::FocalPointGain::Create(center + pos);
    autd->AddSTMGain(g);
  }

  autd->StartSTModulation(1).unwrap();  // 1 Hz
}
