// File: simple.hpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"
#include "primitive_gain.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SIZE_MM;

inline void SimpleTest(const autd::ControllerPtr& autd) {
  autd->SetSilentMode(true);

  const auto m = autd::modulation::SineModulation::Create(150);  // 150Hz AM
  autd->AppendModulationSync(m);

  const auto center = autd::Vector3(TRANS_SIZE_MM * ((NUM_TRANS_X - 1) / 2.0f), TRANS_SIZE_MM * ((NUM_TRANS_Y - 1) / 2.0f), 150.0f);
  const auto g = autd::gain::FocalPointGain::Create(center);
  autd->AppendGainSync(g);
}
