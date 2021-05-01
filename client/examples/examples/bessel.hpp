// File: bessel.hpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SIZE_MM, autd::PI;

inline void BesselTest(const autd::ControllerPtr& autd) {
  autd->SetSilentMode(true);

  const auto m = autd::modulation::SineModulation::Create(150);  // 150Hz AM
  autd->AppendModulationSync(m).unwrap();

  const auto center = autd::Vector3(TRANS_SIZE_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SIZE_MM * ((NUM_TRANS_Y - 1) / 2.0), 0);
  const auto g = autd::gain::BesselBeamGain::Create(center, autd::Vector3::UnitZ(), 13.0 / 180.0 * PI);
  autd->AppendGainSync(g).unwrap();
}
