// File: bessel.hpp
// Project: examples
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 18/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"
#include "primitive_gain.hpp"
#include "primitive_modulation.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void BesselTest(autd::Controller& autd) {
  autd.silent_mode() = true;

  const auto m = autd::modulation::Sine::Create(150);  // 150Hz AM

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto g = autd::gain::BesselBeam::Create(center, autd::Vector3::UnitZ(), 13.0 / 180.0 * M_PI);
  autd.Send(g, m).unwrap();
}
