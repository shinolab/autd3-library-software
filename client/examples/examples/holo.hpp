// File: holo.hpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "autd3.hpp"

using autd::NUM_TRANS_X;
using autd::NUM_TRANS_Y;
using autd::TRANS_SIZE_MM;

void holo_test(autd::ControllerPtr autd) {
  auto m = autd::SineModulation::Create(150);  // 150Hz AM
  autd->AppendModulationSync(m);

  auto center = autd::Vector3(TRANS_SIZE_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SIZE_MM * ((NUM_TRANS_Y - 1) / 2.0), 150);
  auto foci = {
      center - autd::Vector3::unit_x() * 20.0,
      center + autd::Vector3::unit_x() * 20.0,
  };
  auto amps = {1.0, 1.0};
  auto g = autd::HoloGain::Create(foci, amps);
  autd->AppendGainSync(g);
}
