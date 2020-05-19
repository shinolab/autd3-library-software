// File: stm.hpp
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

void stm_test(autd::ControllerPtr autd) {
  auto m = autd::Modulation::Create(255);
  autd->AppendModulationSync(m);

  auto center = autd::Vector3(TRANS_SIZE_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SIZE_MM * ((NUM_TRANS_Y - 1) / 2.0), 150);
  auto radius = 30.0;
  auto point_num = 200;
  for (int i = 0; i < point_num; i++) {
    auto theta = 2 * M_PI * i / point_num;
    auto pos = center + radius * autd::Vector3(cos(theta), sin(theta), 0.0);
    auto g = autd::FocalPointGain::Create(pos);
    autd->AppendSTMGain(g);
  }

  autd->StartSTModulation(1.0);  // 1 Hz
}
