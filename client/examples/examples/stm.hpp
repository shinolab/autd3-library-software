// File: stm.hpp
// Project: examples
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"
#include "primitive_gain.hpp"
#include "primitive_modulation.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void STMTest(autd::Controller& autd) {
  autd.silent_mode() = true;

  const auto m = autd::modulation::Static::create(255);
  autd.send(nullptr, m).unwrap();

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto point_num = 200;
  for (auto i = 0; i < point_num; i++) {
    const auto radius = 30.0;
    const auto theta = 2.0 * M_PI * static_cast<double>(i) / point_num;
    const autd::Vector3 pos(radius * cos(theta), radius * sin(theta), 0.0);
    const auto g = autd::gain::FocalPoint::create(center + pos);
    autd.stm()->add_gain(g);
  }

  autd.stm()->start(1).unwrap();  // 1 Hz
}
