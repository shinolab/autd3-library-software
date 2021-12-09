// File: simple.hpp
// Project: examples
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <autd3.hpp>

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void simple_test(autd::Controller& autd) {
  autd.silent_mode() = true;

  autd::modulation::Sine m(150);  // 150Hz AM

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  autd::gain::FocalPoint g(center);

  autd.send(g, m);
}
