// File: group.hpp
// Project: examples
// Created Date: 23/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "autd3.hpp"
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;
using autd::gain::holo::EigenBackend;

inline void group_test(autd::Controller& autd) {
  autd.silent_mode() = true;

  autd::modulation::Sine m(150);  // 150Hz AM

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto g1 = std::make_shared<autd::gain::FocalPoint>(center);

  const autd::Vector3 apex(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 0);
  const auto g2 = std::make_shared<autd::gain::BesselBeam>(apex, autd::Vector3::UnitZ(), 13.0 / 180.0 * M_PI);

  autd::gain::Grouped g;
  g.add(0, g1);
  g.add(1, g2);

  autd.send(g, m);
}
