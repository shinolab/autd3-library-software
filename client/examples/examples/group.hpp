// File: group.hpp
// Project: examples
// Created Date: 23/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 03/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"
#include "eigen_backend.hpp"
#include "holo_gain.hpp"
#include "primitive_gain.hpp"
#include "primitive_modulation.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void group_test(autd::ControllerPtr& autd) {
  autd->silent_mode() = true;

  const auto m = autd::modulation::Sine::create(150);  // 150Hz AM

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto g1 = autd::gain::FocalPoint::create(center);

  std::vector<autd::Vector3> foci = {center - autd::Vector3::UnitX() * 30.0, center + autd::Vector3::UnitX() * 30.0};
  std::vector<double> amps = {1, 1};
  const auto backend = autd::gain::holo::Eigen3Backend::create();
  const auto g2 = autd::gain::holo::HoloGainSDP::create(backend, foci, amps);

  const auto g = autd::gain::Grouped::create();
  g->add(0, g1);
  g->add(1, g2);

  autd->send(g, m).unwrap();
}
