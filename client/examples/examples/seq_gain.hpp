// File: seq_gain.hpp
// Project: examples
// Created Date: 20/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/08/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <vector>

#include "autd3.hpp"
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;
using autd::gain::holo::Eigen3Backend;

inline void seq_gain_test(const autd::ControllerPtr& autd) {
  autd->silent_mode() = false;

  const auto m = autd::modulation::Static::create();
  autd->send(m);

  const auto seq = autd::sequence::GainSequence::create();
  const auto backend = Eigen3Backend::create();

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  constexpr auto point_num = 200;
  for (auto i = 0; i < point_num; i++) {
    constexpr auto radius = 30.0;
    const auto theta = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(point_num);
    const autd::Vector3 p(radius * std::cos(theta), radius * std::sin(theta), 0);
    std::vector<autd::Vector3> foci = {center + p, center - p};
    std::vector<double> amps = {1, 1};
    const auto g = autd::gain::holo::SDP::create(backend, foci, amps);
    // const auto g = autd::gain::FocalPoint::create(center + p);
    seq->add_gain(g);
  }

  const auto actual_freq = seq->set_frequency(1);
  std::cout << "Actual frequency is " << actual_freq << " Hz\n";
  autd->send(seq);
}
