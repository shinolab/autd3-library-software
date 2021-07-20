// File: seq_gain.hpp
// Project: examples
// Created Date: 20/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <autd3.hpp>
#include <autd3/gain/eigen_backend.hpp>
#include <autd3/gain/holo.hpp>
#include <iostream>
#include <string>

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;
using autd::gain::holo::Eigen3Backend;

inline void seq_gain_test(autd::ControllerPtr& autd) {
  autd->silent_mode() = false;

  const auto m = autd::modulation::Static::create();
  autd->send(m);

  auto seq = autd::sequence::GainSequence::create();
  const auto backend = Eigen3Backend::create();

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);

  std::vector<autd::Vector3> foci1 = {center - autd::Vector3::UnitX() * 30.0, center + autd::Vector3::UnitX() * 30.0};
  std::vector<double> amps1 = {1, 1};
  auto g1 = autd::gain::holo::HoloSDP::create(backend, foci1, amps1);

  std::vector<autd::Vector3> foci2 = {center - autd::Vector3::UnitY() * 30.0, center + autd::Vector3::UnitY() * 30.0};
  std::vector<double> amps2 = {1, 1};
  auto g2 = autd::gain::holo::HoloSDP::create(backend, foci2, amps2);

  seq->add_gain(g1);
  // seq->add_gain(g2);

  const auto actual_freq = seq->set_frequency(1);
  std::cout << "Actual frequency is " << actual_freq << " Hz\n";
  std::cout << "Actual frequency is " << seq->sampling_frequency_division() << " \n";
  autd->send(seq);
}
