// File: holo.hpp
// Project: examples
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <string>

#include "autd3.hpp"
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;
using autd::gain::holo::Eigen3Backend;

inline autd::GainPtr select_opt(std::vector<autd::Vector3>& foci, std::vector<double>& amps) {
  std::cout << "Select Optimization Method (default is SDP)" << std::endl;
  const std::vector<std::string> opts = {"SDP", "EVD", "GS", "GS-PAT", "NAIVE", "LM", "Greedy"};
  for (size_t i = 0; i < opts.size(); i++) std::cout << "[" << i << "]: " << opts[i] << std::endl;

  std::string in;
  size_t idx;
  getline(std::cin, in);
  std::stringstream s(in);
  if (const auto empty = in == "\n"; !(s >> idx) || idx >= opts.size() || empty) idx = 0;

  const auto backend = Eigen3Backend::create();
  switch (idx) {
    case 0:
      return autd::gain::holo::HoloSDP::create(backend, foci, amps);
    case 1:
      return autd::gain::holo::HoloEVD::create(backend, foci, amps);
    case 2:
      return autd::gain::holo::HoloGS::create(backend, foci, amps);
    case 3:
      return autd::gain::holo::HoloGSPAT::create(backend, foci, amps);
    case 4:
      return autd::gain::holo::HoloNaive::create(backend, foci, amps);
    case 5:
      return autd::gain::holo::HoloLM::create(backend, foci, amps);
    case 6:
      return autd::gain::holo::HoloGreedy::create(foci, amps);
    default:
      return autd::gain::holo::HoloSDP::create(backend, foci, amps);
  }
}

inline void holo_test(autd::ControllerPtr& autd) {
  autd->silent_mode() = true;

  const auto m = autd::modulation::Sine::create(150);  // 150Hz AM

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  std::vector<autd::Vector3> foci = {center - autd::Vector3::UnitX() * 30.0, center + autd::Vector3::UnitX() * 30.0};
  std::vector<double> amps = {1, 1};

  const auto g = select_opt(foci, amps);
  autd->send(g, m);
}
