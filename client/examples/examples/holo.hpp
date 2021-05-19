// File: holo.hpp
// Project: examples
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <string>

#include "autd3.hpp"
#include "eigen_backend.hpp"
#include "holo_gain.hpp"
#include "primitive_modulation.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;
using autd::gain::holo::Eigen3Backend;

inline autd::GainPtr select_opt(std::vector<autd::Vector3>& foci, std::vector<double>& amps) {
  std::cout << "Select Optimization Method (default is SDP)" << std::endl;
  const std::vector<std::string> opts = {"SDP", "EVD", "GS", "GS-PAT", "NAIVE", "LM"};
  for (size_t i = 0; i < opts.size(); i++) std::cout << "[" << i << "]: " << opts[i] << std::endl;

  std::string in;
  size_t idx;
  getline(std::cin, in);
  std::stringstream s(in);
  if (const auto empty = in == "\n"; !(s >> idx) || idx >= opts.size() || empty) idx = 0;

  const auto backend = Eigen3Backend::create();
  switch (idx) {
    case 0:
      return autd::gain::holo::HoloGainSDP::create(backend, foci, amps);
    case 1:
      return autd::gain::holo::HoloGainEVD::create(backend, foci, amps);
    case 2:
      return autd::gain::holo::HoloGainGS::create(backend, foci, amps);
    case 3:
      return autd::gain::holo::HoloGainGSPAT::create(backend, foci, amps);
    case 4:
      return autd::gain::holo::HoloGainNaive::create(backend, foci, amps);
    case 5:
      return autd::gain::holo::HoloGainLM::create(backend, foci, amps);
    default:
      return autd::gain::holo::HoloGainSDP::create(backend, foci, amps);
  }
}

inline void holo_test(autd::Controller& autd) {
  autd.silent_mode() = true;

  const auto m = autd::modulation::Sine::create(150);  // 150Hz AM

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  std::vector<autd::Vector3> foci = {center - autd::Vector3::UnitX() * 30.0, center + autd::Vector3::UnitX() * 30.0};
  std::vector<double> amps = {1, 1};

  const auto g = select_opt(foci, amps);
  autd.send(g, m).unwrap();
}
