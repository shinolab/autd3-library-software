// File: holo.hpp
// Project: examples
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/05/2021
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

inline autd::GainPtr SelectOpt(std::vector<autd::Vector3>& foci, std::vector<double>& amps) {
  std::cout << "Select Optimization Method (default is SDP)" << std::endl;
  const std::vector<std::string> opts = {"SDP", "EVD", "GS", "GS-PAT", "NAIVE", "LM"};
  for (size_t i = 0; i < opts.size(); i++) {
    const auto& name = opts[i];
    std::cout << "[" << i << "]: " << name << std::endl;
  }

  std::string in;
  size_t idx = 0;
  getline(std::cin, in);
  std::stringstream s(in);
  if (const auto empty = in == "\n"; !(s >> idx) || idx >= opts.size() || empty) {
    idx = 0;
  }

  const auto backend = Eigen3Backend::Create();
  switch (idx) {
    case 0:
      return autd::gain::holo::HoloGainSDP::Create(backend, foci, amps);
    case 1:
      return autd::gain::holo::HoloGainEVD::Create(backend, foci, amps);
    case 2:
      return autd::gain::holo::HoloGainGS::Create(backend, foci, amps);
    case 3:
      return autd::gain::holo::HoloGainGSPAT::Create(backend, foci, amps);
    case 4:
      return autd::gain::holo::HoloGainNaive::Create(backend, foci, amps);
    case 5:
      return autd::gain::holo::HoloGainLM::Create(backend, foci, amps);
    default:
      return autd::gain::holo::HoloGainSDP::Create(backend, foci, amps);
  }
}

inline void HoloTest(autd::Controller& autd) {
  autd.silent_mode() = true;

  const auto m = autd::modulation::Sine::Create(150);  // 150Hz AM

  const auto center = autd::Vector3(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  std::vector<autd::Vector3> foci = {center - autd::Vector3::UnitX() * 30.0, center + autd::Vector3::UnitX() * 30.0};
  std::vector<double> amps = {1, 1};

  const auto g = SelectOpt(foci, amps);
  autd.Send(g, m).unwrap();
}
