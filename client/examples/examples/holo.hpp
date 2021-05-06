// File: holo.hpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 06/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <string>

#include "autd3.hpp"
#include "gain/holo.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SIZE_MM;
using autd::gain::holo::Eigen3Backend;

inline autd::GainPtr SelectOpt(const std::vector<autd::Vector3> foci, const std::vector<autd::Float> amps) {
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

  switch (idx) {
    case 0:
      return autd::gain::holo::HoloGainSDP<Eigen3Backend>::Create(foci, amps);
    case 1:
      return autd::gain::holo::HoloGainEVD<Eigen3Backend>::Create(foci, amps);
    case 2:
      return autd::gain::holo::HoloGainGS<Eigen3Backend>::Create(foci, amps);
    case 3:
      return autd::gain::holo::HoloGainGSPAT<Eigen3Backend>::Create(foci, amps);
    case 4:
      return autd::gain::holo::HoloGainNaive<Eigen3Backend>::Create(foci, amps);
    case 5:
      return autd::gain::holo::HoloGainLM<Eigen3Backend>::Create(foci, amps);
    default:
      return autd::gain::holo::HoloGainSDP<Eigen3Backend>::Create(foci, amps);
  }
}

inline void HoloTest(const autd::ControllerPtr& autd) {
  autd->SetSilentMode(true);

  const auto m = autd::modulation::SineModulation::Create(150);  // 150Hz AM
  autd->AppendModulationSync(m).unwrap();

  const auto center = autd::Vector3(TRANS_SIZE_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SIZE_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const std::vector<autd::Vector3> foci = {center - autd::Vector3::UnitX() * 30.0, center + autd::Vector3::UnitX() * 30.0};
  const std::vector<autd::Float> amps = {1, 1};

  const auto g = SelectOpt(foci, amps);
  autd->AppendGainSync(g).unwrap();
}
