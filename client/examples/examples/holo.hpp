// File: holo.hpp
// Project: examples
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "autd3.hpp"
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline autd::GainPtr select_opt(const std::vector<autd::Vector3>& foci, const std::vector<double>& amps) {
  using Backend = autd::gain::holo::EigenBackend;

  std::cout << "Select Optimization Method (default is SDP)" << std::endl;

  std::vector<std::tuple<std::string, autd::GainPtr>> opts;
  opts.emplace_back(std::make_tuple("SDP", autd::gain::holo::SDP<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("EVD", autd::gain::holo::EVD<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("GS", autd::gain::holo::GS<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("GSPAT", autd::gain::holo::GSPAT<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("Naive", autd::gain::holo::Naive<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("LM", autd::gain::holo::LM<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("GaussNewton (slow)", autd::gain::holo::GaussNewton<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("GradientDescent", autd::gain::holo::GradientDescent<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("APO", autd::gain::holo::APO<Backend>::create(foci, amps)));
  opts.emplace_back(std::make_tuple("Greedy", autd::gain::holo::Greedy::create(foci, amps)));

  size_t i = 0;
  for (const auto& [name, _opt] : opts) std::cout << "[" << i++ << "]: " << name << std::endl;

  std::string in;
  size_t idx;
  getline(std::cin, in);
  std::stringstream s(in);
  if (const auto empty = in == "\n"; !(s >> idx) || idx >= opts.size() || empty) idx = 0;

  const auto [_name, opt] = opts[idx];
  return opt;
}

inline void holo_test(const autd::ControllerPtr& autd) {
  autd->silent_mode() = true;

  const auto m = autd::modulation::Sine::create(150);  // 150Hz AM

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const std::vector<autd::Vector3> foci = {center - autd::Vector3::UnitX() * 30.0, center + autd::Vector3::UnitX() * 30.0};
  const std::vector<double> amps = {1, 1};

  const auto g = select_opt(foci, amps);
  autd->send(g, m);
}
