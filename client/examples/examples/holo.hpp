// File: holo.hpp
// Project: examples
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "autd3.hpp"
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline std::unique_ptr<autd::Gain> select_opt(const std::vector<autd::Vector3>& foci, const std::vector<double>& amps) {
  std::cout << "Select Optimization Method (default is SDP)" << std::endl;

  const auto backend = EigenBackend::create();

  std::vector<std::tuple<std::string, std::unique_ptr<autd::Gain>>> opts;
  opts.emplace_back(std::make_tuple("SDP", std::make_unique<autd::gain::holo::SDP>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("EVD", std::make_unique<autd::gain::holo::EVD>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("GS", std::make_unique<autd::gain::holo::GS>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("GSPAT", std::make_unique<autd::gain::holo::GSPAT>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("Naive", std::make_unique<autd::gain::holo::Naive>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("LM", std::make_unique<autd::gain::holo::LM>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("GaussNewton (slow)", std::make_unique<autd::gain::holo::GaussNewton>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("GradientDescent", std::make_unique<autd::gain::holo::GradientDescent>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("APO", std::make_unique<autd::gain::holo::APO>(backend, foci, amps)));
  opts.emplace_back(std::make_tuple("Greedy", std::make_unique<autd::gain::holo::Greedy>(backend, foci, amps)));

  size_t i = 0;
  for (const auto& [name, _opt] : opts) std::cout << "[" << i++ << "]: " << name << std::endl;

  std::string in;
  size_t idx;
  getline(std::cin, in);
  std::stringstream s(in);
  if (const auto empty = in == "\n"; !(s >> idx) || idx >= opts.size() || empty) idx = 0;

  std::string _name;
  std::unique_ptr<autd::Gain> opt;
  std::tie(_name, opt) = std::move(opts[idx]);
  return std::move(opt);
}

inline void holo_test(autd::Controller& autd) {
  autd.silent_mode() = true;

  autd::modulation::Sine m(150);  // 150Hz AM

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const std::vector<autd::Vector3> foci = {center - autd::Vector3::UnitX() * 30.0, center + autd::Vector3::UnitX() * 30.0};
  const std::vector<double> amps = {1, 1};

  const auto g = select_opt(foci, amps);
  autd.send(*g, m);
}
