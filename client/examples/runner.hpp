// File: runner.hpp
// Project: examples
// Created Date: 03/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 18/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "autd3.hpp"
#include "examples/bessel.hpp"
#include "examples/seq.hpp"
#include "examples/simple.hpp"
#include "examples/stm.hpp"
#ifdef BUILD_HOLO_GAIN
#include "examples/holo.hpp"
#endif

inline int Run(autd::Controller& autd) {
  using F = std::function<void(autd::Controller&)>;
  std::vector<std::pair<F, std::string>> examples = {
      std::pair(F{SimpleTest}, "Single Focal Point Test"),      std::pair(F{BesselTest}, "BesselBeam Test"),
#ifdef BUILD_HOLO_GAIN
      std::pair(F{HoloTest}, "HoloGain (multiple foci) Test"),
#endif
      std::pair(F{STMTest}, "Spatio-Temporal Modulation Test"), std::pair(F{SeqTest}, "Sequence (hardware STM) Test"),
  };

  autd.geometry()->wavelength() = 8.5;  // mm

  autd.Clear().unwrap();
  autd.Synchronize().unwrap();

  auto firm_info_list = autd.firmware_info_list().unwrap();
  for (auto&& firm_info : firm_info_list) std::cout << firm_info << std::endl;

  while (true) {
    for (size_t i = 0; i < examples.size(); i++) std::cout << "[" << i << "]: " << examples[i].second << std::endl;
    std::cout << "[Others]: finish." << std::endl;

    std::cout << "Choose number: ";
    std::string in;
    size_t idx;
    getline(std::cin, in);
    std::stringstream s(in);
    if (const auto empty = in == "\n"; !(s >> idx) || idx >= examples.size() || empty) break;

    examples[idx].first(autd);

    std::cout << "press any key to finish..." << std::endl;
    std::cin.ignore();

    std::cout << "finish." << std::endl;
    autd.Stop().unwrap();
  }

  autd.Clear().unwrap();
  autd.Close().unwrap();

  return 0;
}
