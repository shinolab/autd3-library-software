// File: runner.hpp
// Project: examples
// Created Date: 03/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/07/2021
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
#include "examples/advanced.hpp"
#include "examples/bessel.hpp"
#include "examples/group.hpp"
#include "examples/seq.hpp"
#include "examples/simple.hpp"
#include "examples/stm.hpp"
#include "examples/trans_test.hpp"
#ifdef BUILD_HOLO_GAIN
#include "examples/holo.hpp"
#endif
#ifdef BUILD_FROM_FILE_MOD
#include "examples/mod_from_file.hpp"
#endif

inline int run(autd::ControllerPtr autd) {
  using F = std::function<void(autd::ControllerPtr&)>;
  std::vector<std::pair<F, std::string>> examples = {std::pair(F{simple_test}, "Single Focal Point Test"),
                                                     std::pair(F{bessel_test}, "BesselBeam Test"),
#ifdef BUILD_HOLO_GAIN
                                                     std::pair(F{holo_test}, "Holo (multiple foci) Test"),
#endif
#ifdef BUILD_FROM_FILE_MOD
                                                     std::pair(F{mod_from_file_test}, "Modulation from wave file Test"),
#endif
                                                     std::pair(F{stm_test}, "Spatio-Temporal Modulation Test"),
                                                     std::pair(F{seq_test}, "Sequence (hardware STM) Test"),
                                                     std::pair(F{advanced_test}, "Advanced Test (custom Gain, Modulation, and set output delay)"),
                                                     std::pair(F{trans_test_test}, "TransducerTest Test")};
  if (autd->geometry()->num_devices() == 2) examples.emplace_back(std::pair(F{group_test}, "Group gain Test"));

  autd->geometry()->wavelength() = 8.5;  // mm

  autd->clear();

  for (auto&& firm_info : autd->firmware_info_list()) std::cout << firm_info << std::endl;

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
    autd->stop();
    autd->clear();
  }

  autd->close();

  return 0;
}
