// File: runner.hpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
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
#ifdef BUILD_HOLO_GAIN
#include "examples/holo.hpp"
#endif
#include "examples/seq.hpp"
#include "examples/simple.hpp"
#include "examples/stm.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::function;
using std::pair;
using std::string;
using std::vector;

constexpr auto ULTRASOUND_WAVELENGTH = 8.5;

inline int Run(autd::ControllerPtr& autd) {
  using F = function<void(autd::ControllerPtr&)>;
  vector<pair<F, string>> examples = {pair(F{SimpleTest}, "Single Focal Point Test"), pair(F{BesselTest}, "BesselBeam Test"),
                                      pair(F{STMTest}, "Spatio-Temporal Modulation Test"),
#ifdef BUILD_HOLO_GAIN
                                      pair(F{HoloTest}, "Multiple Focal Points Test"),
#endif
                                      pair(F{SeqTest}, "Point Sequence Test (Hardware STM)")};

  autd->geometry()->set_wavelength(ULTRASOUND_WAVELENGTH);

  autd->Clear().unwrap();
  autd->Synchronize().unwrap();

  auto firm_info_list = autd->firmware_info_list().unwrap();
  for (auto&& firm_info : firm_info_list) cout << firm_info << endl;

  while (true) {
    for (size_t i = 0; i < examples.size(); i++) cout << "[" << i << "]: " << examples[i].second << endl;

    cout << "[Others]: finish." << endl;

    cout << "Choose number: ";
    string in;
    size_t idx = 0;
    getline(cin, in);
    std::stringstream s(in);
    if (const auto empty = in == "\n"; !(s >> idx) || idx >= examples.size() || empty) break;

    auto fn = examples[idx].first;
    fn(autd);

    cout << "press any key to finish..." << endl;
    cin.ignore();

    cout << "finish." << endl;
    autd->FinishSTModulation().unwrap();
    autd->Stop().unwrap();
  }

  autd->Clear().unwrap();
  autd->Close().unwrap();

  return 0;
}
