// File: runner.hpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "autd3.hpp"
#include "bessel.hpp"
#include "holo.hpp"
#include "seq.hpp"
#include "simple.hpp"
#include "stm.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::function;
using std::pair;
using std::string;
using std::vector;

inline int Run(autd::ControllerPtr& autd) {
  using F = function<void(autd::ControllerPtr)>;
  vector<pair<F, string>> examples = {
      pair(F{SimpleTest}, "Single Focal Point Test"),         pair(F{BesselTest}, "BesselBeam Test"),
      pair(F{HoloTest}, "Multiple Focal Points Test"),        pair(F{STMTest}, "Spatio-Temporal Modulation Test"),
      pair(F{SeqTest}, "Point Sequence Test (Hardware STM)"),
  };

  autd->Clear();

  auto config = autd::Configuration::GetDefaultConfiguration();
  config.set_mod_buf_size(autd::MOD_BUF_SIZE::BUF_4000);
  config.set_mod_sampling_freq(autd::MOD_SAMPLING_FREQ::SMPL_4_KHZ);
  autd->Calibrate(config);

  auto firm_info_list = autd->firmware_info_list();
  for (auto& firm_info : firm_info_list) cout << firm_info << endl;

  while (true) {
    for (size_t i = 0; i < examples.size(); i++) {
      cout << "[" << i << "]: " << examples[i].second << endl;
    }
    cout << "[Others]: finish." << endl;

    cout << "Choose number: ";
    string in;
    size_t idx = 0;
    getline(cin, in);
    std::stringstream s(in);
    const auto empty = in == "\n";
    if (!(s >> idx) || idx >= examples.size() || empty) {
      break;
    }

    auto fn = examples[idx].first;
    fn(autd);

    cout << "press any key to finish..." << endl;
    auto _ = getchar();

    cout << "finish." << endl;
    autd->FinishSTModulation();
    autd->Stop();
  }

  autd->Clear();
  autd->Close();

  return 0;
}
