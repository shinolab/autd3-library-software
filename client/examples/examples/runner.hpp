// File: runner.hpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include <functional>
#include <sstream>
#include <utility>
#include <vector>

#include "autd3.hpp"
#include "bessel.hpp"
#include "holo.hpp"
#include "simple.hpp"
#include "stm.hpp"

using std::cin;
using std::cout;
using std::endl;
using std::function;
using std::pair;
using std::string;
using std::vector;

int run(autd::ControllerPtr autd) {
  using F = function<void(autd::ControllerPtr autd)>;
  vector<pair<F, string>> examples = {
      pair(F{simple_test}, "Single Focal Point Test"),
      pair(F{bessel_test}, "BesselBeam Test"),
      pair(F{holo_test}, "Multiple Focal Points Test"),
      pair(F{stm_test}, "Spatio-Temporal Modulation Test"),
  };

  auto firm_info_list = autd->firmware_info_list();
  for (auto firm_info : firm_info_list) cout << firm_info << endl;

  while (true) {
    for (int i = 0; i < examples.size(); i++) {
      cout << "[" << i << "]: " << examples[i].second << endl;
    }
    cout << "[Others]: finish." << endl;

    cout << "Choose number: ";
    string in = "";
    int idx = 0;
    getline(cin, in);
    std::stringstream s(in);
    auto empty = in == "\n";
    if (!(s >> idx) || idx >= examples.size() || empty) {
      break;
    }

    auto fn = examples[idx].first;
    fn(autd);

    cout << "press any key to finish..." << endl;
    getchar();

    cout << "finish." << endl;
    autd->Stop();
  }

  autd->Close();

  return 0;
}
