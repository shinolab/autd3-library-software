// File: twincat.cpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "link/twincat.hpp"

#include "autd3.hpp"
#include "runner.hpp"

using namespace std;

int main() {
  try {
    auto autd = autd::Controller::Create();
    autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

    if (auto res = autd->OpenWith(autd::link::LocalTwinCATLink::Create()); res.is_err()) {
      std::cerr << res.unwrap_err() << std::endl;
      return ENXIO;
    }
    return Run(autd);
  } catch (exception& e) {
    std::cerr << e.what() << std::endl;
    return ENXIO;
  }
}
