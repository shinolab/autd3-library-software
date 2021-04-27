// File: emulator.cpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "link/emulator.hpp"

#include "autd3.hpp"
#include "runner.hpp"

using namespace std;

int main() {
  try {
    auto autd = autd::Controller::Create();
    autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

    auto res = autd->OpenWith(autd::link::EmulatorLink::Create("127.0.0.1", 50632, autd->geometry()));
    if (res.is_err()) {
      std::cerr << res.unwrap_err() << std::endl;
      return ENXIO;
    }
    if (!res.unwrap()) {
      std::cerr << "Failed to open." << std::endl;
      return ENXIO;
    }

    return Run(autd);
  } catch (exception& e) {
    std::cerr << e.what() << std::endl;
    return ENXIO;
  }
}
