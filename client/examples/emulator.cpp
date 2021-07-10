// File: emulator.cpp
// Project: examples
// Created Date: 05/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <autd3/link/emulator.hpp>

#include "autd3.hpp"
#include "runner.hpp"

int main() {
  try {
    // Please see `dist/emulator/README.md`
    auto autd = autd::Controller::create();
    autd->geometry()->add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

    autd->open(autd::link::Emulator::create(50632, autd->geometry()));
    return run(std::move(autd));
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }
}
