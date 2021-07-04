// File: twincat.cpp
// Project: examples
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 04/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <autd3/link/twincat.hpp>

#include "autd3.hpp"
#include "runner.hpp"

int main() {
  try {
    auto autd = autd::Controller::create();
    autd->geometry()->add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

    autd->open(autd::link::TwinCAT::create());
    return run(std::move(autd));
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }
}
