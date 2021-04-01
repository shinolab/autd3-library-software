// File: twincat.cpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 01/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "link/twincat.hpp"

#include "autd3.hpp"
#include "runner.hpp"

using namespace std;

int main() {
  auto autd = autd::Controller::Create();
  autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

  try {
    autd->OpenWith(autd::link::LocalTwinCATLink::Create());
    if (!autd->is_open()) return ENXIO;
  } catch (exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return Run(autd);
}
