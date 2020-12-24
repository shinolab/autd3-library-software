// File: debug.cpp
// Project: examples
// Created Date: 22/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include <iostream>

#include "autd3.hpp"
#include "debug_link.hpp"
#include "runner.hpp"

using namespace std;

int main() {
  auto autd = autd::Controller::Create();
  autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

  auto link = autd::link::DebugLink::Create(cout);

  autd->OpenWith(link);
  if (!autd->is_open()) return ENXIO;

  return run(autd);
}
