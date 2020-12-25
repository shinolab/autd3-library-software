// File: emulator.cpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "autd3.hpp"
#include "emulator_link.hpp"
#include "runner.hpp"

using namespace std;

int main() {
  auto autd = autd::Controller::Create();
  autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

  autd->OpenWith(autd::link::EmulatorLink::Create("127.0.0.1", 50632, autd->geometry()));
  if (!autd->is_open()) return ENXIO;

  return run(autd);
}
