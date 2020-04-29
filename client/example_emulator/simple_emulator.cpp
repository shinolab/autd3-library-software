// File: simple_emulator.cpp
// Project: example_emulator
// Created Date: 29/04/2020
// Author: Shun Suzuki
// -----
// Last Modified: 29/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include <iostream>

#include "autd3.hpp"

using namespace std;

int main() {
  auto autd = autd::Controller::Create();

  autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

  autd->Open(autd::LinkType::EMULATOR, "127.0.0.1:50632");
  if (!autd->is_open()) return ENXIO;

  auto gain = autd::FocalPointGain::Create(autd::Vector3(10.18 * 8.5, 10.18 * 6.5, 150.0));
  autd->AppendGainSync(gain);

  std::cout << "press any key to finish..." << std::endl;
  getchar();

  std::cout << "disconnecting..." << std::endl;
  autd->Close();
  return 0;
}