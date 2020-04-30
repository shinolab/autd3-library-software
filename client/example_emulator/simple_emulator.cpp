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
  const auto AUTD_WIDTH = autd::TRANS_SIZE_MM * autd::NUM_TRANS_X;
  const auto AUTD_HEIGHT = autd::TRANS_SIZE_MM * autd::NUM_TRANS_Y;
  auto autd = autd::Controller::Create();

  autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
  //autd->geometry()->AddDevice(autd::Vector3(AUTD_WIDTH, 0, 0),
  //                            autd::Vector3(0, -M_PI_2, 0));
  //autd->geometry()->AddDevice(autd::Vector3(AUTD_WIDTH, 0, AUTD_WIDTH),
  //                            autd::Vector3(0, M_PI, 0));
  //autd->geometry()->AddDevice(autd::Vector3(0, 0, AUTD_WIDTH),
  //                            autd::Vector3(0, M_PI_2, 0));

  autd->Open(autd::LinkType::EMULATOR, "127.0.0.1:50632");
  if (!autd->is_open()) return ENXIO;

  auto center = autd::Vector3(
      autd::TRANS_SIZE_MM * (autd::NUM_TRANS_X - 1) / 2.0,
      autd::TRANS_SIZE_MM * (autd::NUM_TRANS_Y - 1) / 2.0, AUTD_WIDTH / 2.0);
  auto gain = autd::FocalPointGain::Create(center);
  autd->AppendGainSync(gain);

  std::cout << "press any key to finish..." << std::endl;
  getchar();

  std::cout << "disconnecting..." << std::endl;
  autd->Close();
  return 0;
}