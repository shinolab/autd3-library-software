// File: simple.cpp
// Project: example
// Created Date: 18/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 20/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "autd3.hpp"

#include <iostream>

using namespace std;

int main() {
  auto autd = autd::Controller::Create();

  autd->Open(autd::LinkType::ETHERCAT);
  if (!autd->is_open()) return ENXIO;

  autd->geometry()->AddDevice(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));

  auto gain = autd::FocalPointGain::Create(Eigen::Vector3d(90, 70, 200));

  autd->AppendGainSync(gain);
  autd->AppendModulationSync(autd::SineModulation::Create(150));  // 150Hz AM

  std::cout << "press any key to finish..." << std::endl;
  getchar();

  std::cout << "disconnecting..." << std::endl;
  autd->Close();
  return 0;
}