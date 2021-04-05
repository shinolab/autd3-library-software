// File: soem.cpp
// Project: examples
// Created Date: 19/05/2020
// Author: Shun Suzuki
// -----
// Last Modified: 05/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "link/soem.hpp"

#include <iostream>

#include "autd3.hpp"
#include "runner.hpp"

using namespace std;

string GetAdapterName() {
  size_t size;
  auto adapters = autd::link::SOEMLink::EnumerateAdapters(&size);
  for (size_t i = 0; i < size; i++) {
    auto& [fst, snd] = adapters[i];
    cout << "[" << i << "]: " << fst << ", " << snd << endl;
  }

  int index;
  cout << "Choose number: ";
  cin >> index;
  cin.ignore();

  return adapters[index].second;
}

int main() {
  try {
    auto autd = autd::Controller::Create();
    autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd->geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

    // If you have already recognized the EtherCAT adapter name, you can write it directly like below.
    // auto ifname = "\\Device\\NPF_{B5B631C6-ED16-4780-9C4C-3941AE8120A6}";

    const auto ifname = GetAdapterName();
    auto res = autd->OpenWith(autd::link::SOEMLink::Create(ifname, autd->geometry()->num_devices()));
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
