// File: soem.cpp
// Project: examples
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 18/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <iostream>

#include "autd3.hpp"
#include "runner.hpp"
#include "soem_link.hpp"

std::string GetAdapterName() {
  size_t i = 0;
  const auto adapters = autd::link::SOEMLink::EnumerateAdapters();
  for (auto&& [desc, name] : adapters) std::cout << "[" << i++ << "]: " << desc << ", " << name << std::endl;

  std::cout << "Choose number: ";
  std::string in;
  getline(std::cin, in);
  std::stringstream s(in);
  if (const auto empty = in == "\n"; !(s >> i) || i >= adapters.size() || empty) return "";

  return adapters[i].name;
}

int main() {
  try {
    autd::Controller autd;
    autd.geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
    // autd.geometry()->AddDevice(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

    // If you have already recognized the EtherCAT adapter name, you can write it directly like below.
    // auto ifname = "\\Device\\NPF_{B5B631C6-ED16-4780-9C4C-3941AE8120A6}";
    const auto ifname = GetAdapterName();
    if (auto res = autd.OpenWith(autd::link::SOEMLink::Create(ifname, autd.geometry()->num_devices())); res.is_err()) {
      std::cerr << res.unwrap_err() << std::endl;
      return ENXIO;
    }
    return Run(autd);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return ENXIO;
  }
}
