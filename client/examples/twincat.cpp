// File: twincat.cpp
// Project: examples
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3.hpp"
#include "runner.hpp"
#include "twincat_link.hpp"

int main() {
  try {
    auto autd = autd::Controller::create();
    autd->geometry()->add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

    if (auto res = autd->open(autd::link::TwinCATLink::create()); res.is_err()) {
      std::cerr << res.unwrap_err() << std::endl;
      return ENXIO;
    }
    return run(std::move(autd));
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return ENXIO;
  }
}
