// File: stm.hpp
// Project: examples
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 05/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <autd3.hpp>

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void stm_test(autd::ControllerPtr& autd) {
  autd->silent_mode() = true;

  const auto m = autd::modulation::Static::create();
  autd->send(nullptr, m);

  auto stm = autd->stm();

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto point_num = 100;
  for (auto i = 0; i < point_num; i++) {
    const auto radius = 20.0;
    const auto theta = 2.0 * M_PI * static_cast<double>(i) / point_num;
    const autd::Vector3 pos(radius * cos(theta), radius * sin(theta), 0.0);
    const auto g = autd::gain::FocalPoint::create(center + pos);
    stm->add_gain(g);
  }

  stm->start(0.5);  // 0.5 Hz

  std::cout << "press any key to stop..." << std::endl;
  std::cin.ignore();

  stm->stop();
  stm->finish();
}
