// File: trans_test.hpp
// Project: examples
// Created Date: 05/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void trans_test_test(autd::ControllerPtr& autd) {
  autd->silent_mode() = false;

  // Duty ratio in FPGA is (duty + duty_offset)/512 %
  // The default duty_offset is set to 1, so you should set the offset to 0 for transducers you don't want to drive
  std::vector<std::array<uint8_t, autd::NUM_TRANS_IN_UNIT>> duty_offset;
  duty_offset.resize(autd->geometry()->num_devices());
  duty_offset[0][0] = 1;  // 0-th transducer of 0-th device offset is 1, and the others are 0
  autd->set_duty_offset(duty_offset);

  const auto m = autd::modulation::Static::create();
  const auto g = autd::gain::TransducerTest::create(0, 0xFF, 0x00);
  autd->send(g, m);
}
