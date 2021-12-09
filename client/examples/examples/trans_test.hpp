// File: trans_test.hpp
// Project: examples
// Created Date: 05/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "autd3.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void trans_test_test(autd::Controller& autd) {
  autd.silent_mode() = false;

  // Duty ratio in FPGA is (duty + duty_offset)/512 %
  // The default duty_offset is set to 1, so you should set the offset to 0 for transducers you don't want to drive

  // 0-th transducer of 0-th device offset is 1, and the others are 0
  for (const auto& dev : autd.geometry())
    for (const auto& transducer : dev) autd.delay_offset()[transducer.id()].offset = 0;
  autd.delay_offset()[0].offset = 1;
  autd.set_delay_offset();  // apply

  autd::modulation::Static m;
  autd::gain::TransducerTest g(0, 0xFF, 0x00);
  autd.send(g, m);
}
