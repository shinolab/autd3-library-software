// File: seq.hpp
// Project: examples
// Created Date: 14/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"
#include "primitive_modulation.hpp"
#include "primitive_sequence.hpp"

using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void SeqTest(autd::Controller& autd) {
  autd.silent_mode() = false;

  const auto m = autd::modulation::Static::create();
  autd.send(m).unwrap();

  auto seq = autd::sequence::PointSequence::create();

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto point_num = 200;
  for (auto i = 0; i < point_num; i++) {
    const auto radius = 30.0;
    const auto theta = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(point_num);
    const autd::Vector3 p(radius * std::cos(theta), radius * std::sin(theta), 0);
    seq->add_point(center + p).unwrap();
  }

  const auto actual_freq = seq->set_frequency(1);
  std::cout << "Actual frequency is " << actual_freq << " Hz\n";
  autd.send(seq).unwrap();
}
