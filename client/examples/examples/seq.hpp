// File: seq.hpp
// Project: examples
// Created Date: 01/07/2020
// Author: Shun Suzuki
// -----
// Last Modified: 01/07/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "autd3.hpp"

using autd::NUM_TRANS_X;
using autd::NUM_TRANS_Y;
using autd::TRANS_SIZE_MM;

void seq_test(autd::ControllerPtr autd) {
  auto center = autd::Vector3(TRANS_SIZE_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SIZE_MM * ((NUM_TRANS_Y - 1) / 2.0), 150);
  auto radius = 30.0;
  auto point_num = 200;

  auto circum = autd::sequence::CircumSeq::Create(center, autd::Vector3::unit_z(), radius, point_num);
  auto freq = 200.0;
  // For some reasons, the frequency may differ from the specified frequency.
  // See the documentation for details.
  auto actual_freq = circum->SetFrequency(freq);
  std::cout << "Actual frequency is " << actual_freq << "." << std::endl;

  autd->SetSilentMode(false);
  autd->AppendSequence(circum);
}
