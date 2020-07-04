// File: seq.hpp
// Project: examples
// Created Date: 01/07/2020
// Author: Shun Suzuki
// -----
// Last Modified: 04/07/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "autd3.hpp"

void seq_test(autd::ControllerPtr autd) {
  autd->SetSilentMode(false);

  auto m = autd::modulation::Modulation::Create(255);
  autd->AppendModulationSync(m);

  auto center = autd::Vector3(autd::AUTD_WIDTH / 2, autd::AUTD_HEIGHT / 2, 150);
  auto radius = 30.0;
  auto point_num = 200;

  auto circum = autd::sequence::CircumSeq::Create(center, autd::Vector3::unit_z(), radius, point_num);
  auto freq = 200.0;
  // For some reasons, the frequency may differ from the specified frequency.
  // See the documentation for details.
  auto actual_freq = circum->SetFrequency(freq);
  std::cout << "Actual frequency is " << actual_freq << "." << std::endl;

  autd->AppendSequence(circum);
}
