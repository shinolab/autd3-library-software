// File: c_api.cpp
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "./from_file_modulation.h"
#include "autd3/modulation/from_file.hpp"

void AUTDModulationRawPCM(void** mod, const char* filename, const double sampling_freq, const uint16_t mod_sampling_freq_div) {
  const auto filename_ = std::string(filename);
  auto* m = new autd::modulation::RawPCM(filename_, sampling_freq, mod_sampling_freq_div);
  *mod = m;
}
void AUTDModulationWav(void** mod, const char* filename, const uint16_t mod_sampling_freq_div) {
  const auto filename_ = std::string(filename);
  auto* m = new autd::modulation::Wav(filename_, mod_sampling_freq_div);
  *mod = m;
}
