// File: c_api.cpp
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_modulation.hpp"
#include "./from_file_modulation.h"
#include "autd3/modulation/from_file.hpp"

void AUTDModulationRawPCM(void** mod, const char* filename, double sampling_freq, uint16_t mod_sampling_freq_div) {
  const auto filename_ = std::string(filename);
  auto* m = ModulationCreate(autd::modulation::RawPCM::create(filename_, sampling_freq, mod_sampling_freq_div));
  *mod = m;
}
void AUTDModulationWav(void** mod, const char* filename, uint16_t mod_sampling_freq_div) {
  const auto filename_ = std::string(filename);
  auto* m = ModulationCreate(autd::modulation::Wav::create(filename_, mod_sampling_freq_div));
  *mod = m;
}
