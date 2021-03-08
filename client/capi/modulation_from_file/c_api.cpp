// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <cstring>

#include "../base/wrapper_modulation.hpp"
#include "./modulation_from_file.h"
#include "modulation/from_file.hpp"

void AUTDRawPCMModulation(VOID_PTR* mod, const char* filename, const float sampling_freq) {
  auto* m = ModulationCreate(autd::modulation::RawPCMModulation::Create(std::string(filename), sampling_freq));
  *mod = m;
}
void AUTDWavModulation(VOID_PTR* mod, const char* filename) {
  auto* m = ModulationCreate(autd::modulation::WavModulation::Create(std::string(filename)));
  *mod = m;
}
