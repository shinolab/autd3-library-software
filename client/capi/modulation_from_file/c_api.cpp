// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 02/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_modulation.hpp"
#include "./modulation_from_file.h"
#include "modulation/from_file.hpp"

bool AUTDRawPCMModulation(void** mod, const char* filename, const float sampling_freq, char* error) {
  auto res = autd::modulation::RawPCMModulation::Create(std::string(filename), static_cast<autd::Float>(sampling_freq));
  if (res.is_err()) {
    const auto e = res.unwrap_err();
    std::char_traits<char>::copy(error, e.c_str(), e.size() + 1);
    return false;
  }
  auto* m = ModulationCreate(res.unwrap());
  *mod = m;
  return true;
}
bool AUTDWavModulation(void** mod, const char* filename, char* error) {
  auto res = autd::modulation::WavModulation::Create(std::string(filename));
  if (res.is_err()) {
    const auto e = res.unwrap_err();
    std::char_traits<char>::copy(error, e.c_str(), e.size() + 1);
    return false;
  }
  auto* m = ModulationCreate(res.unwrap());
  *mod = m;
  return true;
}
