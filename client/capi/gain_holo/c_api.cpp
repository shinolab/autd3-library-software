// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 01/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_gain.hpp"
#include "./gain_holo.h"
#include "gain/holo.hpp"

void AUTDHoloGain(void** gain, float* points, float* amps, const int32_t size, int32_t method, void* params) {
  std::vector<autd::Vector3> holo;
  std::vector<autd::Float> amps_;
  for (auto i = 0; i < size; i++) {
    const auto x = static_cast<autd::Float>(points[3 * i]);
    const auto y = static_cast<autd::Float>(points[3 * i + 1]);
    const auto z = static_cast<autd::Float>(points[3 * i + 2]);
    holo.emplace_back(autd::Vector3(x, y, z));
    amps_.emplace_back(static_cast<autd::Float>(amps[i]));
  }
  const auto method_ = static_cast<autd::gain::holo::OPT_METHOD>(method);
  auto* g = GainCreate(autd::gain::holo::HoloGain<autd::gain::holo::Eigen3Backend>::Create(holo, amps_, method_, params));
  *gain = g;
}
