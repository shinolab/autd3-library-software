﻿// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_gain.hpp"
#include "./autd3_c_api_gain_holo.h"
#include "holo_gain.hpp"

void AUTDHoloGain(VOID_PTR* gain, const float* points, const float* amps, const int32_t size, int32_t method, VOID_PTR params) {
  std::vector<autd::Vector3> holo;
  std::vector<float> amps_;
  for (auto i = 0; i < size; i++) {
    autd::Vector3 v(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    holo.emplace_back(v);
    amps_.emplace_back(amps[i]);
  }

  const auto method_ = static_cast<autd::gain::OPT_METHOD>(method);
  auto* g = GainCreate(autd::gain::HoloGain::Create(holo, amps_, method_, params));
  *gain = g;
}
