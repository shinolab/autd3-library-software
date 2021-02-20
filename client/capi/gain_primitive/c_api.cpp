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

#include "../base/wrapper_gain.hpp"
#include "./autd3_c_api_gain_primitive.h"
#include "primitive_gain.hpp"

void AUTDFocalPointGain(VOID_PTR* gain, const float x, const float y, const float z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::FocalPointGain::Create(autd::Vector3(x, y, z), duty));
  *gain = g;
}

void AUTDBesselBeamGain(VOID_PTR* gain, const float x, const float y, const float z, const float n_x, const float n_y, const float n_z,
                        const float theta_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::BesselBeamGain::Create(autd::Vector3(x, y, z), autd::Vector3(n_x, n_y, n_z), theta_z, duty));
  *gain = g;
}
void AUTDPlaneWaveGain(VOID_PTR* gain, const float n_x, const float n_y, const float n_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::PlaneWaveGain::Create(autd::Vector3(n_x, n_y, n_z), duty));
  *gain = g;
}
void AUTDCustomGain(VOID_PTR* gain, const uint16_t* data, const int32_t data_length) {
  auto* g = GainCreate(autd::gain::CustomGain::Create(data, data_length));
  *gain = g;
}

void AUTDTransducerTestGain(VOID_PTR* gain, const int32_t idx, const uint8_t duty, const uint8_t phase) {
  auto* g = GainCreate(autd::gain::TransducerTestGain::Create(idx, duty, phase));
  *gain = g;
}
