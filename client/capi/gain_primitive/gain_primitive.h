// File: autd3_c_api_gain_primitive.h
// Project: gain_primitive
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cstdint>

#if WIN32
#define EXPORT_AUTD __declspec(dllexport)
#else
#define EXPORT_AUTD __attribute__((visibility("default")))
#endif

#define VOID_PTR void*

extern "C" {
EXPORT_AUTD void AUTDFocalPointGain(VOID_PTR* gain, float x, float y, float z, uint8_t duty);
EXPORT_AUTD void AUTDBesselBeamGain(VOID_PTR* gain, float x, float y, float z, float n_x, float n_y, float n_z, float theta_z, uint8_t duty);
EXPORT_AUTD void AUTDPlaneWaveGain(VOID_PTR* gain, float n_x, float n_y, float n_z, uint8_t duty);
EXPORT_AUTD void AUTDCustomGain(VOID_PTR* gain, const uint16_t* data, int32_t data_length);
EXPORT_AUTD void AUTDTransducerTestGain(VOID_PTR* gain, int32_t idx, uint8_t duty, uint8_t phase);
}
