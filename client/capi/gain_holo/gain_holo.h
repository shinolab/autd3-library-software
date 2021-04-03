// File: autd3_c_api_gain_holo.h
// Project: gain_holo
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
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

using VOID_PTR = void*;

extern "C" {
EXPORT_AUTD void AUTDHoloGain(VOID_PTR* gain, const float* points, const float* amps, int32_t size, int32_t method, VOID_PTR params);
}
