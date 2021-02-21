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
EXPORT_AUTD void AUTDCustomModulation(VOID_PTR* mod, const uint8_t* buf, uint32_t size);
EXPORT_AUTD void AUTDSawModulation(VOID_PTR* mod, int32_t freq);
EXPORT_AUTD void AUTDSineModulation(VOID_PTR* mod, int32_t freq, float amp, float offset);
EXPORT_AUTD void AUTDSquareModulation(VOID_PTR* mod, int32_t freq, uint8_t low, uint8_t high);
}
