// File: autd3_c_api_gain_primitive.h
// Project: gain_primitive
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#if WIN32
#define EXPORT_AUTD __declspec(dllexport)
#else
#define EXPORT_AUTD __attribute__((visibility("default")))
#endif

using VOID_PTR = void*;

extern "C" {
EXPORT_AUTD void AUTDRawPCMModulation(VOID_PTR* mod, const char* filename, float sampling_freq);
EXPORT_AUTD void AUTDWavModulation(VOID_PTR* mod, const char* filename);
}
