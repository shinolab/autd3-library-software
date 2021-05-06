// File: autd3_c_api_gain_holo.h
// Project: gain_holo
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/05/2021
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

extern "C" {
EXPORT_AUTD void AUTDHoloGainSDP(void** gain, float* points, float* amps, int32_t size, float alpha, float lambda, uint64_t repeat, bool normalize);
EXPORT_AUTD void AUTDHoloGainEVD(void** gain, float* points, float* amps, int32_t size, float gamma, bool normalize);
EXPORT_AUTD void AUTDHoloGainNaive(void** gain, float* points, float* amps, int32_t size);
EXPORT_AUTD void AUTDHoloGainGS(void** gain, float* points, float* amps, int32_t size, uint64_t repeat);
EXPORT_AUTD void AUTDHoloGainGSPAT(void** gain, float* points, float* amps, int32_t size, uint64_t repeat);
EXPORT_AUTD void AUTDHoloGainLM(void** gain, float* points, float* amps, int32_t size, float eps_1, float eps_2, float tau, uint64_t k_max,
                                float* initial, int32_t initial_size);
}
