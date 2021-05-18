// File: holo_gain.h
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 18/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cstdint>

#include "../base/header.h"

extern "C" {
EXPORT_AUTD void AUTDEigen3Backend(void** out);
EXPORT_AUTD void AUTDDeleteBackend(void* backend);
EXPORT_AUTD void AUTDHoloGainSDP(void** gain, void* backend, float* points, float* amps, int32_t size, float alpha, float lambda, uint64_t repeat,
                                 bool normalize);
EXPORT_AUTD void AUTDHoloGainEVD(void** gain, void* backend, float* points, float* amps, int32_t size, float gamma, bool normalize);
EXPORT_AUTD void AUTDHoloGainNaive(void** gain, void* backend, float* points, float* amps, int32_t size);
EXPORT_AUTD void AUTDHoloGainGS(void** gain, void* backend, float* points, float* amps, int32_t size, uint64_t repeat);
EXPORT_AUTD void AUTDHoloGainGSPAT(void** gain, void* backend, float* points, float* amps, int32_t size, uint64_t repeat);
EXPORT_AUTD void AUTDHoloGainLM(void** gain, void* backend, float* points, float* amps, int32_t size, float eps_1, float eps_2, float tau,
                                uint64_t k_max, float* initial, int32_t initial_size);
}
