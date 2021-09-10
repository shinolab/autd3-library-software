// File: holo_gain.h
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cstdint>

#include "../base/header.h"

extern "C" {
EXPORT_AUTD void AUTDEigen3Backend(void** out);
EXPORT_AUTD void AUTDDeleteBackend(const void* backend);
EXPORT_AUTD void AUTDGainHoloSDP(void** gain, const void* backend, const double* points, const double* amps, int32_t size, double alpha,
                                 double lambda, uint64_t repeat, bool normalize);
EXPORT_AUTD void AUTDGainHoloEVD(void** gain, const void* backend, const double* points, const double* amps, int32_t size, double gamma,
                                 bool normalize);
EXPORT_AUTD void AUTDGainHoloNaive(void** gain, const void* backend, const double* points, const double* amps, int32_t size);
EXPORT_AUTD void AUTDGainHoloGS(void** gain, const void* backend, const double* points, const double* amps, int32_t size, uint64_t repeat);
EXPORT_AUTD void AUTDGainHoloGSPAT(void** gain, const void* backend, const double* points, const double* amps, int32_t size, uint64_t repeat);
EXPORT_AUTD void AUTDGainHoloLM(void** gain, const void* backend, const double* points, const double* amps, int32_t size, double eps_1, double eps_2,
                                double tau, uint64_t k_max, const double* initial, int32_t initial_size);
EXPORT_AUTD void AUTDGainHoloGaussNewton(void** gain, const void* backend, const double* points, const double* amps, int32_t size, double eps_1,
                                         double eps_2, uint64_t k_max, const double* initial, int32_t initial_size);
EXPORT_AUTD void AUTDGainHoloGradientDescent(void** gain, const void* backend, const double* points, const double* amps, int32_t size, double eps,
                                             double step, uint64_t k_max, const double* initial, int32_t initial_size);
EXPORT_AUTD void AUTDGainHoloAPO(void** gain, const void* backend, const double* points, const double* amps, int32_t size, double eps, double lambda,
                                 uint64_t k_max);
EXPORT_AUTD void AUTDGainHoloGreedy(void** gain, const void* backend, const double* points, const double* amps, int32_t size, int32_t phase_div);
}
