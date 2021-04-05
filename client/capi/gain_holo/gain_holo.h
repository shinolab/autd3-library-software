﻿// File: autd3_c_api_gain_holo.h
// Project: gain_holo
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd_types.hpp"

#if WIN32
#define EXPORT_AUTD __declspec(dllexport)
#else
#define EXPORT_AUTD __attribute__((visibility("default")))
#endif

extern "C" {
EXPORT_AUTD void AUTDHoloGain(void** gain, autd::Float* points, autd::Float* amps, int32_t size, int32_t method, void* params);
}
