// File: holo_gain.h
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "../base/header.h"

#ifdef __cplusplus
extern "C" {
#endif
EXPORT_AUTD void AUTDModulationRawPCM(void** mod, const char* filename, double sampling_freq, uint16_t mod_sampling_freq_div);
EXPORT_AUTD void AUTDModulationWav(void** mod, const char* filename, uint16_t mod_sampling_freq_div);
#ifdef __cplusplus
}
#endif
