// File: cuda_backend.h
// Project: cuda
// Created Date: 11/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "../../base/header.h"

extern "C" {
EXPORT_AUTD void AUTDCUDABackend(void** out);
}
