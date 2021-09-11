// File: c_api.cpp
// Project: cuda
// Created Date: 11/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../../base/wrapper.hpp"
#include "./cuda_backend.h"
#include "autd3/gain/cuda_backend.hpp"

void AUTDCUDABackend(void** out) {
  auto* b = BackendCreate(autd::gain::holo::CUDABackend::create());
  *out = b;
}
