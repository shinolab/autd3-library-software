// File: c_api.cpp
// Project: arrayfire
// Created Date: 11/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../../base/wrapper.hpp"
#include "./arrayfire_backend.h"
#include "autd3/gain/arrayfire_backend.hpp"

void AUTDArrayFireBackend(void** out) {
  auto* b = backend_create(autd::gain::holo::ArrayFireBackend::create());
  *out = b;
}
