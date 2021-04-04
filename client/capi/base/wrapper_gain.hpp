// File: wrapper.hpp
// Project: capi
// Created Date: 09/06/2020
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "gain.hpp"

typedef struct {
  autd::GainPtr ptr;
} GainWrapper;

inline GainWrapper* GainCreate(const autd::GainPtr& ptr) { return new GainWrapper{ptr}; }
inline void GainDelete(GainWrapper* ptr) { delete ptr; }
