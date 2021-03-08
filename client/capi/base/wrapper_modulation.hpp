// File: wrapper.hpp
// Project: capi
// Created Date: 09/06/2020
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "modulation.hpp"

typedef struct {
  autd::ModulationPtr ptr;
} ModulationWrapper;

inline ModulationWrapper* ModulationCreate(const autd::ModulationPtr& ptr) { return new ModulationWrapper{ptr}; }
inline void ModulationDelete(ModulationWrapper* ptr) { delete ptr; }
