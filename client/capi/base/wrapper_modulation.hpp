// File: wrapper_modulation.hpp
// Project: base
// Created Date: 05/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/08/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3/core/modulation.hpp"

typedef struct {
  autd::core::ModulationPtr ptr;
} ModulationWrapper;

inline ModulationWrapper* ModulationCreate(const autd::core::ModulationPtr& ptr) { return new ModulationWrapper{ptr}; }
inline void ModulationDelete(const ModulationWrapper* ptr) { delete ptr; }
