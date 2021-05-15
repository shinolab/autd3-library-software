// File: wrapper_modulation.hpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

typedef struct {
  autd::ModulationPtr ptr;
} ModulationWrapper;

inline ModulationWrapper* ModulationCreate(const autd::ModulationPtr& ptr) { return new ModulationWrapper{ptr}; }
inline void ModulationDelete(ModulationWrapper* ptr) { delete ptr; }
