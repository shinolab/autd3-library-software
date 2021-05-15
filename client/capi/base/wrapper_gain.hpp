// File: wrapper_gain.hpp
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
  autd::GainPtr ptr;
} GainWrapper;

inline GainWrapper* GainCreate(const autd::GainPtr& ptr) { return new GainWrapper{ptr}; }
inline void GainDelete(GainWrapper* ptr) { delete ptr; }
