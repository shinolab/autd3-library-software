// File: wrapper_backend.hpp
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 17/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <utility>

#include "linalg_backend.hpp"

typedef struct {
  autd::gain::holo::BackendPtr ptr;
} BackendWrapper;

inline BackendWrapper* BackendCreate(autd::gain::holo::BackendPtr ptr) { return new BackendWrapper{std::move(ptr)}; }
inline void BackendDelete(BackendWrapper* ptr) { delete ptr; }
