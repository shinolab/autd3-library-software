// File: wrapper_link.hpp
// Project: base
// Created Date: 18/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <utility>

#include "core/link.hpp"

typedef struct {
  autd::core::LinkPtr ptr;
} LinkWrapper;

inline LinkWrapper* LinkCreate(autd::core::LinkPtr ptr) { return new LinkWrapper{std::move(ptr)}; }
inline void LinkDelete(LinkWrapper* ptr) { delete ptr; }
