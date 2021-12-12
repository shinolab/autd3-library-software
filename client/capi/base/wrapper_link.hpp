// File: wrapper_link.hpp
// Project: base
// Created Date: 18/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <utility>

#include "autd3/core/link.hpp"

typedef struct {
  autd::core::LinkPtr ptr;
} LinkWrapper;

inline LinkWrapper* link_create(autd::core::LinkPtr ptr) { return new LinkWrapper{std::move(ptr)}; }
inline void link_delete(const LinkWrapper* ptr) { delete ptr; }
