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

#include <utility>

#include "link.hpp"

typedef struct {
  autd::LinkPtr ptr;
} LinkWrapper;

inline LinkWrapper* LinkCreate(autd::LinkPtr ptr) { return new LinkWrapper{std::move(ptr)}; }

inline void LinkDelete(LinkWrapper* ptr) { delete ptr; }
