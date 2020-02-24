// File: core.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autd {
namespace internal {
class Link;
}

enum class LinkType : int { ETHERCAT, TwinCAT, SOEM };

class Controller;
class AUTDController;
class Geometry;
class Timer;

template <class T>
#if DLL_FOR_CAPI
static T *CreateHelper() {
  struct impl : T {
    impl() : T() {}
  };
  return new impl;
}
#else
static std::shared_ptr<T> CreateHelper() {
  struct impl : T {
    impl() : T() {}
  };
  auto p = std::make_shared<impl>();
  return std::move(p);
}
#endif
}  // namespace autd
