﻿// File: core.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 18/02/2020
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

static std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> tokens;
  std::string token;
  for (char ch : s) {
    if (ch == delim) {
      if (!token.empty()) tokens.push_back(token);
      token.clear();
    } else {
      token += ch;
    }
  }
  if (!token.empty()) tokens.push_back(token);
  return tokens;
}
}  // namespace autd
