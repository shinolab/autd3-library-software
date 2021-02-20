// File: autd_types.hpp
// Project: include
// Created Date: 26/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

namespace autd {
#ifdef USE_DOUBLE_AUTD
using Float = double;
constexpr Float ToFloat(const double f) { return f; }
#else
using Float = float;
constexpr Float ToFloat(const double f) { return static_cast<Float>(f); }
#endif
}  // namespace autd
