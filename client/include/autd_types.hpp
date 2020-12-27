// File: autd_types.hpp
// Project: include
// Created Date: 26/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 27/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#if WIN32
#include <codeanalysis/warnings.h>  // NOLINT
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Dense>
#if WIN32
#pragma warning(pop)
#endif

namespace autd {
#ifdef USE_DOUBLE_AUTD
using Float = double;
constexpr Float ToFloat(const double f) { return f; }
#else
using Float = float;
constexpr Float ToFloat(const double f) { return static_cast<Float>(f); }
#endif

using Vector3 = Eigen::Matrix<Float, 3, 1>;
using Quaternion = Eigen::Quaternion<Float>;
}  // namespace autd
