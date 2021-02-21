// File: linalg.hpp
// Project: include
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
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

#include "autd_types.hpp"

namespace autd {
using Vector3 = Eigen::Matrix<Float, 3, 1>;
using Quaternion = Eigen::Quaternion<Float>;
}  // namespace autd
