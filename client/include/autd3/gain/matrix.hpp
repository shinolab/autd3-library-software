// File: matrix.hpp
// Project: holo
// Created Date: 10/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <complex>

namespace autd {
namespace gain {
namespace holo {
enum class TRANSPOSE { NO_TRANS = 111, TRANS = 112, CONJ_TRANS = 113 };
using complex = std::complex<double>;
constexpr complex ONE = complex(1.0, 0.0);
constexpr complex ZERO = complex(0.0, 0.0);
}  // namespace holo
}  // namespace gain
}  // namespace autd
