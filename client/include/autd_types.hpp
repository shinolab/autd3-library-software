// File: autd_types.hpp
// Project: include
// Created Date: 26/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 27/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

namespace autd {
#ifdef USE_DOUBLE_AUTD
using Float = double;
#else
using Float = float;
#endif
}  // namespace autd
