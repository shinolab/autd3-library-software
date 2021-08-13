// File: test_utils.hpp
// Project: tests
// Created Date: 13/08/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/08/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#define ASSERT_NEAR_COMPLEX(a, b, eps)  \
  ASSERT_NEAR(a.real(), b.real(), eps); \
  ASSERT_NEAR(a.imag(), b.imag(), eps)
