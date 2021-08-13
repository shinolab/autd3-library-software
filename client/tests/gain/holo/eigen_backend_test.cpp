// File: eigen_backend_test.cpp
// Project: holo
// Created Date: 13/08/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/08/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/eigen_backend.hpp"

#include <gtest/gtest.h>

#include "test_utils.hpp"

TEST(HoloGainEigenBackend, hadamard_product) {
  const auto backend = autd::gain::holo::Eigen3Backend::create();

  autd::gain::holo::Backend::MatrixXc a(2, 2);
  autd::gain::holo::Backend::MatrixXc b(2, 2);
  a << std::complex<double>(0, 1), std::complex<double>(2, 3), std::complex<double>(4, 5), std::complex<double>(6, 7);
  b << std::complex<double>(8, 9), std::complex<double>(10, 11), std::complex<double>(12, 13), std::complex<double>(14, 15);

  autd::gain::holo::Backend::MatrixXc c(2, 2);
  backend->hadamard_product(a, b, &c);

  ASSERT_NEAR_COMPLEX(c(0, 0), std::complex(-9, 8), 1e-6);
  ASSERT_NEAR_COMPLEX(c(0, 1), std::complex(-13, 52), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 0), std::complex(-17, 112), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 1), std::complex(-21, 188), 1e-6);
}
