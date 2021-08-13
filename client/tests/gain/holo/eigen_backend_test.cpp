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

#include <random>

#include "test_utils.hpp"

using Backend = autd::gain::holo::Eigen3Backend;

template <typename T>
std::vector<double> random_vector(T n, const double minimum = -1.0, const double maximum = 1.0) {
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution dist(minimum, maximum);
  std::vector<double> v;
  v.reserve(n);
  for (Eigen::Index i = 0; i < n; i++) v.emplace_back(dist(engine));
  return v;
}

template <typename T>
Backend::MatrixXc random_square_matrix(T n, const double minimum = -1.0, const double maximum = 1.0) {
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution dist(minimum, maximum);

  const auto n_ = static_cast<Eigen::Index>(n);
  Backend::MatrixXc a(n_, n_);
  for (Eigen::Index i = 0; i < n_; i++)
    for (Eigen::Index j = 0; j < n_; j++) a(i, j) = std::complex<double>(dist(engine), dist(engine));
  return a;
}

TEST(HoloGainEigenBackend, hadamard_product) {
  autd::gain::holo::Backend::MatrixXc a(2, 2);
  autd::gain::holo::Backend::MatrixXc b(2, 2);
  a << std::complex<double>(0, 1), std::complex<double>(2, 3), std::complex<double>(4, 5), std::complex<double>(6, 7);
  b << std::complex<double>(8, 9), std::complex<double>(10, 11), std::complex<double>(12, 13), std::complex<double>(14, 15);

  autd::gain::holo::Backend::MatrixXc c(2, 2);
  Backend::create()->hadamard_product(a, b, &c);

  ASSERT_NEAR_COMPLEX(c(0, 0), std::complex(-9, 8), 1e-6);
  ASSERT_NEAR_COMPLEX(c(0, 1), std::complex(-13, 52), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 0), std::complex(-17, 112), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 1), std::complex(-21, 188), 1e-6);
}

TEST(HoloGainEigenBackend, real) {
  Backend::MatrixXc a(2, 2);
  a << std::complex<double>(0, 1), std::complex<double>(2, 3), std::complex<double>(4, 5), std::complex<double>(6, 7);

  Backend::MatrixX b(2, 2);

  Backend::create()->real(a, &b);

  ASSERT_EQ(b(0, 0), 0.0);
  ASSERT_EQ(b(0, 1), 2.0);
  ASSERT_EQ(b(1, 0), 4.0);
  ASSERT_EQ(b(1, 1), 6.0);
}

TEST(HoloGainEigenBackend, pseudo_inverse_svd) {
  constexpr auto n = 5;
  Backend::MatrixXc a = random_square_matrix(n);

  Backend::MatrixXc b(n, n);
  Backend::create()->pseudo_inverse_svd(&a, 1e-4, &b);

  Backend::MatrixXc c = a * b;
  for (Eigen::Index i = 0; i < n; i++)
    for (Eigen::Index j = 0; j < n; j++) {
      if (i == j)
        ASSERT_NEAR(c(i, j).real(), 1.0, 0.01);
      else
        ASSERT_NEAR(c(i, j).real(), 0.0, 0.01);
      ASSERT_NEAR(c(i, j).imag(), 0.0, 0.01);
    }
}

TEST(HoloGainEigenBackend, max_eigen_vector) {
  constexpr Eigen::Index n = 5;

  // generate matrix 'a' from given eigen value 'lambda' and eigen vectors 'v'
  Backend::MatrixXc v = random_square_matrix(n);
  while (std::abs(v.determinant()) < 0.1) v = random_square_matrix(n);  // ensure v has inverse matrix
  auto lambda_vals = random_vector(n, 1.0, 10.0);
  std::sort(lambda_vals.begin(), lambda_vals.end());  // maximum eigen value is placed at last
  Backend::MatrixXc lambda = Backend::MatrixXc::Zero(n, n);
  for (Eigen::Index i = 0; i < n; i++) lambda(i, i) = lambda_vals[i];
  Backend::MatrixXc a = v * lambda * v.inverse();

  const Backend::VectorXc b = Backend::create()->max_eigen_vector(&a);

  const auto k = b(0) / v.col(n - 1)(0);
  const Backend::VectorXc expected = v.col(n - 1) * k;

  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(b(i), expected(i), 1e-6);
}

TEST(HoloGainEigenBackend, matrix_add) {}

TEST(HoloGainEigenBackend, matrix_mul) {}

TEST(HoloGainEigenBackend, matrix_vector_mul) {}

TEST(HoloGainEigenBackend, vector_add) {}

TEST(HoloGainEigenBackend, solve_ch) {}

TEST(HoloGainEigenBackend, solve_g) {}

TEST(HoloGainEigenBackend, dot) {}

TEST(HoloGainEigenBackend, dot_c) {}

TEST(HoloGainEigenBackend, max_coefficient) {}

TEST(HoloGainEigenBackend, max_coefficient_c) {}

TEST(HoloGainEigenBackend, concat_row) {}

TEST(HoloGainEigenBackend, concat_col) {}

TEST(HoloGainEigenBackend, mat_cpy) {}

TEST(HoloGainEigenBackend, vec_cpy_c) {}
