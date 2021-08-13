// File: eigen_backend_test.cpp
// Project: holo
// Created Date: 13/08/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/08/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/eigen_backend.hpp"

#include <gtest/gtest.h>

#include <random>

#include "test_utils.hpp"

using Backend = autd::gain::holo::Eigen3Backend;
using autd::gain::holo::TRANSPOSE;
using complex = std::complex<double>;

template <typename T>
std::vector<double> random_vector_raw(T n, const double minimum = -1.0, const double maximum = 1.0) {
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution dist(minimum, maximum);
  std::vector<double> v;
  v.reserve(n);
  for (Eigen::Index i = 0; i < n; i++) v.emplace_back(dist(engine));
  return v;
}

template <typename T>
Backend::VectorX random_vector(T n, const double minimum = -1.0, const double maximum = 1.0) {
  auto v = random_vector_raw(n, minimum, maximum);
  Backend::VectorX vv(n);
  for (Eigen::Index i = 0; i < n; i++) vv[i] = v[i];
  return vv;
}

template <typename T>
Backend::VectorXc random_vector_complex(T n, const double minimum = -1.0, const double maximum = 1.0) {
  auto re = random_vector_raw(n, minimum, maximum);
  auto im = random_vector_raw(n, minimum, maximum);
  Backend::VectorXc v(n);
  for (Eigen::Index i = 0; i < n; i++) v[i] = complex(re[i], im[i]);
  return v;
}

template <typename T>
Backend::MatrixXc random_square_matrix_complex(T n, const double minimum = -1.0, const double maximum = 1.0) {
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution dist(minimum, maximum);
  const auto n_ = static_cast<Eigen::Index>(n);
  Backend::MatrixXc a(n_, n_);
  for (Eigen::Index i = 0; i < n_; i++)
    for (Eigen::Index j = 0; j < n_; j++) a(i, j) = std::complex<double>(dist(engine), dist(engine));
  return a;
}

template <typename T>
Backend::MatrixX random_square_matrix(T n, const double minimum = -1.0, const double maximum = 1.0) {
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution dist(minimum, maximum);
  const auto n_ = static_cast<Eigen::Index>(n);
  Backend::MatrixX a(n_, n_);
  for (Eigen::Index i = 0; i < n_; i++)
    for (Eigen::Index j = 0; j < n_; j++) a(i, j) = dist(engine);
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
  Backend::MatrixXc a = random_square_matrix_complex(n);

  Backend::MatrixXc b(n, n);
  Backend::create()->pseudo_inverse_svd(&a, 1e-4, &b);

  Backend::MatrixXc c = a * b;
  for (Eigen::Index i = 0; i < n; i++)
    for (Eigen::Index j = 0; j < n; j++) {
      if (i == j)
        ASSERT_NEAR(c(i, j).real(), 1.0, 0.05);
      else
        ASSERT_NEAR(c(i, j).real(), 0.0, 0.05);
      ASSERT_NEAR(c(i, j).imag(), 0.0, 0.05);
    }
}

TEST(HoloGainEigenBackend, max_eigen_vector) {
  constexpr Eigen::Index n = 5;

  // generate matrix 'a' from given eigen value 'lambda' and eigen vectors 'v'
  Backend::MatrixXc v = random_square_matrix_complex(n);
  while (std::abs(v.determinant()) < 0.1) v = random_square_matrix_complex(n);  // ensure v has inverse matrix
  auto lambda_vals = random_vector_raw(n, 1.0, 10.0);
  std::sort(lambda_vals.begin(), lambda_vals.end());  // maximum eigen value is placed at last
  Backend::MatrixXc lambda = Backend::MatrixXc::Zero(n, n);
  for (Eigen::Index i = 0; i < n; i++) lambda(i, i) = lambda_vals[i];
  Backend::MatrixXc a = v * lambda * v.inverse();

  const Backend::VectorXc b = Backend::create()->max_eigen_vector(&a);

  const auto k = b(0) / v.col(n - 1)(0);
  const Backend::VectorXc expected = v.col(n - 1) * k;

  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(b(i), expected(i), 1e-6);
}

TEST(HoloGainEigenBackend, matrix_add) {
  Backend::MatrixX a(2, 2);
  a << 0, 1, 2, 3;

  Backend::MatrixX b = Backend::MatrixX::Zero(2, 2);

  const auto backend = Backend::create();

  backend->matrix_add(0.0, a, 0.0, &b);
  ASSERT_NEAR(b(0, 0), 0.0, 1e-6);
  ASSERT_NEAR(b(0, 1), 0.0, 1e-6);
  ASSERT_NEAR(b(1, 0), 0.0, 1e-6);
  ASSERT_NEAR(b(1, 1), 0.0, 1e-6);

  backend->matrix_add(1.0, a, 1.0, &b);
  ASSERT_NEAR(b(0, 0), 0.0, 1e-6);
  ASSERT_NEAR(b(0, 1), 1.0, 1e-6);
  ASSERT_NEAR(b(1, 0), 2.0, 1e-6);
  ASSERT_NEAR(b(1, 1), 3.0, 1e-6);

  backend->matrix_add(0.0, a, 2.0, &b);
  ASSERT_NEAR(b(0, 0), 0.0, 1e-6);
  ASSERT_NEAR(b(0, 1), 2.0, 1e-6);
  ASSERT_NEAR(b(1, 0), 4.0, 1e-6);
  ASSERT_NEAR(b(1, 1), 6.0, 1e-6);
}

TEST(HoloGainEigenBackend, matrix_mul) {
  Backend::MatrixXc a(2, 2);
  Backend::MatrixXc b(2, 2);
  a << std::complex<double>(0, 1), std::complex<double>(2, 3), std::complex<double>(4, 5), std::complex<double>(6, 7);
  b << std::complex<double>(8, 9), std::complex<double>(10, 11), std::complex<double>(12, 13), std::complex<double>(14, 15);

  const auto backend = Backend::create();

  Backend::MatrixXc c = Backend::MatrixXc::Zero(2, 2);

  backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, complex(1, 0), a, b, complex(0, 0), &c);
  ASSERT_NEAR_COMPLEX(c(0, 0), std::complex(-24, 70), 1e-6);
  ASSERT_NEAR_COMPLEX(c(0, 1), std::complex(-28, 82), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 0), std::complex(-32, 238), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 1), std::complex(-36, 282), 1e-6);

  backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::TRANS, complex(0, 0), a, b, complex(1, 0), &c);
  ASSERT_NEAR_COMPLEX(c(0, 0), std::complex(-24, 70), 1e-6);
  ASSERT_NEAR_COMPLEX(c(0, 1), std::complex(-28, 82), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 0), std::complex(-32, 238), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 1), std::complex(-36, 282), 1e-6);

  backend->matrix_mul(TRANSPOSE::TRANS, TRANSPOSE::CONJ_NO_TRANS, complex(1, 0), a, b, complex(0, 0), &c);
  ASSERT_NEAR_COMPLEX(c(0, 0), std::complex(122, 16), 1e-6);
  ASSERT_NEAR_COMPLEX(c(0, 1), std::complex(142, 20), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 0), std::complex(206, 12), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 1), std::complex(242, 16), 1e-6);

  backend->matrix_mul(TRANSPOSE::CONJ_NO_TRANS, TRANSPOSE::CONJ_TRANS, complex(1, 0), a, b, complex(1, 0), &c);
  ASSERT_NEAR_COMPLEX(c(0, 0), std::complex(100, -44), 1e-6);
  ASSERT_NEAR_COMPLEX(c(0, 1), std::complex(112, -64), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 0), std::complex(176, -200), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 1), std::complex(204, -284), 1e-6);
}

TEST(HoloGainEigenBackend, matrix_vector_mul) {
  Backend::MatrixXc a(2, 2);
  Backend::VectorXc b(2);
  a << std::complex<double>(0, 1), std::complex<double>(2, 3), std::complex<double>(4, 5), std::complex<double>(6, 7);
  b << std::complex<double>(8, 9), std::complex<double>(10, 11);

  const auto backend = Backend::create();

  Backend::VectorXc c = Backend::VectorXc::Zero(2);

  backend->matrix_vector_mul(TRANSPOSE::NO_TRANS, complex(1, 0), a, b, complex(0, 0), &c);
  ASSERT_NEAR_COMPLEX(c(0), std::complex(-22, 60), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1), std::complex(-30, 212), 1e-6);

  backend->matrix_vector_mul(TRANSPOSE::TRANS, complex(0, 0), a, b, complex(2, 0), &c);
  ASSERT_NEAR_COMPLEX(c(0), std::complex(-44, 120), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1), std::complex(-60, 424), 1e-6);

  backend->matrix_vector_mul(TRANSPOSE::CONJ_TRANS, complex(1, 0), a, b, complex(1, 0), &c);
  ASSERT_NEAR_COMPLEX(c(0), std::complex(60, 106), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1), std::complex(120, 414), 1e-6);

  backend->matrix_vector_mul(TRANSPOSE::CONJ_NO_TRANS, complex(0, 1), a, b, complex(0, 1), &c);
  ASSERT_NEAR_COMPLEX(c(0), std::complex(-90, 122), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1), std::complex(-406, 334), 1e-6);
}

TEST(HoloGainEigenBackend, vector_add) {
  Backend::VectorX a(2);
  Backend::VectorX b(2);
  a << 0, 1;
  b << 8, 9;

  const auto backend = Backend::create();

  backend->vector_add(1, a, 0, &b);
  ASSERT_NEAR(b(0), 0, 1e-6);
  ASSERT_NEAR(b(1), 1, 1e-6);

  backend->vector_add(2, a, 1, &b);
  ASSERT_NEAR(b(0), 0, 1e-6);
  ASSERT_NEAR(b(1), 3, 1e-6);
}

TEST(HoloGainEigenBackend, solve_ch) {
  constexpr Eigen::Index n = 5;

  const auto a = random_square_matrix_complex(n);
  Backend::MatrixXc h = a * a.adjoint();
  auto x_expected = random_vector_complex(n);
  Backend::VectorXc b = h * x_expected;

  Backend::create()->solve_ch(&h, &b);

  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(b(i), x_expected(i), 1e-6);
}

TEST(HoloGainEigenBackend, solve_g) {
  constexpr Eigen::Index n = 5;

  auto a = random_square_matrix(n);
  auto x_expected = random_vector(n);
  Backend::VectorX b = a * x_expected;

  Backend::VectorX x(n);
  Backend::create()->solve_g(&a, &b, &x);

  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR(x(i), x_expected(i), 1e-6);
}

TEST(HoloGainEigenBackend, dot) {
  constexpr Eigen::Index n = 10;
  auto a = random_vector(n, 1.0, 10.0);
  auto b = random_vector(n, 1.0, 10.0);

  auto expected = 0.0;
  for (Eigen::Index i = 0; i < n; i++) expected += a(i) * b(i);

  ASSERT_NEAR(Backend::create()->dot(a, b), expected, 1e-6);
}

TEST(HoloGainEigenBackend, dot_c) {
  constexpr Eigen::Index n = 10;
  auto a = random_vector_complex(n, 1.0, 10.0);
  auto b = random_vector_complex(n, 1.0, 10.0);

  auto expected = complex(0, 0);
  for (Eigen::Index i = 0; i < n; i++) expected += a(i) * b(i);

  ASSERT_NEAR_COMPLEX(Backend::create()->dot_c(a, b), expected, 1e-6);
}

TEST(HoloGainEigenBackend, max_coefficient) {
  constexpr Eigen::Index n = 10;
  auto vals = random_vector_raw(n, 1.0, 10.0);
  std::sort(vals.begin(), vals.end());
  Backend::VectorX v(n);
  for (Eigen::Index i = 0; i < n; i++) v(i) = vals[i];

  ASSERT_EQ(Backend::create()->max_coefficient(v), vals[n - 1]);
}

TEST(HoloGainEigenBackend, max_coefficient_c) {
  constexpr Eigen::Index n = 10;
  auto vals = random_vector_raw(n, 1.0, 10.0);
  std::sort(vals.begin(), vals.end());
  Backend::VectorXc v(n);
  for (Eigen::Index i = 0; i < n; i++) v(i) = complex(vals[i], 0);

  ASSERT_EQ(Backend::create()->max_coefficient_c(v), complex(vals[n - 1], 0));
}

TEST(HoloGainEigenBackend, concat_row) {
  Backend::MatrixXc a(1, 2);
  Backend::MatrixXc b(1, 2);
  a << std::complex<double>(0, 1), std::complex<double>(2, 3);
  b << std::complex<double>(8, 9), std::complex<double>(10, 11);

  auto c = Backend::create()->concat_row(a, b);
  ASSERT_NEAR_COMPLEX(c(0, 0), std::complex(0, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(c(0, 1), std::complex(2, 3), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 0), std::complex(8, 9), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 1), std::complex(10, 11), 1e-6);
}

TEST(HoloGainEigenBackend, concat_col) {
  Backend::MatrixXc a(2, 1);
  Backend::MatrixXc b(2, 1);
  a << std::complex<double>(0, 1), std::complex<double>(2, 3);
  b << std::complex<double>(8, 9), std::complex<double>(10, 11);

  auto c = Backend::create()->concat_col(a, b);
  ASSERT_NEAR_COMPLEX(c(0, 0), std::complex(0, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(c(0, 1), std::complex(8, 9), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 0), std::complex(2, 3), 1e-6);
  ASSERT_NEAR_COMPLEX(c(1, 1), std::complex(10, 11), 1e-6);
}

TEST(HoloGainEigenBackend, mat_cpy) {
  Backend::VectorX a(2);
  a << 0, 1;

  Backend::VectorX b = Backend::VectorX::Zero(2);

  Backend::create()->vec_cpy(a, &b);

  ASSERT_NEAR(b(0), 0, 1e-6);
  ASSERT_NEAR(b(1), 1, 1e-6);
}

TEST(HoloGainEigenBackend, vec_cpy_c) {
  Backend::VectorXc a(2);
  a << std::complex<double>(0, 1), std::complex<double>(2, 3);

  Backend::VectorXc b = Backend::VectorXc::Zero(2);

  Backend::create()->vec_cpy_c(a, &b);

  ASSERT_NEAR_COMPLEX(b(0), std::complex(0, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(b(1), std::complex(2, 3), 1e-6);
}
