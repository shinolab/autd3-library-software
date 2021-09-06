// File: eigen_backend_test.cpp
// Project: holo
// Created Date: 13/08/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <cmath>
#include <random>

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26439 26495 26812)
#endif
#include <gtest/gtest.h>
#if _MSC_VER
#pragma warning(pop)
#endif

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6031 6294 6255 26451 26495 26812)
#endif
#include <unsupported/Eigen/MatrixFunctions>
#if _MSC_VER
#pragma warning(pop)
#endif

#include "autd3/gain/eigen_backend.hpp"
#include "test_utils.hpp"

using autd::gain::holo::Backend;
using autd::gain::holo::complex;
using autd::gain::holo::One;
using autd::gain::holo::TRANSPOSE;
using autd::gain::holo::Zero;

template <typename B>
class BackendTest : public testing::Test {
 protected:
  BackendTest() : backend(B::create()) {}

  std::shared_ptr<Backend> backend;
};

using testing::Types;

// FIXME: make more elegant
#ifdef TEST_BLAS_BACKEND
#include "autd3/gain/blas_backend.hpp"
#ifdef TEST_CUDA_BACKEND
#include "autd3/gain/cuda_backend.hpp"
typedef Types<autd::gain::holo::Eigen3Backend, autd::gain::holo::BLASBackend, autd::gain::holo::CUDABackend> Implementations;
#else
typedef Types<autd::gain::holo::Eigen3Backend, autd::gain::holo::BLASBackend> Implementations;
#endif
#else
#ifdef TEST_CUDA_BACKEND
#include "autd3/gain/cuda_backend.hpp"
typedef Types<autd::gain::holo::Eigen3Backend, autd::gain::holo::CUDABackend> Implementations;
#else
typedef Types<autd::gain::holo::Eigen3Backend> Implementations;
#endif
#endif

TYPED_TEST_SUITE(BackendTest, Implementations);

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
std::vector<complex> random_vector_complex(T n, const double minimum = -1.0, const double maximum = 1.0) {
  const auto re = random_vector(n, minimum, maximum);
  const auto im = random_vector(n, minimum, maximum);
  std::vector<complex> v;
  v.reserve(n);
  for (Eigen::Index i = 0; i < n; i++) v.emplace_back(complex(re[i], im[i]));
  return v;
}

TYPED_TEST(BackendTest, make_complex) {
  auto r = this->backend->allocate_matrix("r", 2, 1);
  auto i = this->backend->allocate_matrix("i", 2, 1);

  r->copy_from({0, 1});
  i->copy_from({2, 3});

  auto a = this->backend->allocate_matrix_c("a", 2, 1);

  this->backend->make_complex(r, i, a);

  a->copy_to_host();
  ASSERT_NEAR_COMPLEX(a->data(0, 0), complex(0, 2), 1e-6);
  ASSERT_NEAR_COMPLEX(a->data(1, 0), complex(1, 3), 1e-6);
}

TYPED_TEST(BackendTest, exp) {
  auto a = this->backend->allocate_matrix_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  this->backend->exp(a);

  a->copy_to_host();
  ASSERT_NEAR_COMPLEX(a->data(0, 0), std::exp(complex(0, 1)), 1e-6);
  ASSERT_NEAR_COMPLEX(a->data(1, 0), std::exp(complex(2, 3)), 1e-6);
}

TYPED_TEST(BackendTest, scale) {
  auto a = this->backend->allocate_matrix_c("a", 4, 1);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});

  this->backend->scale(a, complex(1, 1));

  a->copy_to_host();
  ASSERT_NEAR_COMPLEX(a->data(0, 0), complex(-1, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(a->data(1, 0), complex(-1, 5), 1e-6);
  ASSERT_NEAR_COMPLEX(a->data(2, 0), complex(-1, 9), 1e-6);
  ASSERT_NEAR_COMPLEX(a->data(3, 0), complex(-1, 13), 1e-6);
}

TYPED_TEST(BackendTest, hadamard_product) {
  auto a = this->backend->allocate_matrix_c("a", 2, 2);
  auto b = this->backend->allocate_matrix_c("b", 2, 2);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});
  b->copy_from({complex(8, 9), complex(10, 11), complex(12, 13), complex(14, 15)});

  auto c = this->backend->allocate_matrix_c("c", 2, 2);
  this->backend->hadamard_product(a, b, c);

  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->data(0, 0), complex(-9, 8), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 0), complex(-13, 52), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(0, 1), complex(-17, 112), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 1), complex(-21, 188), 1e-6);
}

TYPED_TEST(BackendTest, real) {
  auto a = this->backend->allocate_matrix_c("a", 2, 2);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});

  auto b = this->backend->allocate_matrix("b", 2, 2);

  this->backend->real(a, b);

  b->copy_to_host();
  ASSERT_EQ(b->data(0, 0), 0.0);
  ASSERT_EQ(b->data(1, 0), 2.0);
  ASSERT_EQ(b->data(0, 1), 4.0);
  ASSERT_EQ(b->data(1, 1), 6.0);
}

TYPED_TEST(BackendTest, arg) {
  auto a = this->backend->allocate_matrix_c("a", 2, 1);
  a->copy_from({std::exp(complex(0, 1)) * 2.0, std::exp(complex(0, 2)) * 4.0});

  auto b = this->backend->allocate_matrix_c("b", 2, 1);

  this->backend->arg(a, b);

  b->copy_to_host();
  ASSERT_NEAR_COMPLEX(b->data(0, 0), std::exp(complex(0, 1)), 1e-6);
  ASSERT_NEAR_COMPLEX(b->data(1, 0), std::exp(complex(0, 2)), 1e-6);
}

TYPED_TEST(BackendTest, pseudo_inverse_svd) {
  constexpr auto n = 5;
  auto a = this->backend->allocate_matrix_c("a", n, n);
  a->copy_from(random_vector_complex(n * n));

  auto b = this->backend->allocate_matrix_c("b", n, n);
  this->backend->pseudo_inverse_svd(a, 1e-4, b);

  auto c = this->backend->allocate_matrix_c("c", n, n);
  this->backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, One, a, b, Zero, c);
  c->copy_to_host();
  for (Eigen::Index i = 0; i < n; i++)
    for (Eigen::Index j = 0; j < n; j++) {
      if (i == j)
        ASSERT_NEAR(c->data(i, j).real(), 1.0, 0.05);
      else
        ASSERT_NEAR(c->data(i, j).real(), 0.0, 0.05);
      ASSERT_NEAR(c->data(i, j).imag(), 0.0, 0.05);
    }
}

TYPED_TEST(BackendTest, max_eigen_vector) {
  constexpr Eigen::Index n = 5;

  auto gen_unitary = [](const Eigen::Index size) {
    Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> tmp(size, size);
    const auto rand = random_vector_complex(size * size);
    std::memcpy(tmp.data(), rand.data(), rand.size() * sizeof(complex));

    const Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> hermite = tmp.adjoint() * tmp;
    Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> u = (complex(0.0, 1.0) * hermite).exp();
    return u;
  };

  // generate matrix 'a' from given eigen value 'lambda' and eigen vectors 'u'
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> u = gen_unitary(n);
  auto lambda_vals = random_vector(n, 1.0, 10.0);
  std::sort(lambda_vals.begin(), lambda_vals.end());  // maximum eigen value is placed at last
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> lambda = Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>::Zero(n, n);
  for (Eigen::Index i = 0; i < n; i++) lambda(i, i) = lambda_vals[i];
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> a_vals = u * lambda * u.adjoint();

  auto a = this->backend->allocate_matrix_c("a", n, n);
  a->copy_from(a_vals.data());
  const auto b = this->backend->max_eigen_vector(a);

  b->copy_to_host();
  const auto k = b->data(0) / u.col(n - 1)(0);
  const Eigen::Matrix<complex, -1, 1, Eigen::ColMajor> expected = u.col(n - 1) * k;

  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(b->data(i), expected(i), 1e-6);
}

TYPED_TEST(BackendTest, matrix_add) {
  auto a = this->backend->allocate_matrix("a", 2, 2);
  a->copy_from({0, 2, 1, 3});

  auto b = this->backend->allocate_matrix("b", 2, 2);
  b->fill(0.0);

  this->backend->matrix_add(0.0, a, b);
  b->copy_to_host();
  ASSERT_NEAR(b->data(0, 0), 0.0, 1e-6);
  ASSERT_NEAR(b->data(0, 1), 0.0, 1e-6);
  ASSERT_NEAR(b->data(1, 0), 0.0, 1e-6);
  ASSERT_NEAR(b->data(1, 1), 0.0, 1e-6);

  this->backend->matrix_add(2.0, a, b);
  b->copy_to_host();
  ASSERT_NEAR(b->data(0, 0), 0.0, 1e-6);
  ASSERT_NEAR(b->data(0, 1), 2.0, 1e-6);
  ASSERT_NEAR(b->data(1, 0), 4.0, 1e-6);
  ASSERT_NEAR(b->data(1, 1), 6.0, 1e-6);
}

TYPED_TEST(BackendTest, matrix_mul_c) {
  auto a = this->backend->allocate_matrix_c("a", 2, 2);
  auto b = this->backend->allocate_matrix_c("b", 2, 2);
  a->copy_from({complex(0, 1), complex(4, 5), complex(2, 3), complex(6, 7)});
  b->copy_from({complex(8, 9), complex(12, 13), complex(10, 11), complex(14, 15)});

  auto c = this->backend->allocate_matrix_c("c", 2, 2);

  this->backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, One, a, b, Zero, c);
  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->data(0, 0), complex(-24, 70), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(0, 1), complex(-28, 82), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 0), complex(-32, 238), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 1), complex(-36, 282), 1e-6);

  this->backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::TRANS, Zero, a, b, One, c);
  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->data(0, 0), complex(-24, 70), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(0, 1), complex(-28, 82), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 0), complex(-32, 238), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 1), complex(-36, 282), 1e-6);

  this->backend->matrix_mul(TRANSPOSE::TRANS, TRANSPOSE::CONJ_NO_TRANS, One, a, b, Zero, c);
  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->data(0, 0), complex(122, 16), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(0, 1), complex(142, 20), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 0), complex(206, 12), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 1), complex(242, 16), 1e-6);

  this->backend->matrix_mul(TRANSPOSE::CONJ_NO_TRANS, TRANSPOSE::CONJ_TRANS, One, a, b, One, c);
  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->data(0, 0), complex(100, -44), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(0, 1), complex(112, -64), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 0), complex(176, -200), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 1), complex(204, -284), 1e-6);
}

TYPED_TEST(BackendTest, matrix_mul) {
  auto a = this->backend->allocate_matrix("a", 2, 2);
  auto b = this->backend->allocate_matrix("b", 2, 2);
  a->copy_from({0, 2, 1, 3});
  b->copy_from({4, 6, 5, 7});

  auto c = this->backend->allocate_matrix("c", 2, 2);

  this->backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::TRANS, 1.0, a, b, 0.0, c);
  c->copy_to_host();
  ASSERT_NEAR(c->data(0, 0), 5.0, 1e-6);
  ASSERT_NEAR(c->data(0, 1), 7.0, 1e-6);
  ASSERT_NEAR(c->data(1, 0), 23.0, 1e-6);
  ASSERT_NEAR(c->data(1, 1), 33.0, 1e-6);

  this->backend->matrix_mul(TRANSPOSE::TRANS, TRANSPOSE::NO_TRANS, 1.0, a, b, 1, c);
  c->copy_to_host();
  ASSERT_NEAR(c->data(0, 0), 17.0, 1e-6);
  ASSERT_NEAR(c->data(0, 1), 21.0, 1e-6);
  ASSERT_NEAR(c->data(1, 0), 45.0, 1e-6);
  ASSERT_NEAR(c->data(1, 1), 59.0, 1e-6);
}

TYPED_TEST(BackendTest, solve_ch) {
  constexpr Eigen::Index n = 5;

  auto tmp = this->backend->allocate_matrix_c("tmp", n, n);
  tmp->copy_from(random_vector_complex(n * n));

  auto a = this->backend->allocate_matrix_c("a", n, n);
  this->backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, One, tmp, tmp, Zero, a);
  auto x_expected = random_vector_complex(n);

  auto x = this->backend->allocate_matrix_c("x", n, 1);
  x->copy_from(x_expected);

  auto b = this->backend->allocate_matrix_c("b", n, 1);
  this->backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, One, a, x, Zero, b);

  this->backend->solve_ch(a, b);

  b->copy_to_host();
  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(b->data(i), x_expected[i], 1e-6);
}

TYPED_TEST(BackendTest, solve_g) {
  constexpr Eigen::Index n = 5;

  auto tmp = this->backend->allocate_matrix("tmp", n, n);
  tmp->copy_from(random_vector(n * n));

  auto a = this->backend->allocate_matrix("a", n, n);
  this->backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::TRANS, 1.0, tmp, tmp, 0.0, a);
  auto x_expected = random_vector(n);

  auto x = this->backend->allocate_matrix("x", n, 1);
  x->copy_from(x_expected);

  auto b = this->backend->allocate_matrix("b", n, 1);
  this->backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, 1.0, a, x, 0.0, b);

  auto xs = this->backend->allocate_matrix("xs", n, 1);
  this->backend->solve_g(a, b, xs);

  xs->copy_to_host();
  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR(xs->data(i), x_expected[i], 1e-6);
}

TYPED_TEST(BackendTest, dot) {
  constexpr Eigen::Index n = 10;
  auto a_vals = random_vector(n, 1.0, 10.0);
  auto b_vals = random_vector(n, 1.0, 10.0);

  auto expected = 0.0;
  for (Eigen::Index i = 0; i < n; i++) expected += a_vals[i] * b_vals[i];

  auto a = this->backend->allocate_matrix("a", n, 1);
  a->copy_from(a_vals);
  auto b = this->backend->allocate_matrix("b", n, 1);
  b->copy_from(b_vals);

  ASSERT_NEAR(this->backend->dot(a, b), expected, 1e-6);
}

TYPED_TEST(BackendTest, dot_c) {
  constexpr Eigen::Index n = 10;
  auto a_vals = random_vector_complex(n, 1.0, 10.0);
  auto b_vals = random_vector_complex(n, 1.0, 10.0);

  auto expected = complex(0, 0);
  for (Eigen::Index i = 0; i < n; i++) expected += std::conj(a_vals[i]) * b_vals[i];

  auto a = this->backend->allocate_matrix_c("a", n, 1);
  a->copy_from(a_vals);
  auto b = this->backend->allocate_matrix_c("b", n, 1);
  b->copy_from(b_vals);

  ASSERT_NEAR_COMPLEX(this->backend->dot(a, b), expected, 1e-6);
}

TYPED_TEST(BackendTest, max_coefficient) {
  constexpr Eigen::Index n = 100;
  auto vals = random_vector(n, -20.0, 2.0);
  std::sort(vals.begin(), vals.end());
  auto v = this->backend->allocate_matrix("v", n, 1);
  v->copy_from(vals);

  ASSERT_EQ(this->backend->max_coefficient(v), complex(vals[n - 1], 0));
}

TYPED_TEST(BackendTest, max_coefficient_c) {
  constexpr Eigen::Index n = 100;
  auto vals = random_vector(n, 0.0, 10.0);
  std::sort(vals.begin(), vals.end());
  std::vector<complex> vals_c;
  for (const auto v : vals) vals_c.emplace_back(complex(v, 0.0));
  auto v = this->backend->allocate_matrix_c("v", n, 1);
  v->copy_from(vals_c);

  ASSERT_EQ(this->backend->max_coefficient(v), complex(vals[n - 1], 0));
}

TYPED_TEST(BackendTest, concat_row) {
  auto a = this->backend->allocate_matrix_c("a", 1, 2);
  auto b = this->backend->allocate_matrix_c("b", 1, 2);
  a->copy_from({complex(0, 1), complex(2, 3)});
  b->copy_from({complex(8, 9), complex(10, 11)});

  auto c = this->backend->concat_row(a, b);

  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->data(0, 0), complex(0, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 0), complex(8, 9), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(0, 1), complex(2, 3), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 1), complex(10, 11), 1e-6);
}

TYPED_TEST(BackendTest, concat_col) {
  auto a = this->backend->allocate_matrix_c("a", 2, 1);
  auto b = this->backend->allocate_matrix_c("b", 2, 1);
  a->copy_from({complex(0, 1), complex(8, 9)});
  b->copy_from({complex(2, 3), complex(10, 11)});

  auto c = this->backend->concat_col(a, b);

  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->data(0, 0), complex(0, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 0), complex(8, 9), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(0, 1), complex(2, 3), 1e-6);
  ASSERT_NEAR_COMPLEX(c->data(1, 1), complex(10, 11), 1e-6);
}

TYPED_TEST(BackendTest, mat_cpy) {
  auto a = this->backend->allocate_matrix("a", 2, 2);
  a->copy_from({0.0, 1.0, 2.0, 3.0});

  auto b = this->backend->allocate_matrix("b", 2, 2);

  this->backend->mat_cpy(a, b);

  b->copy_to_host();
  ASSERT_EQ(b->data(0, 0), 0.0);
  ASSERT_EQ(b->data(1, 0), 1.0);
  ASSERT_EQ(b->data(0, 1), 2.0);
  ASSERT_EQ(b->data(1, 1), 3.0);
}

TYPED_TEST(BackendTest, mat_cpy_c) {
  auto a = this->backend->allocate_matrix_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(8, 9)});

  auto b = this->backend->allocate_matrix_c("b", 2, 1);

  this->backend->mat_cpy(a, b);

  b->copy_to_host();
  ASSERT_EQ(b->data(0, 0), complex(0, 1));
  ASSERT_EQ(b->data(1, 0), complex(8, 9));
}
