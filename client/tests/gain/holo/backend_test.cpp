// File: eigen_backend_test.cpp
// Project: holo
// Created Date: 13/08/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
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

#include "autd3/core/utils.hpp"
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/linalg_backend.hpp"
#include "autd3/utils.hpp"
#include "test_utils.hpp"

using autd::gain::holo::complex;
using autd::gain::holo::ONE;
using autd::gain::holo::TRANSPOSE;
using autd::gain::holo::ZERO;

template <typename P>
class BackendTest : public testing::Test {
 protected:
  BackendTest() : _pool() {}

  P _pool;
};

using testing::Types;

#define Eigen3BackendType autd::gain::holo::EigenBackend

#ifdef TEST_BLAS_BACKEND
#include "autd3/gain/blas_backend.hpp"
#define BLASBackendType , autd::gain::holo::BLASBackend
#else
#define BLASBackendType
#endif

#ifdef TEST_CUDA_BACKEND
#include "autd3/gain/cuda_backend.hpp"
#define CUDABackendType , autd::gain::holo::CUDABackend
#else
#define CUDABackendType
#endif

#ifdef TEST_ARRAYFIRE_BACKEND
#include "autd3/gain/arrayfire_backend.hpp"
#define ArrayFireBackendType , autd::gain::holo::ArrayFireBackend
#else
#define ArrayFireBackendType
#endif

typedef Types<Eigen3BackendType BLASBackendType CUDABackendType ArrayFireBackendType> Implementations;

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
  auto r = this->_pool.rent("r", 2, 1);
  auto i = this->_pool.rent("i", 2, 1);

  r->copy_from({0, 1});
  i->copy_from({2, 3});

  auto a = this->_pool.rent_c("a", 2, 1);

  a->make_complex(r, i);

  a->copy_to_host();
  ASSERT_NEAR_COMPLEX(a->at(0, 0), complex(0, 2), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(1, 0), complex(1, 3), 1e-6);
}

TYPED_TEST(BackendTest, exp) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  a->exp();

  a->copy_to_host();
  ASSERT_NEAR_COMPLEX(a->at(0, 0), std::exp(complex(0, 1)), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(1, 0), std::exp(complex(2, 3)), 1e-6);
}

TYPED_TEST(BackendTest, scale) {
  auto a = this->_pool.rent_c("a", 4, 1);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});

  a->scale(complex(1, 1));

  a->copy_to_host();
  ASSERT_NEAR_COMPLEX(a->at(0, 0), complex(-1, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(1, 0), complex(-1, 5), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(2, 0), complex(-1, 9), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(3, 0), complex(-1, 13), 1e-6);
}

TYPED_TEST(BackendTest, reciprocal) {
  auto a = this->_pool.rent("a", 2, 1);
  a->copy_from({1, 2});

  auto b = this->_pool.rent("b", 2, 1);
  b->reciprocal(a);

  b->copy_to_host();
  ASSERT_NEAR(b->at(0, 0), 1.0 / 1.0, 1e-6);
  ASSERT_NEAR(b->at(1, 0), 1.0 / 2.0, 1e-6);
}

TYPED_TEST(BackendTest, reciprocal_c) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  auto b = this->_pool.rent_c("b", 2, 1);
  b->reciprocal(a);

  b->copy_to_host();
  ASSERT_NEAR_COMPLEX(b->at(0, 0), complex(0, -1), 1e-6);
  ASSERT_NEAR_COMPLEX(b->at(1, 0), complex(2.0 / 13.0, -3.0 / 13.0), 1e-6);
}

TYPED_TEST(BackendTest, abs_c) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->abs(a);

  b->copy_to_host();
  ASSERT_NEAR(b->at(0, 0).real(), std::abs(complex(0, 1)), 1e-6);
  ASSERT_NEAR(b->at(1, 0).real(), std::abs(complex(2, 3)), 1e-6);
}

TYPED_TEST(BackendTest, real) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});

  auto b = this->_pool.rent("b", 2, 2);

  b->real(a);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), 0.0);
  ASSERT_EQ(b->at(1, 0), 2.0);
  ASSERT_EQ(b->at(0, 1), 4.0);
  ASSERT_EQ(b->at(1, 1), 6.0);
}

TYPED_TEST(BackendTest, arg) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({std::exp(complex(0, 1)) * 2.0, std::exp(complex(0, 2)) * 4.0});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->arg(a);

  b->copy_to_host();
  ASSERT_NEAR_COMPLEX(b->at(0, 0), std::exp(complex(0, 1)), 1e-6);
  ASSERT_NEAR_COMPLEX(b->at(1, 0), std::exp(complex(0, 2)), 1e-6);
}

TYPED_TEST(BackendTest, hadamard_product) {
  auto a = this->_pool.rent_c("a", 2, 2);
  auto b = this->_pool.rent_c("b", 2, 2);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});
  b->copy_from({complex(8, 9), complex(10, 11), complex(12, 13), complex(14, 15)});

  auto c = this->_pool.rent_c("c", 2, 2);
  c->hadamard_product(a, b);

  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->at(0, 0), complex(-9, 8), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 0), complex(-13, 52), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(0, 1), complex(-17, 112), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 1), complex(-21, 188), 1e-6);
}

TYPED_TEST(BackendTest, pseudo_inverse_svd) {
  constexpr auto n = 5;
  auto a = this->_pool.rent("a", n, n);
  a->copy_from(random_vector(n * n));

  auto b = this->_pool.rent("b", n, n);
  auto u = this->_pool.rent("u", n, n);
  auto s = this->_pool.rent("s", n, n);
  auto vt = this->_pool.rent("vt", n, n);
  auto buf = this->_pool.rent("buf", n, n);
  auto mat = this->_pool.rent("mat", n, n);
  mat->copy_from(a);
  b->pseudo_inverse_svd(mat, 0.0, u, s, vt, buf);

  auto c = this->_pool.rent("c", n, n);
  c->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, 1.0, a, b, 0.0);
  c->copy_to_host();
  for (Eigen::Index i = 0; i < n; i++)
    for (Eigen::Index j = 0; j < n; j++) {
      if (i == j)
        ASSERT_NEAR(c->at(i, j), 1.0, 0.1);
      else
        ASSERT_NEAR(c->at(i, j), 0.0, 0.1);
    }
}

TYPED_TEST(BackendTest, pseudo_inverse_svd_c) {
  constexpr auto n = 5;
  auto a = this->_pool.rent_c("a", n, n);
  a->copy_from(random_vector_complex(n * n));

  auto b = this->_pool.rent_c("b", n, n);
  auto u = this->_pool.rent_c("u", n, n);
  auto s = this->_pool.rent_c("s", n, n);
  auto vt = this->_pool.rent_c("vt", n, n);
  auto buf = this->_pool.rent_c("buf", n, n);
  auto mat = this->_pool.rent_c("mat", n, n);
  mat->copy_from(a);
  b->pseudo_inverse_svd(mat, 0.0, u, s, vt, buf);

  auto c = this->_pool.rent_c("c", n, n);
  c->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, 1.0, a, b, 0.0);
  c->copy_to_host();
  for (Eigen::Index i = 0; i < n; i++)
    for (Eigen::Index j = 0; j < n; j++) {
      if (i == j)
        ASSERT_NEAR_COMPLEX(c->at(i, j), ONE, 0.1);
      else
        ASSERT_NEAR_COMPLEX(c->at(i, j), ZERO, 0.1);
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

  auto a = this->_pool.rent_c("a", n, n);
  a->copy_from(a_vals.data());
  const auto b = this->_pool.rent_c("b", n, 1);
  a->max_eigen_vector(b);

  b->copy_to_host();
  const auto k = b->at(0, 0) / u.col(n - 1)(0);
  const Eigen::Matrix<complex, -1, 1, Eigen::ColMajor> expected = u.col(n - 1) * k;

  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(b->at(i, 0), expected(i), 1e-6);
}

TYPED_TEST(BackendTest, matrix_add) {
  auto a = this->_pool.rent("a", 2, 2);
  a->copy_from({0, 2, 1, 3});

  auto b = this->_pool.rent("b", 2, 2);
  b->fill(0.0);

  b->add(0.0, a);
  b->copy_to_host();
  ASSERT_NEAR(b->at(0, 0), 0.0, 1e-6);
  ASSERT_NEAR(b->at(0, 1), 0.0, 1e-6);
  ASSERT_NEAR(b->at(1, 0), 0.0, 1e-6);
  ASSERT_NEAR(b->at(1, 1), 0.0, 1e-6);

  b->add(2.0, a);
  b->copy_to_host();
  ASSERT_NEAR(b->at(0, 0), 0.0, 1e-6);
  ASSERT_NEAR(b->at(0, 1), 2.0, 1e-6);
  ASSERT_NEAR(b->at(1, 0), 4.0, 1e-6);
  ASSERT_NEAR(b->at(1, 1), 6.0, 1e-6);
}

TYPED_TEST(BackendTest, matrix_mul_c) {
  auto a = this->_pool.rent_c("a", 2, 2);
  auto b = this->_pool.rent_c("b", 2, 2);
  a->copy_from({complex(0, 1), complex(4, 5), complex(2, 3), complex(6, 7)});
  b->copy_from({complex(8, 9), complex(12, 13), complex(10, 11), complex(14, 15)});

  auto c = this->_pool.rent_c("c", 2, 2);

  c->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, a, b, ZERO);

  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->at(0, 0), complex(-24, 70), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(0, 1), complex(-28, 82), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 0), complex(-32, 238), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 1), complex(-36, 282), 1e-6);

  c->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::TRANS, ZERO, a, b, ONE);
  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->at(0, 0), complex(-24, 70), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(0, 1), complex(-28, 82), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 0), complex(-32, 238), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 1), complex(-36, 282), 1e-6);
}

TYPED_TEST(BackendTest, matrix_mul) {
  auto a = this->_pool.rent("a", 2, 2);
  auto b = this->_pool.rent("b", 2, 2);
  a->copy_from({0, 2, 1, 3});
  b->copy_from({4, 6, 5, 7});

  auto c = this->_pool.rent("c", 2, 2);

  c->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::TRANS, 1.0, a, b, 0.0);
  c->copy_to_host();
  ASSERT_NEAR(c->at(0, 0), 5.0, 1e-6);
  ASSERT_NEAR(c->at(0, 1), 7.0, 1e-6);
  ASSERT_NEAR(c->at(1, 0), 23.0, 1e-6);
  ASSERT_NEAR(c->at(1, 1), 33.0, 1e-6);

  c->mul(TRANSPOSE::TRANS, TRANSPOSE::NO_TRANS, 1.0, a, b, 1);

  c->copy_to_host();
  ASSERT_NEAR(c->at(0, 0), 17.0, 1e-6);
  ASSERT_NEAR(c->at(0, 1), 21.0, 1e-6);
  ASSERT_NEAR(c->at(1, 0), 45.0, 1e-6);
  ASSERT_NEAR(c->at(1, 1), 59.0, 1e-6);
}

TYPED_TEST(BackendTest, solve_c) {
  constexpr Eigen::Index n = 5;

  auto tmp = this->_pool.rent_c("tmp", n, n);
  tmp->copy_from(random_vector_complex(n * n));

  auto a = this->_pool.rent_c("a", n, n);
  a->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, tmp, tmp, ZERO);
  auto x_expected = random_vector_complex(n);

  auto x = this->_pool.rent_c("x", n, 1);
  x->copy_from(x_expected);

  auto b = this->_pool.rent_c("b", n, 1);
  b->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, a, x, ZERO);

  a->solve(b);

  b->copy_to_host();
  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(b->at(i, 0), x_expected[i], 1e-6);
}

TYPED_TEST(BackendTest, solve) {
  constexpr Eigen::Index n = 5;

  auto tmp = this->_pool.rent("tmp", n, n);
  tmp->copy_from(random_vector(n * n));

  auto a = this->_pool.rent("a", n, n);
  a->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::TRANS, 1.0, tmp, tmp, 0.0);
  auto x_expected = random_vector(n);

  auto x = this->_pool.rent("x", n, 1);
  x->copy_from(x_expected);

  auto b = this->_pool.rent("b", n, 1);
  b->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, 1.0, a, x, 0.0);

  a->solve(b);

  b->copy_to_host();
  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR(b->at(i, 0), x_expected[i], 1e-6);
}

TYPED_TEST(BackendTest, dot) {
  constexpr Eigen::Index n = 10;
  auto a_vals = random_vector(n, 1.0, 10.0);
  auto b_vals = random_vector(n, 1.0, 10.0);

  auto expected = 0.0;
  for (Eigen::Index i = 0; i < n; i++) expected += a_vals[i] * b_vals[i];

  auto a = this->_pool.rent("a", n, 1);
  a->copy_from(a_vals);
  auto b = this->_pool.rent("b", n, 1);
  b->copy_from(b_vals);

  ASSERT_NEAR(a->dot(b), expected, 1e-6);
}

TYPED_TEST(BackendTest, dot_c) {
  constexpr Eigen::Index n = 10;
  auto a_vals = random_vector_complex(n, 1.0, 10.0);
  auto b_vals = random_vector_complex(n, 1.0, 10.0);

  auto expected = complex(0, 0);
  for (Eigen::Index i = 0; i < n; i++) expected += std::conj(a_vals[i]) * b_vals[i];

  auto a = this->_pool.rent_c("a", n, 1);
  a->copy_from(a_vals);
  auto b = this->_pool.rent_c("b", n, 1);
  b->copy_from(b_vals);

  ASSERT_NEAR_COMPLEX(a->dot(b), expected, 1e-6);
}

TYPED_TEST(BackendTest, max_element) {
  constexpr Eigen::Index n = 100;
  auto vals = random_vector(n, -20.0, 2.0);
  std::sort(vals.begin(), vals.end());
  auto v = this->_pool.rent("v", n, 1);
  v->copy_from(vals);

  ASSERT_EQ(v->max_element(), complex(vals[n - 1], 0));
}

TYPED_TEST(BackendTest, max_element_c) {
  constexpr Eigen::Index n = 100;
  auto vals = random_vector(n, 0.0, 10.0);
  std::sort(vals.begin(), vals.end());
  std::vector<complex> vals_c;
  vals_c.reserve(vals.size());
  for (const auto v : vals) vals_c.emplace_back(v, 0.0);
  auto v = this->_pool.rent_c("v", n, 1);
  v->copy_from(vals_c);

  ASSERT_NEAR(v->max_element(), vals[n - 1], 1e-6);
}

TYPED_TEST(BackendTest, concat_row) {
  auto a = this->_pool.rent_c("a", 1, 2);
  auto b = this->_pool.rent_c("b", 1, 2);
  a->copy_from({complex(0, 1), complex(2, 3)});
  b->copy_from({complex(8, 9), complex(10, 11)});

  auto c = this->_pool.rent_c("c", 2, 2);

  c->concat_row(a, b);

  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->at(0, 0), complex(0, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 0), complex(8, 9), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(0, 1), complex(2, 3), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 1), complex(10, 11), 1e-6);
}

TYPED_TEST(BackendTest, concat_col) {
  auto a = this->_pool.rent_c("a", 2, 1);
  auto b = this->_pool.rent_c("b", 2, 1);
  a->copy_from({complex(0, 1), complex(8, 9)});
  b->copy_from({complex(2, 3), complex(10, 11)});

  auto c = this->_pool.rent_c("c", 2, 2);

  c->concat_col(a, b);

  c->copy_to_host();
  ASSERT_NEAR_COMPLEX(c->at(0, 0), complex(0, 1), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 0), complex(8, 9), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(0, 1), complex(2, 3), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 1), complex(10, 11), 1e-6);
}

TYPED_TEST(BackendTest, mat_cpy) {
  auto a = this->_pool.rent("a", 2, 2);
  a->copy_from({0.0, 1.0, 2.0, 3.0});

  auto b = this->_pool.rent("b", 2, 2);

  b->copy_from(a);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), 0.0);
  ASSERT_EQ(b->at(1, 0), 1.0);
  ASSERT_EQ(b->at(0, 1), 2.0);
  ASSERT_EQ(b->at(1, 1), 3.0);
}

TYPED_TEST(BackendTest, mat_cpy_c) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(8, 9)});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->copy_from(a);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), complex(0, 1));
  ASSERT_EQ(b->at(1, 0), complex(8, 9));
}

TYPED_TEST(BackendTest, set) {
  auto a = this->_pool.rent("a", 1, 1);
  a->set(0, 0, 10.0);
  a->copy_to_host();
  ASSERT_EQ(a->at(0, 0), 10.0);
}

TYPED_TEST(BackendTest, set_c) {
  auto a = this->_pool.rent_c("a", 1, 1);
  a->set(0, 0, complex(10.0, 5.0));
  a->copy_to_host();
  ASSERT_EQ(a->at(0, 0), complex(10.0, 5.0));
}

TYPED_TEST(BackendTest, get_col) {
  auto a = this->_pool.rent("a", 2, 2);
  a->copy_from({0.0, 1.0, 2.0, 3.0});

  auto b = this->_pool.rent("b", 2, 1);

  b->get_col(a, 0);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), 0.0);
  ASSERT_EQ(b->at(1, 0), 1.0);
}

TYPED_TEST(BackendTest, get_col_c) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0.0, 1.0), complex(2.0, 3.0), complex(4.0, 5.0), complex(6.0, 71.0)});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->get_col(a, 0);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), complex(0.0, 1.0));
  ASSERT_EQ(b->at(1, 0), complex(2.0, 3.0));
}

TYPED_TEST(BackendTest, fill) {
  auto a = this->_pool.rent("a", 1, 1);
  a->fill(10.0);
  a->copy_to_host();
  ASSERT_EQ(a->at(0, 0), 10.0);
}

TYPED_TEST(BackendTest, fill_c) {
  auto a = this->_pool.rent_c("a", 1, 1);
  a->fill(complex(10.0, 5.0));
  a->copy_to_host();
  ASSERT_EQ(a->at(0, 0), complex(10.0, 5.0));
}

TYPED_TEST(BackendTest, get_diagonal) {
  auto a = this->_pool.rent("a", 2, 2);
  a->copy_from({0.0, 1.0, 2.0, 3.0});

  auto b = this->_pool.rent("b", 2, 1);

  b->get_diagonal(a);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), 0.0);
  ASSERT_EQ(b->at(1, 0), 3.0);
}

TYPED_TEST(BackendTest, get_diagonal_c) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0.0, 1.0), complex(2.0, 3.0), complex(4.0, 5.0), complex(6.0, 7.0)});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->get_diagonal(a);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), complex(0.0, 1.0));
  ASSERT_EQ(b->at(1, 0), complex(6.0, 7.0));
}

TYPED_TEST(BackendTest, create_diagonal) {
  auto a = this->_pool.rent("a", 2, 1);
  a->copy_from({0.0, 1.0});

  auto b = this->_pool.rent("b", 2, 2);

  b->create_diagonal(a);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), 0.0);
  ASSERT_EQ(b->at(1, 0), 0.0);
  ASSERT_EQ(b->at(0, 1), 0.0);
  ASSERT_EQ(b->at(1, 1), 1.0);
}

TYPED_TEST(BackendTest, create_diagonal_c) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0.0, 1.0), complex(2.0, 3.0)});

  auto b = this->_pool.rent_c("b", 2, 2);

  b->create_diagonal(a);

  b->copy_to_host();
  ASSERT_EQ(b->at(0, 0), complex(0.0, 1.0));
  ASSERT_EQ(b->at(1, 0), ZERO);
  ASSERT_EQ(b->at(0, 1), ZERO);
  ASSERT_EQ(b->at(1, 1), complex(2.0, 3.0));
}

TYPED_TEST(BackendTest, set_bcd_result) {
  auto a = this->_pool.rent_c("a", 3, 3);
  auto v = this->_pool.rent_c("v", 3, 1);
  v->copy_from({complex(0.0, 1.0), complex(2.0, 3.0), complex(4.0, 5.0)});

  a->fill(ZERO);

  a->set_bcd_result(v, 1);

  a->copy_to_host();
  ASSERT_EQ(a->at(0, 0), ZERO);
  ASSERT_EQ(a->at(1, 0), complex(0.0, -1.0));
  ASSERT_EQ(a->at(2, 0), ZERO);
  ASSERT_EQ(a->at(0, 1), complex(0.0, 1.0));
  ASSERT_EQ(a->at(1, 1), ZERO);
  ASSERT_EQ(a->at(2, 1), complex(4.0, 5.0));
  ASSERT_EQ(a->at(0, 2), ZERO);
  ASSERT_EQ(a->at(1, 2), complex(4.0, -5.0));
  ASSERT_EQ(a->at(2, 2), ZERO);
}

TYPED_TEST(BackendTest, set_from_complex_drive) {
  const auto n = 249;
  constexpr bool normalize = false;
  auto a = this->_pool.rent_c("a", n, 1);
  std::vector drive = random_vector_complex(n);
  a->copy_from(drive);

  std::vector<autd::core::DataArray> data;
  data.emplace_back(autd::core::DataArray{});

  auto max_coef = a->max_element();
  a->set_from_complex_drive(data, normalize, max_coef);

  for (auto i = 0; i < n; i++) {
    const auto f_amp = normalize ? 1.0 : std::abs(drive[i]) / max_coef;
    const auto f_phase = std::arg(drive[i]) / (2.0 * M_PI);
    const auto phase = autd::core::Utilities::to_phase(f_phase);
    const auto duty = autd::core::Utilities::to_duty(f_amp);
    const auto p = autd::core::Utilities::pack_to_u16(duty, phase);
    ASSERT_EQ(data[0][i], p);
  }
}

TYPED_TEST(BackendTest, set_from_arg) {
  const auto n = 3;
  auto a = this->_pool.rent("a", n, 1);
  std::vector args = {0.0, M_PI, 2.0 * M_PI};
  a->copy_from(args);

  std::vector<autd::core::DataArray> data;
  data.emplace_back(autd::core::DataArray{});

  a->set_from_arg(data, n);

  for (auto i = 0; i < n; i++) {
    const auto f_phase = args[i] / (2 * M_PI);
    const auto phase = autd::core::Utilities::to_phase(f_phase);
    const auto p = autd::core::Utilities::pack_to_u16(0xFF, phase);
    ASSERT_EQ(data[0][i], p);
  }
}

TYPED_TEST(BackendTest, back_prop) {
  const auto m = 2;
  const auto dev = 4;
  const auto n = autd::core::NUM_TRANS_IN_UNIT * dev;

  auto tmp_t = random_vector_complex(m * n, 0.0, 1.0);
  auto tmp_a = random_vector_complex(m, 0.0, 1.0);
  std::vector<complex> expected;

  std::vector<double> denominator;
  for (auto i = 0; i < m; i++) {
    auto tmp = 0.0;
    for (auto j = 0; j < n; j++) tmp += std::abs(tmp_t[i + m * j]);
    denominator.emplace_back(tmp);
  }

  for (auto i = 0; i < m; i++) {
    auto c = tmp_a[i] / denominator[i];
    for (auto j = 0; j < n; j++) expected.emplace_back(c * std::conj(tmp_t[i + m * j]));
  }

  auto t = this->_pool.rent_c("a", m, n);
  t->copy_from(tmp_t);

  auto a = this->_pool.rent_c("b", m, 1);
  a->copy_from(tmp_a);

  auto s = this->_pool.rent_c("s", n, m);
  s->back_prop(t, a);

  s->copy_to_host();

  for (auto i = 0; i < n; i++)
    for (auto j = 0; j < m; j++) ASSERT_NEAR_COMPLEX(s->at(i, j), expected[i + j * n], 1e-6);
}

TYPED_TEST(BackendTest, sigma_regularization) {
  const auto m = 2;
  const auto n = 3;

  double gamma = 1.5;

  auto tmp_t = random_vector_complex(m * n);
  auto tmp_a = random_vector_complex(m);
  std::vector<complex> expected;
  for (auto j = 0; j < n; j++) {
    double tmp = 0;
    for (auto i = 0; i < m; i++) tmp += std::abs(tmp_t[i + j * m] * tmp_a[i]);
    expected.emplace_back(std::pow(std::sqrt(tmp / static_cast<double>(m)), gamma), 0.0);
  }

  auto t = this->_pool.rent_c("a", m, n);
  t->copy_from(tmp_t);

  auto a = this->_pool.rent_c("b", m, 1);
  a->copy_from(tmp_a);

  auto s = this->_pool.rent_c("s", n, n);
  s->sigma_regularization(t, a, gamma);

  s->copy_to_host();

  for (auto j = 0; j < n; j++) {
    for (auto i = 0; i < m; i++)
      if (i == j)
        ASSERT_NEAR_COMPLEX(s->at(i, j), expected[i], 1e-6);
      else
        ASSERT_EQ(s->at(i, j), ZERO);
  }
}

TYPED_TEST(BackendTest, col_sum_imag) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0.0, 1.0), complex(2.0, 3.0), complex(4.0, 5.0), complex(6.0, 7.0)});

  auto b = this->_pool.rent("b", 2, 1);

  b->col_sum_imag(a);

  b->copy_to_host();
  ASSERT_NEAR(b->at(0, 0), 6.0, 1e-6);
  ASSERT_NEAR(b->at(1, 0), 10.0, 1e-6);
}
