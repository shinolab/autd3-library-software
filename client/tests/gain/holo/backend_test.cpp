// File: eigen_backend_test.cpp
// Project: holo
// Created Date: 13/08/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/12/2021
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
#include "eigen/eigen_matrix.hpp"
#include "matrix_pool.hpp"
#include "test_utils.hpp"

using autd::gain::holo::complex;
using autd::gain::holo::ONE;
using autd::gain::holo::TRANSPOSE;
using autd::gain::holo::ZERO;

template <typename P>
class BackendTest : public testing::Test {
 public:
  BackendTest() : _pool() {}
  ~BackendTest() override {}
  BackendTest(const BackendTest& v) noexcept = default;
  BackendTest& operator=(const BackendTest& obj) = default;
  BackendTest(BackendTest&& obj) = default;
  BackendTest& operator=(BackendTest&& obj) = default;

 protected:
  P _pool;
};

using testing::Types;

#define Eigen3BackendType autd::gain::holo::MatrixBufferPool<autd::gain::holo::EigenMatrix<double>, autd::gain::holo::EigenMatrix<complex>>

#ifdef TEST_BLAS_BACKEND
#include "blas/blas_matrix.hpp"
#define BLASBackendType , autd::gain::holo::MatrixBufferPool<autd::gain::holo::BLASMatrix<double>, autd::gain::holo::BLASMatrix<complex>>
#else
#define BLASBackendType
#endif

#ifdef TEST_CUDA_BACKEND
#include "cuda/cuda_matrix.hpp"
#define CUDABackendType , autd::gain::holo::MatrixBufferPool<autd::gain::holo::CuMatrix<double>, autd::gain::holo::CuMatrix<complex>>
template <>
BackendTest<autd::gain::holo::MatrixBufferPool<autd::gain::holo::CuMatrix<double>, autd::gain::holo::CuMatrix<complex>>>::BackendTest() {
  autd::gain::holo::CuContext::init(0);
}
template <>
BackendTest<autd::gain::holo::MatrixBufferPool<autd::gain::holo::CuMatrix<double>, autd::gain::holo::CuMatrix<complex>>>::~BackendTest() {
  autd::gain::holo::CuContext::free();
}
#else
#define CUDABackendType
#endif

#ifdef TEST_ARRAYFIRE_BACKEND
#include "array_fire/arrayfire_matrix.hpp"
#define ArrayFireBackendType , autd::gain::holo::MatrixBufferPool<autd::gain::holo::AFMatrix<double>, autd::gain::holo::AFMatrix<complex>>
#else
#define ArrayFireBackendType
#endif

typedef Types<Eigen3BackendType BLASBackendType CUDABackendType ArrayFireBackendType> Implementations;

TYPED_TEST_SUITE(BackendTest, Implementations, );

template <typename T>
std::vector<double> random_vector(T n, const double minimum = -1.0, const double maximum = 1.0) {
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution dist(minimum, maximum);
  std::vector<double> v;
  v.reserve(n);
  for (T i = 0; i < n; ++i) v.emplace_back(dist(engine));
  return v;
}

template <typename T>
std::vector<complex> random_vector_complex(T n, const double minimum = -1.0, const double maximum = 1.0) {
  const auto re = random_vector(n, minimum, maximum);
  const auto im = random_vector(n, minimum, maximum);
  std::vector<complex> v;
  v.reserve(n);
  for (T i = 0; i < n; ++i) v.emplace_back(complex(re[i], im[i]));
  return v;
}

TYPED_TEST(BackendTest, make_complex) {
  auto r = this->_pool.rent("r", 2, 1);
  auto i = this->_pool.rent("i", 2, 1);

  r->copy_from({0.0, 1.0});
  i->copy_from({2.0, 3.0});

  auto a = this->_pool.rent_c("a", 2, 1);

  a->make_complex(r, i);

  ASSERT_NEAR_COMPLEX(a->at(0, 0), complex(0, 2), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(1, 0), complex(1, 3), 1e-6);
}

TYPED_TEST(BackendTest, exp) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  a->exp();

  ASSERT_NEAR_COMPLEX(a->at(0, 0), std::exp(complex(0, 1)), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(1, 0), std::exp(complex(2, 3)), 1e-6);
}

TYPED_TEST(BackendTest, pow) {
  auto a = this->_pool.rent("a", 4, 1);
  a->copy_from({0, 1, 2, 3});

  a->pow(0.5);

  ASSERT_NEAR(a->at(0, 0), std::pow(0, 0.5), 1e-6);
  ASSERT_NEAR(a->at(1, 0), std::pow(1, 0.5), 1e-6);
  ASSERT_NEAR(a->at(2, 0), std::pow(2, 0.5), 1e-6);
  ASSERT_NEAR(a->at(3, 0), std::pow(3, 0.5), 1e-6);
}

TYPED_TEST(BackendTest, pow_c) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  a->pow(0.5);

  ASSERT_NEAR_COMPLEX(a->at(0, 0), std::pow(complex(0, 1), 0.5), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(1, 0), std::pow(complex(2, 3), 0.5), 1e-6);
}

TYPED_TEST(BackendTest, sqrt) {
  auto a = this->_pool.rent("a", 4, 1);
  a->copy_from({0, 1, 2, 3});

  a->sqrt();

  ASSERT_NEAR(a->at(0, 0), std::sqrt(0), 1e-6);
  ASSERT_NEAR(a->at(1, 0), std::sqrt(1), 1e-6);
  ASSERT_NEAR(a->at(2, 0), std::sqrt(2), 1e-6);
  ASSERT_NEAR(a->at(3, 0), std::sqrt(3), 1e-6);
}

TYPED_TEST(BackendTest, sqrt_c) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  a->sqrt();

  ASSERT_NEAR_COMPLEX(a->at(0, 0), std::sqrt(complex(0, 1)), 1e-6);
  ASSERT_NEAR_COMPLEX(a->at(1, 0), std::sqrt(complex(2, 3)), 1e-6);
}

TYPED_TEST(BackendTest, scale) {
  auto a = this->_pool.rent_c("a", 4, 1);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});

  a->scale(complex(1, 1));

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

  ASSERT_NEAR(b->at(0, 0), 1.0 / 1.0, 1e-6);
  ASSERT_NEAR(b->at(1, 0), 1.0 / 2.0, 1e-6);
}

TYPED_TEST(BackendTest, reciprocal_c) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  auto b = this->_pool.rent_c("b", 2, 1);
  b->reciprocal(a);

  ASSERT_NEAR_COMPLEX(b->at(0, 0), complex(0, -1), 1e-6);
  ASSERT_NEAR_COMPLEX(b->at(1, 0), complex(2.0 / 13.0, -3.0 / 13.0), 1e-6);
}

TYPED_TEST(BackendTest, abs_c) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({complex(0, 1), complex(2, 3)});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->abs(a);

  ASSERT_NEAR(b->at(0, 0).real(), std::abs(complex(0, 1)), 1e-6);
  ASSERT_NEAR(b->at(1, 0).real(), std::abs(complex(2, 3)), 1e-6);
}

TYPED_TEST(BackendTest, real) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});

  auto b = this->_pool.rent("b", 2, 2);

  b->real(a);

  ASSERT_EQ(b->at(0, 0), 0.0);
  ASSERT_EQ(b->at(1, 0), 2.0);
  ASSERT_EQ(b->at(0, 1), 4.0);
  ASSERT_EQ(b->at(1, 1), 6.0);
}

TYPED_TEST(BackendTest, imag) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});

  auto b = this->_pool.rent("b", 2, 2);

  b->imag(a);

  ASSERT_EQ(b->at(0, 0), 1.0);
  ASSERT_EQ(b->at(1, 0), 3.0);
  ASSERT_EQ(b->at(0, 1), 5.0);
  ASSERT_EQ(b->at(1, 1), 7.0);
}

TYPED_TEST(BackendTest, conj) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0, 1), complex(2, 3), complex(4, 5), complex(6, 7)});

  auto b = this->_pool.rent_c("b", 2, 2);

  b->conj(a);

  ASSERT_EQ(b->at(0, 0), complex(0, -1));
  ASSERT_EQ(b->at(1, 0), complex(2, -3));
  ASSERT_EQ(b->at(0, 1), complex(4, -5));
  ASSERT_EQ(b->at(1, 1), complex(6, -7));
}

TYPED_TEST(BackendTest, arg) {
  auto a = this->_pool.rent_c("a", 2, 1);
  a->copy_from({std::exp(complex(0, 1)) * 2.0, std::exp(complex(0, 2)) * 4.0});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->arg(a);

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

  Eigen::MatrixXf::Index max_idx;
  u.col(n - 1).cwiseAbs2().maxCoeff(&max_idx);
  const auto k = b->at(max_idx, 0) / u.col(n - 1)(max_idx);
  const Eigen::Matrix<complex, -1, 1, Eigen::ColMajor> expected = u.col(n - 1) * k;

  for (Eigen::Index i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(b->at(i, 0), expected(i), 1e-6);
}

TYPED_TEST(BackendTest, matrix_add) {
  auto a = this->_pool.rent("a", 2, 2);
  a->copy_from({0.0, 2.0, 1.0, 3.0});

  auto b = this->_pool.rent("b", 2, 2);
  b->fill(0.0);

  b->add(0.0, a);

  ASSERT_NEAR(b->at(0, 0), 0.0, 1e-6);
  ASSERT_NEAR(b->at(0, 1), 0.0, 1e-6);
  ASSERT_NEAR(b->at(1, 0), 0.0, 1e-6);
  ASSERT_NEAR(b->at(1, 1), 0.0, 1e-6);

  b->add(2.0, a);

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

  ASSERT_NEAR_COMPLEX(c->at(0, 0), complex(-24, 70), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(0, 1), complex(-28, 82), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 0), complex(-32, 238), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 1), complex(-36, 282), 1e-6);

  c->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::TRANS, ZERO, a, b, ONE);

  ASSERT_NEAR_COMPLEX(c->at(0, 0), complex(-24, 70), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(0, 1), complex(-28, 82), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 0), complex(-32, 238), 1e-6);
  ASSERT_NEAR_COMPLEX(c->at(1, 1), complex(-36, 282), 1e-6);
}

TYPED_TEST(BackendTest, matrix_mul) {
  auto a = this->_pool.rent("a", 2, 2);
  auto b = this->_pool.rent("b", 2, 2);
  a->copy_from({0.0, 2.0, 1.0, 3.0});
  b->copy_from({4.0, 6.0, 5.0, 7.0});

  auto c = this->_pool.rent("c", 2, 2);

  c->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::TRANS, 1.0, a, b, 0.0);

  ASSERT_NEAR(c->at(0, 0), 5.0, 1e-6);
  ASSERT_NEAR(c->at(0, 1), 7.0, 1e-6);
  ASSERT_NEAR(c->at(1, 0), 23.0, 1e-6);
  ASSERT_NEAR(c->at(1, 1), 33.0, 1e-6);

  c->mul(TRANSPOSE::TRANS, TRANSPOSE::NO_TRANS, 1.0, a, b, 1);

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

  ASSERT_EQ(b->at(0, 0), complex(0, 1));
  ASSERT_EQ(b->at(1, 0), complex(8, 9));
}

TYPED_TEST(BackendTest, set) {
  auto a = this->_pool.rent("a", 1, 1);
  a->set(0, 0, 10.0);

  ASSERT_EQ(a->at(0, 0), 10.0);
}

TYPED_TEST(BackendTest, set_c) {
  auto a = this->_pool.rent_c("a", 1, 1);
  a->set(0, 0, complex(10.0, 5.0));

  ASSERT_EQ(a->at(0, 0), complex(10.0, 5.0));
}

TYPED_TEST(BackendTest, set_col) {
  constexpr size_t n = 10;
  auto a = this->_pool.rent("a", n, n);
  a->fill(0);

  auto b = this->_pool.rent("b", n, 1);
  auto vals = random_vector(n);
  b->copy_from(vals);

  a->set_col(3, 1, 6, b);
  ASSERT_EQ(a->at(0, 3), 0.0);
  ASSERT_EQ(a->at(1, 3), vals[1]);
  ASSERT_EQ(a->at(2, 3), vals[2]);
  ASSERT_EQ(a->at(3, 3), vals[3]);
  ASSERT_EQ(a->at(4, 3), vals[4]);
  ASSERT_EQ(a->at(5, 3), vals[5]);
  ASSERT_EQ(a->at(6, 3), 0.0);
  ASSERT_EQ(a->at(7, 3), 0.0);
  ASSERT_EQ(a->at(8, 3), 0.0);
  ASSERT_EQ(a->at(9, 3), 0.0);
}

TYPED_TEST(BackendTest, set_col_c) {
  constexpr size_t n = 10;
  auto a = this->_pool.rent_c("a", n, n);
  a->fill(0);

  auto b = this->_pool.rent_c("b", n, 1);
  auto vals = random_vector_complex(n);
  b->copy_from(vals);

  a->set_col(9, 7, 10, b);
  ASSERT_EQ(a->at(0, 9), ZERO);
  ASSERT_EQ(a->at(1, 9), ZERO);
  ASSERT_EQ(a->at(2, 9), ZERO);
  ASSERT_EQ(a->at(3, 9), ZERO);
  ASSERT_EQ(a->at(4, 9), ZERO);
  ASSERT_EQ(a->at(5, 9), ZERO);
  ASSERT_EQ(a->at(6, 9), ZERO);
  ASSERT_EQ(a->at(7, 9), vals[7]);
  ASSERT_EQ(a->at(8, 9), vals[8]);
  ASSERT_EQ(a->at(9, 9), vals[9]);
}

TYPED_TEST(BackendTest, set_row) {
  constexpr size_t n = 10;
  auto a = this->_pool.rent("a", n, n);
  a->fill(0);

  auto b = this->_pool.rent("b", n, 1);
  auto vals = random_vector(n);
  b->copy_from(vals);

  a->set_row(7, 1, 6, b);
  ASSERT_EQ(a->at(7, 0), 0.0);
  ASSERT_EQ(a->at(7, 1), vals[1]);
  ASSERT_EQ(a->at(7, 2), vals[2]);
  ASSERT_EQ(a->at(7, 3), vals[3]);
  ASSERT_EQ(a->at(7, 4), vals[4]);
  ASSERT_EQ(a->at(7, 5), vals[5]);
  ASSERT_EQ(a->at(7, 6), 0.0);
  ASSERT_EQ(a->at(7, 7), 0.0);
  ASSERT_EQ(a->at(7, 8), 0.0);
  ASSERT_EQ(a->at(7, 9), 0.0);
}

TYPED_TEST(BackendTest, set_row_c) {
  constexpr size_t n = 10;
  auto a = this->_pool.rent_c("a", n, n);
  a->fill(0);

  auto b = this->_pool.rent_c("b", n, 1);
  auto vals = random_vector_complex(n);
  b->copy_from(vals);

  a->set_row(0, 7, 10, b);
  ASSERT_EQ(a->at(0, 0), ZERO);
  ASSERT_EQ(a->at(0, 1), ZERO);
  ASSERT_EQ(a->at(0, 2), ZERO);
  ASSERT_EQ(a->at(0, 3), ZERO);
  ASSERT_EQ(a->at(0, 4), ZERO);
  ASSERT_EQ(a->at(0, 5), ZERO);
  ASSERT_EQ(a->at(0, 6), ZERO);
  ASSERT_EQ(a->at(0, 7), vals[7]);
  ASSERT_EQ(a->at(0, 8), vals[8]);
  ASSERT_EQ(a->at(0, 9), vals[9]);
}
TYPED_TEST(BackendTest, get_col) {
  auto a = this->_pool.rent("a", 2, 2);
  a->copy_from({0.0, 1.0, 2.0, 3.0});

  auto b = this->_pool.rent("b", 2, 1);

  b->get_col(a, 0);

  ASSERT_EQ(b->at(0, 0), 0.0);
  ASSERT_EQ(b->at(1, 0), 1.0);
}

TYPED_TEST(BackendTest, get_col_c) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0.0, 1.0), complex(2.0, 3.0), complex(4.0, 5.0), complex(6.0, 71.0)});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->get_col(a, 0);

  ASSERT_EQ(b->at(0, 0), complex(0.0, 1.0));
  ASSERT_EQ(b->at(1, 0), complex(2.0, 3.0));
}

TYPED_TEST(BackendTest, fill) {
  auto a = this->_pool.rent("a", 1, 1);
  a->fill(10.0);

  ASSERT_EQ(a->at(0, 0), 10.0);
}

TYPED_TEST(BackendTest, fill_c) {
  auto a = this->_pool.rent_c("a", 1, 1);
  a->fill(complex(10.0, 5.0));

  ASSERT_EQ(a->at(0, 0), complex(10.0, 5.0));
}

TYPED_TEST(BackendTest, get_diagonal) {
  auto a = this->_pool.rent("a", 2, 2);
  a->copy_from({0.0, 1.0, 2.0, 3.0});

  auto b = this->_pool.rent("b", 2, 1);

  b->get_diagonal(a);

  ASSERT_EQ(b->at(0, 0), 0.0);
  ASSERT_EQ(b->at(1, 0), 3.0);
}

TYPED_TEST(BackendTest, get_diagonal_c) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0.0, 1.0), complex(2.0, 3.0), complex(4.0, 5.0), complex(6.0, 7.0)});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->get_diagonal(a);

  ASSERT_EQ(b->at(0, 0), complex(0.0, 1.0));
  ASSERT_EQ(b->at(1, 0), complex(6.0, 7.0));
}

TYPED_TEST(BackendTest, create_diagonal) {
  auto a = this->_pool.rent("a", 2, 1);
  a->copy_from({0.0, 1.0});

  auto b = this->_pool.rent("b", 2, 2);

  b->create_diagonal(a);

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

  ASSERT_EQ(b->at(0, 0), complex(0.0, 1.0));
  ASSERT_EQ(b->at(1, 0), ZERO);
  ASSERT_EQ(b->at(0, 1), ZERO);
  ASSERT_EQ(b->at(1, 1), complex(2.0, 3.0));
}

TYPED_TEST(BackendTest, set_from_complex_drive) {
  constexpr size_t dev = 4;
  constexpr auto n = dev * autd::core::NUM_TRANS_IN_UNIT;
  constexpr bool normalize = false;
  auto a = this->_pool.rent_c("a", n, 1);
  std::vector drive = random_vector_complex(n, 0.0, 1.0);
  a->copy_from(drive);

  std::vector<autd::core::Drive> data;
  for (size_t d = 0; d < n; d++) data.emplace_back(autd::core::Drive{});

  auto max_coef = a->max_element();
  a->set_from_complex_drive(data, normalize, max_coef);

  for (size_t i = 0; i < n; i++) {
    const auto f_amp = normalize ? 1.0 : std::abs(drive[i]) / max_coef;
    ASSERT_EQ(data[i].duty, autd::core::utils::to_duty(f_amp));
    ASSERT_EQ(data[i].phase, autd::core::utils::to_phase(std::arg(drive[i])));
  }
}

TYPED_TEST(BackendTest, set_from_arg) {
  constexpr auto n = 3;
  auto a = this->_pool.rent("a", n, 1);
  std::vector args = {0.0, M_PI, 2.0 * M_PI};
  a->copy_from(args);

  std::vector<autd::core::Drive> data;
  for (auto i = 0; i < n; i++) data.emplace_back(autd::core::Drive{});

  a->set_from_arg(data, n);

  for (auto i = 0; i < n; i++) {
    ASSERT_EQ(data[i].duty, 0xFF);
    ASSERT_EQ(data[i].phase, autd::core::utils::to_phase(args[i]));
  }
}

TYPED_TEST(BackendTest, reduce_col) {
  auto a = this->_pool.rent("a", 2, 2);
  a->copy_from({0.0, 1.0, 2.0, 3.0});

  auto b = this->_pool.rent("b", 2, 1);

  b->reduce_col(a);

  ASSERT_NEAR(b->at(0, 0), 2.0, 1e-6);
  ASSERT_NEAR(b->at(1, 0), 4.0, 1e-6);
}

TYPED_TEST(BackendTest, reduce_col_c) {
  auto a = this->_pool.rent_c("a", 2, 2);
  a->copy_from({complex(0.0, 1.0), complex(2.0, 3.0), complex(4.0, 5.0), complex(6.0, 7.0)});

  auto b = this->_pool.rent_c("b", 2, 1);

  b->reduce_col(a);

  ASSERT_NEAR_COMPLEX(b->at(0, 0), complex(4.0, 6.0), 1e-6);
  ASSERT_NEAR_COMPLEX(b->at(1, 0), complex(8.0, 10.0), 1e-6);
}
