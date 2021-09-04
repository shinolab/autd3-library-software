// File: blas_backend.cpp
// Project: src
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/blas_backend.hpp"

#include <iostream>

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#ifdef USE_BLAS_MKL
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

namespace autd::gain::holo {

constexpr auto AUTD_GESVD = LAPACKE_zgesdd;
constexpr auto AUTD_HEEV = LAPACKE_zheev;
constexpr auto AUTD_AXPY = cblas_daxpy;
constexpr auto AUTD_GEMM = cblas_zgemm;
constexpr auto AUTD_GEMV = cblas_zgemv;
constexpr auto AUTD_DOTC = cblas_zdotu_sub;
constexpr auto AUTD_DOT = cblas_ddot;
constexpr auto AUTD_IMAXC = cblas_izamax;
constexpr auto AUTD_SYSV = LAPACKE_dsysv;
constexpr auto AUTD_POSVC = LAPACKE_zposv;
constexpr auto AUTD_CPY = LAPACKE_dlacpy;
constexpr auto AUTD_CPYC = LAPACKE_zlacpy;

BackendPtr BLASBackend::create() { return std::make_shared<BLASBackend>(); }

bool BLASBackend::supports_svd() { return true; }
bool BLASBackend::supports_evd() { return true; }
bool BLASBackend::supports_solve() { return true; }

void BLASBackend::hadamard_product(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) { (*c).noalias() = a.cwiseProduct(b); }
void BLASBackend::real(const MatrixXc& a, MatrixX* b) { (*b).noalias() = a.real(); }
void BLASBackend::pseudo_inverse_svd(const MatrixXc& matrix, const double alpha, MatrixXc* result) {
  const auto nc = matrix.cols();
  const auto nr = matrix.rows();

  const auto lda = static_cast<int>(nr);
  const auto ldu = static_cast<int>(nr);
  const auto ldvt = static_cast<int>(nc);

  const auto s_size = std::min(nr, nc);
  const auto s = std::make_unique<double[]>(s_size);
  auto u = MatrixXc(nr, nr);
  auto vt = MatrixXc(nc, nc);

  MatrixXc m = matrix;
  auto r = AUTD_GESVD(LAPACK_COL_MAJOR, 'A', static_cast<int>(nr), static_cast<int>(nc), m.data(), lda, s.get(), u.data(), ldu, vt.data(), ldvt);
  MatrixXc singular_inv = MatrixXc::Zero(nc, nr);
  for (int i = 0; i < s_size; i++) singular_inv(i, i) = s[i] / (s[i] * s[i] + alpha);

  auto tmp = MatrixXc(nc, nr);
  BLASBackend::matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, std::complex<double>(1, 0), singular_inv, u, std::complex<double>(0, 0), &tmp);
  BLASBackend::matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, std::complex<double>(1, 0), vt, tmp, std::complex<double>(0, 0), result);
}

BLASBackend::VectorXc BLASBackend::max_eigen_vector(MatrixXc* matrix) {
  const auto size = matrix->cols();
  const auto eigenvalues = std::make_unique<double[]>(size);
  AUTD_HEEV(CblasColMajor, 'V', 'U', static_cast<int>(size), matrix->data(), static_cast<int>(size), eigenvalues.get());
  return matrix->col(size - 1);
}

void BLASBackend::matrix_add(const double alpha, const MatrixX& a, MatrixX* b) {
  AUTD_AXPY(static_cast<int>(a.size()), alpha, a.data(), 1, b->data(), 1);
}

void BLASBackend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const std::complex<double> alpha, const MatrixXc& a, const MatrixXc& b,
                             const std::complex<double> beta, MatrixXc* c) {
  const auto lda = static_cast<int>(a.rows());
  const auto ldb = static_cast<int>(b.rows());
  const auto ldc = trans_a == TRANSPOSE::NO_TRANS || trans_a == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(a.rows()) : static_cast<int>(a.cols());
  const auto n = trans_b == TRANSPOSE::NO_TRANS || trans_b == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(b.cols()) : static_cast<int>(b.rows());
  const auto k = trans_a == TRANSPOSE::NO_TRANS || trans_a == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(a.cols()) : static_cast<int>(a.rows());
  AUTD_GEMM(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), static_cast<CBLAS_TRANSPOSE>(trans_b), ldc, n, k, &alpha, a.data(), lda, b.data(),
            ldb, &beta, c->data(), ldc);
}

void BLASBackend::matrix_vector_mul(const TRANSPOSE trans_a, const std::complex<double> alpha, const MatrixXc& a, const VectorXc& b,
                                    const std::complex<double> beta, VectorXc* c) {
  const auto lda = static_cast<int>(a.rows());
  const auto m = static_cast<int>(a.rows());
  const auto n = static_cast<int>(a.cols());
  AUTD_GEMV(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), m, n, &alpha, a.data(), lda, b.data(), 1, &beta, c->data(), 1);
}

void BLASBackend::vector_add(const double alpha, const VectorX& a, VectorX* b) {
  AUTD_AXPY(static_cast<int>(a.size()), alpha, a.data(), 1, b->data(), 1);
}

void BLASBackend::solve_g(MatrixX* a, VectorX* b, VectorX* c) {
  const auto n = static_cast<int>(a->cols());
  const auto lda = static_cast<int>(a->rows());
  const auto ldb = static_cast<int>(b->size());
  std::memcpy(c->data(), b->data(), ldb * sizeof(double));
  const auto ipiv = std::make_unique<int[]>(n);
  AUTD_SYSV(CblasColMajor, 'U', n, 1, a->data(), lda, ipiv.get(), c->data(), ldb);
}
void BLASBackend::solve_ch(MatrixXc* a, VectorXc* b) {
  const auto n = static_cast<int>(a->cols());
  const auto lda = static_cast<int>(a->rows());
  const auto ldb = static_cast<int>(b->size());
  auto ipiv = std::make_unique<int[]>(n);
  AUTD_POSVC(CblasColMajor, 'U', n, 1, a->data(), lda, b->data(), ldb);
}
double BLASBackend::dot(const VectorX& a, const VectorX& b) { return AUTD_DOT(static_cast<int>(a.size()), a.data(), 1, b.data(), 1); }
std::complex<double> BLASBackend::dot_c(const VectorXc& a, const VectorXc& b) {
  std::complex<double> d;
  AUTD_DOTC(static_cast<int>(a.size()), a.data(), 1, b.data(), 1, &d);
  return d;
}
double BLASBackend::max_coefficient_c(const VectorXc& v) {
  const auto idx = AUTD_IMAXC(static_cast<int>(v.size()), v.data(), 1);
  return abs(v(idx));
}
double BLASBackend::max_coefficient(const VectorX& v) { return v.maxCoeff(); }
BLASBackend::MatrixXc BLASBackend::concat_row(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows() + b.rows(), b.cols());
  c << a, b;
  return c;
}
BLASBackend::MatrixXc BLASBackend::concat_col(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows(), a.cols() + b.cols());
  c << a, b;
  return c;
}
void BLASBackend::mat_cpy(const MatrixX& a, MatrixX* b) {
  AUTD_CPY(LAPACK_COL_MAJOR, 'A', static_cast<int>(a.rows()), static_cast<int>(a.cols()), a.data(), static_cast<int>(a.rows()), b->data(),
           static_cast<int>(b->rows()));
}
void BLASBackend::vec_cpy(const VectorX& a, VectorX* b) {
  AUTD_CPY(LAPACK_COL_MAJOR, 'A', static_cast<int>(a.size()), 1, a.data(), static_cast<int>(a.size()), b->data(), 1);
}
void BLASBackend::vec_cpy_c(const VectorXc& a, VectorXc* b) {
  AUTD_CPYC(LAPACK_COL_MAJOR, 'A', static_cast<int>(a.size()), 1, a.data(), static_cast<int>(a.size()), b->data(), 1);
}
}  // namespace autd::gain::holo
