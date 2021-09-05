// File: blas_backend.cpp
// Project: src
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/blas_backend.hpp"

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#ifdef USE_BLAS_MKL
#include "./mkl_cblas.h"
#include "./mkl_lapacke.h"
#else
#include "./cblas.h"
#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4190)
#endif
#include "./lapacke.h"
#if _MSC_VER
#pragma warning(pop)
#endif
#endif

namespace autd::gain::holo {

constexpr auto AUTD_GESVD = LAPACKE_zgesdd;
constexpr auto AUTD_HEEV = LAPACKE_zheev;
constexpr auto AUTD_AXPY = cblas_daxpy;
constexpr auto AUTD_DGEMM = cblas_dgemm;
constexpr auto AUTD_ZGEMM = cblas_zgemm;
constexpr auto AUTD_DGEMV = cblas_dgemv;
constexpr auto AUTD_ZGEMV = cblas_zgemv;
constexpr auto AUTD_DOTC = cblas_zdotu_sub;
constexpr auto AUTD_DOT = cblas_ddot;
constexpr auto AUTD_IMAX = cblas_idamax;
constexpr auto AUTD_IMAXC = cblas_izamax;
constexpr auto AUTD_SYSV = LAPACKE_dsysv;
constexpr auto AUTD_POSVC = LAPACKE_zposv;
constexpr auto AUTD_CPY = LAPACKE_dlacpy;
constexpr auto AUTD_CPYC = LAPACKE_zlacpy;

BackendPtr BLASBackend::create() { return std::make_shared<BLASBackend>(); }

void BLASBackend::pseudo_inverse_svd(const std::shared_ptr<MatrixXc> matrix, const double alpha, std::shared_ptr<MatrixXc> result) {
  const auto nc = matrix->data.cols();
  const auto nr = matrix->data.rows();

  const auto lda = static_cast<int>(nr);
  const auto ldu = static_cast<int>(nr);
  const auto ldvt = static_cast<int>(nc);

  const auto s_size = std::min(nr, nc);
  const auto s = std::make_unique<double[]>(s_size);
  auto u = this->allocate_matrix_c("_pis_u", nr, nr);
  auto vt = this->allocate_matrix_c("_pis_vt", nc, nc);

  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> m = matrix->data;
  auto r = AUTD_GESVD(LAPACK_COL_MAJOR, 'A', static_cast<int>(nr), static_cast<int>(nc), m.data(), lda, s.get(), u->data.data(), ldu, vt->data.data(),
                      ldvt);
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> singular_inv = Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>::Zero(nc, nr);
  for (int i = 0; i < s_size; i++) singular_inv(i, i) = s[i] / (s[i] * s[i] + alpha);

  auto si = std::make_shared<MatrixXc>(singular_inv);
  auto tmp = this->allocate_matrix_c("_pis_tmp", nc, nr);
  BLASBackend::matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, One, si, u, Zero, tmp);
  BLASBackend::matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, One, vt, tmp, Zero, result);
}

std::shared_ptr<VectorXc> BLASBackend::max_eigen_vector(std::shared_ptr<MatrixXc> matrix) {
  const auto size = matrix->data.cols();
  const auto eigenvalues = std::make_unique<double[]>(size);
  AUTD_HEEV(CblasColMajor, 'V', 'U', static_cast<int>(size), matrix->data.data(), static_cast<int>(size), eigenvalues.get());
  return std::make_shared<VectorXc>(matrix->data.col(size - 1));
}

void BLASBackend::matrix_add(const double alpha, const std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) {
  AUTD_AXPY(static_cast<int>(a->data.size()), alpha, a->data.data(), 1, b->data.data(), 1);
}

void BLASBackend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const complex alpha, const std::shared_ptr<MatrixXc> a,
                             const std::shared_ptr<MatrixXc> b, const complex beta, std::shared_ptr<MatrixXc> c) {
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.rows());
  const auto ldc =
      trans_a == TRANSPOSE::NO_TRANS || trans_a == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(a->data.rows()) : static_cast<int>(a->data.cols());
  const auto n =
      trans_b == TRANSPOSE::NO_TRANS || trans_b == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(b->data.cols()) : static_cast<int>(b->data.rows());
  const auto k =
      trans_a == TRANSPOSE::NO_TRANS || trans_a == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(a->data.cols()) : static_cast<int>(a->data.rows());
  AUTD_ZGEMM(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), static_cast<CBLAS_TRANSPOSE>(trans_b), ldc, n, k, &alpha, a->data.data(), lda,
             b->data.data(), ldb, &beta, c->data.data(), ldc);
}

void BLASBackend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const double alpha, const std::shared_ptr<MatrixX> a,
                             const std::shared_ptr<MatrixX> b, const double beta, std::shared_ptr<MatrixX> c) {
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.rows());
  const auto ldc =
      trans_a == TRANSPOSE::NO_TRANS || trans_a == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(a->data.rows()) : static_cast<int>(a->data.cols());
  const auto n =
      trans_b == TRANSPOSE::NO_TRANS || trans_b == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(b->data.cols()) : static_cast<int>(b->data.rows());
  const auto k =
      trans_a == TRANSPOSE::NO_TRANS || trans_a == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(a->data.cols()) : static_cast<int>(a->data.rows());
  AUTD_DGEMM(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), static_cast<CBLAS_TRANSPOSE>(trans_b), ldc, n, k, alpha, a->data.data(), lda,
             b->data.data(), ldb, beta, c->data.data(), ldc);
}

void BLASBackend::matrix_vector_mul(const TRANSPOSE trans_a, const complex alpha, const std::shared_ptr<MatrixXc> a,
                                    const std::shared_ptr<VectorXc> b, const complex beta, std::shared_ptr<VectorXc> c) {
  const auto lda = static_cast<int>(a->data.rows());
  const auto m = static_cast<int>(a->data.rows());
  const auto n = static_cast<int>(a->data.cols());
  AUTD_ZGEMV(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), m, n, &alpha, a->data.data(), lda, b->data.data(), 1, &beta, c->data.data(), 1);
}

void BLASBackend::matrix_vector_mul(const TRANSPOSE trans_a, const double alpha, const std::shared_ptr<MatrixX> a, const std::shared_ptr<VectorX> b,
                                    const double beta, std::shared_ptr<VectorX> c) {
  const auto lda = static_cast<int>(a->data.rows());
  const auto m = static_cast<int>(a->data.rows());
  const auto n = static_cast<int>(a->data.cols());
  AUTD_DGEMV(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), m, n, alpha, a->data.data(), lda, b->data.data(), 1, beta, c->data.data(), 1);
}

void BLASBackend::vector_add(const double alpha, const std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) {
  AUTD_AXPY(static_cast<int>(a->data.size()), alpha, a->data.data(), 1, b->data.data(), 1);
}

void BLASBackend::solve_g(std::shared_ptr<MatrixX> a, std::shared_ptr<VectorX> b, std::shared_ptr<VectorX> c) {
  const auto n = static_cast<int>(a->data.cols());
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.size());
  std::memcpy(c->data.data(), b->data.data(), ldb * sizeof(double));
  const auto ipiv = std::make_unique<int[]>(n);
  AUTD_SYSV(CblasColMajor, 'U', n, 1, a->data.data(), lda, ipiv.get(), c->data.data(), ldb);
}
void BLASBackend::solve_ch(std::shared_ptr<MatrixXc> a, std::shared_ptr<VectorXc> b) {
  const auto n = static_cast<int>(a->data.cols());
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.size());
  auto ipiv = std::make_unique<int[]>(n);
  AUTD_POSVC(CblasColMajor, 'U', n, 1, a->data.data(), lda, b->data.data(), ldb);
}
double BLASBackend::dot(const std::shared_ptr<VectorX> a, const std::shared_ptr<VectorX> b) {
  return AUTD_DOT(static_cast<int>(a->data.size()), a->data.data(), 1, b->data.data(), 1);
}
complex BLASBackend::dot(const std::shared_ptr<VectorXc> a, const std::shared_ptr<VectorXc> b) {
  complex d;
  AUTD_DOTC(static_cast<int>(a->data.size()), a->data.data(), 1, b->data.data(), 1, &d);
  return d;
}
double BLASBackend::max_coefficient(const std::shared_ptr<VectorXc> v) {
  const auto idx = AUTD_IMAXC(static_cast<int>(v->data.size()), v->data.data(), 1);
  return abs(v->data(idx));
}
double BLASBackend::max_coefficient(const std::shared_ptr<VectorX> v) {
  const auto idx = AUTD_IMAX(static_cast<int>(v->data.size()), v->data.data(), 1);
  return v->data(idx);
}

void BLASBackend::mat_cpy(const std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) {
  AUTD_CPY(LAPACK_COL_MAJOR, 'A', static_cast<int>(a->data.rows()), static_cast<int>(a->data.cols()), a->data.data(),
           static_cast<int>(a->data.rows()), b->data.data(), static_cast<int>(b->data.rows()));
}
void BLASBackend::vec_cpy(const std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) {
  AUTD_CPY(LAPACK_COL_MAJOR, 'A', static_cast<int>(a->data.size()), 1, a->data.data(), static_cast<int>(a->data.size()), b->data.data(), 1);
}
void BLASBackend::vec_cpy(const std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b) {
  AUTD_CPYC(LAPACK_COL_MAJOR, 'A', static_cast<int>(a->data.size()), 1, a->data.data(), static_cast<int>(a->data.size()), b->data.data(), 1);
}
}  // namespace autd::gain::holo
