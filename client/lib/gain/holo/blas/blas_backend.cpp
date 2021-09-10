// File: blas_backend.cpp
// Project: src
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
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

constexpr auto AUTD_GESVD = LAPACKE_dgesdd;
constexpr auto AUTD_GESVDC = LAPACKE_zgesdd;
constexpr auto AUTD_HEEV = LAPACKE_zheev;
constexpr auto AUTD_ZSCAL = cblas_zscal;
constexpr auto AUTD_AXPY = cblas_daxpy;
constexpr auto AUTD_AXPYC = cblas_zaxpy;
constexpr auto AUTD_DGEMM = cblas_dgemm;
constexpr auto AUTD_ZGEMM = cblas_zgemm;
constexpr auto AUTD_DOTC = cblas_zdotc_sub;
constexpr auto AUTD_DOT = cblas_ddot;
constexpr auto AUTD_IMAXC = cblas_izamax;
constexpr auto AUTD_SYSV = LAPACKE_dsysv;
constexpr auto AUTD_POSVC = LAPACKE_zposv;
constexpr auto AUTD_CPY = cblas_dcopy;
constexpr auto AUTD_CPYC = cblas_zcopy;

void BLASMatrix<complex>::pseudo_inverse_svd(const std::shared_ptr<EigenMatrix<complex>>& matrix, const double alpha,
                                             const std::shared_ptr<EigenMatrix<complex>>& u, const std::shared_ptr<EigenMatrix<complex>>& s,
                                             const std::shared_ptr<EigenMatrix<complex>>& vt, const std::shared_ptr<EigenMatrix<complex>>& buf) {
  const auto nc = matrix->cols();
  const auto nr = matrix->rows();

  const auto lda = static_cast<int>(nr);
  const auto ldu = static_cast<int>(nr);
  const auto ldvt = static_cast<int>(nc);

  const auto s_size = std::min(nr, nc);
  const auto sigma = std::make_unique<double[]>(s_size);

  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> m = matrix->data;
  AUTD_GESVDC(LAPACK_COL_MAJOR, 'A', static_cast<int>(nr), static_cast<int>(nc), m.data(), lda, sigma.get(), u->data.data(), ldu, vt->data.data(),
              ldvt);
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> singular_inv =
      Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>::Zero(static_cast<Eigen::Index>(nc), static_cast<Eigen::Index>(nr));
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(s_size); i++) singular_inv(i, i) = sigma[i] / (sigma[i] * sigma[i] + alpha);

  s->copy_from(singular_inv.data());
  buf->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, s, u, ZERO);
  this->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, vt, buf, ZERO);
}

void BLASMatrix<double>::pseudo_inverse_svd(const std::shared_ptr<EigenMatrix<double>>& matrix, const double alpha,
                                            const std::shared_ptr<EigenMatrix<double>>& u, const std::shared_ptr<EigenMatrix<double>>& s,
                                            const std::shared_ptr<EigenMatrix<double>>& vt, const std::shared_ptr<EigenMatrix<double>>& buf) {
  const auto nc = matrix->data.cols();
  const auto nr = matrix->data.rows();

  const auto lda = static_cast<int>(nr);
  const auto ldu = static_cast<int>(nr);
  const auto ldvt = static_cast<int>(nc);

  const auto s_size = std::min(nr, nc);
  const auto sigma = std::make_unique<double[]>(s_size);

  Eigen::Matrix<double, -1, -1, Eigen::ColMajor> m = matrix->data;
  AUTD_GESVD(LAPACK_COL_MAJOR, 'A', static_cast<int>(nr), static_cast<int>(nc), m.data(), lda, sigma.get(), u->data.data(), ldu, vt->data.data(),
             ldvt);
  Eigen::Matrix<double, -1, -1, Eigen::ColMajor> singular_inv = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>::Zero(nc, nr);
  for (int i = 0; i < s_size; i++) singular_inv(i, i) = sigma[i] / (sigma[i] * sigma[i] + alpha);

  s->copy_from(singular_inv.data());
  buf->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::TRANS, 1.0, s, u, 0.0);
  this->mul(TRANSPOSE::TRANS, TRANSPOSE::NO_TRANS, 1.0, vt, buf, 0.0);
}
void BLASMatrix<complex>::max_eigen_vector(const std::shared_ptr<EigenMatrix<complex>>& ev) {
  const auto size = data.cols();
  const auto eigenvalues = std::make_unique<double[]>(size);
  AUTD_HEEV(CblasColMajor, 'V', 'U', static_cast<int>(size), data.data(), static_cast<int>(size), eigenvalues.get());
  std::memcpy(ev->data.data(), data.data() + size * (size - 1), size * sizeof(complex));
}
void BLASMatrix<double>::max_eigen_vector(const std::shared_ptr<EigenMatrix<double>>& ev) {}

void BLASMatrix<double>::add(const double alpha, const std::shared_ptr<EigenMatrix<double>>& a) {
  AUTD_AXPY(static_cast<int>(a->data.size()), alpha, a->data.data(), 1, data.data(), 1);
}
void BLASMatrix<complex>::add(const complex alpha, const std::shared_ptr<EigenMatrix<complex>>& a) {
  AUTD_AXPYC(static_cast<int>(a->data.size()), &alpha, a->data.data(), 1, data.data(), 1);
}
template <>
void BLASMatrix<double>::mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const double alpha,
                             const std::shared_ptr<const EigenMatrix<double>>& a, const std::shared_ptr<const EigenMatrix<double>>& b,
                             const double beta) {
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.rows());
  const auto ldc = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->data.rows()) : static_cast<int>(a->data.cols());
  const auto n = trans_b == TRANSPOSE::NO_TRANS ? static_cast<int>(b->data.cols()) : static_cast<int>(b->data.rows());
  const auto k = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->data.cols()) : static_cast<int>(a->data.rows());
  AUTD_DGEMM(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), static_cast<CBLAS_TRANSPOSE>(trans_b), ldc, n, k, alpha, a->data.data(), lda,
             b->data.data(), ldb, beta, data.data(), ldc);
}
template <>
void BLASMatrix<complex>::mul(TRANSPOSE trans_a, TRANSPOSE trans_b, const complex alpha, const std::shared_ptr<const EigenMatrix<complex>>& a,
                              const std::shared_ptr<const EigenMatrix<complex>>& b, const complex beta) {
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.rows());
  const auto ldc = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->data.rows()) : static_cast<int>(a->data.cols());
  const auto n = trans_b == TRANSPOSE::NO_TRANS ? static_cast<int>(b->data.cols()) : static_cast<int>(b->data.rows());
  const auto k = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->data.cols()) : static_cast<int>(a->data.rows());
  AUTD_ZGEMM(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), static_cast<CBLAS_TRANSPOSE>(trans_b), ldc, n, k, &alpha, a->data.data(), lda,
             b->data.data(), ldb, &beta, data.data(), ldc);
}

void BLASMatrix<double>::solve(const std::shared_ptr<EigenMatrix<double>>& b) {
  const auto n = static_cast<int>(data.cols());
  const auto lda = static_cast<int>(data.rows());
  const auto ldb = static_cast<int>(b->data.size());
  const auto ipiv = std::make_unique<int[]>(n);
  AUTD_SYSV(CblasColMajor, 'U', n, 1, data.data(), lda, ipiv.get(), b->data.data(), ldb);
}

void BLASMatrix<complex>::solve(const std::shared_ptr<EigenMatrix<complex>>& b) {
  const auto n = static_cast<int>(data.cols());
  const auto lda = static_cast<int>(data.rows());
  const auto ldb = static_cast<int>(b->data.size());
  auto ipiv = std::make_unique<int[]>(n);
  AUTD_POSVC(CblasColMajor, 'U', n, 1, data.data(), lda, b->data.data(), ldb);
}
double BLASMatrix<double>::dot(const std::shared_ptr<const EigenMatrix<double>>& a) {
  return AUTD_DOT(static_cast<int>(a->data.size()), data.data(), 1, a->data.data(), 1);
}
complex BLASMatrix<complex>::dot(const std::shared_ptr<const EigenMatrix<complex>>& a) {
  complex d;
  AUTD_DOTC(static_cast<int>(a->data.size()), data.data(), 1, a->data.data(), 1, &d);
  return d;
}

template <>
double BLASMatrix<std::complex<double>>::max_element() const {
  const auto idx = AUTD_IMAXC(static_cast<int>(data.size()), data.data(), 1);
  return std::abs(data(static_cast<Eigen::Index>(idx), 0));
}

template <>
double BLASMatrix<double>::max_element() const {
  return this->data.maxCoeff();
  // idamax return the first occurrence of the the maximum 'absolute' value
  // const auto idx = AUTD_IMAX(static_cast<int>(v->data.size()), v->data.data(), 1);
  // return v->data(idx);
}

void BLASMatrix<double>::copy_from(const double* v, const size_t n) { AUTD_CPY(static_cast<int>(n), v, 1, data.data(), 1); }
void BLASMatrix<complex>::copy_from(const complex* v, const size_t n) { AUTD_CPYC(static_cast<int>(n), v, 1, data.data(), 1); }
}  // namespace autd::gain::holo
