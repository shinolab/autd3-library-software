// File: linalg_backend.cpp
// Project: holo
// Created Date: 06/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "gain/linalg_backend.hpp"

#include "gain/holo.hpp"

#ifdef ENABLE_BLAS
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#ifdef USE_BLAS_MKL
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif
#endif

namespace autd::gain::holo {

#ifndef DISABLE_EIGEN
void Eigen3Backend::HadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) { (*c).noalias() = a.cwiseProduct(b); }
void Eigen3Backend::Real(const MatrixXc& a, MatrixX* b) { (*b).noalias() = a.real(); }
void Eigen3Backend::PseudoInverseSvd(MatrixXc* matrix, const Float alpha, MatrixXc* result) {
  const Eigen::BDCSVD<MatrixXc> svd(*matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singular_values_inv = svd.singularValues();
  for (auto i = 0; i < singular_values_inv.size(); i++) {
    singular_values_inv(i) = singular_values_inv(i) / (singular_values_inv(i) * singular_values_inv(i) + alpha);
  }
  (*result).noalias() = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().adjoint();
}
Eigen3Backend::VectorXc Eigen3Backend::MaxEigenVector(MatrixXc* matrix) {
  const Eigen::ComplexEigenSolver<MatrixXc> ces(*matrix);
  auto idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  return ces.eigenvectors().col(idx);
}
void Eigen3Backend::MatAdd(const Float alpha, const MatrixX& a, const Float beta, MatrixX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::MatMul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b,
                           const std::complex<Float> beta, MatrixXc* c) {
  *c *= beta;
  switch (trans_a) {
    case TRANSPOSE::CONJ_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          (*c).noalias() += alpha * (a.adjoint() * b.adjoint());
          break;
        case TRANSPOSE::TRANS:
          (*c).noalias() += alpha * (a.adjoint() * b.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          (*c).noalias() += alpha * (a.adjoint() * b.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          (*c).noalias() += alpha * (a.adjoint() * b);
          break;
      }
      break;
    case TRANSPOSE::TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          (*c).noalias() += alpha * (a.transpose() * b.adjoint());
          break;
        case TRANSPOSE::TRANS:
          (*c).noalias() += alpha * (a.transpose() * b.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          (*c).noalias() += alpha * (a.transpose() * b.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          (*c).noalias() += alpha * (a.transpose() * b);
          break;
      }
      break;
    case TRANSPOSE::CONJ_NO_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          (*c).noalias() += alpha * (a.conjugate() * b.adjoint());
          break;
        case TRANSPOSE::TRANS:
          (*c).noalias() += alpha * (a.conjugate() * b.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          (*c).noalias() += alpha * (a.conjugate() * b.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          (*c).noalias() += alpha * (a.conjugate() * b);
          break;
      }
      break;
    case TRANSPOSE::NO_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          (*c).noalias() += alpha * (a * b.adjoint());
          break;
        case TRANSPOSE::TRANS:
          (*c).noalias() += alpha * (a * b.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          (*c).noalias() += alpha * (a * b.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          (*c).noalias() += alpha * (a * b);
          break;
      }
      break;
  }
}
void Eigen3Backend::MatVecMul(const TRANSPOSE trans_a, const std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b,
                              const std::complex<Float> beta, VectorXc* c) {
  *c *= beta;
  switch (trans_a) {
    case TRANSPOSE::CONJ_TRANS:
      (*c).noalias() += alpha * (a.adjoint() * b);
      break;
    case TRANSPOSE::TRANS:
      (*c).noalias() += alpha * (a.transpose() * b);
      break;
    case TRANSPOSE::CONJ_NO_TRANS:
      (*c).noalias() += alpha * (a.conjugate() * b);
      break;
    case TRANSPOSE::NO_TRANS:
      (*c).noalias() += alpha * (a * b);
      break;
  }
}
void Eigen3Backend::VecAdd(const Float alpha, const VectorX& a, const Float beta, VectorX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::Solveg(MatrixX* a, VectorX* b, VectorX* c) {
  const Eigen::HouseholderQR<MatrixX> qr(*a);
  (*c).noalias() = qr.solve(*b);
}
void Eigen3Backend::SolveCh(MatrixXc* a, VectorXc* b) {
  const Eigen::LLT<MatrixXc> llt(*a);
  llt.solveInPlace(*b);
}
Float Eigen3Backend::Dot(const VectorX& a, const VectorX& b) { return a.dot(b); }
std::complex<Float> Eigen3Backend::DotC(const VectorXc& a, const VectorXc& b) { return a.conjugate().dot(b); }
Float Eigen3Backend::MaxCoeffC(const VectorXc& v) { return sqrt(v.cwiseAbs2().maxCoeff()); }
Float Eigen3Backend::MaxCoeff(const VectorX& v) { return v.maxCoeff(); }
Eigen3Backend::MatrixXc Eigen3Backend::ConcatRow(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows() + b.rows(), b.cols());
  c << a, b;
  return c;
}
Eigen3Backend::MatrixXc Eigen3Backend::ConcatCol(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows(), a.cols() + b.cols());
  c << a, b;
  return c;
}
void Eigen3Backend::MatCpy(const MatrixX& a, MatrixX* b) { *b = a; }
void Eigen3Backend::VecCpy(const VectorX& a, VectorX* b) { *b = a; }
void Eigen3Backend::VecCpyC(const VectorXc& a, VectorXc* b) { *b = a; }
#endif

#ifdef ENABLE_BLAS

#ifdef USE_DOUBLE_AUTD
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
#else
constexpr auto AUTD_GESVD = LAPACKE_cgesdd;
constexpr auto AUTD_HEEV = LAPACKE_cheev;
constexpr auto AUTD_AXPY = cblas_saxpy;
constexpr auto AUTD_GEMM = cblas_cgemm;
constexpr auto AUTD_GEMV = cblas_cgemv;
constexpr auto AUTD_DOTC = cblas_cdotu_sub;
constexpr auto AUTD_DOT = cblas_sdot;
constexpr auto AUTD_IMAXC = cblas_icamax;
constexpr auto AUTD_SYSV = LAPACKE_ssysv;
constexpr auto AUTD_POSVC = LAPACKE_cposv;
constexpr auto AUTD_CPY = LAPACKE_slacpy;
constexpr auto AUTD_CPYC = LAPACKE_clacpy;
#endif

void BLASBackend::HadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) {
  const auto* ap = a.data();
  const auto* bp = b.data();
  auto* cp = c->data();
  for (size_t i = 0; i < a.size(); i++) {
    *cp++ = *ap++ * *bp++;
  }
}
void BLASBackend::Real(const MatrixXc& a, MatrixX* b) {
  const auto* ap = a.data();
  auto* bp = b->data();
  for (size_t i = 0; i < a.size(); i++) {
    *bp++ = (*ap++).real();
  }
}
void BLASBackend::PseudoInverseSvd(MatrixXc* matrix, const Float alpha, MatrixXc* result) {
  const auto nc = matrix->cols();
  const auto nr = matrix->rows();

  const auto lda = static_cast<int>(nr);
  const auto ldu = static_cast<int>(nr);
  const auto ldvt = static_cast<int>(nc);

  const auto s_size = std::min(nr, nc);
  const auto s = std::make_unique<Float[]>(s_size);
  auto u = MatrixXc(nr, nr);
  auto vt = MatrixXc(nc, nc);

  AUTD_GESVD(LAPACK_COL_MAJOR, 'A', static_cast<int>(nr), static_cast<int>(nc), matrix->data(), lda, s.get(), u.data(), ldu, vt.data(), ldvt);

  auto singular_inv = MatrixXc::Zero(nc, nr);
  for (size_t i = 0; i < s_size; i++) singular_inv(i, i) = s[i] / (s[i] * s[i] + alpha);

  auto tmp = MatrixXc(nc, nr);
  BLASBackend::MatMul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, std::complex<Float>(1, 0), singular_inv, u, std::complex<Float>(0, 0), &tmp);
  BLASBackend::MatMul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, std::complex<Float>(1, 0), vt, tmp, std::complex<Float>(0, 0), result);
}

BLASBackend::VectorXc BLASBackend::MaxEigenVector(MatrixXc* matrix) {
  const auto size = matrix->cols();
  const auto eigenvalues = std::make_unique<Float[]>(size);
  AUTD_HEEV(CblasColMajor, 'V', 'U', static_cast<int>(size), matrix->data(), static_cast<int>(size), eigenvalues.get());

  return matrix->col(size - 1);
}

void BLASBackend::MatAdd(const Float alpha, const MatrixX& a, Float beta, MatrixX* b) {
  AUTD_AXPY(static_cast<int>(a.size()), alpha, a.data(), 1, b->data(), 1);
}

void BLASBackend::MatMul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b,
                         std::complex<Float> beta, MatrixXc* c) {
  const auto lda = static_cast<int>(a.rows());
  const auto ldb = static_cast<int>(b.rows());
  const auto ldc = trans_a == TRANSPOSE::NO_TRANS || trans_a == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(a.rows()) : static_cast<int>(a.cols());
  const auto n = trans_b == TRANSPOSE::NO_TRANS || trans_b == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(b.cols()) : static_cast<int>(b.rows());
  const auto k = trans_a == TRANSPOSE::NO_TRANS || trans_a == TRANSPOSE::CONJ_NO_TRANS ? static_cast<int>(a.cols()) : static_cast<int>(a.rows());
  AUTD_GEMM(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), static_cast<CBLAS_TRANSPOSE>(trans_b), ldc, n, k, &alpha, a.data(), lda, b.data(),
            ldb, &beta, c->data(), ldc);
}

void BLASBackend::MatVecMul(const TRANSPOSE trans_a, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta,
                            VectorXc* c) {
  const auto lda = static_cast<int>(a.rows());
  const auto m = static_cast<int>(a.rows());
  const auto n = static_cast<int>(a.cols());
  AUTD_GEMV(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(trans_a), m, n, &alpha, a.data(), lda, b.data(), 1, &beta, c->data(), 1);
}
void BLASBackend::VecAdd(const Float alpha, const VectorX& a, Float beta, VectorX* b) {
  AUTD_AXPY(static_cast<int>(a.size()), alpha, a.data(), 1, b->data(), 1);
}
void BLASBackend::Solveg(MatrixX* a, VectorX* b, VectorX* c) {
  const auto n = static_cast<int>(a->cols());
  const auto lda = static_cast<int>(a->rows());
  const auto ldb = static_cast<int>(b->size());
  std::memcpy(c->data(), b->data(), ldb * sizeof(Float));
  const auto ipiv = std::make_unique<int[]>(n);
  AUTD_SYSV(CblasColMajor, 'U', n, 1, a->data(), lda, ipiv.get(), c->data(), ldb);
}
void BLASBackend::SolveCh(MatrixXc* a, VectorXc* b) {
  const auto n = static_cast<int>(a->cols());
  const auto lda = static_cast<int>(a->rows());
  const auto ldb = static_cast<int>(b->size());
  auto ipiv = std::make_unique<int[]>(n);
  AUTD_POSVC(CblasColMajor, 'U', n, 1, a->data(), lda, b->data(), ldb);
}

Float BLASBackend::Dot(const VectorX& a, const VectorX& b) { return AUTD_DOT(static_cast<int>(a.size()), a.data(), 1, b.data(), 1); }

std::complex<Float> BLASBackend::DotC(const VectorXc& a, const VectorXc& b) {
  std::complex<Float> d;
  AUTD_DOTC(static_cast<int>(a.size()), a.data(), 1, b.data(), 1, &d);
  return d;
}

Float BLASBackend::MaxCoeff(const VectorX& v) {
  auto max_value = v(0);
  for (size_t i = 1; i < v.size(); i++) {
    max_value = std::max(max_value, v(i));
  }
  return max_value;
}
Float BLASBackend::MaxCoeffC(const VectorXc& v) {
  const auto idx = AUTD_IMAXC(static_cast<int>(v.size()), v.data(), 1);
  return abs(v(idx));
}
BLASBackend::MatrixXc BLASBackend::ConcatRow(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows() + b.rows(), b.cols());
  const auto* ap = a.data();
  const auto* bp = b.data();
  auto* cp = c.data();
  for (size_t i = 0; i < a.cols(); i++) {
    std::memcpy(cp, ap, a.rows() * sizeof(std::complex<float>));
    ap += a.rows();
    cp += a.rows();
    std::memcpy(cp, bp, b.rows() * sizeof(std::complex<float>));
    bp += b.rows();
    cp += b.rows();
  }
  return c;
}
BLASBackend::MatrixXc BLASBackend::ConcatCol(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows(), a.cols() + b.cols());
  auto* cp = c.data();
  std::memcpy(cp, a.data(), a.size() * sizeof(std::complex<float>));
  cp += a.size();
  std::memcpy(cp, b.data(), b.size() * sizeof(std::complex<float>));
  return c;
}
void BLASBackend::MatCpy(const MatrixX& a, MatrixX* b) {
  AUTD_CPY(LAPACK_COL_MAJOR, 'A', static_cast<int>(a.rows()), static_cast<int>(a.cols()), a.data(), static_cast<int>(a.rows()), b->data(),
           static_cast<int>(b->rows()));
}
void BLASBackend::VecCpy(const VectorX& a, VectorX* b) {
  AUTD_CPY(LAPACK_COL_MAJOR, 'A', static_cast<int>(a.size()), 1, a.data(), static_cast<int>(a.size()), b->data(), 1);
}
void BLASBackend::VecCpyC(const VectorXc& a, VectorXc* b) {
  AUTD_CPYC(LAPACK_COL_MAJOR, 'A', static_cast<int>(a.size()), 1, a.data(), static_cast<int>(a.size()), b->data(), 1);
}
#endif

}  // namespace autd::gain::holo
