// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 05/03/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "gain/holo.hpp"

#ifdef ENABLE_BLAS
#if WIN32
#include <codeanalysis\warnings.h>
#define lapack_complex_float _C_float_complex
#define lapack_complex_double _C_double_complex
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include "cblas.h"
#include "lapacke.h"
#if WIN32
#pragma warning(pop)
#endif
#endif

namespace autd::gain {

void Eigen3Backend::hadamardProduct(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::MatrixXc& b, Eigen3Backend::MatrixXc* c) {
  (*c).noalias() = a.cwiseProduct(b);
}
void Eigen3Backend::real(const Eigen3Backend::MatrixXc& a, Eigen3Backend::MatrixX* b) { (*b).noalias() = a.real(); }
void Eigen3Backend::pseudoInverseSVD(Eigen3Backend::MatrixXc* matrix, Float alpha, Eigen3Backend::MatrixXc* result) {
  Eigen::JacobiSVD<MatrixXc> svd(*matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::JacobiSVD<MatrixXc>::SingularValuesType singularValues_inv = svd.singularValues();
  for (auto i = 0; i < singularValues_inv.size(); i++) {
    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha);
  }
  (*result).noalias() = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());
}
Eigen3Backend::VectorXc Eigen3Backend::maxEigenVector(Eigen3Backend::MatrixXc* matrix) {
  const Eigen::ComplexEigenSolver<MatrixXc> ces(*matrix);
  int idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  return ces.eigenvectors().col(idx);
}
void Eigen3Backend::matadd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::matmul(const TRANSPOSE transa, const TRANSPOSE transb, std::complex<Float> alpha, const Eigen3Backend::MatrixXc& a,
                           const Eigen3Backend::MatrixXc& b, std::complex<Float> beta, Eigen3Backend::MatrixXc* c) {
  *c *= beta;
  switch (transa) {
    case TRANSPOSE::ConjTrans:
      switch (transb) {
        case TRANSPOSE::ConjTrans:
          (*c).noalias() += alpha * (a.adjoint() * b.adjoint());
          break;
        case TRANSPOSE::Trans:
          (*c).noalias() += alpha * (a.adjoint() * b.transpose());
          break;
        case TRANSPOSE::ConjNoTrans:
          (*c).noalias() += alpha * (a.adjoint() * b.conjugate());
          break;
        case TRANSPOSE::NoTrans:
          (*c).noalias() += alpha * (a.adjoint() * b);
          break;
      }
      break;
    case TRANSPOSE::Trans:
      switch (transb) {
        case TRANSPOSE::ConjTrans:
          (*c).noalias() += alpha * (a.transpose() * b.adjoint());
          break;
        case TRANSPOSE::Trans:
          (*c).noalias() += alpha * (a.transpose() * b.transpose());
          break;
        case TRANSPOSE::ConjNoTrans:
          (*c).noalias() += alpha * (a.transpose() * b.conjugate());
          break;
        case TRANSPOSE::NoTrans:
          (*c).noalias() += alpha * (a.transpose() * b);
          break;
      }
      break;
    case TRANSPOSE::ConjNoTrans:
      switch (transb) {
        case TRANSPOSE::ConjTrans:
          (*c).noalias() += alpha * (a.conjugate() * b.adjoint());
          break;
        case TRANSPOSE::Trans:
          (*c).noalias() += alpha * (a.conjugate() * b.transpose());
          break;
        case TRANSPOSE::ConjNoTrans:
          (*c).noalias() += alpha * (a.conjugate() * b.conjugate());
          break;
        case TRANSPOSE::NoTrans:
          (*c).noalias() += alpha * (a.conjugate() * b);
          break;
      }
      break;
    case TRANSPOSE::NoTrans:
      switch (transb) {
        case TRANSPOSE::ConjTrans:
          (*c).noalias() += alpha * (a * b.adjoint());
          break;
        case TRANSPOSE::Trans:
          (*c).noalias() += alpha * (a * b.transpose());
          break;
        case TRANSPOSE::ConjNoTrans:
          (*c).noalias() += alpha * (a * b.conjugate());
          break;
        case TRANSPOSE::NoTrans:
          (*c).noalias() += alpha * (a * b);
          break;
      }
      break;
  }
}
void Eigen3Backend::matvecmul(const TRANSPOSE transa, std::complex<Float> alpha, const Eigen3Backend::MatrixXc& a, const Eigen3Backend::VectorXc& b,
                              std::complex<Float> beta, Eigen3Backend::VectorXc* c) {
  *c *= beta;
  switch (transa) {
    case TRANSPOSE::ConjTrans:
      (*c).noalias() += alpha * (a.adjoint() * b);
      break;
    case TRANSPOSE::Trans:
      (*c).noalias() += alpha * (a.transpose() * b);
      break;
    case TRANSPOSE::ConjNoTrans:
      (*c).noalias() += alpha * (a.conjugate() * b);
      break;
    case TRANSPOSE::NoTrans:
      (*c).noalias() += alpha * (a * b);
      break;
  }
}
void Eigen3Backend::vecadd(Float alpha, const Eigen3Backend::VectorX& a, Float beta, Eigen3Backend::VectorX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::solveg(Eigen3Backend::MatrixX* a, Eigen3Backend::VectorX* b, Eigen3Backend::VectorX* c) {
  Eigen::HouseholderQR<Eigen3Backend::MatrixX> qr(*a);
  (*c).noalias() = qr.solve(*b);
}
void Eigen3Backend::csolveh(Eigen3Backend::MatrixXc* a, Eigen3Backend::VectorXc* b) {
  Eigen::LLT<Eigen3Backend::MatrixXc> llt(*a);
  llt.solveInPlace(*b);
}
Float Eigen3Backend::dot(const Eigen3Backend::VectorX& a, const Eigen3Backend::VectorX& b) { return a.dot(b); }
std::complex<Float> Eigen3Backend::cdot(const Eigen3Backend::VectorXc& a, const Eigen3Backend::VectorXc& b) { return a.conjugate().dot(b); }
Float Eigen3Backend::cmaxCoeff(const Eigen3Backend::VectorXc& v) { return sqrt(v.cwiseAbs2().maxCoeff()); }
Float Eigen3Backend::maxCoeff(const Eigen3Backend::VectorX& v) { return v.maxCoeff(); }
Eigen3Backend::MatrixXc Eigen3Backend::concat_in_row(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::MatrixXc& b) {
  Eigen3Backend::MatrixXc c(a.rows() + b.rows(), b.cols());
  c << a, b;
  return c;
}
Eigen3Backend::MatrixXc Eigen3Backend::concat_in_col(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::MatrixXc& b) {
  Eigen3Backend::MatrixXc c(a.rows(), a.cols() + b.cols());
  c << a, b;
  return c;
}

#ifdef ENABLE_BLAS

#ifdef USE_DOUBLE_AUTD
#define AUTD_gesvd LAPACKE_zgesvd
#define AUTD_heev LAPACKE_zheev
#define AUTD_axpy cblas_daxpy
#define AUTD_gemm cblas_zgemm
#define AUTD_gemv cblas_zgemv
#define AUTD_dotc cblas_zdotc
#define AUTD_dot cblas_ddot
#define AUTD_imaxc cblas_izamax
#define AUTD_imax cblas_idamax
#define AUTD_gesv LAPACKE_dgesv
#define AUTD_posvc LAPACKE_zposv
#else
#define AUTD_gesvd LAPACKE_cgesvd
#define AUTD_heev LAPACKE_cheev
#define AUTD_axpy cblas_saxpy
#define AUTD_gemm cblas_cgemm
#define AUTD_gemv cblas_cgemv
#define AUTD_dotc cblas_cdotc
#define AUTD_dot cblas_sdot
#define AUTD_imaxc cblas_icamax
#define AUTD_imax cblas_isamax
#define AUTD_gesv LAPACKE_sgesv
#define AUTD_posvc LAPACKE_cposv
#endif

inline void blas_matmul(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, complex alpha, BLASBackend::MatrixXc& a, BLASBackend::MatrixXc& b,
                        complex beta, BLASBackend::MatrixXc* c) {
  const blasint M = static_cast<blasint>(a.rows());
  const blasint N = static_cast<blasint>(b.cols());
  const blasint K = static_cast<blasint>(a.cols());
  AUTD_gemm(CblasColMajor, transa, transb, M, N, K, &alpha, a.data(), M, b.data(), K, &beta, c->data(), M);
}

inline void blas_matmul(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, BLASBackend::MatrixXc& a, BLASBackend::MatrixXc& b,
                        BLASBackend::MatrixXc* c) {
  complex one = complex(1.0, 0.0);
  complex zero = complex(0.0, 0.0);
  blas_matmul(transa, transb, one, a, b, zero, c);
}

inline void blas_matmul(BLASBackend::MatrixXc& a, BLASBackend::MatrixXc& b, BLASBackend::MatrixXc* c) {
  blas_matmul(CblasNoTrans, CblasNoTrans, a, b, c);
}
void BLASBackend::hadamardProduct(const BLASBackend::MatrixXc& a, const BLASBackend::MatrixXc& b, BLASBackend::MatrixXc* c) {
  const auto* ap = a.data();
  const auto* bp = b.data();
  auto* cp = c->data();
  for (size_t i = 0; i < a.size(); i++) {
    (*cp++) = (*ap++) * (*bp++);
  }
}
void BLASBackend::real(const BLASBackend::MatrixXc& a, BLASBackend::MatrixX* b) {
  const auto* ap = a.data();
  auto* bp = b->data();
  for (size_t i = 0; i < a.size(); i++) {
    (*bp++) = (*ap++).real();
  }
}
void BLASBackend::pseudoInverseSVD(BLASBackend::MatrixXc* matrix, Float alpha, BLASBackend::MatrixXc* result) {
  const size_t nc = matrix->cols();
  const size_t nr = matrix->rows();

  const blasint LDA = static_cast<blasint>(nr);
  const blasint LDU = static_cast<blasint>(nr);
  const blasint LDVT = static_cast<blasint>(nc);

  const size_t s_size = std::min(nr, nc);
  auto s = std::make_unique<Float[]>(s_size);
  BLASBackend::MatrixXc u = BLASBackend::MatrixXc(nr, nr);
  BLASBackend::MatrixXc vt = BLASBackend::MatrixXc(nc, nc);
  auto superb = std::make_unique<Float[]>(s_size - 1);

  AUTD_gesvd(LAPACK_COL_MAJOR, 'A', 'A', static_cast<blasint>(nr), static_cast<blasint>(nc), matrix->data(), LDA, s.get(), u.data(), LDU, vt.data(),
             LDVT, superb.get());

  BLASBackend::MatrixXc sinv = BLASBackend::MatrixXc::Zero(nc, nr);
  for (size_t i = 0; i < s_size; i++) {
    sinv(i, i) = s[i] / (s[i] * s[i] + alpha);
  }

  BLASBackend::MatrixXc tmp = BLASBackend::MatrixXc(nc, nr);
  blas_matmul(CblasNoTrans, CblasConjTrans, sinv, u, &tmp);
  blas_matmul(CblasConjTrans, CblasNoTrans, vt, tmp, result);
}

BLASBackend::VectorXc BLASBackend::maxEigenVector(BLASBackend::MatrixXc* matrix) {
  const size_t size = matrix->cols();
  auto eigenvalues = std::make_unique<Float[]>(size);
  AUTD_heev(CblasColMajor, 'V', 'U', static_cast<blasint>(size), matrix->data(), static_cast<blasint>(size), eigenvalues.get());

  return matrix->col(size - 1);
}

void BLASBackend::matadd(Float alpha, const BLASBackend::MatrixX& a, Float beta, BLASBackend::MatrixX* b) {
  AUTD_axpy(static_cast<blasint>(a.size()), alpha, a.data(), 1, b->data(), 1);
}

void BLASBackend::matmul(const TRANSPOSE transa, const TRANSPOSE transb, std::complex<Float> alpha, const BLASBackend::MatrixXc& a,
                         const BLASBackend::MatrixXc& b, std::complex<Float> beta, BLASBackend::MatrixXc* c) {
  const blasint LDA = static_cast<blasint>(a.rows());
  const blasint LDB = static_cast<blasint>(b.rows());
  const blasint M =
      (transa == TRANSPOSE::NoTrans || transa == TRANSPOSE::ConjNoTrans) ? static_cast<blasint>(a.rows()) : static_cast<blasint>(a.cols());
  const blasint N =
      (transb == TRANSPOSE::NoTrans || transb == TRANSPOSE::ConjNoTrans) ? static_cast<blasint>(b.cols()) : static_cast<blasint>(b.rows());
  const blasint K =
      (transa == TRANSPOSE::NoTrans || transa == TRANSPOSE::ConjNoTrans) ? static_cast<blasint>(a.cols()) : static_cast<blasint>(a.rows());
  AUTD_gemm(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(transa), static_cast<CBLAS_TRANSPOSE>(transb), M, N, K, &alpha, a.data(), LDA, b.data(), LDB,
            &beta, c->data(), M);
}

void BLASBackend::matvecmul(const TRANSPOSE transa, std::complex<Float> alpha, const BLASBackend::MatrixXc& a, const BLASBackend::VectorXc& b,
                            std::complex<Float> beta, BLASBackend::VectorXc* c) {
  const blasint LDA = static_cast<blasint>(a.rows());
  const blasint M = static_cast<blasint>(a.rows());
  const blasint N = static_cast<blasint>(a.cols());
  AUTD_gemv(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(transa), M, N, &alpha, a.data(), LDA, b.data(), 1, &beta, c->data(), 1);
}
void BLASBackend::vecadd(Float alpha, const BLASBackend::VectorX& a, Float beta, BLASBackend::VectorX* b) {
  AUTD_axpy(static_cast<blasint>(a.size()), alpha, a.data(), 1, b->data(), 1);
}
void BLASBackend::solveg(BLASBackend::MatrixX* a, BLASBackend::VectorX* b, BLASBackend::VectorX* c) {
  const blasint N = static_cast<blasint>(a->cols());
  const blasint LDA = static_cast<blasint>(a->rows());
  const blasint LDB = static_cast<blasint>(b->size());
  std::memcpy(c->data(), b->data(), LDB * sizeof(Float));
  std::unique_ptr<blasint[]> ipiv = std::make_unique<blasint[]>(N);
  AUTD_gesv(CblasColMajor, N, 1, a->data(), LDA, ipiv.get(), c->data(), LDB);
}

void BLASBackend::csolveh(BLASBackend::MatrixXc* a, BLASBackend::VectorXc* b) {
  const blasint N = static_cast<blasint>(a->cols());
  const blasint LDA = static_cast<blasint>(a->rows());
  const blasint LDB = static_cast<blasint>(b->size());
  std::unique_ptr<blasint[]> ipiv = std::make_unique<blasint[]>(N);
  AUTD_posvc(CblasColMajor, 'U', N, 1, a->data(), LDA, b->data(), LDB);
}

Float BLASBackend::dot(const BLASBackend::VectorX& a, const BLASBackend::VectorX& b) {
  return AUTD_dot(static_cast<blasint>(a.size()), a.data(), 1, b.data(), 1);
}

std::complex<Float> BLASBackend::cdot(const BLASBackend::VectorXc& a, const BLASBackend::VectorXc& b) {
  auto d = AUTD_dotc(static_cast<blasint>(a.size()), a.data(), 1, b.data(), 1);
  return std::complex<Float>(d.real, d.imag);
}

Float BLASBackend::maxCoeff(const BLASBackend::VectorX& v) {
  auto idx = AUTD_imax(static_cast<blasint>(v.size()), v.data(), 1);
  return v(idx);
}
Float BLASBackend::cmaxCoeff(const BLASBackend::VectorXc& v) {
  auto idx = AUTD_imaxc(static_cast<blasint>(v.size()), v.data(), 1);
  return abs(v(idx));
}
BLASBackend::MatrixXc BLASBackend::concat_in_row(const BLASBackend::MatrixXc& a, const BLASBackend::MatrixXc& b) {
  BLASBackend::MatrixXc c(a.rows() + b.rows(), b.cols());
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
BLASBackend::MatrixXc BLASBackend::concat_in_col(const BLASBackend::MatrixXc& a, const BLASBackend::MatrixXc& b) {
  BLASBackend::MatrixXc c(a.rows(), a.cols() + b.cols());
  auto* cp = c.data();
  std::memcpy(cp, a.data(), a.size() * sizeof(std::complex<float>));
  cp += a.size();
  std::memcpy(cp, b.data(), b.size() * sizeof(std::complex<float>));
  return c;
}
#endif

}  // namespace autd::gain
