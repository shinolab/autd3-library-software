// File: linalg_backend.cpp
// Project: holo
// Created Date: 06/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/03/2021
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

#ifdef ENABLE_EIGEN
void Eigen3Backend::hadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) { (*c).noalias() = a.cwiseProduct(b); }
void Eigen3Backend::real(const MatrixXc& a, MatrixX* b) { (*b).noalias() = a.real(); }
void Eigen3Backend::pseudoInverseSVD(MatrixXc* matrix, const Float alpha, MatrixXc* result) {
  const Eigen::JacobiSVD<MatrixXc> svd(*matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singularValues_inv = svd.singularValues();
  for (auto i = 0; i < singularValues_inv.size(); i++) {
    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha);
  }
  (*result).noalias() = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());
}
Eigen3Backend::VectorXc Eigen3Backend::maxEigenVector(MatrixXc* matrix) {
  const Eigen::ComplexEigenSolver<MatrixXc> ces(*matrix);
  auto idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  return ces.eigenvectors().col(idx);
}
void Eigen3Backend::matAdd(const Float alpha, const MatrixX& a, const Float beta, MatrixX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::matMul(const TRANSPOSE transA, const TRANSPOSE transB, const std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b,
                           const std::complex<Float> beta, MatrixXc* c) {
  *c *= beta;
  switch (transA) {
    case TRANSPOSE::ConjTrans:
      switch (transB) {
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
      switch (transB) {
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
      switch (transB) {
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
      switch (transB) {
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
void Eigen3Backend::matVecMul(const TRANSPOSE transA, const std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b,
                              const std::complex<Float> beta, VectorXc* c) {
  *c *= beta;
  switch (transA) {
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
void Eigen3Backend::vecAdd(const Float alpha, const VectorX& a, const Float beta, VectorX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::solveg(MatrixX* a, VectorX* b, VectorX* c) {
  const Eigen::HouseholderQR<MatrixX> qr(*a);
  (*c).noalias() = qr.solve(*b);
}
void Eigen3Backend::csolveh(MatrixXc* a, VectorXc* b) {
  const Eigen::LLT<MatrixXc> llt(*a);
  llt.solveInPlace(*b);
}
Float Eigen3Backend::dot(const VectorX& a, const VectorX& b) { return a.dot(b); }
std::complex<Float> Eigen3Backend::cdot(const VectorXc& a, const VectorXc& b) { return a.conjugate().dot(b); }
Float Eigen3Backend::cmaxCoeff(const VectorXc& v) { return sqrt(v.cwiseAbs2().maxCoeff()); }
Float Eigen3Backend::maxCoeff(const VectorX& v) { return v.maxCoeff(); }
Eigen3Backend::MatrixXc Eigen3Backend::concatRow(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows() + b.rows(), b.cols());
  c << a, b;
  return c;
}
Eigen3Backend::MatrixXc Eigen3Backend::concatCol(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows(), a.cols() + b.cols());
  c << a, b;
  return c;
}
void Eigen3Backend::matCpy(const MatrixX& a, MatrixX* b) { *b = a; }
void Eigen3Backend::vecCpy(const VectorX& a, VectorX* b) { *b = a; }

#endif

#ifdef ENABLE_BLAS

#ifdef USE_DOUBLE_AUTD
#define AUTD_gesvd LAPACKE_zgesvd
#define AUTD_heev LAPACKE_zheev
#define AUTD_axpy cblas_daxpy
#define AUTD_gemm cblas_zgemm
#define AUTD_gemv cblas_zgemv
#define AUTD_dotc cblas_zdotu_sub
#define AUTD_dot cblas_ddot
#define AUTD_imaxc cblas_izamax
#define AUTD_sysv LAPACKE_dsysv
#define AUTD_posvc LAPACKE_zposv
#define AUTD_cpy LAPACKE_dlacpy
#else
#define AUTD_gesvd LAPACKE_cgesvd
#define AUTD_heev LAPACKE_cheev
#define AUTD_axpy cblas_saxpy
#define AUTD_gemm cblas_cgemm
#define AUTD_gemv cblas_cgemv
#define AUTD_dotc cblas_cdotu_sub
#define AUTD_dot cblas_sdot
#define AUTD_imaxc cblas_icamax
#define AUTD_sysv LAPACKE_ssysv
#define AUTD_posvc LAPACKE_cposv
#define AUTD_cpy LAPACKE_slacpy
#endif

void BLASBackend::hadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) {
  const auto* ap = a.data();
  const auto* bp = b.data();
  auto* cp = c->data();
  for (size_t i = 0; i < a.size(); i++) {
    (*cp++) = (*ap++) * (*bp++);
  }
}
void BLASBackend::real(const MatrixXc& a, MatrixX* b) {
  const auto* ap = a.data();
  auto* bp = b->data();
  for (size_t i = 0; i < a.size(); i++) {
    (*bp++) = (*ap++).real();
  }
}
void BLASBackend::pseudoInverseSVD(MatrixXc* matrix, const Float alpha, MatrixXc* result) {
  const auto nc = matrix->cols();
  const auto nr = matrix->rows();

  const auto LDA = static_cast<int>(nr);
  const auto LDU = static_cast<int>(nr);
  const auto LDVT = static_cast<int>(nc);

  const auto s_size = std::min(nr, nc);
  const auto s = std::make_unique<Float[]>(s_size);
  auto u = MatrixXc(nr, nr);
  auto vt = MatrixXc(nc, nc);
  const auto superb = std::make_unique<Float[]>(s_size - 1);

  AUTD_gesvd(LAPACK_COL_MAJOR, 'A', 'A', static_cast<int>(nr), static_cast<int>(nc), matrix->data(), LDA, s.get(), u.data(), LDU, vt.data(), LDVT,
             superb.get());

  auto singularInv = MatrixXc::Zero(nc, nr);
  for (size_t i = 0; i < s_size; i++) {
    singularInv(i, i) = s[i] / (s[i] * s[i] + alpha);
  }

  auto tmp = MatrixXc(nc, nr);
  BLASBackend::matMul(TRANSPOSE::NoTrans, TRANSPOSE::ConjTrans, std::complex<Float>(1, 0), singularInv, u, std::complex<Float>(0, 0), &tmp);
  BLASBackend::matMul(TRANSPOSE::ConjTrans, TRANSPOSE::NoTrans, std::complex<Float>(1, 0), vt, tmp, std::complex<Float>(0, 0), result);
}

BLASBackend::VectorXc BLASBackend::maxEigenVector(MatrixXc* matrix) {
  const auto size = matrix->cols();
  const auto eigenvalues = std::make_unique<Float[]>(size);
  AUTD_heev(CblasColMajor, 'V', 'U', static_cast<int>(size), matrix->data(), static_cast<int>(size), eigenvalues.get());

  return matrix->col(size - 1);
}

void BLASBackend::matAdd(const Float alpha, const MatrixX& a, Float beta, MatrixX* b) {
  AUTD_axpy(static_cast<int>(a.size()), alpha, a.data(), 1, b->data(), 1);
}

void BLASBackend::matMul(const TRANSPOSE transA, const TRANSPOSE transB, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b,
                         std::complex<Float> beta, MatrixXc* c) {
  const auto LDA = static_cast<int>(a.rows());
  const auto LDB = static_cast<int>(b.rows());
  const auto M = (transA == TRANSPOSE::NoTrans || transA == TRANSPOSE::ConjNoTrans) ? static_cast<int>(a.rows()) : static_cast<int>(a.cols());
  const auto N = (transB == TRANSPOSE::NoTrans || transB == TRANSPOSE::ConjNoTrans) ? static_cast<int>(b.cols()) : static_cast<int>(b.rows());
  const auto K = (transA == TRANSPOSE::NoTrans || transA == TRANSPOSE::ConjNoTrans) ? static_cast<int>(a.cols()) : static_cast<int>(a.rows());
  AUTD_gemm(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(transA), static_cast<CBLAS_TRANSPOSE>(transB), M, N, K, &alpha, a.data(), LDA, b.data(), LDB,
            &beta, c->data(), M);
}

void BLASBackend::matVecMul(const TRANSPOSE transA, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta,
                            VectorXc* c) {
  const auto LDA = static_cast<int>(a.rows());
  const auto M = static_cast<int>(a.rows());
  const auto N = static_cast<int>(a.cols());
  AUTD_gemv(CblasColMajor, static_cast<CBLAS_TRANSPOSE>(transA), M, N, &alpha, a.data(), LDA, b.data(), 1, &beta, c->data(), 1);
}
void BLASBackend::vecAdd(const Float alpha, const VectorX& a, Float beta, VectorX* b) {
  AUTD_axpy(static_cast<int>(a.size()), alpha, a.data(), 1, b->data(), 1);
}
void BLASBackend::solveg(MatrixX* a, VectorX* b, VectorX* c) {
  const auto N = static_cast<int>(a->cols());
  const auto LDA = static_cast<int>(a->rows());
  const auto LDB = static_cast<int>(b->size());
  std::memcpy(c->data(), b->data(), LDB * sizeof(Float));
  const auto ipiv = std::make_unique<int[]>(N);
  AUTD_sysv(CblasColMajor, 'U', N, 1, a->data(), LDA, ipiv.get(), c->data(), LDB);
}
void BLASBackend::csolveh(MatrixXc* a, VectorXc* b) {
  const auto N = static_cast<int>(a->cols());
  const auto LDA = static_cast<int>(a->rows());
  const auto LDB = static_cast<int>(b->size());
  auto ipiv = std::make_unique<int[]>(N);
  AUTD_posvc(CblasColMajor, 'U', N, 1, a->data(), LDA, b->data(), LDB);
}

Float BLASBackend::dot(const VectorX& a, const VectorX& b) { return AUTD_dot(static_cast<int>(a.size()), a.data(), 1, b.data(), 1); }

std::complex<Float> BLASBackend::cdot(const VectorXc& a, const VectorXc& b) {
  std::complex<Float> d;
  AUTD_dotc(static_cast<int>(a.size()), a.data(), 1, b.data(), 1, &d);
  return d;
}

Float BLASBackend::maxCoeff(const VectorX& v) {
  auto maxValue = v(0);
  for (size_t i = 1; i < v.size(); i++) {
    maxValue = std::max(maxValue, v(i));
  }
  return maxValue;
}
Float BLASBackend::cmaxCoeff(const VectorXc& v) {
  const auto idx = AUTD_imaxc(static_cast<int>(v.size()), v.data(), 1);
  return abs(v(idx));
}
BLASBackend::MatrixXc BLASBackend::concatRow(const MatrixXc& a, const MatrixXc& b) {
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
BLASBackend::MatrixXc BLASBackend::concatCol(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows(), a.cols() + b.cols());
  auto* cp = c.data();
  std::memcpy(cp, a.data(), a.size() * sizeof(std::complex<float>));
  cp += a.size();
  std::memcpy(cp, b.data(), b.size() * sizeof(std::complex<float>));
  return c;
}
void BLASBackend::matCpy(const MatrixX& a, MatrixX* b) {
  AUTD_cpy(LAPACK_COL_MAJOR, 'A', static_cast<int>(a.rows()), static_cast<int>(a.cols()), a.data(), static_cast<int>(a.rows()), b->data(),
           static_cast<int>(b->rows()));
}
void BLASBackend::vecCpy(const VectorX& a, VectorX* b) {
  AUTD_cpy(LAPACK_COL_MAJOR, 'A', static_cast<int>(a.size()), 1, a.data(), static_cast<int>(a.size()), b->data(), 1);
}
#endif

}  // namespace autd::gain::holo
