// File: linalg_backend.hpp
// Project: holo
// Created Date: 06/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/03/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <complex>

#include "autd_types.hpp"
#include "linalg.hpp"

#ifdef ENABLE_BLAS
#include "linalg/matrix.hpp"
#include "linalg/vector.hpp"
#endif

namespace autd::gain::holo {

enum class TRANSPOSE { NoTrans = 111, Trans = 112, ConjTrans = 113, ConjNoTrans = 114 };

template <typename MCx, typename VCx, typename Mx, typename Vx>
class Backend {
 public:
  using MatrixXc = MCx;
  using VectorXc = VCx;
  using MatrixX = Mx;
  using VectorX = Vx;

  virtual bool supports_SVD() = 0;
  virtual bool supports_EVD() = 0;
  virtual bool supports_solve() = 0;
  virtual void hadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) = 0;
  virtual void real(const MatrixXc& a, MatrixX* b) = 0;
  virtual void pseudoInverseSVD(MatrixXc* matrix, Float alpha, MatrixXc* result) = 0;
  virtual VectorXc maxEigenVector(MatrixXc* matrix) = 0;
  virtual void matadd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) = 0;
  virtual void matmul(const TRANSPOSE transa, const TRANSPOSE transb, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b,
                      std::complex<Float> beta, MatrixXc* c) = 0;
  virtual void matvecmul(const TRANSPOSE transa, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta,
                         VectorXc* c) = 0;
  virtual void vecadd(Float alpha, const VectorX& a, Float beta, VectorX* b) = 0;
  virtual void csolveh(MatrixXc* a, VectorXc* b) = 0;
  virtual void solveg(MatrixX* a, VectorX* b, VectorX* c) = 0;
  virtual Float dot(const VectorX& a, const VectorX& b) = 0;
  virtual std::complex<Float> cdot(const VectorXc& a, const VectorXc& b) = 0;
  virtual Float maxCoeff(const VectorX& v) = 0;
  virtual Float cmaxCoeff(const VectorXc& v) = 0;
  virtual MatrixXc concat_in_row(const MatrixXc& a, const MatrixXc& b) = 0;
  virtual MatrixXc concat_in_col(const MatrixXc& a, const MatrixXc& b) = 0;
  virtual void matcpy(const MatrixX& a, MatrixX* b) = 0;
  virtual void veccpy(const VectorX& a, VectorX* b) = 0;

  virtual ~Backend() {}
};

#ifdef ENABLE_EIGEN
class Eigen3Backend final : public Backend<Eigen::Matrix<std::complex<Float>, -1, -1>, Eigen::Matrix<std::complex<Float>, -1, 1>,
                                           Eigen::Matrix<Float, -1, -1>, Eigen::Matrix<Float, -1, 1>> {
 public:
  bool supports_SVD() override { return true; }
  bool supports_EVD() override { return true; }
  bool supports_solve() override { return true; }
  void hadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c);
  void real(const MatrixXc& a, MatrixX* b);
  void pseudoInverseSVD(MatrixXc* matrix, Float alpha, MatrixXc* result) override;
  VectorXc maxEigenVector(MatrixXc* matrix) override;
  void matadd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) override;
  void matmul(const TRANSPOSE transa, const TRANSPOSE transb, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b,
              std::complex<Float> beta, MatrixXc* c) override;
  void matvecmul(const TRANSPOSE transa, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta,
                 VectorXc* c) override;
  void vecadd(Float alpha, const VectorX& a, Float beta, VectorX* b) override;
  void csolveh(MatrixXc* a, VectorXc* b) override;
  void solveg(MatrixX* a, VectorX* b, VectorX* c) override;
  Float dot(const VectorX& a, const VectorX& b) override;
  std::complex<Float> cdot(const VectorXc& a, const VectorXc& b) override;
  Float maxCoeff(const VectorX& v) override;
  Float cmaxCoeff(const VectorXc& v) override;
  MatrixXc concat_in_row(const MatrixXc& a, const MatrixXc& b) override;
  MatrixXc concat_in_col(const MatrixXc& a, const MatrixXc& b) override;
  void matcpy(const MatrixX& a, MatrixX* b) override;
  void veccpy(const VectorX& a, VectorX* b) override;
};
#endif

#ifdef ENABLE_BLAS
class BLASBackend final
    : public Backend<_utils::MatrixX<std::complex<Float>>, _utils::VectorX<std::complex<Float>>, _utils::MatrixX<Float>, _utils::VectorX<Float>> {
 public:
  bool supports_SVD() override { return true; }
  bool supports_EVD() override { return true; }
  bool supports_solve() override { return true; }
  void hadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c);
  void real(const MatrixXc& a, MatrixX* b);
  void pseudoInverseSVD(MatrixXc* matrix, Float alpha, MatrixXc* result) override;
  VectorXc maxEigenVector(MatrixXc* matrix) override;
  void matadd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) override;
  void matmul(const TRANSPOSE transa, const TRANSPOSE transb, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b,
              std::complex<Float> beta, MatrixXc* c) override;
  void matvecmul(const TRANSPOSE transa, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta,
                 VectorXc* c) override;
  void vecadd(Float alpha, const VectorX& a, Float beta, VectorX* b) override;
  void csolveh(MatrixXc* a, VectorXc* b) override;
  void solveg(MatrixX* a, VectorX* b, VectorX* c) override;
  Float dot(const VectorX& a, const VectorX& b) override;
  std::complex<Float> cdot(const VectorXc& a, const VectorXc& b) override;
  Float maxCoeff(const VectorX& v) override;
  Float cmaxCoeff(const VectorXc& v) override;
  MatrixXc concat_in_row(const MatrixXc& a, const MatrixXc& b) override;
  MatrixXc concat_in_col(const MatrixXc& a, const MatrixXc& b) override;
  void matcpy(const MatrixX& a, MatrixX* b) override;
  void veccpy(const VectorX& a, VectorX* b) override;
};
#endif
}  // namespace autd::gain::holo
