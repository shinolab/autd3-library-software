// File: linalg_backend.hpp
// Project: holo
// Created Date: 06/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/04/2021
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

enum class TRANSPOSE { NO_TRANS = 111, TRANS = 112, CONJ_TRANS = 113, CONJ_NO_TRANS = 114 };

template <typename MCx, typename VCx, typename Mx, typename Vx>
class Backend {
 public:
  using MatrixXc = MCx;
  using VectorXc = VCx;
  using MatrixX = Mx;
  using VectorX = Vx;

  virtual bool SupportsSvd() = 0;
  virtual bool SupportsEVD() = 0;
  virtual bool SupportsSolve() = 0;
  virtual void HadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) = 0;
  virtual void Real(const MatrixXc& a, MatrixX* b) = 0;
  virtual void PseudoInverseSvd(MatrixXc* matrix, Float alpha, MatrixXc* result) = 0;
  virtual VectorXc MaxEigenVector(MatrixXc* matrix) = 0;
  virtual void MatAdd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) = 0;
  virtual void MatMul(TRANSPOSE trans_a, TRANSPOSE trans_b, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b, std::complex<Float> beta,
                      MatrixXc* c) = 0;
  virtual void MatVecMul(TRANSPOSE trans_a, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta,
                         VectorXc* c) = 0;
  virtual void VecAdd(Float alpha, const VectorX& a, Float beta, VectorX* b) = 0;
  virtual void SolveCh(MatrixXc* a, VectorXc* b) = 0;
  virtual void Solveg(MatrixX* a, VectorX* b, VectorX* c) = 0;
  virtual Float Dot(const VectorX& a, const VectorX& b) = 0;
  virtual std::complex<Float> DotC(const VectorXc& a, const VectorXc& b) = 0;
  virtual Float MaxCoeff(const VectorX& v) = 0;
  virtual Float MaxCoeffC(const VectorXc& v) = 0;
  virtual MatrixXc ConcatRow(const MatrixXc& a, const MatrixXc& b) = 0;
  virtual MatrixXc ConcatCol(const MatrixXc& a, const MatrixXc& b) = 0;
  virtual void MatCpy(const MatrixX& a, MatrixX* b) = 0;
  virtual void VecCpy(const VectorX& a, VectorX* b) = 0;
  virtual void VecCpyC(const VectorXc& a, VectorXc* b) = 0;

  Backend() = default;
  virtual ~Backend() = default;
  Backend(const Backend& obj) = delete;
  Backend& operator=(const Backend& obj) = delete;
  Backend(const Backend&& v) = delete;
  Backend& operator=(Backend&& obj) = delete;
};

#ifndef DISABLE_EIGEN
class Eigen3Backend final : public Backend<Eigen::Matrix<std::complex<Float>, -1, -1>, Eigen::Matrix<std::complex<Float>, -1, 1>,
                                           Eigen::Matrix<Float, -1, -1>, Eigen::Matrix<Float, -1, 1>> {
 public:
  bool SupportsSvd() override { return true; }
  bool SupportsEVD() override { return true; }
  bool SupportsSolve() override { return true; }
  void HadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) override;
  void Real(const MatrixXc& a, MatrixX* b) override;
  void PseudoInverseSvd(MatrixXc* matrix, Float alpha, MatrixXc* result) override;
  VectorXc MaxEigenVector(MatrixXc* matrix) override;
  void MatAdd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) override;
  void MatMul(TRANSPOSE trans_a, TRANSPOSE trans_b, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b, std::complex<Float> beta,
              MatrixXc* c) override;
  void MatVecMul(TRANSPOSE trans_a, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta, VectorXc* c) override;
  void VecAdd(Float alpha, const VectorX& a, Float beta, VectorX* b) override;
  void SolveCh(MatrixXc* a, VectorXc* b) override;
  void Solveg(MatrixX* a, VectorX* b, VectorX* c) override;
  Float Dot(const VectorX& a, const VectorX& b) override;
  std::complex<Float> DotC(const VectorXc& a, const VectorXc& b) override;
  Float MaxCoeff(const VectorX& v) override;
  Float MaxCoeffC(const VectorXc& v) override;
  MatrixXc ConcatRow(const MatrixXc& a, const MatrixXc& b) override;
  MatrixXc ConcatCol(const MatrixXc& a, const MatrixXc& b) override;
  void MatCpy(const MatrixX& a, MatrixX* b) override;
  void VecCpy(const VectorX& a, VectorX* b) override;
  void VecCpyC(const VectorXc& a, VectorXc* b) override;
};
#endif

#ifdef ENABLE_BLAS
class BLASBackend final
    : public Backend<utils::MatrixX<std::complex<Float>>, utils::VectorX<std::complex<Float>>, utils::MatrixX<Float>, utils::VectorX<Float>> {
 public:
  bool SupportsSvd() override { return true; }
  bool SupportsEVD() override { return true; }
  bool SupportsSolve() override { return true; }
  void HadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) override;
  void Real(const MatrixXc& a, MatrixX* b) override;
  void PseudoInverseSvd(MatrixXc* matrix, Float alpha, MatrixXc* result) override;
  VectorXc MaxEigenVector(MatrixXc* matrix) override;
  void MatAdd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) override;
  void MatMul(TRANSPOSE trans_a, TRANSPOSE trans_b, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b, std::complex<Float> beta,
              MatrixXc* c) override;
  void MatVecMul(TRANSPOSE trans_a, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta, VectorXc* c) override;
  void VecAdd(Float alpha, const VectorX& a, Float beta, VectorX* b) override;
  void SolveCh(MatrixXc* a, VectorXc* b) override;
  void Solveg(MatrixX* a, VectorX* b, VectorX* c) override;
  Float Dot(const VectorX& a, const VectorX& b) override;
  std::complex<Float> DotC(const VectorXc& a, const VectorXc& b) override;
  Float MaxCoeff(const VectorX& v) override;
  Float MaxCoeffC(const VectorXc& v) override;
  MatrixXc ConcatRow(const MatrixXc& a, const MatrixXc& b) override;
  MatrixXc ConcatCol(const MatrixXc& a, const MatrixXc& b) override;
  void MatCpy(const MatrixX& a, MatrixX* b) override;
  void VecCpy(const VectorX& a, VectorX* b) override;
  void VecCpyC(const VectorXc& a, VectorXc* b) override;
};
#endif
}  // namespace autd::gain::holo
