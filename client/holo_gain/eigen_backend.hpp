// File: eigen_backend.hpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "linalg_backend.hpp"

namespace autd::gain::holo {

class Eigen3Backend final : public Backend {
 public:
  bool SupportsSvd() override { return true; }
  bool SupportsEVD() override { return true; }
  bool SupportsSolve() override { return true; }
  void HadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) override;
  void Real(const MatrixXc& a, MatrixX* b) override;
  void PseudoInverseSVD(MatrixXc* matrix, double alpha, MatrixXc* result) override;
  VectorXc MaxEigenVector(MatrixXc* matrix) override;
  void MatAdd(double alpha, const MatrixX& a, double beta, MatrixX* b) override;
  void MatMul(TRANSPOSE trans_a, TRANSPOSE trans_b, std::complex<double> alpha, const MatrixXc& a, const MatrixXc& b, std::complex<double> beta,
              MatrixXc* c) override;
  void MatVecMul(TRANSPOSE trans_a, std::complex<double> alpha, const MatrixXc& a, const VectorXc& b, std::complex<double> beta,
                 VectorXc* c) override;
  void VecAdd(double alpha, const VectorX& a, double beta, VectorX* b) override;
  void SolveCh(MatrixXc* a, VectorXc* b) override;
  void Solveg(MatrixX* a, VectorX* b, VectorX* c) override;
  double Dot(const VectorX& a, const VectorX& b) override;
  std::complex<double> DotC(const VectorXc& a, const VectorXc& b) override;
  double MaxCoeff(const VectorX& v) override;
  double MaxCoeffC(const VectorXc& v) override;
  MatrixXc ConcatRow(const MatrixXc& a, const MatrixXc& b) override;
  MatrixXc ConcatCol(const MatrixXc& a, const MatrixXc& b) override;
  void MatCpy(const MatrixX& a, MatrixX* b) override;
  void VecCpy(const VectorX& a, VectorX* b) override;
  void VecCpyC(const VectorXc& a, VectorXc* b) override;
};
}  // namespace autd::gain::holo
