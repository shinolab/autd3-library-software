// File: linalg_backend.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <complex>

namespace autd::gain::holo {

class Backend;
using BackendPtr = std::shared_ptr<Backend>;

enum class TRANSPOSE { NO_TRANS = 111, TRANS = 112, CONJ_TRANS = 113, CONJ_NO_TRANS = 114 };

class Backend {
 public:
  using MatrixXc = Eigen::Matrix<std::complex<double>, -1, -1>;
  using VectorXc = Eigen::Matrix<std::complex<double>, -1, 1>;
  using MatrixX = Eigen::Matrix<double, -1, -1>;
  using VectorX = Eigen::Matrix<double, -1, 1>;

  virtual bool SupportsSvd() = 0;
  virtual bool SupportsEVD() = 0;
  virtual bool SupportsSolve() = 0;
  virtual void HadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) = 0;
  virtual void Real(const MatrixXc& a, MatrixX* b) = 0;
  virtual void PseudoInverseSVD(MatrixXc* matrix, double alpha, MatrixXc* result) = 0;
  virtual VectorXc MaxEigenVector(MatrixXc* matrix) = 0;
  virtual void MatAdd(double alpha, const MatrixX& a, double beta, MatrixX* b) = 0;
  virtual void MatMul(TRANSPOSE trans_a, TRANSPOSE trans_b, std::complex<double> alpha, const MatrixXc& a, const MatrixXc& b,
                      std::complex<double> beta, MatrixXc* c) = 0;
  virtual void MatVecMul(TRANSPOSE trans_a, std::complex<double> alpha, const MatrixXc& a, const VectorXc& b, std::complex<double> beta,
                         VectorXc* c) = 0;
  virtual void VecAdd(double alpha, const VectorX& a, double beta, VectorX* b) = 0;
  virtual void SolveCh(MatrixXc* a, VectorXc* b) = 0;
  virtual void Solveg(MatrixX* a, VectorX* b, VectorX* c) = 0;
  virtual double Dot(const VectorX& a, const VectorX& b) = 0;
  virtual std::complex<double> DotC(const VectorXc& a, const VectorXc& b) = 0;
  virtual double MaxCoeff(const VectorX& v) = 0;
  virtual double MaxCoeffC(const VectorXc& v) = 0;
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
}  // namespace autd::gain::holo
