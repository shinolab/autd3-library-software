// File: blas_backend.hpp
// Project: include
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "linalg_backend.hpp"

namespace autd::gain::holo {

class BLASBackend final : public Backend {
 public:
  static BackendPtr create();

  bool supports_svd() override;
  bool supports_evd() override;
  bool supports_solve() override;
  void hadamard_product(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) override;
  void real(const MatrixXc& a, MatrixX* b) override;
  void pseudo_inverse_svd(const MatrixXc& matrix, double alpha, MatrixXc* result) override;
  VectorXc max_eigen_vector(MatrixXc* matrix) override;
  void matrix_add(double alpha, const MatrixX& a, MatrixX* b) override;
  void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, std::complex<double> alpha, const MatrixXc& a, const MatrixXc& b, std::complex<double> beta,
                  MatrixXc* c) override;
  void matrix_vector_mul(TRANSPOSE trans_a, std::complex<double> alpha, const MatrixXc& a, const VectorXc& b, std::complex<double> beta,
                         VectorXc* c) override;
  void vector_add(double alpha, const VectorX& a, VectorX* b) override;
  void solve_ch(MatrixXc* a, VectorXc* b) override;
  void solve_g(MatrixX* a, VectorX* b, VectorX* c) override;
  double dot(const VectorX& a, const VectorX& b) override;
  std::complex<double> dot_c(const VectorXc& a, const VectorXc& b) override;
  double max_coefficient(const VectorX& v) override;
  double max_coefficient_c(const VectorXc& v) override;
  MatrixXc concat_row(const MatrixXc& a, const MatrixXc& b) override;
  MatrixXc concat_col(const MatrixXc& a, const MatrixXc& b) override;
  void mat_cpy(const MatrixX& a, MatrixX* b) override;
  void vec_cpy(const VectorX& a, VectorX* b) override;
  void vec_cpy_c(const VectorXc& a, VectorXc* b) override;
};
}  // namespace autd::gain::holo
