// File: blas_backend.hpp
// Project: include
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "eigen_backend.hpp"

namespace autd::gain::holo {

class BLASBackend final : public Eigen3Backend {
 public:
  static BackendPtr create();

  void scale(std::shared_ptr<MatrixXc> a, complex s) override;
  void pseudo_inverse_svd(std::shared_ptr<MatrixXc> matrix, double alpha, std::shared_ptr<MatrixXc> result) override;
  std::shared_ptr<MatrixXc> max_eigen_vector(std::shared_ptr<MatrixXc> matrix) override;
  void matrix_add(double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) override;
  void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, complex beta,
                  std::shared_ptr<MatrixXc> c) override;
  void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b, double beta,
                  std::shared_ptr<MatrixX> c) override;
  void solve_ch(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) override;
  void solve_g(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b, std::shared_ptr<MatrixX> c) override;
  double dot(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) override;
  complex dot(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) override;
  double max_coefficient(std::shared_ptr<MatrixX> v) override;
  double max_coefficient(std::shared_ptr<MatrixXc> v) override;
  void mat_cpy(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) override;
  void mat_cpy(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) override;
};
}  // namespace autd::gain::holo
