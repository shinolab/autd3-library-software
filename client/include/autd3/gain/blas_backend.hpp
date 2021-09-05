// File: blas_backend.hpp
// Project: include
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/09/2021
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

  void pseudo_inverse_svd(std::shared_ptr<MatrixXc> matrix, double alpha, std::shared_ptr<MatrixXc> result) override;
  std::shared_ptr<VectorXc> max_eigen_vector(std::shared_ptr<MatrixXc> matrix) override;
  void matrix_add(double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) override;
  void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, complex beta,
                  std::shared_ptr<MatrixXc> c) override;
  void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b, double beta,
                  std::shared_ptr<MatrixX> c) override;
  void matrix_vector_mul(TRANSPOSE trans_a, complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<VectorXc> b, complex beta,
                         std::shared_ptr<VectorXc> c) override;
  void matrix_vector_mul(TRANSPOSE trans_a, double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<VectorX> b, double beta,
                         std::shared_ptr<VectorX> c) override;
  void vector_add(double alpha, std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) override;
  void solve_ch(std::shared_ptr<MatrixXc> a, std::shared_ptr<VectorXc> b) override;
  void solve_g(std::shared_ptr<MatrixX> a, std::shared_ptr<VectorX> b, std::shared_ptr<VectorX> c) override;
  double dot(std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) override;
  complex dot(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b) override;
  double max_coefficient(std::shared_ptr<VectorX> v) override;
  double max_coefficient(std::shared_ptr<VectorXc> v) override;
  void mat_cpy(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) override;
  void vec_cpy(std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) override;
  void vec_cpy(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b) override;
};
}  // namespace autd::gain::holo
