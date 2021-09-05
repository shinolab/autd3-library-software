// File: eigen_backend.hpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "linalg_backend.hpp"

namespace autd::gain::holo {

/**
 * \brief Linear algebra calculation backend using Eigen3.
 */
class Eigen3Backend : public Backend {
 public:
  static BackendPtr create();

  void make_complex(std::shared_ptr<VectorX> r, std::shared_ptr<VectorX> i, std::shared_ptr<VectorXc> c) override;
  void exp(std::shared_ptr<VectorXc> a) override;
  void scale(std::shared_ptr<VectorXc> a, complex s) override;
  void hadamard_product(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, std::shared_ptr<MatrixXc> c) override;
  void hadamard_product(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b, std::shared_ptr<VectorXc> c) override;
  void real(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixX> b) override;
  void arg(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> c) override;
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
  std::shared_ptr<MatrixXc> concat_row(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) override;
  std::shared_ptr<MatrixXc> concat_col(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) override;
  void mat_cpy(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) override;
  void vec_cpy(std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) override;
  void vec_cpy(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b) override;

  void set_from_complex_drive(std::vector<core::DataArray>& data, std::shared_ptr<VectorXc> drive, bool normalize, double max_coefficient) override;
  std::shared_ptr<MatrixXc> transfer_matrix(const std::vector<core::Vector3>& foci, const core::GeometryPtr& geometry) override;

  void set_bcd_result(std::shared_ptr<MatrixXc> mat, std::shared_ptr<VectorXc> vec, Eigen::Index idx) override;
  std::shared_ptr<MatrixXc> back_prop(std::shared_ptr<MatrixXc> transfer, const std::vector<complex>& amps) override;
  std::shared_ptr<MatrixXc> sigma_regularization(std::shared_ptr<MatrixXc> transfer, const std::vector<complex>& amps, double gamma) override;
  void col_sum_imag(std::shared_ptr<MatrixXc> mat, std::shared_ptr<VectorX> dst) override;
};
}  // namespace autd::gain::holo
