// File: cuda_backend.hpp
// Project: gain
// Created Date: 04/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cublas_v2.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "linalg_backend.hpp"

namespace autd {
namespace gain {
namespace holo {

class CUDABackend final : public Backend {
  std::unordered_map<std::string, std::shared_ptr<MatrixX>> _cache_mat;
  std::unordered_map<std::string, std::shared_ptr<MatrixXc>> _cache_mat_c;

 public:
  CUDABackend();
  ~CUDABackend() override;
  CUDABackend(const CUDABackend& obj) = delete;
  CUDABackend& operator=(const CUDABackend& obj) = delete;
  CUDABackend(const CUDABackend&& v) = delete;
  CUDABackend& operator=(CUDABackend&& obj) = delete;

  std::shared_ptr<MatrixX> allocate_matrix(const std::string& name, size_t row, size_t col) override;
  std::shared_ptr<MatrixXc> allocate_matrix_c(const std::string& name, size_t row, size_t col) override;

  static BackendPtr create();

  void make_complex(std::shared_ptr<MatrixX> r, std::shared_ptr<MatrixX> i, std::shared_ptr<MatrixXc> c) override;
  void exp(std::shared_ptr<MatrixXc> a) override;
  void scale(std::shared_ptr<MatrixXc> a, complex s) override;
  void hadamard_product(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, std::shared_ptr<MatrixXc> c) override;
  void real(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixX> b) override;
  void arg(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> c) override;
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
  std::shared_ptr<MatrixXc> concat_row(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) override;
  std::shared_ptr<MatrixXc> concat_col(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) override;
  void mat_cpy(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) override;
  void mat_cpy(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) override;

  void set_from_complex_drive(std::vector<core::DataArray>& data, std::shared_ptr<MatrixXc> drive, bool normalize, double max_coefficient) override;
  std::shared_ptr<MatrixXc> transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions,
                                            const std::vector<const double*>& directions, double wavelength, double attenuation) override;

  void set_bcd_result(std::shared_ptr<MatrixXc> mat, std::shared_ptr<MatrixXc> vec, size_t index) override;
  std::shared_ptr<MatrixXc> back_prop(std::shared_ptr<MatrixXc> transfer, const std::vector<complex>& amps) override;
  std::shared_ptr<MatrixXc> sigma_regularization(std::shared_ptr<MatrixXc> transfer, const std::vector<complex>& amps, double gamma) override;
  void col_sum_imag(std::shared_ptr<MatrixXc> mat, std::shared_ptr<MatrixX> dst) override;

 private:
  cublasHandle_t _handle;
};
}  // namespace holo
}  // namespace gain
}  // namespace autd
