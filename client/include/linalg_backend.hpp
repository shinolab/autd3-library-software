// File: linalg_backend.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <complex>
#include <memory>

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26450 26454 26495 26812)
#endif
#if defined(__GNUC__) && !defined(__llvm__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Dense>
#if _MSC_VER
#pragma warning(pop)
#endif
#if defined(__GNUC__) && !defined(__llvm__)
#pragma GCC diagnostic pop
#endif

namespace autd::gain::holo {

class Backend;
using BackendPtr = std::shared_ptr<Backend>;

enum class TRANSPOSE { NO_TRANS = 111, TRANS = 112, CONJ_TRANS = 113, CONJ_NO_TRANS = 114 };

/**
 * \brief Linear algebra calculation backend
 */
class Backend {
 public:
  using MatrixXc = Eigen::Matrix<std::complex<double>, -1, -1>;
  using VectorXc = Eigen::Matrix<std::complex<double>, -1, 1>;
  using MatrixX = Eigen::Matrix<double, -1, -1>;
  using VectorX = Eigen::Matrix<double, -1, 1>;

  virtual bool supports_svd() = 0;
  virtual bool supports_evd() = 0;
  virtual bool supports_solve() = 0;
  virtual void hadamard_product(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) = 0;
  virtual void real(const MatrixXc& a, MatrixX* b) = 0;
  virtual void pseudo_inverse_svd(MatrixXc* matrix, double alpha, MatrixXc* result) = 0;
  virtual VectorXc max_eigen_vector(MatrixXc* matrix) = 0;
  virtual void matrix_add(double alpha, const MatrixX& a, double beta, MatrixX* b) = 0;
  virtual void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, std::complex<double> alpha, const MatrixXc& a, const MatrixXc& b,
                          std::complex<double> beta, MatrixXc* c) = 0;
  virtual void matrix_vector_mul(TRANSPOSE trans_a, std::complex<double> alpha, const MatrixXc& a, const VectorXc& b, std::complex<double> beta,
                                 VectorXc* c) = 0;
  virtual void vector_add(double alpha, const VectorX& a, double beta, VectorX* b) = 0;
  virtual void solve_ch(MatrixXc* a, VectorXc* b) = 0;
  virtual void solve_g(MatrixX* a, VectorX* b, VectorX* c) = 0;
  virtual double dot(const VectorX& a, const VectorX& b) = 0;
  virtual std::complex<double> dot_c(const VectorXc& a, const VectorXc& b) = 0;
  virtual double max_coefficient(const VectorX& v) = 0;
  virtual double max_coefficient_c(const VectorXc& v) = 0;
  virtual MatrixXc concat_row(const MatrixXc& a, const MatrixXc& b) = 0;
  virtual MatrixXc concat_col(const MatrixXc& a, const MatrixXc& b) = 0;
  virtual void mat_cpy(const MatrixX& a, MatrixX* b) = 0;
  virtual void vec_cpy(const VectorX& a, VectorX* b) = 0;
  virtual void vec_cpy_c(const VectorXc& a, VectorXc* b) = 0;

  Backend() = default;
  virtual ~Backend() = default;
  Backend(const Backend& obj) = delete;
  Backend& operator=(const Backend& obj) = delete;
  Backend(const Backend&& v) = delete;
  Backend& operator=(Backend&& obj) = delete;
};
}  // namespace autd::gain::holo
