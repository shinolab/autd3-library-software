// File: linalg_backend.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6031 26450 26451 26454 26495 26812)
#endif
#if defined(__GNUC__) && !defined(__llvm__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Core>
#if _MSC_VER
#pragma warning(pop)
#endif
#if defined(__GNUC__) && !defined(__llvm__)
#pragma GCC diagnostic pop
#endif

#include "autd3/core/hardware_defined.hpp"

namespace autd {
namespace gain {
namespace holo {

class Backend;
using BackendPtr = std::shared_ptr<Backend>;

enum class TRANSPOSE { NO_TRANS = 111, TRANS = 112, CONJ_TRANS = 113 };

/**
 * \brief Matrix wrapper for Backend
 */
template <typename T>
struct Matrix {
  explicit Matrix(const Eigen::Index row, const Eigen::Index col) : data(row, col) {}
  explicit Matrix(Eigen::Matrix<T, -1, -1, Eigen::ColMajor> mat) : data(std::move(mat)) {}
  virtual ~Matrix() = default;
  Matrix(const Matrix& obj) = delete;
  Matrix& operator=(const Matrix& obj) = delete;
  Matrix(const Matrix&& v) = delete;
  Matrix& operator=(Matrix&& obj) = delete;

  [[nodiscard]] virtual T at(size_t row, size_t col) const = 0;
  [[nodiscard]] virtual const T* ptr() const = 0;
  virtual T* ptr() = 0;
  virtual void set(Eigen::Index row, Eigen::Index col, T v) = 0;
  virtual void get_col(Eigen::Index i, std::shared_ptr<Matrix<T>> dst) = 0;
  [[nodiscard]] virtual double max_element() const = 0;
  virtual void fill(T v) = 0;
  virtual void get_diagonal(std::shared_ptr<Matrix<T>> v) = 0;
  virtual void set_diagonal(std::shared_ptr<Matrix<T>> v) = 0;
  virtual void copy_from(const std::vector<T>& v) = 0;
  virtual void copy_from(const T* v) = 0;
  virtual void copy_to_host() = 0;

  Eigen::Matrix<T, -1, -1, Eigen::ColMajor> data;
};

using complex = std::complex<double>;
constexpr complex ONE = complex(1.0, 0.0);
constexpr complex ZERO = complex(0.0, 0.0);
using MatrixXc = Matrix<complex>;
using MatrixX = Matrix<double>;

/**
 * \brief Linear algebra calculation backend
 */
class Backend {
 public:
  Backend() = default;
  virtual ~Backend() = default;
  Backend(const Backend& obj) = delete;
  Backend& operator=(const Backend& obj) = delete;
  Backend(const Backend&& v) = delete;
  Backend& operator=(Backend&& obj) = delete;

  virtual std::shared_ptr<MatrixX> allocate_matrix(const std::string& name, size_t row, size_t col) = 0;
  virtual std::shared_ptr<MatrixXc> allocate_matrix_c(const std::string& name, size_t row, size_t col) = 0;

  virtual void make_complex(std::shared_ptr<MatrixX> r, std::shared_ptr<MatrixX> i, std::shared_ptr<MatrixXc> c) = 0;
  virtual void exp(std::shared_ptr<MatrixXc> a) = 0;
  virtual void scale(std::shared_ptr<MatrixXc> a, complex s) = 0;
  virtual void hadamard_product(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, std::shared_ptr<MatrixXc> c) = 0;
  virtual void real(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixX> b) = 0;
  virtual void arg(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> c) = 0;
  virtual void pseudo_inverse_svd(std::shared_ptr<MatrixXc> matrix, double alpha, std::shared_ptr<MatrixXc> result) = 0;
  virtual void pseudo_inverse_svd(std::shared_ptr<MatrixX> matrix, double alpha, std::shared_ptr<MatrixX> result) = 0;
  virtual void max_eigen_vector(std::shared_ptr<MatrixXc> matrix, std::shared_ptr<MatrixXc> ev) = 0;
  virtual void matrix_add(double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) = 0;
  virtual void matrix_add(complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) = 0;
  virtual void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, complex beta,
                          std::shared_ptr<MatrixXc> c) = 0;
  virtual void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b, double beta,
                          std::shared_ptr<MatrixX> c) = 0;
  virtual void solve_ch(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) = 0;
  virtual void solve_g(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b, std::shared_ptr<MatrixX> c) = 0;
  virtual double dot(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) = 0;
  virtual complex dot(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) = 0;
  virtual double max_coefficient(std::shared_ptr<MatrixX> v) = 0;
  virtual double max_coefficient(std::shared_ptr<MatrixXc> v) = 0;
  virtual std::shared_ptr<MatrixXc> concat_row(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) = 0;
  virtual std::shared_ptr<MatrixXc> concat_col(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) = 0;
  virtual void mat_cpy(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) = 0;
  virtual void mat_cpy(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) = 0;
  virtual void set_from_complex_drive(std::vector<core::DataArray>& data, std::shared_ptr<MatrixXc> drive, bool normalize,
                                      double max_coefficient) = 0;
  virtual std::shared_ptr<MatrixXc> transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions,
                                                    const std::vector<const double*>& directions, double wavelength, double attenuation) = 0;

  // FIXME: following functions are too specialized
  virtual void set_bcd_result(std::shared_ptr<MatrixXc> mat, std::shared_ptr<MatrixXc> vec, size_t index) = 0;
  virtual std::shared_ptr<MatrixXc> back_prop(std::shared_ptr<MatrixXc> transfer, std::shared_ptr<MatrixXc> amps) = 0;
  virtual std::shared_ptr<MatrixXc> sigma_regularization(std::shared_ptr<MatrixXc> transfer, std::shared_ptr<MatrixXc> amps, double gamma) = 0;
  virtual void col_sum_imag(std::shared_ptr<MatrixXc> mat, std::shared_ptr<MatrixX> dst) = 0;
};
}  // namespace holo
}  // namespace gain
}  // namespace autd
