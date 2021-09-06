// File: eigen_backend.hpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 07/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
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
#include <Eigen/Dense>
#if _MSC_VER
#pragma warning(pop)
#endif
#if defined(__GNUC__) && !defined(__llvm__)
#pragma GCC diagnostic pop
#endif

#include "linalg_backend.hpp"

namespace autd::gain::holo {
template <typename T>
struct EigenMatrix final : Matrix<T> {
  explicit EigenMatrix(const Eigen::Index row, const Eigen::Index col) : Matrix<T>(row, col) {}
  explicit EigenMatrix(Eigen::Matrix<T, -1, -1, Eigen::ColMajor> mat) : Matrix<T>(std::move(mat)) {}
  ~EigenMatrix() override = default;
  EigenMatrix(const EigenMatrix& obj) = delete;
  EigenMatrix& operator=(const EigenMatrix& obj) = delete;
  EigenMatrix(const EigenMatrix&& v) = delete;
  EigenMatrix& operator=(EigenMatrix&& obj) = delete;

  [[nodiscard]] T at(size_t row, size_t col) const override { return data(row, col); }
  [[nodiscard]] const T* ptr() const override { return data.data(); }
  T* ptr() override { return data.data(); }

  [[nodiscard]] double max_element() const override;
  void set(const Eigen::Index row, const Eigen::Index col, T v) override { data(row, col) = v; }
  void get_col(const Eigen::Index i, std::shared_ptr<Matrix<T>> dst) override {
    const auto& col = data.col(i);
    std::memcpy(dst->data.data(), col.data(), sizeof(T) * col.size());
  }
  void fill(T v) override { data.fill(v); }
  void get_diagonal(std::shared_ptr<Matrix<T>> v) override {
    for (Eigen::Index i = 0; i < (std::min)(data.rows(), data.cols()); i++) v->data(i) = data(i, i);
  }
  void set_diagonal(std::shared_ptr<Matrix<T>> v) override { data.diagonal() = v->data; }
  void copy_from(const std::vector<T>& v) override { std::memcpy(data.data(), v.data(), sizeof(T) * v.size()); }
  void copy_from(const T* v) override { std::memcpy(data.data(), v, sizeof(T) * data.size()); }
  void copy_to_host() override {}
};

template <>
inline double EigenMatrix<double>::max_element() const {
  return data.maxCoeff();
}

template <>
inline double EigenMatrix<complex>::max_element() const {
  return std::sqrt(data.cwiseAbs2().maxCoeff());
}

/**
 * \brief Linear algebra calculation backend using Eigen3.
 */
class Eigen3Backend : public Backend {
  std::unordered_map<std::string, std::shared_ptr<MatrixX>> _cache_mat;
  std::unordered_map<std::string, std::shared_ptr<MatrixXc>> _cache_mat_c;

  template <typename T, typename E>
  static std::shared_ptr<T> allocate_matrix_impl(const std::string& name, const size_t row, const size_t col,
                                                 std::unordered_map<std::string, std::shared_ptr<T>>& cache) {
    if (const auto it = cache.find(name); it != cache.end()) {
      if (static_cast<size_t>(it->second->data.rows()) == row && static_cast<size_t>(it->second->data.cols()) == col) return it->second;
      cache.erase(name);
    }
    std::shared_ptr<T> v = std::make_shared<E>(row, col);
    cache.emplace(name, v);
    return v;
  }

 public:
  static BackendPtr create();

  std::shared_ptr<MatrixX> allocate_matrix(const std::string& name, const size_t row, const size_t col) override {
    return allocate_matrix_impl<MatrixX, EigenMatrix<double>>(name, row, col, _cache_mat);
  }

  std::shared_ptr<MatrixXc> allocate_matrix_c(const std::string& name, const size_t row, const size_t col) override {
    return allocate_matrix_impl<MatrixXc, EigenMatrix<complex>>(name, row, col, _cache_mat_c);
  }

  void make_complex(std::shared_ptr<MatrixX> r, std::shared_ptr<MatrixX> i, std::shared_ptr<MatrixXc> c) override;
  void exp(std::shared_ptr<MatrixXc> a) override;
  void scale(std::shared_ptr<MatrixXc> a, complex s) override;
  void hadamard_product(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, std::shared_ptr<MatrixXc> c) override;
  void real(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixX> b) override;
  void arg(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> c) override;
  void pseudo_inverse_svd(std::shared_ptr<MatrixXc> matrix, double alpha, std::shared_ptr<MatrixXc> result) override;
  void max_eigen_vector(std::shared_ptr<MatrixXc> matrix, std::shared_ptr<MatrixXc> ev) override;
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
  std::shared_ptr<MatrixXc> back_prop(std::shared_ptr<MatrixXc> transfer, std::shared_ptr<MatrixXc> amps) override;
  std::shared_ptr<MatrixXc> sigma_regularization(std::shared_ptr<MatrixXc> transfer, std::shared_ptr<MatrixXc> amps, double gamma) override;
  void col_sum_imag(std::shared_ptr<MatrixXc> mat, std::shared_ptr<MatrixX> dst) override;
};
}  // namespace autd::gain::holo
