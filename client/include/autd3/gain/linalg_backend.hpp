// File: linalg_backend.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <complex>
#include <memory>

#include "autd3/core/geometry.hpp"
#include "autd3/core/hardware_defined.hpp"

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26450 26451 26454 26495 26812)
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

namespace autd {
namespace gain {
namespace holo {

class Backend;
using BackendPtr = std::shared_ptr<Backend>;

enum class TRANSPOSE { NO_TRANS = 111, TRANS = 112, CONJ_TRANS = 113, CONJ_NO_TRANS = 114 };

template <typename T>
struct Vector {
  explicit Vector(const Eigen::Index n) : data(n) {}
  explicit Vector(Eigen::Matrix<T, -1, 1, Eigen::ColMajor> vec) : data(std::move(vec)) {}
  Vector(const Vector& obj) = delete;
  Vector& operator=(const Vector& obj) = delete;
  Vector(const Vector&& v) = delete;
  Vector& operator=(Vector&& obj) = delete;

  virtual ~Vector() = default;

  virtual void fill(T v) { data.fill(v); }
  virtual void copy_from(const std::vector<T>& v) { std::memcpy(data.data(), v.data(), sizeof(T) * data.size()); }
  virtual void copy_from(const T* v) { std::memcpy(data.data(), v, sizeof(T) * data.size()); }
  virtual void copy_to_host() {}

  Eigen::Matrix<T, -1, 1, Eigen::ColMajor> data;
};

template <typename T>
struct Matrix {
  explicit Matrix(const Eigen::Index row, const Eigen::Index col) : data(row, col) {}
  explicit Matrix(Eigen::Matrix<T, -1, -1, Eigen::ColMajor> mat) : data(std::move(mat)) {}
  virtual ~Matrix() = default;
  Matrix(const Matrix& obj) = delete;
  Matrix& operator=(const Matrix& obj) = delete;
  Matrix(const Matrix&& v) = delete;
  Matrix& operator=(Matrix&& obj) = delete;

  virtual void fill(T v) { data.fill(v); }
  virtual void copy_from(const std::vector<T>& v) { std::memcpy(data.data(), v.data(), sizeof(T) * data.size()); }
  virtual void copy_from(const T* v) { std::memcpy(data.data(), v, sizeof(T) * data.size()); }
  virtual void copy_to_host() {}

  Eigen::Matrix<T, -1, -1, Eigen::ColMajor> data;
};

using complex = std::complex<double>;
constexpr complex One = complex(1.0, 0.0);
constexpr complex Zero = complex(0.0, 0.0);
using MatrixXc = Matrix<complex>;
using VectorXc = Vector<complex>;
using MatrixX = Matrix<double>;
using VectorX = Vector<double>;

/**
 * \brief Linear algebra calculation backend
 */
class Backend {
 protected:
  std::unordered_map<std::string, std::shared_ptr<VectorX>> _cache_vec;
  std::unordered_map<std::string, std::shared_ptr<VectorXc>> _cache_vec_c;
  std::unordered_map<std::string, std::shared_ptr<MatrixX>> _cache_mat;
  std::unordered_map<std::string, std::shared_ptr<MatrixXc>> _cache_mat_c;

  template <typename T>
  static std::shared_ptr<T> allocate_vector_impl(const std::string& name, const Eigen::Index n,
                                                 std::unordered_map<std::string, std::shared_ptr<T>>& cache) {
    if (const auto it = cache.find(name); it != cache.end()) {
      if (it->second->data.size() == n) return it->second;
      cache.erase(name);
    }
    auto v = std::make_shared<T>(n);
    cache.emplace(name, v);
    return v;
  }

  template <typename T>
  static std::shared_ptr<T> allocate_matrix_impl(const std::string& name, const Eigen::Index row, const Eigen::Index col,
                                                 std::unordered_map<std::string, std::shared_ptr<T>>& cache) {
    if (const auto it = cache.find(name); it != cache.end()) {
      if (it->second->data.rows() == row && it->second->data.cols() == col) return it->second;
      cache.erase(name);
    }
    auto v = std::make_shared<T>(row, col);
    cache.emplace(name, v);
    return v;
  }

 public:
  virtual std::shared_ptr<VectorX> allocate_vector(const std::string& name, const Eigen::Index n) {
    return allocate_vector_impl(name, n, _cache_vec);
  }

  virtual std::shared_ptr<VectorXc> allocate_vector_c(const std::string& name, const Eigen::Index n) {
    return allocate_vector_impl(name, n, _cache_vec_c);
  }

  virtual std::shared_ptr<MatrixX> allocate_matrix(const std::string& name, const Eigen::Index row, const Eigen::Index col) {
    return allocate_matrix_impl(name, row, col, _cache_mat);
  }

  virtual std::shared_ptr<MatrixXc> allocate_matrix_c(const std::string& name, const Eigen::Index row, const Eigen::Index col) {
    return allocate_matrix_impl(name, row, col, _cache_mat_c);
  }

  virtual void hadamard_product(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, std::shared_ptr<MatrixXc> c) = 0;
  virtual void hadamard_product(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b, std::shared_ptr<VectorXc> c) = 0;
  virtual void real(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixX> b) = 0;
  virtual void arg(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> c) = 0;
  virtual void pseudo_inverse_svd(std::shared_ptr<MatrixXc> matrix, double alpha, std::shared_ptr<MatrixXc> result) = 0;
  virtual std::shared_ptr<VectorXc> max_eigen_vector(std::shared_ptr<MatrixXc> matrix) = 0;
  virtual void matrix_add(double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) = 0;
  virtual void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b, complex beta,
                          std::shared_ptr<MatrixXc> c) = 0;
  virtual void matrix_mul(TRANSPOSE trans_a, TRANSPOSE trans_b, double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b, double beta,
                          std::shared_ptr<MatrixX> c) = 0;
  virtual void matrix_vector_mul(TRANSPOSE trans_a, complex alpha, std::shared_ptr<MatrixXc> a, std::shared_ptr<VectorXc> b, complex beta,
                                 std::shared_ptr<VectorXc> c) = 0;
  virtual void matrix_vector_mul(TRANSPOSE trans_a, double alpha, std::shared_ptr<MatrixX> a, std::shared_ptr<VectorX> b, double beta,
                                 std::shared_ptr<VectorX> c) = 0;
  virtual void vector_add(double alpha, std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) = 0;
  virtual void solve_ch(std::shared_ptr<MatrixXc> a, std::shared_ptr<VectorXc> b) = 0;
  virtual void solve_g(std::shared_ptr<MatrixX> a, std::shared_ptr<VectorX> b, std::shared_ptr<VectorX> c) = 0;
  virtual double dot(std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) = 0;
  virtual complex dot(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b) = 0;
  virtual double max_coefficient(std::shared_ptr<VectorX> v) = 0;
  virtual double max_coefficient(std::shared_ptr<VectorXc> v) = 0;
  virtual std::shared_ptr<MatrixXc> concat_row(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) = 0;
  virtual std::shared_ptr<MatrixXc> concat_col(std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixXc> b) = 0;
  virtual void mat_cpy(std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) = 0;
  virtual void vec_cpy(std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) = 0;
  virtual void vec_cpy(std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b) = 0;

  virtual void set_from_complex_drive(std::vector<core::DataArray>& data, std::shared_ptr<VectorXc> drive, bool normalize,
                                      double max_coefficient) = 0;
  virtual std::shared_ptr<MatrixXc> transfer_matrix(const std::vector<core::Vector3>& foci, const core::GeometryPtr& geometry) = 0;

  Backend() = default;
  virtual ~Backend() = default;
  Backend(const Backend& obj) = delete;
  Backend& operator=(const Backend& obj) = delete;
  Backend(const Backend&& v) = delete;
  Backend& operator=(Backend&& obj) = delete;
};
}  // namespace holo
}  // namespace gain
}  // namespace autd
