// File: eigen_matrix.hpp
// Project: eigen
// Created Date: 06/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6031 6255 6294 26450 26451 26454 26495 26812)
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

#include "autd3/core/hardware_defined.hpp"
#include "autd3/gain/matrix.hpp"
#include "autd3/utils.hpp"

namespace autd::gain::holo {

/**
 * \brief Eigen3 matrix
 */
template <typename T>
struct EigenMatrix {
  Eigen::Matrix<T, -1, -1, Eigen::ColMajor> data;

  explicit EigenMatrix(const size_t row, const size_t col) : data(row, col) {}
  explicit EigenMatrix(Eigen::Matrix<T, -1, -1, Eigen::ColMajor> other) : data(std::move(other)) {}
  virtual ~EigenMatrix() = default;
  EigenMatrix(const EigenMatrix& obj) = delete;
  EigenMatrix& operator=(const EigenMatrix& obj) = delete;
  EigenMatrix(const EigenMatrix&& v) = delete;
  EigenMatrix& operator=(EigenMatrix&& obj) = delete;

  void make_complex(const std::shared_ptr<const EigenMatrix<double>>& r, const std::shared_ptr<const EigenMatrix<double>>& i);
  void exp() { data = data.array().exp(); }
  void scale(const T s) { data *= s; }
  void reciprocal(const std::shared_ptr<const EigenMatrix<T>>& src) {
    data = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>::Ones(src->data.rows(), src->data.cols()).cwiseQuotient(src->data);
  }
  void abs(const std::shared_ptr<const EigenMatrix<T>>& src) { data = src->data.cwiseAbs(); }
  void real(const std::shared_ptr<const EigenMatrix<complex>>& src);
  void arg(const std::shared_ptr<const EigenMatrix<complex>>& src);
  void hadamard_product(const std::shared_ptr<const EigenMatrix<T>>& a, const std::shared_ptr<const EigenMatrix<T>>& b) {
    data.noalias() = a->data.cwiseProduct(b->data);
  }
  virtual void pseudo_inverse_svd(const std::shared_ptr<EigenMatrix<T>>& matrix, double alpha, const std::shared_ptr<EigenMatrix<T>>&,
                                  const std::shared_ptr<EigenMatrix<T>>&, const std::shared_ptr<EigenMatrix<T>>&,
                                  const std::shared_ptr<EigenMatrix<T>>&) {
    const Eigen::BDCSVD svd(matrix->data, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto singular_values_inv = svd.singularValues();
    const auto size = singular_values_inv.size();
    for (Eigen::Index i = 0; i < size; i++)
      singular_values_inv(i) = singular_values_inv(i) / (singular_values_inv(i) * singular_values_inv(i) + alpha);
    data.noalias() = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().adjoint();
  }
  virtual void max_eigen_vector(const std::shared_ptr<EigenMatrix<T>>& ev);
  virtual void add(const T alpha, const std::shared_ptr<EigenMatrix<T>>& a) { data.noalias() += alpha * a->data; }
  virtual void mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const T alpha, const std::shared_ptr<const EigenMatrix<T>>& a,
                   const std::shared_ptr<const EigenMatrix<T>>& b, const T beta) {
    data *= beta;
    switch (trans_a) {
      case TRANSPOSE::CONJ_TRANS:
        switch (trans_b) {
          case TRANSPOSE::CONJ_TRANS:
            data.noalias() += alpha * (a->data.adjoint() * b->data.adjoint());
            break;
          case TRANSPOSE::TRANS:
            data.noalias() += alpha * (a->data.adjoint() * b->data.transpose());
            break;
          case TRANSPOSE::NO_TRANS:
            data.noalias() += alpha * (a->data.adjoint() * b->data);
            break;
        }
        break;
      case TRANSPOSE::TRANS:
        switch (trans_b) {
          case TRANSPOSE::CONJ_TRANS:
            data.noalias() += alpha * (a->data.transpose() * b->data.adjoint());
            break;
          case TRANSPOSE::TRANS:
            data.noalias() += alpha * (a->data.transpose() * b->data.transpose());
            break;
          case TRANSPOSE::NO_TRANS:
            data.noalias() += alpha * (a->data.transpose() * b->data);
            break;
        }
        break;
      case TRANSPOSE::NO_TRANS:
        switch (trans_b) {
          case TRANSPOSE::CONJ_TRANS:
            data.noalias() += alpha * (a->data * b->data.adjoint());
            break;
          case TRANSPOSE::TRANS:
            data.noalias() += alpha * (a->data * b->data.transpose());
            break;
          case TRANSPOSE::NO_TRANS:
            data.noalias() += alpha * (a->data * b->data);
            break;
        }
        break;
    }
  }
  virtual void solve(const std::shared_ptr<EigenMatrix<T>>& b) {
    const Eigen::LLT<Eigen::Matrix<T, -1, -1, Eigen::ColMajor>> llt(data);
    llt.solveInPlace(b->data);
  }
  virtual T dot(const std::shared_ptr<const EigenMatrix<T>>& a) { return (data.adjoint() * a->data)(0); }
  [[nodiscard]] virtual double max_element() const;
  void concat_row(const std::shared_ptr<const EigenMatrix<T>>& a, const std::shared_ptr<const EigenMatrix<T>>& b) { data << a->data, b->data; }
  void concat_col(const std::shared_ptr<const EigenMatrix<T>>& a, const std::shared_ptr<const EigenMatrix<T>>& b) { data << a->data, b->data; }

  [[nodiscard]] T at(const size_t row, const size_t col) const { return data(row, col); }
  [[nodiscard]] size_t rows() const { return data.rows(); }
  [[nodiscard]] size_t cols() const { return data.cols(); }

  void set(const size_t row, const size_t col, T v) { data(row, col) = v; }
  void get_col(const std::shared_ptr<const EigenMatrix<T>>& src, const size_t i) {
    const auto& col = src->data.col(i);
    std::memcpy(data.data(), col.data(), sizeof(T) * col.size());
  }
  void fill(T v) { data.fill(v); }
  void get_diagonal(const std::shared_ptr<const EigenMatrix<T>>& src) {
    for (Eigen::Index i = 0; i < (std::min)(src->data.rows(), src->data.cols()); i++) data(i, 0) = src->data(i, i);
  }
  void create_diagonal(const std::shared_ptr<const EigenMatrix<T>>& v) {
    fill(0.0);
    data.diagonal() = v->data;
  }
  virtual void copy_from(const std::shared_ptr<const EigenMatrix<T>>& a) { data = a->data; }
  virtual void copy_from(const std::vector<T>& v) { std::memcpy(this->data.data(), v.data(), sizeof(T) * v.size()); }
  virtual void copy_from(const T* v) { std::memcpy(this->data.data(), v, sizeof(T) * this->data.size()); }

  // FIXME: following functions are too specialized
  void transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions, const std::vector<const double*>& directions,
                       double wavelength, double attenuation);
  void set_bcd_result(const std::shared_ptr<const EigenMatrix<T>>& vec, size_t index);
  void set_from_complex_drive(std::vector<core::DataArray>& dst, bool normalize, double max_coefficient);
  void set_from_arg(std::vector<core::DataArray>& dst, size_t n);
  void back_prop(const std::shared_ptr<const EigenMatrix<T>>& transfer, const std::shared_ptr<const EigenMatrix<T>>& amps);
  void sigma_regularization(const std::shared_ptr<const EigenMatrix<T>>& transfer, const std::shared_ptr<const EigenMatrix<T>>& amps, double gamma);
  void col_sum_imag(const std::shared_ptr<EigenMatrix<complex>>& src);
};
}  // namespace autd::gain::holo
