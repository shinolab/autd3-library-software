// File: blas_backend.hpp
// Project: include
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "eigen_backend.hpp"

namespace autd::gain::holo {

/**
 * \brief BLAS matrix
 */
template <typename T>
struct BLASMatrix final : EigenMatrix<T> {
  explicit BLASMatrix(const size_t row, const size_t col) : EigenMatrix(row, col) {}
  explicit BLASMatrix(Eigen::Matrix<T, -1, -1, Eigen::ColMajor> other) : EigenMatrix(std::move(other)) {}
  ~BLASMatrix() override = default;
  BLASMatrix(const BLASMatrix& obj) = delete;
  BLASMatrix& operator=(const BLASMatrix& obj) = delete;
  BLASMatrix(const BLASMatrix&& v) = delete;
  BLASMatrix& operator=(BLASMatrix&& obj) = delete;

  void pseudo_inverse_svd(const std::shared_ptr<EigenMatrix<T>>& matrix, double alpha, const std::shared_ptr<EigenMatrix<T>>& u,
                          const std::shared_ptr<EigenMatrix<T>>& s, const std::shared_ptr<EigenMatrix<T>>& vt,
                          const std::shared_ptr<EigenMatrix<T>>& buf) override;
  void max_eigen_vector(const std::shared_ptr<EigenMatrix<T>>& ev) override;
  void add(T alpha, const std::shared_ptr<EigenMatrix<T>>& a) override;
  void mul(TRANSPOSE trans_a, TRANSPOSE trans_b, T alpha, const std::shared_ptr<const EigenMatrix<T>>& a,
           const std::shared_ptr<const EigenMatrix<T>>& b, T beta) override;
  void solve(const std::shared_ptr<EigenMatrix<T>>& b) override;
  T dot(const std::shared_ptr<const EigenMatrix<T>>& a) override;
  [[nodiscard]] double max_element() const override;

  void copy_from(const std::shared_ptr<const EigenMatrix<T>>& a) override { copy_from(a->data.data(), a->data.size()); }
  void copy_from(const std::vector<T>& v) override { copy_from(v.data(), v.size()); }
  void copy_from(const T* v) override { copy_from(v, this->data.size()); }
  void copy_from(const T* v, size_t);
};

using BLASBackend = MatrixBufferPool<BLASMatrix<double>, BLASMatrix<complex>, Context>;

}  // namespace autd::gain::holo
