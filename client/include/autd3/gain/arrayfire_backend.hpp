// File: arrayfire_backend.hpp
// Project: gain
// Created Date: 08/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "./arrayfire.h"
#include "autd3/core/hardware_defined.hpp"
#include "autd3/gain/linalg_backend.hpp"

namespace autd::gain::holo {
/**
 * \brief ArrayFire matrix (not optimized yet)
 */
template <typename T>
struct AFMatrix {
  af::array af_array;

  explicit AFMatrix(size_t row, size_t col);
  ~AFMatrix() = default;
  AFMatrix(const AFMatrix& obj) = delete;
  AFMatrix& operator=(const AFMatrix& obj) = delete;
  AFMatrix(const AFMatrix&& v) = delete;
  AFMatrix& operator=(AFMatrix&& obj) = delete;

  static double cast(const double v) { return v; }
  static af::cdouble cast(const complex v) { return af::cdouble(v.real(), v.imag()); }

  void make_complex(const std::shared_ptr<const AFMatrix<double>>& r, const std::shared_ptr<const AFMatrix<double>>& i) {
    af_array = af::complex(r->af_array, i->af_array);
  }
  void exp() { af_array = af::exp(af_array); }
  void scale(const T s) { af_array *= cast(s); }
  void reciprocal(const std::shared_ptr<const AFMatrix<T>>& src);
  void abs(const std::shared_ptr<const AFMatrix<T>>& src) { af_array = af::abs(src->af_array); }
  void real(const std::shared_ptr<const AFMatrix<complex>>& src) { af_array = af::real(src->af_array); }
  void arg(const std::shared_ptr<const AFMatrix<complex>>& src) { af_array = src->af_array / af::abs(src->af_array); }
  void hadamard_product(const std::shared_ptr<const AFMatrix<T>>& a, const std::shared_ptr<const AFMatrix<T>>& b) {
    af_array = a->af_array * b->af_array;
  }
  void pseudo_inverse_svd(const std::shared_ptr<AFMatrix<T>>& matrix, double alpha, const std::shared_ptr<AFMatrix<T>>& u,
                          const std::shared_ptr<AFMatrix<T>>& s, const std::shared_ptr<AFMatrix<T>>& vt, const std::shared_ptr<AFMatrix<T>>& buf);
  void max_eigen_vector(const std::shared_ptr<AFMatrix<T>>& ev);
  void add(const T alpha, const std::shared_ptr<AFMatrix<T>>& a) { af_array += cast(alpha) * a->af_array; }
  void mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const T alpha, const std::shared_ptr<const AFMatrix<T>>& a,
           const std::shared_ptr<const AFMatrix<T>>& b, const T beta) {
    af_array *= cast(beta);
    switch (trans_a) {
      case TRANSPOSE::CONJ_TRANS:
        switch (trans_b) {
          case TRANSPOSE::CONJ_TRANS:
            af_array += cast(alpha) * af::matmul(a->af_array, b->af_array, AF_MAT_CTRANS, AF_MAT_CTRANS);
            break;
          case TRANSPOSE::TRANS:
            af_array += cast(alpha) * matmul(a->af_array, b->af_array, AF_MAT_CTRANS, AF_MAT_TRANS);
            break;
          case TRANSPOSE::NO_TRANS:
            af_array += cast(alpha) * matmul(a->af_array, b->af_array, AF_MAT_CTRANS, AF_MAT_NONE);
            break;
        }
        break;
      case TRANSPOSE::TRANS:
        switch (trans_b) {
          case TRANSPOSE::CONJ_TRANS:
            af_array += cast(alpha) * matmul(a->af_array, b->af_array, AF_MAT_TRANS, AF_MAT_CTRANS);
            break;
          case TRANSPOSE::TRANS:
            af_array += cast(alpha) * matmul(a->af_array, b->af_array, AF_MAT_TRANS, AF_MAT_TRANS);
            break;
          case TRANSPOSE::NO_TRANS:
            af_array += cast(alpha) * matmul(a->af_array, b->af_array, AF_MAT_TRANS, AF_MAT_NONE);
            break;
        }
        break;
      case TRANSPOSE::NO_TRANS:
        switch (trans_b) {
          case TRANSPOSE::CONJ_TRANS:
            af_array += cast(alpha) * matmul(a->af_array, b->af_array, AF_MAT_NONE, AF_MAT_CTRANS);
            break;
          case TRANSPOSE::TRANS:
            af_array += cast(alpha) * matmul(a->af_array, b->af_array, AF_MAT_NONE, AF_MAT_TRANS);
            break;
          case TRANSPOSE::NO_TRANS:
            af_array += cast(alpha) * matmul(a->af_array, b->af_array, AF_MAT_NONE, AF_MAT_NONE);
            break;
        }
        break;
    }
  }
  void solve(const std::shared_ptr<AFMatrix<T>>& b) { b->af_array = af::solve(af_array, b->af_array); }
  T dot(const std::shared_ptr<const AFMatrix<T>>& a) {
    T v;
    auto r = af::dot(af_array, a->af_array, AF_MAT_CONJ);
    r.host(&v);
    return v;
  }
  [[nodiscard]] double max_element() const {
    T v;
    (af::max)((af::max)(af_array)).host(&v);
    return std::abs(v);
  }
  void concat_row(const std::shared_ptr<const AFMatrix<T>>& a, const std::shared_ptr<const AFMatrix<T>>& b) {
    af_array = af::join(0, a->af_array, b->af_array);
  }
  void concat_col(const std::shared_ptr<const AFMatrix<T>>& a, const std::shared_ptr<const AFMatrix<T>>& b) {
    af_array = af::join(1, a->af_array, b->af_array);
  }

  [[nodiscard]] T at(const size_t row, const size_t col) const {
    T v;
    af_array(row, col).host(&v);
    return v;
  }
  [[nodiscard]] size_t rows() const { return af_array.dims(0); }
  [[nodiscard]] size_t cols() const { return af_array.dims(1); }

  void set(const size_t row, const size_t col, T v) { af_array(static_cast<int>(row), static_cast<int>(col)) = cast(v); }
  void get_col(const std::shared_ptr<const AFMatrix<T>>& src, const size_t i) { af_array = src->af_array.col(i); }
  void fill(T v) { af_array = cast(v); }
  void get_diagonal(const std::shared_ptr<const AFMatrix<T>>& src) { af_array = af::diag(src->af_array); }
  void create_diagonal(const std::shared_ptr<const AFMatrix<T>>& v) { af_array = af::diag(v->af_array, 0, false); }
  void copy_from(const std::shared_ptr<const AFMatrix<T>>& a) { af_array = af::array(a->af_array); }
  void copy_from(const std::vector<T>& v) { copy_from(v.data(), v.size()); }
  void copy_from(const T* v) { copy_from(v, rows() * cols()); }
  void copy_from(const T* v, const size_t n) {
    if (n == 0) return;
    af_array.write(reinterpret_cast<const void*>(v), n * sizeof(T));
  }
  void copy_to_host() {}

  void transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions, const std::vector<const double*>& directions,
                       double wavelength, double attenuation);
  void set_bcd_result(const std::shared_ptr<const AFMatrix<T>>& vec, size_t index);
  void set_from_complex_drive(std::vector<core::DataArray>& dst, bool normalize, double max_coefficient);
  void set_from_arg(std::vector<core::DataArray>& dst, size_t n);
  void back_prop(const std::shared_ptr<const AFMatrix<T>>& transfer, const std::shared_ptr<const AFMatrix<T>>& amps);
  void sigma_regularization(const std::shared_ptr<const AFMatrix<T>>& transfer, const std::shared_ptr<const AFMatrix<T>>& amps, double gamma);
  void col_sum_imag(const std::shared_ptr<AFMatrix<complex>>& src);
};

using ArrayFireBackend = MatrixBufferPool<AFMatrix<double>, AFMatrix<complex>, Context>;

}  // namespace autd::gain::holo
