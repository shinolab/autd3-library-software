// File: cuda_matrix.hpp
// Project: cuda
// Created Date: 10/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26439 26478 26812 26495)
#endif
#include <cublas_v2.h>
#include <cusolverDn.h>
#if _MSC_VER
#pragma warning(pop)
#endif

#include <memory>
#include <vector>

#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 20014
#define diag_suppress nv_diag_suppress
#endif
#include "autd3/core/gain.hpp"
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#undef diag_suppress
#endif
#include "autd3/gain/matrix.hpp"
#include "matrix_pool.hpp"

namespace autd {
namespace gain {
namespace holo {

struct CuContext {
  static cublasHandle_t handle;
  static cusolverDnHandle_t handle_s;
  static size_t cnt;

  static void init(int device_idx);
  static void free();
};

template <typename T>
struct CuMatrix {
  explicit CuMatrix(size_t row, size_t col);
  ~CuMatrix() = default;
  CuMatrix(const CuMatrix& obj) = delete;
  CuMatrix& operator=(const CuMatrix& obj) = delete;
  CuMatrix(const CuMatrix&& v) = delete;
  CuMatrix& operator=(CuMatrix&& obj) = delete;

  [[nodiscard]] const T* ptr() const;
  T* ptr();

  void make_complex(const std::shared_ptr<const CuMatrix<double>>& r, const std::shared_ptr<const CuMatrix<double>>& i);
  void exp();
  void pow(double s);
  void scale(T s);
  void sqrt();
  void reciprocal(const std::shared_ptr<const CuMatrix<T>>& src);
  void abs(const std::shared_ptr<const CuMatrix<T>>& src);
  void real(const std::shared_ptr<const CuMatrix<complex>>& src);
  void imag(const std::shared_ptr<const CuMatrix<complex>>& src);
  void conj(const std::shared_ptr<const CuMatrix<complex>>& src);
  void arg(const std::shared_ptr<const CuMatrix<complex>>& src);
  void hadamard_product(const std::shared_ptr<const CuMatrix<T>>& a, const std::shared_ptr<const CuMatrix<T>>& b);
  void pseudo_inverse_svd(const std::shared_ptr<CuMatrix<T>>& matrix, double alpha, const std::shared_ptr<CuMatrix<T>>& u,
                          const std::shared_ptr<CuMatrix<T>>& s, const std::shared_ptr<CuMatrix<T>>& vt, const std::shared_ptr<CuMatrix<T>>& buf);
  void max_eigen_vector(const std::shared_ptr<CuMatrix<T>>& ev);
  void add(T alpha, const std::shared_ptr<CuMatrix<T>>& a);
  void mul(TRANSPOSE trans_a, TRANSPOSE trans_b, T alpha, const std::shared_ptr<const CuMatrix<T>>& a, const std::shared_ptr<const CuMatrix<T>>& b,
           T beta);
  void solve(const std::shared_ptr<CuMatrix<T>>& b);
  T dot(const std::shared_ptr<const CuMatrix<T>>& a);
  [[nodiscard]] double max_element() const;
  void concat_row(const std::shared_ptr<const CuMatrix<T>>& a, const std::shared_ptr<const CuMatrix<T>>& b);
  void concat_col(const std::shared_ptr<const CuMatrix<T>>& a, const std::shared_ptr<const CuMatrix<T>>& b);

  [[nodiscard]] T at(size_t row, size_t col);
  [[nodiscard]] size_t rows() const { return _row; }
  [[nodiscard]] size_t cols() const { return _col; }

  void set(size_t row, size_t col, T v);
  void set_col(size_t col, size_t start_row, size_t end_row, const std::shared_ptr<const CuMatrix<T>>& vec);
  void set_row(size_t row, size_t start_col, size_t end_col, const std::shared_ptr<const CuMatrix<T>>& vec);
  void get_col(const std::shared_ptr<const CuMatrix<T>>& src, size_t i);
  void fill(T v);
  void get_diagonal(const std::shared_ptr<const CuMatrix<T>>& src);
  void create_diagonal(const std::shared_ptr<const CuMatrix<T>>& v);
  void reduce_col(const std::shared_ptr<const CuMatrix<T>>& src);
  void copy_from(const std::shared_ptr<const CuMatrix<T>>& a);
  void copy_from(const std::vector<T>& v) { copy_from(v.data(), v.size()); }
  void copy_from(const T* v) { copy_from(v, _row * _col); }
  void copy_from(const T* v, size_t n);

  void transfer_matrix(const double* foci, size_t foci_num, const std::vector<const core::Transducer*>& transducers,
                       const std::vector<const double*>& directions, double wavelength, double attenuation);
  void set_from_complex_drive(std::vector<core::Drive>& dst, bool normalize, double max_coefficient);
  void set_from_arg(std::vector<core::Drive>& dst, size_t n);

 private:
  size_t _row;
  size_t _col;
  struct Impl;
  std::shared_ptr<Impl> _pimpl;
};

template <>
const double* CuMatrix<double>::ptr() const;
template <>
const complex* CuMatrix<complex>::ptr() const;
template <>
double* CuMatrix<double>::ptr();
template <>
complex* CuMatrix<complex>::ptr();

template <>
void CuMatrix<double>::mul(TRANSPOSE trans_a, TRANSPOSE trans_b, double alpha, const std::shared_ptr<const CuMatrix<double>>& a,
                           const std::shared_ptr<const CuMatrix<double>>& b, double beta);
template <>
void CuMatrix<complex>::mul(TRANSPOSE trans_a, TRANSPOSE trans_b, complex alpha, const std::shared_ptr<const CuMatrix<complex>>& a,
                            const std::shared_ptr<const CuMatrix<complex>>& b, complex beta);
}  // namespace holo
}  // namespace gain
}  // namespace autd
