// File: arrayfire_matrix.hpp
// Project: array_fire
// Created Date: 08/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26812)
#endif
#include "./arrayfire.h"
#if _MSC_VER
#pragma warning(pop)
#endif

#include "autd3/core/hardware_defined.hpp"
#include "autd3/core/utils.hpp"
#include "autd3/gain/matrix.hpp"
#include "autd3/utils.hpp"

namespace autd::gain::holo {

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
  static af::cdouble cast(const complex v) { return {v.real(), v.imag()}; }

  void make_complex(const std::shared_ptr<const AFMatrix<double>>& r, const std::shared_ptr<const AFMatrix<double>>& i) {
    af_array = af::complex(r->af_array, i->af_array);
  }
  void exp() { af_array = af::exp(af_array); }
  void pow(const double s) { af_array = af::pow(af_array, s); }
  void scale(const T s) { af_array *= cast(s); }
  void sqrt() { af_array = af::sqrt(af_array); }
  void reciprocal(const std::shared_ptr<const AFMatrix<T>>& src);
  void abs(const std::shared_ptr<const AFMatrix<T>>& src) { af_array = af::abs(src->af_array); }
  void real(const std::shared_ptr<const AFMatrix<complex>>& src) { af_array = af::real(src->af_array); }
  void imag(const std::shared_ptr<const AFMatrix<complex>>& src) { af_array = af::imag(src->af_array); }
  void conj(const std::shared_ptr<const AFMatrix<complex>>& src) { af_array = conjg(src->af_array); }
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
    T v{};
    auto r = af::dot(af_array, a->af_array, AF_MAT_CONJ);
    r.host(&v);
    return v;
  }
  [[nodiscard]] double max_element() const {
    T v{};
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
    T v{};
    af_array(static_cast<int>(row), static_cast<int>(col)).host(&v);
    return v;
  }
  [[nodiscard]] size_t rows() const { return af_array.dims(0); }
  [[nodiscard]] size_t cols() const { return af_array.dims(1); }

  void set(const size_t row, const size_t col, T v) { af_array(static_cast<int>(row), static_cast<int>(col)) = cast(v); }
  void set_col(const size_t col, const size_t start_row, const size_t end_row, const std::shared_ptr<const AFMatrix<T>>& vec) {
    af_array(af::seq(static_cast<double>(start_row), static_cast<double>(end_row) - 1), static_cast<int>(col)) =
        vec->af_array(af::seq(static_cast<double>(start_row), static_cast<double>(end_row) - 1), 0);
  }
  void set_row(const size_t row, const size_t start_col, const size_t end_col, const std::shared_ptr<const AFMatrix<T>>& vec) {
    af_array(static_cast<int>(row), af::seq(static_cast<double>(start_col), static_cast<double>(end_col) - 1)) =
        vec->af_array(af::seq(static_cast<double>(start_col), static_cast<double>(end_col) - 1), 0);
  }
  void get_col(const std::shared_ptr<const AFMatrix<T>>& src, const size_t i) { af_array = src->af_array.col(static_cast<int>(i)); }
  void fill(T v) { af_array = cast(v); }
  void get_diagonal(const std::shared_ptr<const AFMatrix<T>>& src) { af_array = af::diag(src->af_array); }
  void create_diagonal(const std::shared_ptr<const AFMatrix<T>>& v) { af_array = af::diag(v->af_array, 0, false); }
  void reduce_col(const std::shared_ptr<const AFMatrix<T>>& src) { af_array = af::sum(src->af_array, 1); }
  void copy_from(const std::shared_ptr<const AFMatrix<T>>& a) { af_array = af::array(a->af_array); }
  void copy_from(const std::vector<T>& v) { copy_from(v.data(), v.size()); }
  void copy_from(const T* v) { copy_from(v, rows() * cols()); }
  void copy_from(const T* v, const size_t n) {
    if (n == 0) return;
    af_array.write(reinterpret_cast<const void*>(v), n * sizeof(T));
  }

  void transfer_matrix(const double* foci, size_t foci_num, const std::vector<const core::Transducer*>& transducers,
                       const std::vector<const double*>& directions, double wavelength, double attenuation);
  void set_from_complex_drive(std::vector<core::Drive>& dst, bool normalize, double max_coefficient);
  void set_from_arg(std::vector<core::Drive>& dst, size_t n);
};

inline AFMatrix<double>::AFMatrix(const size_t row, const size_t col) : af_array(static_cast<dim_t>(row), static_cast<dim_t>(col), af::dtype::f64) {}
inline AFMatrix<complex>::AFMatrix(const size_t row, const size_t col) : af_array(static_cast<dim_t>(row), static_cast<dim_t>(col), af::dtype::c64) {}

inline void AFMatrix<double>::reciprocal(const std::shared_ptr<const AFMatrix<double>>& src) {
  af_array = af::constant(1.0, static_cast<dim_t>(rows()), static_cast<dim_t>(cols()), af::dtype::f64) / src->af_array;
}
inline void AFMatrix<complex>::reciprocal(const std::shared_ptr<const AFMatrix<complex>>& src) {
  af_array = constant(af::cdouble(1.0, 0.0), static_cast<dim_t>(rows()), static_cast<dim_t>(cols()), af::dtype::c64) / src->af_array;
}

inline void AFMatrix<double>::pseudo_inverse_svd(const std::shared_ptr<AFMatrix<double>>& matrix, const double alpha,
                                                 const std::shared_ptr<AFMatrix<double>>& u, const std::shared_ptr<AFMatrix<double>>& s,
                                                 const std::shared_ptr<AFMatrix<double>>& vt, const std::shared_ptr<AFMatrix<double>>& buf) {
  const auto m = static_cast<dim_t>(matrix->rows());
  const auto n = static_cast<dim_t>(matrix->cols());
  af::array s_vec;
  svd(u->af_array, s_vec, vt->af_array, matrix->af_array);
  s_vec = s_vec / (s_vec * s_vec + af::constant(alpha, s_vec.dims(0), af::dtype::f64));
  const af::array s_mat = diag(s_vec, 0, false);
  const af::array zero = af::constant(0.0, n - m, m, af::dtype::f64);
  s->af_array = join(0, s_mat, zero);
  buf->af_array = matmul(s->af_array, u->af_array, AF_MAT_NONE, AF_MAT_TRANS);
  af_array = matmul(vt->af_array, buf->af_array, AF_MAT_TRANS, AF_MAT_NONE);
}
inline void AFMatrix<complex>::pseudo_inverse_svd(const std::shared_ptr<AFMatrix<complex>>& matrix, const double alpha,
                                                  const std::shared_ptr<AFMatrix<complex>>& u, const std::shared_ptr<AFMatrix<complex>>& s,
                                                  const std::shared_ptr<AFMatrix<complex>>& vt, const std::shared_ptr<AFMatrix<complex>>& buf) {
  const auto m = static_cast<dim_t>(matrix->rows());
  const auto n = static_cast<dim_t>(matrix->cols());
  af::array s_vec;
  svd(u->af_array, s_vec, vt->af_array, matrix->af_array);
  s_vec = s_vec / (s_vec * s_vec + af::constant(alpha, s_vec.dims(0), af::dtype::f64));
  const af::array s_mat = diag(s_vec, 0, false);
  const af::array zero = af::constant(0.0, n - m, m, af::dtype::f64);
  s->af_array = af::complex(join(0, s_mat, zero), 0);
  buf->af_array = matmul(s->af_array, u->af_array, AF_MAT_NONE, AF_MAT_CTRANS);
  af_array = matmul(vt->af_array, buf->af_array, AF_MAT_CTRANS, AF_MAT_NONE);
}

inline void AFMatrix<double>::max_eigen_vector(const std::shared_ptr<AFMatrix<double>>&) {}
inline void AFMatrix<complex>::max_eigen_vector(const std::shared_ptr<AFMatrix<complex>>& ev) {
  /// ArrayFire currently does not support eigen value decomposition?
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> data(rows(), cols());
  af_array.host(data.data());
  const Eigen::ComplexEigenSolver<Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>> ces(data);
  auto idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  const Eigen::Matrix<complex, -1, 1, Eigen::ColMajor>& max_ev = ces.eigenvectors().col(idx);
  ev->copy_from(max_ev.data());
}

inline void AFMatrix<double>::transfer_matrix(const double*, size_t, const std::vector<const core::Transducer*>&, const std::vector<const double*>&,
                                              double, double) {}
inline void AFMatrix<complex>::transfer_matrix(const double* foci, const size_t foci_num, const std::vector<const core::Transducer*>& transducers,
                                               const std::vector<const double*>& directions, double const wavelength, double const attenuation) {
  // FIXME: implement with ArrayFire
  const auto data = std::make_unique<complex[]>(foci_num * transducers.size() * core::NUM_TRANS_IN_UNIT);

  const auto wave_number = 2.0 * M_PI / wavelength;
  size_t k = 0;
  for (size_t dev = 0; dev < transducers.size(); dev++) {
    const auto dir = core::Vector3(directions[dev][0], directions[dev][1], directions[dev][2]);
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(core::NUM_TRANS_IN_UNIT); j++) {
      const auto& transducer = transducers[dev][j];
      for (size_t i = 0; i < foci_num; i++, k++) {
        const auto tp = core::Vector3(foci[3 * i], foci[3 * i + 1], foci[3 * i + 2]);
        data[k] = utils::transfer(transducer, dir, tp, wave_number, attenuation);
      }
    }
  }

  af_array = af::array(static_cast<dim_t>(foci_num), static_cast<dim_t>(transducers.size() * core::NUM_TRANS_IN_UNIT),
                       reinterpret_cast<const af::cdouble*>(data.get()));
}

inline void AFMatrix<double>::set_from_complex_drive(std::vector<core::Drive>&, const bool, const double) {}
inline void AFMatrix<complex>::set_from_complex_drive(std::vector<core::Drive>& dst, const bool normalize, const double max_coefficient) {
  // FIXME: implement with ArrayFire
  const auto n = rows() * cols();
  const auto data = std::make_unique<complex[]>(n);
  af_array.host(data.get());

  for (size_t j = 0; j < n; j++) {
    const auto f_amp = normalize ? 1.0 : std::abs(data[j]) / max_coefficient;
    dst[j].duty = core::utils::to_duty(f_amp);
    dst[j].phase = core::utils::to_phase(std::arg(data[j]));
  }
}

inline void AFMatrix<double>::set_from_arg(std::vector<core::Drive>& dst, const size_t n) {
  // FIXME: implement with ArrayFire
  const auto data = std::make_unique<double[]>(n);
  af_array(af::seq(static_cast<double>(n))).host(data.get());
  for (size_t j = 0; j < n; j++) {
    dst[j].duty = 0xFF;
    dst[j].phase = core::utils::to_phase(data[j]);
  }
}
inline void AFMatrix<complex>::set_from_arg(std::vector<core::Drive>&, const size_t) {}
}  // namespace autd::gain::holo
