// File: eigen_matrix.hpp
// Project: eigen
// Created Date: 06/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <iostream>
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
#include "autd3/core/utils.hpp"
#include "autd3/gain/matrix.hpp"
#include "autd3/utils.hpp"

namespace autd::gain::holo {

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
  void pow(double s) { data = data.array().pow(s); }
  void scale(const T s) { data *= s; }
  void sqrt() { data = data.cwiseSqrt(); }
  void reciprocal(const std::shared_ptr<const EigenMatrix<T>>& src) {
    data = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>::Ones(src->data.rows(), src->data.cols()).cwiseQuotient(src->data);
  }
  void abs(const std::shared_ptr<const EigenMatrix<T>>& src) { data = src->data.cwiseAbs(); }
  void real(const std::shared_ptr<const EigenMatrix<complex>>& src);
  void imag(const std::shared_ptr<const EigenMatrix<complex>>& src);
  void conj(const std::shared_ptr<const EigenMatrix<complex>>& src);
  void arg(const std::shared_ptr<const EigenMatrix<complex>>& src);
  void hadamard_product(const std::shared_ptr<const EigenMatrix<T>>& a, const std::shared_ptr<const EigenMatrix<T>>& b) {
    data.noalias() = a->data.cwiseProduct(b->data);
  }
  virtual void pseudo_inverse_svd(const std::shared_ptr<EigenMatrix<T>>& matrix, double alpha, const std::shared_ptr<EigenMatrix<T>>&,
                                  const std::shared_ptr<EigenMatrix<T>>& s, const std::shared_ptr<EigenMatrix<T>>&,
                                  const std::shared_ptr<EigenMatrix<T>>&) {
    const Eigen::BDCSVD svd(matrix->data, Eigen::ComputeFullU | Eigen::ComputeFullV);
    s->data.fill(0);
    auto singular_values = svd.singularValues();
    const auto size = singular_values.size();
    for (Eigen::Index i = 0; i < size; i++) s->data(i, i) = singular_values(i) / (singular_values(i) * singular_values(i) + alpha);
    data.noalias() = svd.matrixV() * s->data * svd.matrixU().adjoint();
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

  void set_col(const size_t col, const size_t start_row, const size_t end_row, const std::shared_ptr<const EigenMatrix<T>>& vec) {
    data.block(start_row, col, end_row - start_row, 1) = vec->data.block(start_row, 0, end_row - start_row, 1);
  }
  void set_row(const size_t row, const size_t start_col, const size_t end_col, const std::shared_ptr<const EigenMatrix<T>>& vec) {
    data.block(row, start_col, 1, end_col - start_col) = vec->data.block(start_col, 0, end_col - start_col, 1).transpose();
  }

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

  void reduce_col(const std::shared_ptr<const EigenMatrix<T>>& src) {
    const auto n = data.size();
    for (Eigen::Index i = 0; i < n; i++) {
      T tmp = 0;
      for (Eigen::Index k = 0; k < n; k++) tmp += src->data(i, k);
      data(i, 0) = tmp;
    }
  }

  virtual void copy_from(const std::shared_ptr<const EigenMatrix<T>>& a) { data = a->data; }
  virtual void copy_from(const std::vector<T>& v) { std::memcpy(this->data.data(), v.data(), sizeof(T) * v.size()); }
  virtual void copy_from(const T* v) { std::memcpy(this->data.data(), v, sizeof(T) * this->data.size()); }

  void transfer_matrix(const double* foci, size_t foci_num, const std::vector<const core::Transducer*>& transducers,
                       const std::vector<const double*>& directions, double wavelength, double attenuation);
  void set_from_complex_drive(std::vector<core::Drive>& dst, bool normalize, double max_coefficient);
  void set_from_arg(std::vector<core::Drive>& dst, size_t n);
};

template <>
inline void EigenMatrix<complex>::make_complex(const std::shared_ptr<const EigenMatrix<double>>& r,
                                               const std::shared_ptr<const EigenMatrix<double>>& i) {
  data.real() = r->data;
  data.imag() = i->data;
}

template <>
inline void EigenMatrix<double>::real(const std::shared_ptr<const EigenMatrix<complex>>& src) {
  data = src->data.real();
}
template <>
inline void EigenMatrix<double>::imag(const std::shared_ptr<const EigenMatrix<complex>>& src) {
  data = src->data.imag();
}
template <>
inline void EigenMatrix<complex>::conj(const std::shared_ptr<const EigenMatrix<complex>>& src) {
  data = src->data.conjugate();
}
template <>
inline void EigenMatrix<complex>::arg(const std::shared_ptr<const EigenMatrix<complex>>& src) {
  data = src->data.cwiseQuotient(src->data.cwiseAbs());
}
template <>
inline double EigenMatrix<double>::max_element() const {
  return this->data.maxCoeff();
}
template <>
inline double EigenMatrix<complex>::max_element() const {
  return std::sqrt(this->data.cwiseAbs2().maxCoeff());
}
template <>
inline void EigenMatrix<double>::max_eigen_vector(const std::shared_ptr<EigenMatrix<double>>&) {}
template <>
inline void EigenMatrix<complex>::max_eigen_vector(const std::shared_ptr<EigenMatrix<complex>>& ev) {
  const Eigen::ComplexEigenSolver<Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>> ces(data);
  auto idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  const Eigen::Matrix<complex, -1, 1, Eigen::ColMajor>& max_ev = ces.eigenvectors().col(idx);
  ev->copy_from(max_ev.data());
}

template <>
inline void EigenMatrix<complex>::transfer_matrix(const double* foci, const size_t foci_num, const std::vector<const core::Transducer*>& transducers,
                                                  const std::vector<const double*>& directions, const double wavelength, const double attenuation) {
  const auto m = static_cast<Eigen::Index>(foci_num);

  const auto wave_number = 2.0 * M_PI / wavelength;
  for (Eigen::Index i = 0; i < m; i++) {
    const auto tp = core::Vector3(foci[3 * i], foci[3 * i + 1], foci[3 * i + 2]);
    Eigen::Index k = 0;
    for (size_t dev = 0; dev < transducers.size(); dev++) {
      const auto dir = core::Vector3(directions[dev][0], directions[dev][1], directions[dev][2]);
      for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(core::NUM_TRANS_IN_UNIT); j++, k++)
        data(i, k) = utils::transfer(transducers[dev][j], dir, tp, wave_number, attenuation);
    }
  }
}

template <>
inline void EigenMatrix<complex>::set_from_complex_drive(std::vector<core::Drive>& dst, const bool normalize, const double max_coefficient) {
  const Eigen::Index n = data.size();
  for (Eigen::Index j = 0; j < n; j++) {
    const auto f_amp = normalize ? 1.0 : std::abs(data(j, 0)) / max_coefficient;
    dst[j].duty = core::utils::to_duty(f_amp);
    dst[j].phase = core::utils::to_phase(std::arg(data(j, 0)));
  }
}

template <>
inline void EigenMatrix<double>::set_from_arg(std::vector<core::Drive>& dst, const size_t n) {
  for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(n); j++) {
    dst[j].duty = 0xFF;
    dst[j].phase = core::utils::to_phase(data(j, 0));
  }
}

}  // namespace autd::gain::holo
