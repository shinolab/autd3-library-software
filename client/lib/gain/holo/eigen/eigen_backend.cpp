// File: eigen_backend.cpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/eigen_backend.hpp"

#include "autd3/core/utils.hpp"
#include "autd3/utils.hpp"

namespace autd::gain::holo {

template <>
void EigenMatrix<complex>::make_complex(const std::shared_ptr<const EigenMatrix<double>>& r, const std::shared_ptr<const EigenMatrix<double>>& i) {
  data.real() = r->data;
  data.imag() = i->data;
}
template <>
void EigenMatrix<double>::real(const std::shared_ptr<const EigenMatrix<complex>>& src) {
  data = src->data.real();
}
template <>
void EigenMatrix<complex>::arg(const std::shared_ptr<const EigenMatrix<complex>>& src) {
  data = src->data.cwiseQuotient(src->data.cwiseAbs());
}
template <>
double EigenMatrix<double>::max_element() const {
  return this->data.maxCoeff();
}
template <>
double EigenMatrix<complex>::max_element() const {
  return std::sqrt(this->data.cwiseAbs2().maxCoeff());
}
template <>
void EigenMatrix<complex>::max_eigen_vector(const std::shared_ptr<EigenMatrix<complex>>& ev) {
  const Eigen::ComplexEigenSolver<Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>> ces(data);
  auto idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  const Eigen::Matrix<complex, -1, 1, Eigen::ColMajor>& max_ev = ces.eigenvectors().col(idx);
  ev->copy_from(max_ev.data());
}

template <>
void EigenMatrix<complex>::transfer_matrix(const double* foci, const size_t foci_num, const std::vector<const double*>& positions,
                                           const std::vector<const double*>& directions, const double wavelength, const double attenuation) {
  const auto m = static_cast<Eigen::Index>(foci_num);

  const auto wave_number = 2.0 * M_PI / wavelength;
  for (Eigen::Index i = 0; i < m; i++) {
    const auto tp = core::Vector3(foci[3 * i], foci[3 * i + 1], foci[3 * i + 2]);
    Eigen::Index k = 0;
    for (size_t dev = 0; dev < positions.size(); dev++) {
      const double* p = positions[dev];
      const auto dir = core::Vector3(directions[dev][0], directions[dev][1], directions[dev][2]);
      for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(core::NUM_TRANS_IN_UNIT); j++, k++) {
        const auto pos = core::Vector3(p[3 * j], p[3 * j + 1], p[3 * j + 2]);
        data(i, k) = utils::transfer(pos, dir, tp, wave_number, attenuation);
      }
    }
  }
}

template <>
void EigenMatrix<complex>::set_bcd_result(const std::shared_ptr<const EigenMatrix<complex>>& vec, const size_t index) {
  const auto m = vec->data.size();
  const auto idx = static_cast<Eigen::Index>(index);
  for (Eigen::Index i = 0; i < idx; i++) data(idx, i) = std::conj(vec->data(i, 0));
  for (Eigen::Index i = idx + 1; i < m; i++) data(idx, i) = std::conj(vec->data(i, 0));
  for (Eigen::Index i = 0; i < idx; i++) data(i, idx) = vec->data(i, 0);
  for (Eigen::Index i = idx + 1; i < m; i++) data(i, idx) = vec->data(i, 0);
}

template <>
void EigenMatrix<complex>::set_from_complex_drive(std::vector<core::DataArray>& dst, const bool normalize, const double max_coefficient) {
  const Eigen::Index n = data.size();
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (Eigen::Index j = 0; j < n; j++) {
    const auto f_amp = normalize ? 1.0 : std::abs(data(j, 0)) / max_coefficient;
    const auto f_phase = std::arg(data(j, 0)) / (2.0 * M_PI);
    const auto phase = core::Utilities::to_phase(f_phase);
    const auto duty = core::Utilities::to_duty(f_amp);
    dst[dev_idx][trans_idx++] = core::Utilities::pack_to_u16(duty, phase);
    if (trans_idx == core::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}

template <>
void EigenMatrix<double>::set_from_arg(std::vector<core::DataArray>& dst, const size_t n) {
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(n); j++) {
    constexpr uint8_t duty = 0xFF;
    const auto f_phase = data(j, 0) / (2 * M_PI);
    const auto phase = core::Utilities::to_phase(f_phase);
    dst[dev_idx][trans_idx++] = core::Utilities::pack_to_u16(duty, phase);
    if (trans_idx == core::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}

template <>
void EigenMatrix<complex>::back_prop(const std::shared_ptr<const EigenMatrix<complex>>& transfer,
                                     const std::shared_ptr<const EigenMatrix<complex>>& amps) {
  const auto m = transfer->data.rows();
  const auto n = transfer->data.cols();

  Eigen::Matrix<double, -1, 1, Eigen::ColMajor> denominator(m);
  for (Eigen::Index i = 0; i < m; i++) {
    auto tmp = 0.0;
    for (Eigen::Index j = 0; j < n; j++) tmp += std::abs(transfer->data(i, j));
    denominator(i) = tmp;
  }

  for (Eigen::Index i = 0; i < m; i++) {
    auto c = amps->data(i) / denominator(i);
    for (Eigen::Index j = 0; j < n; j++) data(j, i) = c * std::conj(transfer->data(i, j));
  }
}

template <>
void EigenMatrix<complex>::sigma_regularization(const std::shared_ptr<const EigenMatrix<complex>>& transfer,
                                                const std::shared_ptr<const EigenMatrix<complex>>& amps, const double gamma) {
  const auto m = transfer->data.rows();
  const auto n = transfer->data.cols();

  fill(ZERO);
  for (Eigen::Index j = 0; j < n; j++) {
    double tmp = 0;
    for (Eigen::Index i = 0; i < m; i++) tmp += std::abs(transfer->data(i, j) * amps->data(i));
    data(j, j) = complex(std::pow(std::sqrt(tmp / static_cast<double>(m)), gamma), 0.0);
  }
}

template <>
void EigenMatrix<double>::col_sum_imag(const std::shared_ptr<EigenMatrix<complex>>& src) {
  const auto n = data.size();
  for (Eigen::Index i = 0; i < n; i++) {
    double tmp = 0;
    for (Eigen::Index k = 0; k < n; k++) tmp += src->data(i, k).imag();
    data(i, 0) = tmp;
  }
}
}  // namespace autd::gain::holo
