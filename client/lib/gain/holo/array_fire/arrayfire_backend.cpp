// File: arrayfire_backend.cpp
// Project: array_fire
// Created Date: 08/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <iostream>

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

#include "autd3/core/geometry.hpp"
#include "autd3/core/utils.hpp"
#include "autd3/gain/arrayfire_backend.hpp"
#include "autd3/utils.hpp"

namespace autd::gain::holo {
AFMatrix<double>::AFMatrix(const size_t row, const size_t col) : _af_array(static_cast<dim_t>(row), static_cast<dim_t>(col), af::dtype::f64) {}
AFMatrix<complex>::AFMatrix(const size_t row, const size_t col) : _af_array(static_cast<dim_t>(row), static_cast<dim_t>(col), af::dtype::c64) {}

void AFMatrix<double>::reciprocal(const std::shared_ptr<const AFMatrix<double>>& src) {
  _af_array = af::constant(1.0, static_cast<dim_t>(rows()), static_cast<dim_t>(cols()), af::dtype::f64) / src->_af_array;
}
void AFMatrix<complex>::reciprocal(const std::shared_ptr<const AFMatrix<complex>>& src) {
  _af_array = af::constant(af::cdouble(1.0, 0.0), static_cast<dim_t>(rows()), static_cast<dim_t>(cols()), af::dtype::c64) / src->_af_array;
}

void AFMatrix<double>::pseudo_inverse_svd(const std::shared_ptr<AFMatrix<double>>& matrix, const double alpha,
                                          const std::shared_ptr<AFMatrix<double>>& u, const std::shared_ptr<AFMatrix<double>>& s,
                                          const std::shared_ptr<AFMatrix<double>>& vt, const std::shared_ptr<AFMatrix<double>>& buf) {
  const auto m = matrix->rows();
  const auto n = matrix->cols();
  af::array s_vec;
  af::svd(u->_af_array, s_vec, vt->_af_array, matrix->_af_array);
  s_vec = s_vec / (s_vec * s_vec + af::constant(alpha, s_vec.dims(0), af::dtype::f64));
  const af::array s_mat = diag(s_vec, 0, false);
  const af::array zero = af::constant(0.0, n - m, m, af::dtype::f64);
  s->_af_array = af::join(0, s_mat, zero);
  buf->_af_array = af::matmul(s->_af_array, u->_af_array, AF_MAT_NONE, AF_MAT_TRANS);
  _af_array = af::matmul(vt->_af_array, buf->_af_array, AF_MAT_TRANS, AF_MAT_NONE);
}
void AFMatrix<complex>::pseudo_inverse_svd(const std::shared_ptr<AFMatrix<complex>>& matrix, const double alpha,
                                           const std::shared_ptr<AFMatrix<complex>>& u, const std::shared_ptr<AFMatrix<complex>>& s,
                                           const std::shared_ptr<AFMatrix<complex>>& vt, const std::shared_ptr<AFMatrix<complex>>& buf) {
  const auto m = matrix->rows();
  const auto n = matrix->cols();
  af::array s_vec;
  af::svd(u->_af_array, s_vec, vt->_af_array, matrix->_af_array);
  s_vec = s_vec / (s_vec * s_vec + af::constant(alpha, s_vec.dims(0), af::dtype::f64));
  const af::array s_mat = diag(s_vec, 0, false);
  const af::array zero = af::constant(0.0, n - m, m, af::dtype::f64);
  s->_af_array = af::complex(af::join(0, s_mat, zero), 0);
  buf->_af_array = af::matmul(s->_af_array, u->_af_array, AF_MAT_NONE, AF_MAT_CTRANS);
  _af_array = af::matmul(vt->_af_array, buf->_af_array, AF_MAT_CTRANS, AF_MAT_NONE);
}

void AFMatrix<double>::max_eigen_vector(const std::shared_ptr<AFMatrix<double>>&) {}
void AFMatrix<complex>::max_eigen_vector(const std::shared_ptr<AFMatrix<complex>>& ev) {
  /// ArrayFire currently does not support eigen value decomposition?
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> data(rows(), cols());
  _af_array.host(data.data());
  const Eigen::ComplexEigenSolver<Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>> ces(data);
  auto idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  const Eigen::Matrix<complex, -1, 1, Eigen::ColMajor>& max_ev = ces.eigenvectors().col(idx);
  ev->copy_from(max_ev.data());
}

void AFMatrix<double>::transfer_matrix(const double*, size_t, const std::vector<const double*>&, const std::vector<const double*>&, double, double) {}
void AFMatrix<complex>::transfer_matrix(const double* foci, const size_t foci_num, const std::vector<const double*>& positions,
                                        const std::vector<const double*>& directions, double const wavelength, double const attenuation) {
  // FIXME: implement with ArrayFire
  const auto data = std::make_unique<complex[]>(foci_num * positions.size() * core::NUM_TRANS_IN_UNIT);

  const auto wave_number = 2.0 * M_PI / wavelength;
  size_t k = 0;
  for (size_t dev = 0; dev < positions.size(); dev++) {
    const double* p = positions[dev];
    const auto dir = core::Vector3(directions[dev][0], directions[dev][1], directions[dev][2]);
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(core::NUM_TRANS_IN_UNIT); j++) {
      const auto pos = core::Vector3(p[3 * j], p[3 * j + 1], p[3 * j + 2]);
      for (size_t i = 0; i < foci_num; i++, k++) {
        const auto tp = core::Vector3(foci[3 * i], foci[3 * i + 1], foci[3 * i + 2]);
        data[k] = utils::transfer(pos, dir, tp, wave_number, attenuation);
      }
    }
  }

  _af_array = af::array(foci_num, positions.size() * core::NUM_TRANS_IN_UNIT, reinterpret_cast<const af::cdouble*>(data.get()));
}

void AFMatrix<double>::set_bcd_result(const std::shared_ptr<const AFMatrix<double>>&, size_t) {}
void AFMatrix<complex>::set_bcd_result(const std::shared_ptr<const AFMatrix<complex>>& vec, const size_t index) {
  const auto ii = at(index, index);
  const auto vh = vec->_af_array.H();
  _af_array.row(index) = vh;
  _af_array.col(index) = vec->_af_array;
  set(index, index, ii);
}

void AFMatrix<double>::set_from_complex_drive(std::vector<core::DataArray>&, const bool, const double) {}
void AFMatrix<complex>::set_from_complex_drive(std::vector<core::DataArray>& dst, const bool normalize, const double max_coefficient) {
  // FIXME: implement with ArrayFire
  const auto n = rows() * cols();
  const auto data = std::make_unique<complex[]>(n);
  _af_array.host(data.get());

  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (size_t j = 0; j < n; j++) {
    const auto f_amp = normalize ? 1.0 : std::abs(data[j]) / max_coefficient;
    const auto f_phase = std::arg(data[j]) / (2.0 * M_PI);
    const auto phase = core::Utilities::to_phase(f_phase);
    const auto duty = core::Utilities::to_duty(f_amp);
    dst[dev_idx][trans_idx++] = core::Utilities::pack_to_u16(duty, phase);
    if (trans_idx == core::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}

void AFMatrix<double>::set_from_arg(std::vector<core::DataArray>& dst, const size_t n) {
  // FIXME: implement with ArrayFire
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  const auto data = std::make_unique<double[]>(n);
  _af_array(af::seq(n)).host(data.get());
  for (size_t j = 0; j < n; j++) {
    constexpr uint8_t duty = 0xFF;
    const auto f_phase = data[j] / (2 * M_PI);
    const auto phase = core::Utilities::to_phase(f_phase);
    dst[dev_idx][trans_idx++] = core::Utilities::pack_to_u16(duty, phase);
    if (trans_idx == core::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}
void AFMatrix<complex>::set_from_arg(std::vector<core::DataArray>&, const size_t) {}

void AFMatrix<double>::back_prop(const std::shared_ptr<const AFMatrix<double>>&, const std::shared_ptr<const AFMatrix<double>>&) {}
void AFMatrix<complex>::back_prop(const std::shared_ptr<const AFMatrix<complex>>& transfer, const std::shared_ptr<const AFMatrix<complex>>& amps) {
  const auto m = transfer->rows();
  const auto n = transfer->cols();

  af::array t = af::abs(transfer->_af_array);
  af::array c = af::tile(af::moddims(amps->_af_array / af::sum(t, 1), 1, m), n, 1);
  _af_array = c * transfer->_af_array.H();
}

void AFMatrix<complex>::sigma_regularization(const std::shared_ptr<const AFMatrix<complex>>& transfer,
                                             const std::shared_ptr<const AFMatrix<complex>>& amps, const double gamma) {
  const auto m = transfer->rows();
  const auto n = transfer->cols();

  af::array ac = af::tile(amps->_af_array, 1, n);
  ac *= transfer->_af_array;
  const af::array a = af::abs(ac);

  af::array d = af::moddims(af::sum(a, 0), n);
  d /= static_cast<double>(m);
  d = af::sqrt(d);
  d = af::pow(d, gamma);
  _af_array = af::complex(af::diag(d, 0, false), 0);
}

void AFMatrix<double>::col_sum_imag(const std::shared_ptr<AFMatrix<complex>>& src) {
  af::array imag = af::imag(src->_af_array);
  _af_array = af::sum(imag, 1);
}
void AFMatrix<complex>::col_sum_imag(const std::shared_ptr<AFMatrix<complex>>&) {}

}  // namespace autd::gain::holo
