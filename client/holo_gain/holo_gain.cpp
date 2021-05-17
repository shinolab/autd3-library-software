// File: holo_gain.cpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 17/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "holo_gain.hpp"

#include <random>

#include "linalg_backend.hpp"
#include "utils.hpp"

namespace autd::gain::holo {

void HoloGain::MatrixMul(const Backend::MatrixXc& a, const Backend::MatrixXc& b, Backend::MatrixXc* c) const {
  this->_backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, std::complex<double>(1, 0), a, b, std::complex<double>(0, 0), c);
}

void HoloGain::MatrixVecMul(const Backend::MatrixXc& a, const Backend::VectorXc& b, Backend::VectorXc* c) const {
  this->_backend->matrix_vector_mul(TRANSPOSE::NO_TRANS, std::complex<double>(1, 0), a, b, std::complex<double>(0, 0), c);
}

void HoloGain::SetFromComplexDrive(std::vector<core::AUTDDataArray>& data, const Backend::VectorXc& drive, const bool normalize,
                                   const double max_coefficient) {
  const size_t n = drive.size();
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (size_t j = 0; j < n; j++) {
    const auto f_amp = normalize ? 1.0 : std::abs(drive(j)) / max_coefficient;
    const auto f_phase = arg(drive(j)) / (2.0 * M_PI) + 0.5;
    const auto phase = static_cast<uint16_t>((1.0 - f_phase) * 255.0);
    const uint16_t duty = static_cast<uint16_t>(core::ToDuty(f_amp)) << 8 & 0xFF00;
    data[dev_idx][trans_idx++] = duty | phase;
    if (trans_idx == core::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}

std::complex<double> HoloGain::Transfer(const core::Vector3& trans_pos, const core::Vector3& trans_norm, const core::Vector3& target_pos,
                                        const double wave_number, const double attenuation) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = std::atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180.0 / M_PI;
  const auto directivity = utils::DirectivityT4010A1(theta);
  return directivity / dist * exp(std::complex<double>(-dist * attenuation, -wave_number * dist));
}

Backend::MatrixXc HoloGain::TransferMatrix(const std::vector<core::Vector3>& foci, const core::GeometryPtr& geometry) {
  const auto m = foci.size();
  const auto n = geometry->num_transducers();

  Backend::MatrixXc g(m, n);

  const auto wave_number = 2.0 * M_PI / geometry->wavelength();
  const auto attenuation = geometry->attenuation_coefficient();
  for (size_t i = 0; i < m; i++) {
    const auto& tp = foci[i];
    for (size_t j = 0; j < n; j++) {
      const auto pos = geometry->position(j);
      const auto dir = geometry->direction(j / core::NUM_TRANS_IN_UNIT);
      g(i, j) = Transfer(pos, dir, tp, wave_number, attenuation);
    }
  }
  return g;
}

Error HoloGainSDP::Calc(const core::GeometryPtr& geometry) {
  if (!this->_backend->supports_svd() || !this->_backend->supports_evd()) return Err(std::string("This backend does not support this method."));

  auto set_bcd_result = [](Backend::MatrixXc& mat, const Backend::VectorXc& vec, const size_t idx) {
    const size_t m = vec.size();
    for (size_t i = 0; i < idx; i++) mat(idx, i) = std::conj(vec(i));
    for (auto i = idx + 1; i < m; i++) mat(idx, i) = std::conj(vec(i));
    for (size_t i = 0; i < idx; i++) mat(i, idx) = vec(i);
    for (auto i = idx + 1; i < m; i++) mat(i, idx) = vec(i);
  };

  const auto m = this->_foci.size();
  const auto n = geometry->num_transducers();

  Backend::MatrixXc p = Backend::MatrixXc::Zero(m, m);
  for (size_t i = 0; i < m; i++) p(i, i) = std::complex<double>(this->_amps[i], 0);

  auto b = TransferMatrix(this->_foci, geometry);
  Backend::MatrixXc pseudo_inv_b(n, m);
  this->_backend->pseudo_inverse_svd(&b, _alpha, &pseudo_inv_b);

  Backend::MatrixXc mm = Backend::MatrixXc::Identity(m, m);
  this->_backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, std::complex<double>(1, 0), b, pseudo_inv_b, std::complex<double>(-1, 0), &mm);
  Backend::MatrixXc tmp = Backend::MatrixXc::Zero(m, m);
  this->MatrixMul(p, mm, &tmp);
  this->MatrixMul(tmp, p, &mm);
  Backend::MatrixXc x_mat = Backend::MatrixXc::Identity(m, m);

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<double> range(0, 1);
  const Backend::VectorXc zero = Backend::VectorXc::Zero(m);
  Backend::VectorXc x = Backend::VectorXc::Zero(m);
  for (size_t i = 0; i < _repeat; i++) {
    const auto ii = static_cast<size_t>(static_cast<double>(m) * range(mt));

    Backend::VectorXc mmc = mm.col(ii);
    mmc(ii) = 0;

    this->MatrixVecMul(x_mat, mmc, &x);
    if (std::complex<double> gamma = this->_backend->dot_c(x, mmc); gamma.real() > 0) {
      x = -x * sqrt(_lambda / gamma.real());
      set_bcd_result(x_mat, x, ii);
    } else {
      set_bcd_result(x_mat, zero, ii);
    }
  }

  const Backend::VectorXc u = this->_backend->max_eigen_vector(&x_mat);

  Backend::VectorXc ut = Backend::VectorXc::Zero(m);
  this->MatrixVecMul(p, u, &ut);

  Backend::VectorXc q = Backend::VectorXc::Zero(n);
  this->MatrixVecMul(pseudo_inv_b, ut, &q);

  const auto max_coefficient = this->_backend->max_coefficient_c(q);
  SetFromComplexDrive(this->_data, q, _normalize, max_coefficient);

  this->_built = true;
  return Ok();
}

Error HoloGainEVD::Calc(const core::GeometryPtr& geometry) {
  const auto m = this->_foci.size();
  const auto n = geometry->num_transducers();

  const auto g = TransferMatrix(this->_foci, geometry);

  Backend::VectorXc denominator(m);
  for (size_t i = 0; i < m; i++) {
    auto tmp = std::complex<double>(0, 0);
    for (size_t j = 0; j < n; j++) tmp += g(i, j);
    denominator(i) = tmp;
  }

  Backend::MatrixXc x(n, m);
  for (size_t i = 0; i < m; i++) {
    auto c = std::complex<double>(this->_amps[i], 0) / denominator(i);
    for (size_t j = 0; j < n; j++) x(j, i) = c * std::conj(g(i, j));
  }
  Backend::MatrixXc r = Backend::MatrixXc::Zero(m, m);
  this->MatrixMul(g, x, &r);
  Backend::VectorXc max_ev = this->_backend->max_eigen_vector(&r);

  Backend::MatrixXc sigma = Backend::MatrixXc::Zero(n, n);
  for (size_t j = 0; j < n; j++) {
    double tmp = 0;
    for (size_t i = 0; i < m; i++) tmp += std::abs(g(i, j)) * this->_amps[i];
    sigma(j, j) = std::complex<double>(std::pow(std::sqrt(tmp / static_cast<double>(m)), _gamma), 0.0);
  }

  const Backend::MatrixXc gr = this->_backend->concat_row(g, sigma);

  Backend::VectorXc f = Backend::VectorXc::Zero(m + n);
  for (size_t i = 0; i < m; i++) f(i) = this->_amps[i] * max_ev(i) / std::abs(max_ev(i));

  Backend::MatrixXc gtg = Backend::MatrixXc::Zero(n, n);
  this->_backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, std::complex<double>(1, 0), gr, gr, std::complex<double>(0, 0), &gtg);

  Backend::VectorXc gtf = Backend::VectorXc::Zero(n);
  this->_backend->matrix_vector_mul(TRANSPOSE::CONJ_TRANS, std::complex<double>(1, 0), gr, f, std::complex<double>(0, 0), &gtf);

  this->_backend->solve_ch(&gtg, &gtf);

  const auto max_coefficient = this->_backend->max_coefficient_c(gtf);
  SetFromComplexDrive(this->_data, gtf, _normalize, max_coefficient);

  this->_built = true;
  return Ok();
}

Error HoloGainNaive::Calc(const core::GeometryPtr& geometry) {
  const auto m = this->_foci.size();
  const auto n = geometry->num_transducers();

  const auto g = TransferMatrix(this->_foci, geometry);
  Backend::VectorXc p(m);
  for (size_t i = 0; i < m; i++) p(i) = std::complex<double>(this->_amps[i], 0);

  Backend::VectorXc q = Backend::VectorXc::Zero(n);
  this->_backend->matrix_vector_mul(TRANSPOSE::CONJ_TRANS, std::complex<double>(1, 0), g, p, std::complex<double>(0, 0), &q);

  SetFromComplexDrive(this->_data, q, true, 1.0);

  this->_built = true;
  return Ok();
}

Error HoloGainGS::Calc(const core::GeometryPtr& geometry) {
  const auto m = this->_foci.size();
  const auto n = geometry->num_transducers();

  const auto g = TransferMatrix(this->_foci, geometry);

  Backend::VectorXc q0 = Backend::VectorXc::Ones(n);

  Backend::VectorXc q(n);
  this->_backend->vec_cpy_c(q0, &q);

  Backend::VectorXc gamma = Backend::VectorXc::Zero(m);
  Backend::VectorXc p(m);
  Backend::VectorXc xi = Backend::VectorXc::Zero(n);
  for (size_t k = 0; k < _repeat; k++) {
    this->MatrixVecMul(g, q, &gamma);
    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / std::abs(gamma(i)) * this->_amps[i];
    this->_backend->matrix_vector_mul(TRANSPOSE::CONJ_TRANS, std::complex<double>(1, 0), g, p, std::complex<double>(0, 0), &xi);
    for (size_t j = 0; j < n; j++) q(j) = xi(j) / std::abs(xi(j)) * q0(j);
  }

  SetFromComplexDrive(this->_data, q, true, 1.0);

  this->_built = true;
  return Ok();
}

Error HoloGainGSPAT::Calc(const core::GeometryPtr& geometry) {
  const auto m = this->_foci.size();
  const auto n = geometry->num_transducers();

  const auto g = TransferMatrix(this->_foci, geometry);

  Backend::VectorXc denominator(m);
  for (size_t i = 0; i < m; i++) {
    auto tmp = std::complex<double>(0, 0);
    for (size_t j = 0; j < n; j++) tmp += std::abs(g(i, j));
    denominator(i) = tmp;
  }

  Backend::MatrixXc b(n, m);
  for (size_t i = 0; i < m; i++) {
    auto d = denominator(i) * denominator(i);
    for (size_t j = 0; j < n; j++) {
      b(j, i) = std::conj(g(i, j)) / d;
    }
  }

  Backend::MatrixXc r = Backend::MatrixXc::Zero(m, m);
  this->MatrixMul(g, b, &r);

  Backend::VectorXc p(m);
  for (size_t i = 0; i < m; i++) p(i) = std::complex<double>(this->_amps[i], 0);

  Backend::VectorXc gamma = Backend::VectorXc::Zero(m);
  this->MatrixVecMul(r, p, &gamma);
  for (size_t k = 0; k < _repeat; k++) {
    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / std::abs(gamma(i)) * this->_amps[i];
    this->MatrixVecMul(r, p, &gamma);
  }

  for (size_t i = 0; i < m; i++) p(i) = gamma(i) / (std::abs(gamma(i)) * std::abs(gamma(i))) * this->_amps[i] * this->_amps[i];

  Backend::VectorXc q = Backend::VectorXc::Zero(n);
  this->MatrixVecMul(b, p, &q);

  SetFromComplexDrive(this->_data, q, true, 1.0);

  this->_built = true;
  return Ok();
}

Error HoloGainLM::Calc(const core::GeometryPtr& geometry) {
  if (!this->_backend->supports_solve()) return Err(std::string("This backend does not support this method."));

  auto make_bhb = [](const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const core::GeometryPtr& geo,
                     Backend::MatrixXc* bhb) {
    const auto m = foci.size();

    Backend::MatrixXc p = Backend::MatrixXc::Zero(m, m);
    for (size_t i = 0; i < m; i++) p(i, i) = -amps[i];

    const auto g = TransferMatrix(foci, geo);

    const Backend::MatrixXc b = backend->concat_col(g, p);
    backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, std::complex<double>(1, 0), b, b, std::complex<double>(0, 0), bhb);
  };

  auto calc_t_th = [](const BackendPtr& backend, const Backend::VectorX& x, Backend::MatrixXc* tth) {
    const size_t len = x.size();
    Backend::MatrixXc t(len, 1);
    for (size_t i = 0; i < len; i++) t(i, 0) = std::exp(std::complex<double>(0, -x(i)));
    backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, std::complex<double>(1, 0), t, t, std::complex<double>(0, 0), tth);
  };

  const auto m = this->_foci.size();
  const auto n = geometry->num_transducers();
  const auto n_param = n + m;

  Backend::MatrixXc bhb = Backend::MatrixXc::Zero(n_param, n_param);
  make_bhb(this->_backend, this->_foci, this->_amps, geometry, &bhb);

  Backend::VectorX x = Backend::VectorX::Zero(n_param);
  for (size_t i = 0; i < _initial.size(); i++) x[i] = _initial[i];

  auto nu = 2.0;

  Backend::MatrixXc tth = Backend::MatrixXc::Zero(n_param, n_param);
  calc_t_th(this->_backend, x, &tth);

  Backend::MatrixXc bhb_tth(n_param, n_param);
  this->_backend->hadamard_product(bhb, tth, &bhb_tth);

  Backend::MatrixX a(n_param, n_param);
  this->_backend->real(bhb_tth, &a);

  Backend::VectorX g(n_param);
  for (size_t i = 0; i < n_param; i++) {
    double tmp = 0;
    for (size_t k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
    g(i) = tmp;
  }

  double a_max = 0;
  for (size_t i = 0; i < n_param; i++) a_max = std::max(a_max, a(i, i));

  auto mu = _tau * a_max;

  Backend::VectorXc t(n_param);
  for (size_t i = 0; i < n_param; i++) t(i) = std::exp(std::complex<double>(0, x(i)));

  Backend::VectorXc tmp_vec_c = Backend::VectorXc::Zero(n_param);
  this->MatrixVecMul(bhb, t, &tmp_vec_c);
  double fx = this->_backend->dot_c(t, tmp_vec_c).real();

  const Backend::MatrixX identity = Backend::MatrixX::Identity(n_param, n_param);
  Backend::VectorX tmp_vec(n_param);
  Backend::VectorX h_lm(n_param);
  Backend::VectorX x_new(n_param);
  Backend::MatrixX tmp_mat(n_param, n_param);
  for (size_t k = 0; k < _k_max; k++) {
    if (this->_backend->max_coefficient(g) <= _eps_1) break;

    this->_backend->mat_cpy(a, &tmp_mat);
    this->_backend->matrix_add(mu, identity, double{1.0}, &tmp_mat);
    this->_backend->solve_g(&tmp_mat, &g, &h_lm);
    if (h_lm.norm() <= _eps_2 * (x.norm() + _eps_2)) break;

    this->_backend->vec_cpy(x, &x_new);
    this->_backend->vector_add(double{-1.0}, h_lm, double{1.0}, &x_new);
    for (size_t i = 0; i < n_param; i++) t(i) = std::exp(std::complex<double>(0, x_new(i)));

    this->MatrixVecMul(bhb, t, &tmp_vec_c);
    const double fx_new = this->_backend->dot_c(t, tmp_vec_c).real();

    this->_backend->vec_cpy(g, &tmp_vec);
    this->_backend->vector_add(mu, h_lm, 1.0, &tmp_vec);
    const double l0_lhlm = this->_backend->dot(h_lm, tmp_vec) / 2;

    const auto rho = (fx - fx_new) / l0_lhlm;
    fx = fx_new;
    if (rho > 0) {
      this->_backend->vec_cpy(x_new, &x);
      calc_t_th(this->_backend, x, &tth);
      this->_backend->hadamard_product(bhb, tth, &bhb_tth);
      this->_backend->real(bhb_tth, &a);
      for (size_t i = 0; i < n_param; i++) {
        double tmp = 0;
        for (size_t j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
        g(i) = tmp;
      }
      mu *= std::max(1. / 3., std::pow(1 - (2 * rho - 1), 3.0));
      nu = 2;
    } else {
      mu *= nu;
      nu *= 2;
    }
  }

  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (size_t j = 0; j < n; j++) {
    const uint16_t duty = 0xFF00;
    const auto f_phase = std::fmod(x(j), 2 * M_PI) / (2 * M_PI);
    const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
    this->_data[dev_idx][trans_idx++] = duty | phase;
    if (trans_idx == core::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }

  this->_built = true;
  return Ok();
}

}  // namespace autd::gain::holo
