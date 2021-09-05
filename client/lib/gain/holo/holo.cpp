// File: holo_gain.cpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/holo.hpp"

#include <limits>
#include <random>

#include "autd3/core/exception.hpp"
#include "autd3/core/geometry.hpp"
#include "autd3/core/hardware_defined.hpp"
#include "autd3/core/utils.hpp"
#include "autd3/gain/linalg_backend.hpp"

namespace autd::gain::holo {

void SDP::calc(const core::GeometryPtr& geometry) {
  // auto set_bcd_result = [](Backend::MatrixXc& mat, const Backend::VectorXc& vec, const Eigen::Index idx) {
  //  const Eigen::Index m = vec.size();
  //  for (Eigen::Index i = 0; i < idx; i++) mat(idx, i) = std::conj(vec(i));
  //  for (Eigen::Index i = idx + 1; i < m; i++) mat(idx, i) = std::conj(vec(i));
  //  for (Eigen::Index i = 0; i < idx; i++) mat(i, idx) = vec(i);
  //  for (Eigen::Index i = idx + 1; i < m; i++) mat(i, idx) = vec(i);
  //};

  // const auto m = static_cast<Eigen::Index>(this->_foci.size());
  // const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  // Backend::MatrixXc p = Backend::MatrixXc::Zero(m, m);
  // for (Eigen::Index i = 0; i < m; i++) p(i, i) = complex(this->_amps[i], 0);

  // auto b = transfer_matrix(this->_foci, geometry);
  // Backend::MatrixXc pseudo_inv_b(n, m);
  // this->_backend->pseudo_inverse_svd(b, _alpha, &pseudo_inv_b);

  // Backend::MatrixXc mm = Backend::MatrixXc::Identity(m, m);
  // this->_backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, One, b, pseudo_inv_b, complex(-1, 0), &mm);
  // Backend::MatrixXc tmp = Backend::MatrixXc::Zero(m, m);
  // this->matrix_mul(p, mm, &tmp);
  // this->matrix_mul(tmp, p, &mm);
  // Backend::MatrixXc x_mat = Backend::MatrixXc::Identity(m, m);

  // std::random_device rnd;
  // std::mt19937 mt(rnd());
  // std::uniform_real_distribution<double> range(0, 1);
  // const Backend::VectorXc zero = Backend::VectorXc::Zero(m);
  // Backend::VectorXc x = Backend::VectorXc::Zero(m);
  // for (size_t i = 0; i < _repeat; i++) {
  //  const auto ii = static_cast<Eigen::Index>(static_cast<double>(m) * range(mt));

  //  Backend::VectorXc mmc = mm.col(ii);
  //  mmc(ii) = 0;

  //  this->matrix_vec_mul(x_mat, mmc, &x);
  //  if (complex gamma = this->_backend->dot_c(x, mmc); gamma.real() > 0) {
  //    x = -x * sqrt(_lambda / gamma.real());
  //    set_bcd_result(x_mat, x, ii);
  //  } else {
  //    set_bcd_result(x_mat, zero, ii);
  //  }
  //}

  // const Backend::VectorXc u = this->_backend->max_eigen_vector(&x_mat);

  // Backend::VectorXc ut = Backend::VectorXc::Zero(m);
  // this->matrix_vec_mul(p, u, &ut);

  // Backend::VectorXc q = Backend::VectorXc::Zero(n);
  // this->matrix_vec_mul(pseudo_inv_b, ut, &q);

  // const auto max_coefficient = this->_backend->max_coefficient_c(q);
  // set_from_complex_drive(this->_data, q, _normalize, max_coefficient);

  this->_built = true;
}

void EVD::calc(const core::GeometryPtr& geometry) {
  // const auto m = static_cast<Eigen::Index>(this->_foci.size());
  // const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  // const auto g = transfer_matrix(this->_foci, geometry);

  // Backend::VectorXc denominator(m);
  // for (Eigen::Index i = 0; i < m; i++) {
  //  auto tmp = Zero;
  //  for (Eigen::Index j = 0; j < n; j++) tmp += g(i, j);
  //  denominator(i) = tmp;
  //}

  // Backend::MatrixXc x(n, m);
  // for (Eigen::Index i = 0; i < m; i++) {
  //  auto c = complex(this->_amps[i], 0) / denominator(i);
  //  for (Eigen::Index j = 0; j < n; j++) x(j, i) = c * std::conj(g(i, j));
  //}
  // Backend::MatrixXc r = Backend::MatrixXc::Zero(m, m);
  // this->matrix_mul(g, x, &r);
  // Backend::VectorXc max_ev = this->_backend->max_eigen_vector(&r);

  // Backend::MatrixXc sigma = Backend::MatrixXc::Zero(n, n);
  // for (Eigen::Index j = 0; j < n; j++) {
  //  double tmp = 0;
  //  for (Eigen::Index i = 0; i < m; i++) tmp += std::abs(g(i, j)) * this->_amps[i];
  //  sigma(j, j) = complex(std::pow(std::sqrt(tmp / static_cast<double>(m)), _gamma), 0.0);
  //}

  // const Backend::MatrixXc gr = this->_backend->concat_row(g, sigma);

  // Backend::VectorXc f = Backend::VectorXc::Zero(m + n);
  // for (Eigen::Index i = 0; i < m; i++) f(i) = this->_amps[i] * max_ev(i) / std::abs(max_ev(i));

  // Backend::MatrixXc gtg = Backend::MatrixXc::Zero(n, n);
  // this->_backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, One, gr, gr, Zero, &gtg);

  // Backend::VectorXc gtf = Backend::VectorXc::Zero(n);
  // this->_backend->matrix_vector_mul(TRANSPOSE::CONJ_TRANS, One, gr, f, Zero, &gtf);

  // this->_backend->solve_ch(&gtg, &gtf);

  // const auto max_coefficient = this->_backend->max_coefficient_c(gtf);
  // set_from_complex_drive(this->_data, gtf, _normalize, max_coefficient);

  this->_built = true;
}

void Naive::calc(const core::GeometryPtr& geometry) {
  // const auto m = static_cast<Eigen::Index>(this->_foci.size());
  // const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  // const auto g = transfer_matrix(this->_foci, geometry);
  // Backend::VectorXc p(m);
  // for (Eigen::Index i = 0; i < m; i++) p(i) = complex(this->_amps[i], 0);

  // Backend::VectorXc q = Backend::VectorXc::Zero(n);
  // this->_backend->matrix_vector_mul(TRANSPOSE::CONJ_TRANS, One, g, p, Zero, &q);

  // set_from_complex_drive(this->_data, q, true, 1.0);

  this->_built = true;
}

void GS::calc(const core::GeometryPtr& geometry) {
  const auto m = static_cast<Eigen::Index>(this->_foci.size());
  const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  const auto g = _backend->transfer_matrix(this->_foci, geometry);
  const auto amps = _backend->allocate_vector_c("amps", m);
  amps->copy_from(_amps);

  auto q0 = _backend->allocate_vector_c("q0", n);
  q0->fill(One);

  auto q = _backend->allocate_vector_c("q", n);
  this->_backend->vec_cpy(q0, q);

  auto gamma = _backend->allocate_vector_c("gamma", m);
  gamma->fill(Zero);
  auto p = _backend->allocate_vector_c("p", m);
  auto xi = _backend->allocate_vector_c("xi", n);
  xi->fill(complex(0.0, 0.0));
  for (size_t k = 0; k < _repeat; k++) {
    _backend->matrix_vector_mul(TRANSPOSE::NO_TRANS, One, g, q, Zero, gamma);
    _backend->arg(gamma, gamma);
    _backend->hadamard_product(gamma, amps, p);
    this->_backend->matrix_vector_mul(TRANSPOSE::CONJ_TRANS, One, g, p, Zero, xi);
    _backend->arg(xi, xi);
    _backend->hadamard_product(xi, q0, q);
  }

  _backend->set_from_complex_drive(this->_data, q, true, 1.0);

  this->_built = true;
}

void GSPAT::calc(const core::GeometryPtr& geometry) {
  // const auto m = static_cast<Eigen::Index>(this->_foci.size());
  // const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  // const auto g = transfer_matrix(this->_foci, geometry);

  // Backend::VectorXc denominator(m);
  // for (Eigen::Index i = 0; i < m; i++) {
  //  auto tmp = Zero;
  //  for (Eigen::Index j = 0; j < n; j++) tmp += std::abs(g(i, j));
  //  denominator(i) = tmp;
  //}

  // Backend::MatrixXc b(n, m);
  // for (Eigen::Index i = 0; i < m; i++) {
  //  auto d = std::norm(denominator(i));
  //  for (Eigen::Index j = 0; j < n; j++) b(j, i) = std::conj(g(i, j)) / d;
  //}

  // Backend::MatrixXc r = Backend::MatrixXc::Zero(m, m);
  // this->matrix_mul(g, b, &r);

  // Backend::VectorXc p(m);
  // for (Eigen::Index i = 0; i < m; i++) p(i) = complex(this->_amps[i], 0);

  // Backend::VectorXc gamma = Backend::VectorXc::Zero(m);
  // this->matrix_vec_mul(r, p, &gamma);
  // for (size_t k = 0; k < _repeat; k++) {
  //  for (Eigen::Index i = 0; i < m; i++) p(i) = gamma(i) / std::abs(gamma(i)) * this->_amps[i];
  //  this->matrix_vec_mul(r, p, &gamma);
  //}

  // for (Eigen::Index i = 0; i < m; i++) p(i) = gamma(i) / (std::abs(gamma(i)) * std::abs(gamma(i))) * this->_amps[i] * this->_amps[i];

  // Backend::VectorXc q = Backend::VectorXc::Zero(n);
  // this->matrix_vec_mul(b, p, &q);

  // set_from_complex_drive(this->_data, q, true, 1.0);

  this->_built = true;
}

void LM::calc(const core::GeometryPtr& geometry) {
  // if (!this->_backend->supports_solve()) throw core::GainBuildError("This backend does not support this method.");

  // auto make_bhb = [](const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const core::GeometryPtr&
  // geo,
  //                   Backend::MatrixXc* bhb) {
  //  const auto m = static_cast<Eigen::Index>(foci.size());

  //  Backend::MatrixXc p = Backend::MatrixXc::Zero(m, m);
  //  for (Eigen::Index i = 0; i < m; i++) p(i, i) = -amps[i];

  //  const auto g = transfer_matrix(foci, geo);

  //  const Backend::MatrixXc b = backend->concat_col(g, p);
  //  backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, One, b, b, Zero, bhb);
  //};

  // auto calc_t_th = [](const BackendPtr& backend, const Backend::VectorX& x, Backend::MatrixXc* tth) {
  //  const auto len = x.size();
  //  Backend::MatrixXc t(len, 1);
  //  for (Eigen::Index i = 0; i < len; i++) t(i, 0) = std::exp(complex(0, -x(i)));
  //  backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, One, t, t, Zero, tth);
  //};

  // const auto m = static_cast<Eigen::Index>(this->_foci.size());
  // const auto n = static_cast<Eigen::Index>(geometry->num_transducers());
  // const Eigen::Index n_param = n + m;

  // Backend::MatrixXc bhb = Backend::MatrixXc::Zero(n_param, n_param);
  // make_bhb(this->_backend, this->_foci, this->_amps, geometry, &bhb);

  // Backend::VectorX x = Backend::VectorX::Zero(n_param);
  // std::memcpy(x.data(), &_initial[0], _initial.size() * sizeof(double));

  // auto nu = 2.0;

  // Backend::MatrixXc tth = Backend::MatrixXc::Zero(n_param, n_param);
  // calc_t_th(this->_backend, x, &tth);

  // Backend::MatrixXc bhb_tth(n_param, n_param);
  // this->_backend->hadamard_product(bhb, tth, &bhb_tth);

  // Backend::MatrixX a(n_param, n_param);
  // this->_backend->real(bhb_tth, &a);

  // Backend::VectorX g(n_param);
  // for (Eigen::Index i = 0; i < n_param; i++) {
  //  double tmp = 0;
  //  for (Eigen::Index k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
  //  g(i) = tmp;
  //}

  // double a_max = 0;
  // for (Eigen::Index i = 0; i < n_param; i++) a_max = std::max(a_max, a(i, i));

  // auto mu = _tau * a_max;

  // Backend::VectorXc t(n_param);
  // for (Eigen::Index i = 0; i < n_param; i++) t(i) = std::exp(complex(0, x(i)));

  // Backend::VectorXc tmp_vec_c = Backend::VectorXc::Zero(n_param);
  // this->matrix_vec_mul(bhb, t, &tmp_vec_c);
  // double fx = this->_backend->dot_c(t, tmp_vec_c).real();

  // const Backend::MatrixX identity = Backend::MatrixX::Identity(n_param, n_param);
  // Backend::VectorX tmp_vec(n_param);
  // Backend::VectorX h_lm(n_param);
  // Backend::VectorX x_new(n_param);
  // Backend::MatrixX tmp_mat(n_param, n_param);
  // for (size_t k = 0; k < _k_max; k++) {
  //  if (this->_backend->max_coefficient(g) <= _eps_1) break;

  //  this->_backend->mat_cpy(a, &tmp_mat);
  //  this->_backend->matrix_add(mu, identity, &tmp_mat);
  //  this->_backend->solve_g(&tmp_mat, &g, &h_lm);
  //  if (h_lm.norm() <= _eps_2 * (x.norm() + _eps_2)) break;

  //  this->_backend->vec_cpy(x, &x_new);
  //  this->_backend->vector_add(-1.0, h_lm, &x_new);
  //  for (Eigen::Index i = 0; i < n_param; i++) t(i) = std::exp(complex(0, x_new(i)));

  //  this->matrix_vec_mul(bhb, t, &tmp_vec_c);
  //  const double fx_new = this->_backend->dot_c(t, tmp_vec_c).real();

  //  this->_backend->vec_cpy(g, &tmp_vec);
  //  this->_backend->vector_add(mu, h_lm, &tmp_vec);
  //  const double l0_lhlm = this->_backend->dot(h_lm, tmp_vec) / 2;

  //  const auto rho = (fx - fx_new) / l0_lhlm;
  //  fx = fx_new;
  //  if (rho > 0) {
  //    this->_backend->vec_cpy(x_new, &x);
  //    calc_t_th(this->_backend, x, &tth);
  //    this->_backend->hadamard_product(bhb, tth, &bhb_tth);
  //    this->_backend->real(bhb_tth, &a);
  //    for (Eigen::Index i = 0; i < n_param; i++) {
  //      double tmp = 0;
  //      for (Eigen::Index j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
  //      g(i) = tmp;
  //    }
  //    mu *= std::max(1. / 3., std::pow(1 - (2 * rho - 1), 3.0));
  //    nu = 2;
  //  } else {
  //    mu *= nu;
  //    nu *= 2;
  //  }
  //}

  // size_t dev_idx = 0;
  // size_t trans_idx = 0;
  // for (Eigen::Index j = 0; j < n; j++) {
  //  constexpr uint8_t duty = 0xFF;
  //  const auto f_phase = x(j) / (2 * M_PI);
  //  const auto phase = core::Utilities::to_phase(f_phase);
  //  this->_data[dev_idx][trans_idx++] = core::Utilities::pack_to_u16(duty, phase);
  //  if (trans_idx == core::NUM_TRANS_IN_UNIT) {
  //    dev_idx++;
  //    trans_idx = 0;
  //  }
  //}

  this->_built = true;
}

void Greedy::calc(const core::GeometryPtr& geometry) {
  // const auto m = this->_foci.size();

  // const auto wave_num = 2.0 * M_PI / geometry->wavelength();
  // const auto attenuation = geometry->attenuation_coefficient();

  // std::vector<std::unique_ptr<complex[]>> tmp;
  // tmp.reserve(this->_phases.size());
  // for (size_t i = 0; i < this->_phases.size(); i++) tmp.emplace_back(std::make_unique<complex[]>(m));

  // const auto cache = std::make_unique<complex[]>(m);

  // auto transfer_foci = [wave_num, attenuation](const core::Vector3& trans_pos, const core::Vector3& trans_dir, const complex phase,
  //                                             const std::vector<core::Vector3>& foci, complex* res) {
  //  for (size_t i = 0; i < foci.size(); i++) res[i] = transfer(trans_pos, trans_dir, foci[i], wave_num, attenuation) * phase;
  //};

  // for (size_t dev = 0; dev < geometry->num_devices(); dev++) {
  //  for (size_t i = 0; i < core::NUM_TRANS_IN_UNIT; i++) {
  //    auto trans_pos = geometry->position(dev, i);
  //    auto trans_dir = geometry->direction(dev);
  //    size_t min_idx = 0;
  //    auto min_v = std::numeric_limits<double>::infinity();
  //    for (size_t p = 0; p < this->_phases.size(); p++) {
  //      transfer_foci(trans_pos, trans_dir, this->_phases[p], this->_foci, &tmp[p][0]);
  //      auto v = 0.0;
  //      for (size_t j = 0; j < m; j++) v += std::abs(this->_amps[j] - std::abs(tmp[p][j] + cache[j]));
  //      if (v < min_v) {
  //        min_v = v;
  //        min_idx = p;
  //      }
  //    }
  //    for (size_t j = 0; j < m; j++) cache[j] += tmp[min_idx][j];

  //    constexpr uint8_t duty = 0xFF;
  //    const auto f_phase = std::arg(this->_phases[min_idx]) / (2 * M_PI);
  //    const auto phase = core::Utilities::to_phase(f_phase);
  //    this->_data[dev][i] = core::Utilities::pack_to_u16(duty, phase);
  //  }
  //}

  this->_built = true;
}

}  // namespace autd::gain::holo
