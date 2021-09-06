// File: holo_gain.cpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 07/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/holo.hpp"

#include <limits>
#include <random>

#include "autd3/core/geometry.hpp"
#include "autd3/core/hardware_defined.hpp"
#include "autd3/core/utils.hpp"
#include "autd3/gain/linalg_backend.hpp"
#include "autd3/utils.hpp"

namespace {
std::shared_ptr<autd::gain::holo::MatrixXc> generate_transfer_matrix(const autd::gain::holo::BackendPtr& backend,
                                                                     const std::vector<autd::core::Vector3>& foci,
                                                                     const autd::core::GeometryPtr& geometry) {
  std::vector<const double*> positions, directions;
  positions.reserve(geometry->num_devices());
  directions.reserve(geometry->num_devices());
  for (size_t i = 0; i < geometry->num_devices(); i++) {
    positions.emplace_back(geometry->position(i, 0).data());
    directions.emplace_back(geometry->direction(i).data());
  }
  return backend->transfer_matrix(reinterpret_cast<const double*>(foci.data()), foci.size(), positions, directions, geometry->wavelength(),
                                  geometry->attenuation_coefficient());
}
}  // namespace

namespace autd::gain::holo {

void SDP::calc(const core::GeometryPtr& geometry) {
  const auto m = static_cast<Eigen::Index>(_foci.size());
  const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  const auto amps = _backend->allocate_matrix_c("amps", m, 1);
  amps->copy_from(_amps);
  const auto p = _backend->allocate_matrix_c("P", m, m);
  p->fill(0.0);
  p->set_diagonal(amps);

  const auto b = generate_transfer_matrix(_backend, _foci, geometry);
  const auto pseudo_inv_b = _backend->allocate_matrix_c("pinvb", n, m);
  _backend->pseudo_inverse_svd(b, _alpha, pseudo_inv_b);

  const auto mm = _backend->allocate_matrix_c("mm", m, m);
  const auto one = _backend->allocate_matrix_c("onec", m, 1);
  one->fill(ONE);
  mm->set_diagonal(one);

  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, -ONE, b, pseudo_inv_b, ONE, mm);
  const auto tmp = _backend->allocate_matrix_c("tmp", m, m);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, p, mm, ZERO, tmp);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, tmp, p, ZERO, mm);

  const auto x_mat = _backend->allocate_matrix_c("x_mat", m, m);
  x_mat->set_diagonal(one);

  std::random_device rnd;
  std::mt19937 mt(rnd());
  const std::uniform_real_distribution<double> range(0, 1);
  const auto zero = _backend->allocate_matrix_c("zero", m, 1);
  zero->fill(ZERO);
  const auto x = _backend->allocate_matrix_c("x", m, 1);
  const auto mmc = _backend->allocate_matrix_c("mmc", m, 1);

  for (size_t i = 0; i < _repeat; i++) {
    const auto ii = static_cast<Eigen::Index>(static_cast<double>(m) * range(mt));

    mm->get_col(ii, mmc);
    mmc->set(ii, 0, ZERO);

    _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, x_mat, mmc, ZERO, x);
    if (complex gamma = _backend->dot(x, mmc); gamma.real() > 0) {
      _backend->scale(x, complex(-std::sqrt(_lambda / gamma.real()), 0.0));
      _backend->set_bcd_result(x_mat, x, ii);
    } else {
      _backend->set_bcd_result(x_mat, zero, ii);
    }
  }

  const auto u = _backend->allocate_matrix_c("u", m, 1);
  _backend->max_eigen_vector(x_mat, u);

  const auto ut = _backend->allocate_matrix_c("ut", m, 1);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, p, u, ZERO, ut);

  const auto q = _backend->allocate_matrix_c("q", n, 1);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, pseudo_inv_b, ut, ZERO, q);

  const auto max_coefficient = _backend->max_coefficient(q);
  _backend->set_from_complex_drive(_data, q, _normalize, max_coefficient);

  _built = true;
}

void EVD::calc(const core::GeometryPtr& geometry) {
  const auto m = static_cast<Eigen::Index>(_foci.size());
  const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  const auto g = generate_transfer_matrix(_backend, _foci, geometry);
  const auto amps = _backend->allocate_matrix_c("amps", m, 1);
  amps->copy_from(_amps);

  const auto x = _backend->back_prop(g, amps);

  const auto r = _backend->allocate_matrix_c("r", m, m);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, x, ZERO, r);
  const auto max_ev = _backend->allocate_matrix_c("max_ev", m, 1);
  _backend->max_eigen_vector(r, max_ev);

  const auto sigma = _backend->sigma_regularization(g, amps, _gamma);

  const auto gr = _backend->concat_row(g, sigma);

  const auto fm = _backend->allocate_matrix_c("fm", m, 1);
  _backend->arg(max_ev, fm);
  _backend->hadamard_product(amps, fm, fm);
  const auto fn = _backend->allocate_matrix_c("fn", n, 1);
  fn->fill(0.0);
  const auto f = _backend->concat_row(fm, fn);

  const auto gtg = _backend->allocate_matrix_c("gtg", n, n);
  _backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, gr, gr, ZERO, gtg);

  const auto gtf = _backend->allocate_matrix_c("gtf", n, 1);
  _backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, gr, f, ZERO, gtf);

  _backend->solve_ch(gtg, gtf);

  const auto max_coefficient = _backend->max_coefficient(gtf);
  _backend->set_from_complex_drive(_data, gtf, _normalize, max_coefficient);

  _built = true;
}

void Naive::calc(const core::GeometryPtr& geometry) {
  const auto m = static_cast<Eigen::Index>(_foci.size());
  const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  const auto g = generate_transfer_matrix(_backend, _foci, geometry);
  const auto p = _backend->allocate_matrix_c("amps", m, 1);
  p->copy_from(_amps);

  const auto q = _backend->allocate_matrix_c("q", n, 1);

  _backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, p, ZERO, q);

  _backend->set_from_complex_drive(_data, q, true, 1.0);

  _built = true;
}

void GS::calc(const core::GeometryPtr& geometry) {
  const auto m = static_cast<Eigen::Index>(_foci.size());
  const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  const auto g = generate_transfer_matrix(_backend, _foci, geometry);
  const auto amps = _backend->allocate_matrix_c("amps", m, 1);
  amps->copy_from(_amps);

  const auto q0 = _backend->allocate_matrix_c("q0", n, 1);
  q0->fill(ONE);

  const auto q = _backend->allocate_matrix_c("q", n, 1);
  _backend->mat_cpy(q0, q);

  const auto gamma = _backend->allocate_matrix_c("gamma", m, 1);
  const auto p = _backend->allocate_matrix_c("p", m, 1);
  const auto xi = _backend->allocate_matrix_c("xi", n, 1);
  for (size_t k = 0; k < _repeat; k++) {
    _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, q, ZERO, gamma);
    _backend->arg(gamma, gamma);
    _backend->hadamard_product(gamma, amps, p);
    _backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, p, ZERO, xi);
    _backend->arg(xi, xi);
    _backend->hadamard_product(xi, q0, q);
  }

  _backend->set_from_complex_drive(_data, q, true, 1.0);

  _built = true;
}

void GSPAT::calc(const core::GeometryPtr& geometry) {
  const auto m = static_cast<Eigen::Index>(_foci.size());
  const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  const auto g = generate_transfer_matrix(_backend, _foci, geometry);
  const auto amps = _backend->allocate_matrix_c("amps", m, 1);
  amps->copy_from(_amps);

  const auto b = _backend->back_prop(g, amps);

  const auto r = _backend->allocate_matrix_c("r", m, m);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, b, ZERO, r);

  const auto p = _backend->allocate_matrix_c("p", m, 1);
  p->copy_from(_amps);

  const auto gamma = _backend->allocate_matrix_c("gamma", m, 1);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, r, p, ZERO, gamma);
  for (size_t k = 0; k < _repeat; k++) {
    _backend->arg(gamma, gamma);
    _backend->hadamard_product(gamma, amps, p);
    _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, r, p, ZERO, gamma);
  }

  gamma->copy_to_host();
  for (Eigen::Index i = 0; i < m; i++)
    p->data(i, 0) = gamma->data(i, 0) / (std::abs(gamma->data(i, 0)) * std::abs(gamma->data(i, 0))) * std::abs(_amps[i]) * std::abs(_amps[i]);

  const auto q = _backend->allocate_matrix_c("q", n, 1);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, b, p, ZERO, q);

  _backend->set_from_complex_drive(_data, q, true, 1.0);

  _built = true;
}

void LM::calc(const core::GeometryPtr& geometry) {
  auto make_bhb = [](const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::shared_ptr<MatrixXc>& amps,
                     const core::GeometryPtr& geo) {
    const auto m = static_cast<Eigen::Index>(foci.size());
    const auto n = static_cast<Eigen::Index>(geo->num_transducers());
    const Eigen::Index n_param = n + m;

    const auto p = backend->allocate_matrix_c("p", m, m);
    p->fill(0.0);
    backend->scale(amps, complex(-1.0, 0.0));
    p->set_diagonal(amps);

    const auto g = generate_transfer_matrix(backend, foci, geo);

    const auto b = backend->concat_col(g, p);

    auto bhb = backend->allocate_matrix_c("bhb", n_param, n_param);
    backend->matrix_mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, b, b, ZERO, bhb);
    return bhb;
  };

  auto calc_t_th = [](const BackendPtr& backend, const std::shared_ptr<MatrixX>& zero, const std::shared_ptr<MatrixX> x,
                      const std::shared_ptr<MatrixXc>& tth) {
    const auto len = x->data.size();
    const auto t = backend->allocate_matrix_c("T", len, 1);

    backend->make_complex(zero, x, t);
    backend->scale(t, complex(-1, 0));
    backend->exp(t);

    backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, t, t, ZERO, tth);
  };

  const auto m = static_cast<Eigen::Index>(_foci.size());
  const auto n = static_cast<Eigen::Index>(geometry->num_transducers());
  const Eigen::Index n_param = n + m;

  const auto amps = _backend->allocate_matrix_c("amps", m, 1);
  amps->copy_from(_amps);

  const auto bhb = make_bhb(_backend, _foci, amps, geometry);

  const auto x = _backend->allocate_matrix("x", n_param, 1);
  x->fill(0.0);
  x->copy_from(_initial);

  auto nu = 2.0;

  const auto tth = _backend->allocate_matrix_c("tth", n_param, n_param);
  const auto zero = _backend->allocate_matrix("zero", n_param, 1);
  zero->fill(0.0);
  calc_t_th(_backend, zero, x, tth);

  const auto bhb_tth = _backend->allocate_matrix_c("bhb_tth", n_param, n_param);
  _backend->hadamard_product(bhb, tth, bhb_tth);

  const auto a = _backend->allocate_matrix("a", n_param, n_param);
  _backend->real(bhb_tth, a);

  const auto g = _backend->allocate_matrix("g", n_param, 1);
  _backend->col_sum_imag(bhb_tth, g);

  auto a_diag = _backend->allocate_matrix("a_diag", n_param, 1);
  a->get_diagonal(a_diag);
  double a_max = a_diag->max_element();

  auto mu = _tau * a_max;

  const auto t = _backend->allocate_matrix_c("t", n_param, 1);
  _backend->make_complex(zero, x, t);
  _backend->exp(t);

  const auto tmp_vec_c = _backend->allocate_matrix_c("tmp_vec_c", n_param, 1);
  _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, bhb, t, ZERO, tmp_vec_c);
  double fx = _backend->dot(t, tmp_vec_c).real();

  const auto identity = _backend->allocate_matrix("identity", n_param, n_param);
  auto one = _backend->allocate_matrix("one", n_param, 1);
  one->fill(1.0);
  identity->set_diagonal(one);

  const auto tmp_vec = _backend->allocate_matrix("tmp_vec", n_param, 1);
  const auto h_lm = _backend->allocate_matrix("h_lm", n_param, 1);
  const auto x_new = _backend->allocate_matrix("x_new", n_param, 1);
  const auto tmp_mat = _backend->allocate_matrix("tmp_mat", n_param, n_param);
  for (size_t k = 0; k < _k_max; k++) {
    if (_backend->max_coefficient(g) <= _eps_1) break;

    _backend->mat_cpy(a, tmp_mat);
    _backend->matrix_add(mu, identity, tmp_mat);
    _backend->solve_g(tmp_mat, g, h_lm);
    if (std::sqrt(_backend->dot(h_lm, h_lm)) <= _eps_2 * (std::sqrt(_backend->dot(x, x)) + _eps_2)) break;

    _backend->mat_cpy(x, x_new);
    _backend->matrix_add(-1.0, h_lm, x_new);
    _backend->make_complex(zero, x_new, t);
    _backend->exp(t);

    _backend->matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, bhb, t, ZERO, tmp_vec_c);
    const double fx_new = _backend->dot(t, tmp_vec_c).real();

    _backend->mat_cpy(g, tmp_vec);
    _backend->matrix_add(mu, h_lm, tmp_vec);
    const double l0_lhlm = _backend->dot(h_lm, tmp_vec) / 2;

    const auto rho = (fx - fx_new) / l0_lhlm;
    fx = fx_new;
    if (rho > 0) {
      _backend->mat_cpy(x_new, x);
      calc_t_th(_backend, zero, x, tth);
      _backend->hadamard_product(bhb, tth, bhb_tth);
      _backend->real(bhb_tth, a);
      _backend->col_sum_imag(bhb_tth, g);
      mu *= std::max(1. / 3., std::pow(1 - (2 * rho - 1), 3.0));
      nu = 2;
    } else {
      mu *= nu;
      nu *= 2;
    }
  }

  x->copy_to_host();
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (Eigen::Index j = 0; j < n; j++) {
    constexpr uint8_t duty = 0xFF;
    const auto f_phase = x->data(j, 0) / (2 * M_PI);
    const auto phase = core::Utilities::to_phase(f_phase);
    _data[dev_idx][trans_idx++] = core::Utilities::pack_to_u16(duty, phase);
    if (trans_idx == core::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }

  _built = true;
}

void Greedy::calc(const core::GeometryPtr& geometry) {
  const auto m = _foci.size();

  const auto wave_num = 2.0 * M_PI / geometry->wavelength();
  const auto attenuation = geometry->attenuation_coefficient();

  std::vector<std::unique_ptr<complex[]>> tmp;
  tmp.reserve(_phases.size());
  for (size_t i = 0; i < _phases.size(); i++) tmp.emplace_back(std::make_unique<complex[]>(m));

  const auto cache = std::make_unique<complex[]>(m);

  auto transfer_foci = [wave_num, attenuation](const core::Vector3& trans_pos, const core::Vector3& trans_dir, const complex phase,
                                               const std::vector<core::Vector3>& foci, complex* res) {
    for (size_t i = 0; i < foci.size(); i++) res[i] = utils::transfer(trans_pos, trans_dir, foci[i], wave_num, attenuation) * phase;
  };

  for (size_t dev = 0; dev < geometry->num_devices(); dev++) {
    for (size_t i = 0; i < core::NUM_TRANS_IN_UNIT; i++) {
      const auto& trans_pos = geometry->position(dev, i);
      const auto& trans_dir = geometry->direction(dev);
      size_t min_idx = 0;
      auto min_v = std::numeric_limits<double>::infinity();
      for (size_t p = 0; p < _phases.size(); p++) {
        transfer_foci(trans_pos, trans_dir, _phases[p], _foci, &tmp[p][0]);
        auto v = 0.0;
        for (size_t j = 0; j < m; j++) v += std::abs(_amps[j] - std::abs(tmp[p][j] + cache[j]));
        if (v < min_v) {
          min_v = v;
          min_idx = p;
        }
      }
      for (size_t j = 0; j < m; j++) cache[j] += tmp[min_idx][j];

      constexpr uint8_t duty = 0xFF;
      const auto f_phase = std::arg(_phases[min_idx]) / (2 * M_PI);
      const auto phase = core::Utilities::to_phase(f_phase);
      _data[dev][i] = core::Utilities::pack_to_u16(duty, phase);
    }
  }

  _built = true;
}

}  // namespace autd::gain::holo
