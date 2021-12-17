// File: holo_impl.hpp
// Project: holo
// Created Date: 10/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "autd3/core/gain.hpp"
#include "autd3/core/utils.hpp"
#include "autd3/gain/matrix.hpp"
#include "autd3/utils.hpp"

namespace autd {
namespace gain {
namespace holo {

template <typename M>
void generate_transfer_matrix(const std::vector<core::Vector3>& foci, const core::Geometry& geometry, const std::shared_ptr<M> g) {
  std::vector<const core::Transducer*> transducers;
  std::vector<const double*> directions;
  transducers.reserve(geometry.num_devices());
  directions.reserve(geometry.num_devices());
  for (const auto& dev : geometry) {
    transducers.emplace_back(&*dev.begin());
    directions.emplace_back(dev.z_direction().data());
  }
  g->transfer_matrix(reinterpret_cast<const double*>(foci.data()), foci.size(), transducers, directions, geometry.wavelength(),
                     geometry.attenuation_coefficient());
}

template <typename P, typename M>
void back_prop(P& pool, const std::shared_ptr<M>& transfer, const std::shared_ptr<M>& amps, std::shared_ptr<M> b) {
  const auto m = transfer->rows();
  const auto n = transfer->cols();

  const auto denominator = pool.rent_c("denominator", m, 1);

  denominator->reduce_col(transfer);
  denominator->abs(denominator);
  denominator->reciprocal(denominator);
  denominator->hadamard_product(amps, denominator);

  const auto b_tmp = pool.rent_c("back_prop_tmp", m, m);
  b_tmp->create_diagonal(denominator);

  b->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, transfer, b_tmp, ZERO);
}

template <typename P>
void sdp_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps_, double alpha,
              const double lambda, const size_t repeat, bool normalize, std::vector<core::Drive>& dst) {
  const auto m = foci.size();
  const auto n = geometry.num_transducers();

  const auto amps = pool.rent_c("amps", m, 1);
  amps->copy_from(amps_);
  const auto p = pool.rent_c("P", m, m);
  p->create_diagonal(amps);

  const auto b = pool.rent_c("b", m, n);
  generate_transfer_matrix(foci, geometry, b);
  const auto pseudo_inv_b = pool.rent_c("p_inv_b", n, m);
  auto u_ = pool.rent_c("u_", m, m);
  auto s = pool.rent_c("s", n, m);
  auto vt = pool.rent_c("vt", n, n);
  auto buf = pool.rent_c("buf", n, m);
  const auto b_tmp = pool.rent_c("b_tmp", m, n);
  b_tmp->copy_from(b);
  pseudo_inv_b->pseudo_inverse_svd(b_tmp, alpha, u_, s, vt, buf);

  const auto mm = pool.rent_c("mm", m, m);
  const auto one = pool.rent_c("one_c", m, 1);
  one->fill(ONE);
  mm->create_diagonal(one);

  mm->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, -ONE, b, pseudo_inv_b, ONE);
  const auto tmp = pool.rent_c("tmp", m, m);
  tmp->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, p, mm, ZERO);
  mm->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, tmp, p, ZERO);

  const auto x_mat = pool.rent_c("x_mat", m, m);
  x_mat->create_diagonal(one);

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<double> range(0, 1);
  const auto zero = pool.rent_c("zero", m, 1);
  zero->fill(ZERO);
  const auto x = pool.rent_c("x", m, 1);
  const auto x_conj = pool.rent_c("x_conj", m, 1);
  const auto mmc = pool.rent_c("mmc", m, 1);
  for (size_t i = 0; i < repeat; i++) {
    const auto ii = static_cast<size_t>(std::floor(static_cast<double>(m) * range(mt)));

    mmc->get_col(mm, ii);
    mmc->set(ii, 0, ZERO);

    x->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, x_mat, mmc, ZERO);
    if (complex gamma = x->dot(mmc); gamma.real() > 0) {
      x->scale(complex(-std::sqrt(lambda / gamma.real()), 0.0));
      x_conj->conj(x);
      x_mat->set_row(ii, 0, ii, x_conj);
      x_mat->set_row(ii, ii + 1, m, x_conj);
      x_mat->set_col(ii, 0, ii, x);
      x_mat->set_col(ii, ii + 1, m, x);
    } else {
      x_mat->set_row(ii, 0, ii, zero);
      x_mat->set_row(ii, ii + 1, m, zero);
      x_mat->set_col(ii, 0, ii, zero);
      x_mat->set_col(ii, ii + 1, m, zero);
    }
  }

  const auto u = pool.rent_c("u", m, 1);
  x_mat->max_eigen_vector(u);

  const auto ut = pool.rent_c("ut", m, 1);
  ut->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, p, u, ZERO);

  const auto q = pool.rent_c("q", n, 1);
  q->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, pseudo_inv_b, ut, ZERO);

  const auto max_coefficient = q->max_element();
  q->set_from_complex_drive(dst, normalize, max_coefficient);
}

template <typename P>
void evd_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps_, double gamma,
              bool normalize, std::vector<core::Drive>& dst) {
  const auto m = foci.size();
  const auto n = geometry.num_transducers();

  const auto g = pool.rent_c("g", m, n);
  generate_transfer_matrix(foci, geometry, g);
  const auto amps = pool.rent_c("amps", m, 1);
  amps->copy_from(amps_);

  const auto x = pool.rent_c("x", n, m);
  back_prop(pool, g, amps, x);

  const auto r = pool.rent_c("r", m, m);
  r->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, x, ZERO);
  const auto max_ev = pool.rent_c("max_ev", m, 1);
  r->max_eigen_vector(max_ev);

  const auto sigma = pool.rent_c("sigma", n, n);
  {
    const auto sigma_tmp = pool.rent_c("sigma_tmp", n, 1);
    sigma_tmp->mul(TRANSPOSE::TRANS, TRANSPOSE::NO_TRANS, ONE, g, amps, ZERO);
    sigma_tmp->abs(sigma_tmp);
    sigma_tmp->scale(1.0 / static_cast<double>(m));
    sigma_tmp->sqrt();
    sigma_tmp->pow(gamma);
    sigma->create_diagonal(sigma_tmp);
  }

  const auto gr = pool.rent_c("gr", m + n, n);
  gr->concat_row(g, sigma);

  const auto fm = pool.rent_c("fm", m, 1);
  fm->arg(max_ev);
  fm->hadamard_product(amps, fm);
  const auto fn = pool.rent_c("fn", n, 1);
  fn->fill(0.0);
  const auto f = pool.rent_c("f", m + n, 1);
  f->concat_row(fm, fn);

  const auto gtg = pool.rent_c("gtg", n, n);
  gtg->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, gr, gr, ZERO);

  const auto gtf = pool.rent_c("gtf", n, 1);
  gtf->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, gr, f, ZERO);

  gtg->solve(gtf);

  const auto max_coefficient = gtf->max_element();
  gtf->set_from_complex_drive(dst, normalize, max_coefficient);
}

template <typename P>
void naive_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps,
                std::vector<core::Drive>& dst) {
  const auto m = foci.size();
  const auto n = geometry.num_transducers();

  const auto g = pool.rent_c("g", m, n);
  generate_transfer_matrix(foci, geometry, g);
  const auto p = pool.rent_c("amps", m, 1);
  p->copy_from(amps);

  const auto q = pool.rent_c("q", n, 1);

  q->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, p, ZERO);

  q->set_from_complex_drive(dst, true, 1.0);
}

template <typename P>
void gs_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps_, const size_t repeat,
             std::vector<core::Drive>& dst) {
  const auto m = foci.size();
  const auto n = geometry.num_transducers();

  const auto g = pool.rent_c("g", m, n);
  generate_transfer_matrix(foci, geometry, g);

  const auto amps = pool.rent_c("amps", m, 1);
  amps->copy_from(amps_);

  const auto q0 = pool.rent_c("q0", n, 1);
  q0->fill(ONE);

  const auto q = pool.rent_c("q", n, 1);
  q->copy_from(q0);

  const auto gamma = pool.rent_c("gamma", m, 1);
  const auto p = pool.rent_c("p", m, 1);
  const auto xi = pool.rent_c("xi", n, 1);
  for (size_t k = 0; k < repeat; k++) {
    gamma->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, q, ZERO);
    gamma->arg(gamma);
    p->hadamard_product(gamma, amps);
    xi->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, p, ZERO);
    xi->arg(xi);
    q->hadamard_product(xi, q0);
  }

  q->set_from_complex_drive(dst, true, 1.0);
}

template <typename P>
void gspat_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps_,
                const size_t repeat, std::vector<core::Drive>& dst) {
  const auto m = foci.size();
  const auto n = geometry.num_transducers();

  const auto g = pool.rent_c("g", m, n);
  generate_transfer_matrix(foci, geometry, g);

  const auto amps = pool.rent_c("amps", m, 1);
  amps->copy_from(amps_);

  const auto b = pool.rent_c("b", n, m);
  back_prop(pool, g, amps, b);

  const auto r = pool.rent_c("r", m, m);
  r->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, b, ZERO);

  const auto p = pool.rent_c("p", m, 1);
  p->copy_from(amps);

  const auto gamma = pool.rent_c("gamma", m, 1);
  gamma->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, r, p, ZERO);
  for (size_t k = 0; k < repeat; k++) {
    gamma->arg(gamma);
    p->hadamard_product(gamma, amps);
    gamma->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, r, p, ZERO);
  }

  const auto tmp = pool.rent_c("tmp", m, 1);
  tmp->abs(gamma);
  tmp->reciprocal(tmp);
  tmp->hadamard_product(tmp, amps);
  tmp->hadamard_product(tmp, amps);
  gamma->arg(gamma);
  p->hadamard_product(gamma, tmp);

  const auto q = pool.rent_c("q", n, 1);
  q->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, b, p, ZERO);

  q->set_from_complex_drive(dst, true, 1.0);
}

template <typename P>
void make_bhb(P& pool, const std::vector<core::Vector3>& foci, const core::Geometry& geo) {
  const auto m = foci.size();
  const auto n = geo.num_transducers();
  const auto n_param = n + m;

  const auto amps = pool.rent_c("amps", m, 1);

  const auto p = pool.rent_c("p", m, m);
  amps->scale(complex(-1.0, 0.0));
  p->create_diagonal(amps);

  const auto g = pool.rent_c("g", m, n);
  generate_transfer_matrix(foci, geo, g);

  const auto b = pool.rent_c("b", m, m + n);
  b->concat_col(g, p);

  auto bhb = pool.rent_c("bhb", n_param, n_param);
  bhb->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, b, b, ZERO);
}
template <typename P>
void make_t(P& pool, const size_t n_param) {
  const auto x = pool.rent("x", n_param, 1);
  const auto t = pool.rent_c("T", n_param, 1);
  const auto zero = pool.rent("zero", n_param, 1);
  t->make_complex(zero, x);
  t->scale(complex(-1, 0));
  t->exp();
}
template <typename P>
void calc_jtf(P& pool, const size_t n_param) {
  const auto t = pool.rent_c("T", n_param, 1);
  auto bhb = pool.rent_c("bhb", n_param, n_param);
  const auto tth = pool.rent_c("tth", n_param, n_param);
  const auto bhb_tth = pool.rent_c("bhb_tth", n_param, n_param);
  const auto bhb_tth_r = pool.rent("bhb_tth_r", n_param, n_param);
  const auto jtf = pool.rent("jtf", n_param, 1);
  tth->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, t, t, ZERO);
  bhb_tth->hadamard_product(bhb, tth);
  bhb_tth_r->imag(bhb_tth);
  jtf->reduce_col(bhb_tth_r);
}

template <typename P>
void calc_jtj_jtf(P& pool, const size_t n_param) {
  const auto t = pool.rent_c("T", n_param, 1);
  auto bhb = pool.rent_c("bhb", n_param, n_param);
  const auto tth = pool.rent_c("tth", n_param, n_param);
  const auto bhb_tth = pool.rent_c("bhb_tth", n_param, n_param);
  const auto bhb_tth_r = pool.rent("bhb_tth_r", n_param, n_param);
  const auto jtj = pool.rent("jtj", n_param, n_param);
  const auto jtf = pool.rent("jtf", n_param, 1);
  tth->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, t, t, ZERO);
  bhb_tth->hadamard_product(bhb, tth);
  jtj->real(bhb_tth);
  bhb_tth_r->imag(bhb_tth);
  jtf->reduce_col(bhb_tth_r);
}
template <typename P>
double calc_fx(P& pool, const std::string& param_name, const size_t n_param) {
  const auto x = pool.rent(param_name, n_param, 1);
  const auto zero = pool.rent("zero", n_param, 1);
  const auto t = pool.rent_c("t", n_param, 1);
  t->make_complex(zero, x);
  t->exp();
  const auto bhb = pool.rent_c("bhb", n_param, n_param);
  const auto tmp_vec_c = pool.rent_c("tmp_vec_c", n_param, 1);
  tmp_vec_c->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, bhb, t, ZERO);
  return t->dot(tmp_vec_c).real();
}

template <typename P>
void lm_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps_, double eps_1,
             double eps_2, const double tau, const size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) {
  const auto m = foci.size();
  const auto n = geometry.num_transducers();
  const size_t n_param = n + m;

  const auto amps = pool.rent_c("amps", m, 1);
  amps->copy_from(amps_);

  make_bhb(pool, foci, geometry);
  auto bhb = pool.rent_c("bhb", n_param, n_param);

  const auto x = pool.rent("x", n_param, 1);
  x->fill(0.0);
  x->copy_from(initial);

  auto nu = 2.0;

  const auto t = pool.rent_c("T", n_param, 1);
  const auto zero = pool.rent("zero", n_param, 1);
  zero->fill(0.0);
  make_t(pool, n_param);

  const auto tth = pool.rent_c("tth", n_param, n_param);
  const auto bhb_tth = pool.rent_c("bhb_tth", n_param, n_param);
  const auto a = pool.rent("jtj", n_param, n_param);
  const auto g = pool.rent("jtf", n_param, 1);
  calc_jtj_jtf(pool, n_param);

  const auto a_diag = pool.rent("a_diag", n_param, 1);
  a_diag->get_diagonal(a);
  const double a_max = a_diag->max_element();

  auto mu = tau * a_max;

  const auto tmp_vec_c = pool.rent_c("tmp_vec_c", n_param, 1);
  const auto t_ = pool.rent_c("t", n_param, 1);
  double fx = calc_fx(pool, "x", n_param);

  const auto identity = pool.rent("identity", n_param, n_param);
  const auto one = pool.rent("one", n_param, 1);
  one->fill(1.0);
  identity->create_diagonal(one);

  const auto tmp_vec = pool.rent("tmp_vec", n_param, 1);
  const auto h_lm = pool.rent("h_lm", n_param, 1);
  const auto x_new = pool.rent("x_new", n_param, 1);
  const auto tmp_mat = pool.rent("tmp_mat", n_param, n_param);
  for (size_t k = 0; k < k_max; k++) {
    if (g->max_element() <= eps_1) break;

    tmp_mat->copy_from(a);
    tmp_mat->add(mu, identity);
    h_lm->copy_from(g);
    tmp_mat->solve(h_lm);
    if (std::sqrt(h_lm->dot(h_lm)) <= eps_2 * (std::sqrt(x->dot(x)) + eps_2)) break;

    x_new->copy_from(x);
    x_new->add(-1.0, h_lm);

    const double fx_new = calc_fx(pool, "x_new", n_param);

    tmp_vec->copy_from(g);
    tmp_vec->add(mu, h_lm);
    const double l0_lhlm = h_lm->dot(tmp_vec) / 2;

    const auto rho = (fx - fx_new) / l0_lhlm;
    fx = fx_new;
    if (rho > 0) {
      x->copy_from(x_new);

      make_t(pool, n_param);
      calc_jtj_jtf(pool, n_param);

      mu *= (std::max)(1. / 3., std::pow(1 - (2 * rho - 1), 3.0));
      nu = 2;
    } else {
      mu *= nu;
      nu *= 2;
    }
  }

  x->set_from_arg(dst, n);
}

template <typename P>
void gauss_newton_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps_,
                       double eps_1, double eps_2, const size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) {
  const auto m = foci.size();
  const auto n = geometry.num_transducers();
  const size_t n_param = n + m;

  const auto amps = pool.rent_c("amps", m, 1);
  amps->copy_from(amps_);

  make_bhb(pool, foci, geometry);
  auto bhb = pool.rent_c("bhb", n_param, n_param);

  const auto x = pool.rent("x", n_param, 1);
  x->fill(0.0);
  x->copy_from(initial);

  const auto t = pool.rent_c("T", n_param, 1);
  const auto zero = pool.rent("zero", n_param, 1);
  zero->fill(0.0);
  make_t(pool, n_param);

  const auto tth = pool.rent_c("tth", n_param, n_param);
  const auto bhb_tth = pool.rent_c("bhb_tth", n_param, n_param);
  const auto a = pool.rent("jtj", n_param, n_param);
  const auto g = pool.rent("jtf", n_param, 1);
  calc_jtj_jtf(pool, n_param);

  const auto h_lm = pool.rent("h_lm", n_param, 1);
  const auto pia = pool.rent("pis", n_param, n_param);
  auto u = pool.rent("u", n_param, n_param);
  auto s = pool.rent("s", n_param, n_param);
  auto vt = pool.rent("vt", n_param, n_param);
  auto buf = pool.rent("buf", n_param, n_param);
  const auto a_tmp = pool.rent("a_tmp", n_param, n_param);
  for (size_t k = 0; k < k_max; k++) {
    if (g->max_element() <= eps_1) break;

    //_backend->solve_g(a, g, h_lm);
    a_tmp->copy_from(a);
    pia->pseudo_inverse_svd(a_tmp, 1e-3, u, s, vt, buf);
    h_lm->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, 1.0, pia, g, 0.0);
    if (std::sqrt(h_lm->dot(h_lm)) <= eps_2 * (std::sqrt(x->dot(x)) + eps_2)) break;

    x->add(-1.0, h_lm);

    make_t(pool, n_param);
    calc_jtj_jtf(pool, n_param);
  }

  x->set_from_arg(dst, n);
}

template <typename P>
void gradient_descent_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps_,
                           double eps, const double step, const size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) {
  const auto m = foci.size();
  const auto n = geometry.num_transducers();
  const size_t n_param = n + m;

  const auto amps = pool.rent_c("amps", m, 1);
  amps->copy_from(amps_);

  make_bhb(pool, foci, geometry);
  auto bhb = pool.rent_c("bhb", n_param, n_param);

  const auto x = pool.rent("x", n_param, 1);
  x->fill(0.0);
  x->copy_from(initial);

  const auto t = pool.rent_c("T", n_param, 1);
  const auto zero = pool.rent("zero", n_param, 1);
  zero->fill(0.0);

  const auto tth = pool.rent_c("tth", n_param, n_param);
  const auto bhb_tth = pool.rent_c("bhb_tth", n_param, n_param);
  const auto jtf = pool.rent("jtf", n_param, 1);
  for (size_t k = 0; k < k_max; k++) {
    make_t(pool, n_param);
    calc_jtf(pool, n_param);
    if (jtf->max_element() <= eps) break;
    x->add(-step, jtf);
  }
  x->set_from_arg(dst, n);
}

template <typename P>
void apo_impl(P& pool, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps,
              double lambda, const size_t line_search_max, const size_t k_max, std::vector<core::Drive>& dst) {
  auto make_ri = [](P& pool_, const size_t m, const size_t n, const size_t i) {
    const auto g = pool_.rent_c("g", m, n);

    const auto di = pool_.rent_c("di", m, m);
    di->fill(ZERO);
    di->set(i, i, ONE);

    auto ri = pool_.rent_c("ri" + std::to_string(i), n, n);
    const auto tmp = pool_.rent_c("tmp_ri", n, m);
    tmp->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, di, ZERO);
    ri->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, tmp, g, ZERO);
  };

  auto calc_nabla_j = [](P& pool_, const size_t m, const size_t n, const double lambda_, const std::string& nabla_j_name) {
    const auto tmp = pool_.rent_c("cnj_tmp", n, 1);
    const auto q = pool_.rent_c("q", n, 1);
    const auto p2 = pool_.rent_c("p2", m, 1);
    const auto nabla_j = pool_.rent_c(nabla_j_name, n, 1);
    for (size_t i = 0; i < m; i++) {
      auto ri = pool_.rent_c("ri" + std::to_string(i), n, n);
      tmp->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, ri, q, ZERO);
      const auto s = p2->at(i, 0) - q->dot(tmp);
      tmp->scale(s);
      nabla_j->add(ONE, tmp);
    }

    nabla_j->add(complex(lambda_, 0), q);
  };

  auto calc_j = [](P& pool_, const size_t m, const size_t n, const double lambda_) {
    const auto q = pool_.rent_c("q", n, 1);
    const auto p2 = pool_.rent_c("p2", m, 1);
    const auto tmp = pool_.rent_c("cj_tmp", n, 1);
    auto j = 0.0;
    for (size_t i = 0; i < m; i++) {
      auto ri = pool_.rent_c("ri" + std::to_string(i), n, n);
      tmp->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, ri, q, ZERO);
      const auto s = p2->at(i, 0) - q->dot(tmp);
      j += std::norm(s);
    }

    j += std::abs(q->dot(q)) * lambda_;
    return j;
  };

  auto line_search = [&calc_j](P& pool_, const size_t m, const size_t n, const double lambda_, const size_t line_search_max_) {
    auto alpha = 0.0;
    auto min = (std::numeric_limits<double>::max)();
    for (size_t i = 0; i < line_search_max_; i++) {
      const auto a = static_cast<double>(i) / static_cast<double>(line_search_max_);  // FIXME: only for 0-1
      if (const auto v = calc_j(pool_, m, n, lambda_); v < min) {
        alpha = a;
        min = v;
      }
    }
    return alpha;
  };

  const auto m = foci.size();
  const auto n = geometry.num_transducers();

  const auto g = pool.rent_c("g", m, n);
  generate_transfer_matrix(foci, geometry, g);

  const auto p = pool.rent_c("p", m, 1);
  p->copy_from(amps);

  const auto p2 = pool.rent_c("p2", m, 1);
  p2->hadamard_product(p, p);

  const auto one = pool.rent_c("one", n, 1);
  one->fill(ONE);
  const auto h = pool.rent_c("h", n, n);
  h->create_diagonal(one);

  const auto tmp = pool.rent_c("tmp", n, n);
  tmp->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, g, ZERO);
  tmp->add(complex(lambda, 0.0), h);

  const auto q = pool.rent_c("q", n, 1);
  q->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, p, ZERO);
  tmp->solve(q);

  for (size_t i = 0; i < m; i++) make_ri(pool, m, n, i);

  const auto nabla_j = pool.rent_c("nabla_j", n, 1);
  calc_nabla_j(pool, m, n, lambda, "nabla_j");

  const auto d = pool.rent_c("d", n, 1);
  const auto nabla_j_new = pool.rent_c("nabla_j_new", n, 1);
  const auto s = pool.rent_c("s", n, 1);
  const auto hs = pool.rent_c("hs", n, 1);
  for (size_t k = 0; k < k_max; k++) {
    d->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, -ONE, h, nabla_j, ZERO);

    // FIXME
    const auto alpha = line_search(pool, m, n, lambda, line_search_max);

    d->scale(complex(alpha, 0));

    if (std::sqrt(d->dot(d).real()) < eps) break;

    q->add(ONE, d);
    calc_nabla_j(pool, m, n, lambda, "nabla_j_new");

    s->copy_from(nabla_j_new);
    s->add(-ONE, nabla_j);

    const auto ys = ONE / d->dot(s);
    hs->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, h, s, ZERO);
    const auto shs = -ONE / s->dot(hs);

    h->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ys, d, d, ONE);
    h->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, shs, hs, hs, ONE);

    nabla_j->copy_from(nabla_j_new);
  }

  const auto max_coefficient = q->max_element();
  q->set_from_complex_drive(dst, true, max_coefficient);
}

template <typename P>
void greedy_impl(P&, const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const size_t phase_div,
                 std::vector<core::Drive>& dst) {
  const auto m = foci.size();

  std::vector<complex> phases;
  phases.reserve(phase_div);
  for (size_t i = 0; i < phase_div; i++)
    phases.emplace_back(std::exp(complex(0., 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(phase_div))));

  const auto wave_num = 2.0 * M_PI / geometry.wavelength();
  const auto attenuation = geometry.attenuation_coefficient();

  std::vector<std::unique_ptr<complex[]>> tmp;
  tmp.reserve(phases.size());
  for (size_t i = 0; i < phases.size(); i++) tmp.emplace_back(std::make_unique<complex[]>(m));

  const auto cache = std::make_unique<complex[]>(m);

  auto transfer_foci = [wave_num, attenuation](const core::Transducer& transducer, const core::Vector3& trans_dir, const complex phase,
                                               const std::vector<core::Vector3>& foci_, complex* res) {
    for (size_t i = 0; i < foci_.size(); i++) res[i] = utils::transfer(transducer, trans_dir, foci_[i], wave_num, attenuation) * phase;
  };

  for (const auto& dev : geometry) {
    const auto& trans_dir = dev.z_direction();
    for (const auto& transducer : dev) {
      size_t min_idx = 0;
      auto min_v = std::numeric_limits<double>::infinity();
      for (size_t p = 0; p < phases.size(); p++) {
        transfer_foci(transducer, trans_dir, phases[p], foci, &tmp[p][0]);
        auto v = 0.0;
        for (size_t j = 0; j < m; j++) v += std::abs(amps[j] - std::abs(tmp[p][j] + cache[j]));
        if (v < min_v) {
          min_v = v;
          min_idx = p;
        }
      }
      for (size_t j = 0; j < m; j++) cache[j] += tmp[min_idx][j];

      dst[transducer.id()].duty = 0xFF;
      dst[transducer.id()].phase = core::utils::to_phase(std::arg(phases[min_idx]));
    }
  }
}
}  // namespace holo
}  // namespace gain
}  // namespace autd
