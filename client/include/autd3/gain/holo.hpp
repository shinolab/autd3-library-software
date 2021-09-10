// File: holo_gain.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "autd3/core/exception.hpp"
#include "autd3/core/gain.hpp"
#include "autd3/utils.hpp"
#include "linalg_backend.hpp"

namespace autd::gain::holo {
template <typename M>
void generate_transfer_matrix(const std::vector<autd::core::Vector3>& foci, const autd::core::GeometryPtr& geometry, const std::shared_ptr<M> g) {
  std::vector<const double*> positions, directions;
  positions.reserve(geometry->num_devices());
  directions.reserve(geometry->num_devices());
  for (size_t i = 0; i < geometry->num_devices(); i++) {
    positions.emplace_back(geometry->position(i, 0).data());
    directions.emplace_back(geometry->direction(i).data());
  }
  g->transfer_matrix(reinterpret_cast<const double*>(foci.data()), foci.size(), positions, directions, geometry->wavelength(),
                     geometry->attenuation_coefficient());
}

/**
 * @brief Gain to produce multiple focal points
 */
template <typename P>
class Holo : public core::Gain {
 public:
  Holo(std::vector<core::Vector3> foci, const std::vector<double>& amps) : _pool(std::make_unique<P>()), _foci(std::move(foci)) {
    if (_foci.size() != amps.size()) throw core::GainBuildError("The size of foci and amps are not the same");
    _amps.reserve(amps.size());
    for (const auto amp : amps) _amps.emplace_back(complex(amp, 0.0));
  }
  ~Holo() override = default;
  Holo(const Holo& v) noexcept = default;
  Holo& operator=(const Holo& obj) = default;
  Holo(Holo&& obj) = default;
  Holo& operator=(Holo&& obj) = default;

  std::vector<core::Vector3>& foci() { return this->_foci; }
  std::vector<complex>& amplitudes() { return this->_amps; }

 protected:
  std::unique_ptr<P> _pool;
  std::vector<core::Vector3> _foci;
  std::vector<complex> _amps;
};

/**
 * @brief Gain to produce multiple focal points with SDP method.
 * Refer to Inoue, Seki, Yasutoshi Makino, and Hiroyuki Shinoda. "Active touch
 * perception produced by airborne ultrasonic haptic hologram." 2015 IEEE
 * World Haptics Conference (WHC). IEEE, 2015.
 */
template <typename P>
class SDP final : public Holo<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] alpha parameter
   * @param[in] lambda parameter
   * @param[in] repeat parameter
   * @param[in] normalize parameter
   */
  static std::shared_ptr<SDP> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, double alpha = 1e-3,
                                     double lambda = 0.9, size_t repeat = 100, bool normalize = true) {
    return std::make_shared<SDP>(foci, amps, alpha, lambda, repeat, normalize);
  }

  void calc(const core::GeometryPtr& geometry) {
    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());

    const auto amps = _pool->rent_c("amps", m, 1);
    amps->copy_from(_amps);
    const auto p = _pool->rent_c("P", m, m);
    p->create_diagonal(amps);

    const auto b = _pool->rent_c("b", m, n);
    generate_transfer_matrix(_foci, geometry, b);
    const auto pseudo_inv_b = _pool->rent_c("pinvb", n, m);
    auto u_ = this->_pool->rent_c("u_", m, m);
    auto s = this->_pool->rent_c("s", n, m);
    auto vt = this->_pool->rent_c("vt", n, n);
    auto buf = this->_pool->rent_c("buf", n, m);
    const auto btmp = _pool->rent_c("btmp", m, n);
    btmp->copy_from(b);
    pseudo_inv_b->pseudo_inverse_svd(btmp, _alpha, u_, s, vt, buf);

    const auto mm = _pool->rent_c("mm", m, m);
    const auto one = _pool->rent_c("onec", m, 1);
    one->fill(ONE);
    mm->create_diagonal(one);

    mm->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, -ONE, b, pseudo_inv_b, ONE);
    const auto tmp = _pool->rent_c("tmp", m, m);
    tmp->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, p, mm, ZERO);
    mm->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, tmp, p, ZERO);

    const auto x_mat = _pool->rent_c("x_mat", m, m);
    x_mat->create_diagonal(one);

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> range(0, 1);
    const auto zero = _pool->rent_c("zero", m, 1);
    zero->fill(ZERO);
    const auto x = _pool->rent_c("x", m, 1);
    const auto mmc = _pool->rent_c("mmc", m, 1);
    for (size_t i = 0; i < _repeat; i++) {
      const auto ii = (static_cast<double>(m) * range(mt));

      mmc->get_col(mm, ii);
      mmc->set(ii, 0, ZERO);

      x->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, x_mat, mmc, ZERO);
      if (complex gamma = x->dot(mmc); gamma.real() > 0) {
        x->scale(complex(-std::sqrt(_lambda / gamma.real()), 0.0));
        x_mat->set_bcd_result(x, ii);
      } else {
        x_mat->set_bcd_result(zero, ii);
      }
    }

    const auto u = _pool->rent_c("u", m, 1);
    x_mat->max_eigen_vector(u);

    const auto ut = _pool->rent_c("ut", m, 1);
    ut->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, p, u, ZERO);

    const auto q = _pool->rent_c("q", n, 1);
    q->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, pseudo_inv_b, ut, ZERO);

    const auto max_coefficient = q->max_element();
    q->set_from_complex_drive(_data, _normalize, max_coefficient);

    _built = true;
  }

  SDP(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double alpha, const double lambda, const size_t repeat,
      const bool normalize)
      : Holo(foci, amps), _alpha(alpha), _lambda(lambda), _repeat(repeat), _normalize(normalize) {}

 private:
  double _alpha;
  double _lambda;
  size_t _repeat;
  bool _normalize;
};

/**
 * @brief Gain to produce multiple focal points with EVD method.
 * Refer to Long, Benjamin, et al. "Rendering volumetric haptic shapes in mid-air
 * using ultrasound." ACM Transactions on Graphics (TOG) 33.6 (2014): 1-10.
 */
template <typename P>
class EVD final : public Holo<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] gamma parameter
   * @param[in] normalize parameter
   */
  static std::shared_ptr<EVD> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, double gamma = 1,
                                     bool normalize = true) {
    return std::make_shared<EVD>(foci, amps, gamma, normalize);
  }

  void calc(const core::GeometryPtr& geometry) {
    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());

    const auto g = _pool->rent_c("g", m, n);
    generate_transfer_matrix(_foci, geometry, g);
    const auto amps = _pool->rent_c("amps", m, 1);
    amps->copy_from(_amps);

    const auto x = _pool->rent_c("x", n, m);
    x->back_prop(g, amps);

    const auto r = _pool->rent_c("r", m, m);
    r->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, x, ZERO);
    const auto max_ev = _pool->rent_c("max_ev", m, 1);
    r->max_eigen_vector(max_ev);

    const auto sigma = _pool->rent_c("sigma", n, n);
    sigma->sigma_regularization(g, amps, _gamma);

    const auto gr = _pool->rent_c("gr", m + n, n);
    gr->concat_row(g, sigma);

    const auto fm = _pool->rent_c("fm", m, 1);
    fm->arg(max_ev);
    fm->hadamard_product(amps, fm);
    const auto fn = _pool->rent_c("fn", n, 1);
    fn->fill(0.0);
    const auto f = _pool->rent_c("f", m + n, 1);
    f->concat_row(fm, fn);

    const auto gtg = _pool->rent_c("gtg", n, n);
    gtg->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, gr, gr, ZERO);

    const auto gtf = _pool->rent_c("gtf", n, 1);
    gtf->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, gr, f, ZERO);

    gtg->solve(gtf);

    const auto max_coefficient = gtf->max_element();
    gtf->set_from_complex_drive(_data, _normalize, max_coefficient);

    _built = true;
  }

  EVD(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double gamma, const bool normalize)
      : Holo(foci, amps), _gamma(gamma), _normalize(normalize) {}

 private:
  double _gamma;
  bool _normalize;
};

/**
 * @brief Gain to produce multiple focal points with naive method.
 */
template <typename P>
class Naive final : public Holo<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   */
  static std::shared_ptr<Naive> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps) {
    return std::make_shared<Naive>(foci, amps);
  }

  void calc(const core::GeometryPtr& geometry) override {
    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());

    const auto g = _pool->rent_c("g", m, n);
    generate_transfer_matrix(_foci, geometry, g);
    const auto p = _pool->rent_c("amps", m, 1);
    p->copy_from(_amps);

    const auto q = _pool->rent_c("q", n, 1);

    q->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, p, ZERO);

    q->set_from_complex_drive(_data, true, 1.0);

    _built = true;
  }

  Naive(const std::vector<core::Vector3>& foci, const std::vector<double>& amps) : Holo(foci, amps) {}
};

/**
 * @brief Gain to produce multiple focal points with GS method.
 * Refer to Asier Marzo and Bruce W Drinkwater, "Holographic acoustic
 * tweezers," Proceedings of theNational Academy of Sciences, 116(1):84–89, 2019.
 */
template <typename P>
class GS final : public Holo<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<GS> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat = 100) {
    return std::make_shared<GS>(foci, amps, repeat);
  }

  void calc(const core::GeometryPtr& geometry) override {
    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());

    const auto g = _pool->rent_c("g", m, n);
    generate_transfer_matrix(_foci, geometry, g);

    const auto amps = _pool->rent_c("amps", m, 1);
    amps->copy_from(_amps);

    const auto q0 = _pool->rent_c("q0", n, 1);
    q0->fill(ONE);

    const auto q = _pool->rent_c("q", n, 1);
    q->copy_from(q0);

    const auto gamma = _pool->rent_c("gamma", m, 1);
    const auto p = _pool->rent_c("p", m, 1);
    const auto xi = _pool->rent_c("xi", n, 1);
    for (size_t k = 0; k < _repeat; k++) {
      gamma->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, q, ZERO);
      gamma->arg(gamma);
      p->hadamard_product(gamma, amps);
      xi->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, p, ZERO);
      xi->arg(xi);
      q->hadamard_product(xi, q0);
    }

    q->set_from_complex_drive(_data, true, 1.0);

    _built = true;
  }

  GS(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat) : Holo(foci, amps), _repeat(repeat) {}

 private:
  size_t _repeat;
};

/**
 * @brief Gain to produce multiple focal points with GS-PAT method (not yet been implemented with GPU).
 * Refer to Diego Martinez Plasencia et al. "Gs-pat: high-speed multi-point
 * sound-fields for phased arrays of transducers," ACMTrans-actions on
 * Graphics (TOG), 39(4):138–1, 2020.
 */
template <typename P>
class GSPAT final : public Holo<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<GSPAT> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat = 100) {
    return std::make_shared<GSPAT>(foci, amps, repeat);
  }
  void calc(const core::GeometryPtr& geometry) override {
    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());

    const auto g = _pool->rent_c("g", m, n);
    generate_transfer_matrix(_foci, geometry, g);

    const auto amps = _pool->rent_c("amps", m, 1);
    amps->copy_from(_amps);

    const auto b = _pool->rent_c("b", n, m);
    b->back_prop(g, amps);

    const auto r = _pool->rent_c("r", m, m);
    r->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, g, b, ZERO);

    const auto p = _pool->rent_c("p", m, 1);
    p->copy_from(_amps);

    const auto gamma = _pool->rent_c("gamma", m, 1);
    gamma->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, r, p, ZERO);
    for (size_t k = 0; k < _repeat; k++) {
      gamma->arg(gamma);
      p->hadamard_product(gamma, amps);
      gamma->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, r, p, ZERO);
    }

    const auto tmp = _pool->rent_c("tmp", m, 1);
    tmp->abs(gamma);
    tmp->reciprocal(tmp);
    tmp->hadamard_product(tmp, amps);
    tmp->hadamard_product(tmp, amps);
    gamma->arg(gamma);
    p->hadamard_product(gamma, tmp);

    const auto q = _pool->rent_c("q", n, 1);
    q->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, b, p, ZERO);

    q->set_from_complex_drive(_data, true, 1.0);

    _built = true;
  }

  GSPAT(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat) : Holo(foci, amps), _repeat(repeat) {}

 private:
  size_t _repeat;
};

template <typename P>
class NLS : public Holo<P> {
 public:
  NLS(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps, const size_t k_max, std::vector<double> initial)
      : Holo(foci, amps), _eps(eps), _k_max(k_max), _initial(std::move(initial)) {}

 protected:
  void make_bhb(const std::vector<core::Vector3>& foci, const core::GeometryPtr& geo) {
    const auto m = (foci.size());
    const auto n = (geo->num_transducers());
    const auto n_param = n + m;

    const auto amps = _pool->rent_c("amps", m, 1);

    const auto p = _pool->rent_c("p", m, m);
    amps->scale(complex(-1.0, 0.0));
    p->create_diagonal(amps);

    const auto g = _pool->rent_c("g", m, n);
    generate_transfer_matrix(_foci, geo, g);

    const auto b = _pool->rent_c("b", m, m + n);
    b->concat_col(g, p);

    auto bhb = _pool->rent_c("bhb", n_param, n_param);
    bhb->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, b, b, ZERO);
  }
  void make_t(const size_t n_param) {
    const auto x = _pool->rent("x", n_param, 1);
    const auto t = _pool->rent_c("T", n_param, 1);
    const auto zero = _pool->rent("zero", n_param, 1);
    t->make_complex(zero, x);
    t->scale(complex(-1, 0));
    t->exp();
  }
  void calc_jtf(const size_t n_param) {
    const auto t = _pool->rent_c("T", n_param, 1);
    auto bhb = _pool->rent_c("bhb", n_param, n_param);
    const auto tth = _pool->rent_c("tth", n_param, n_param);
    const auto bhb_tth = _pool->rent_c("bhb_tth", n_param, n_param);
    const auto jtf = _pool->rent("jtf", n_param, 1);
    tth->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, t, t, ZERO);
    bhb_tth->hadamard_product(bhb, tth);
    jtf->col_sum_imag(bhb_tth);
  }

  void calc_jtj_jtf(const size_t n_param) {
    const auto t = _pool->rent_c("T", n_param, 1);
    auto bhb = _pool->rent_c("bhb", n_param, n_param);
    const auto tth = _pool->rent_c("tth", n_param, n_param);
    const auto bhb_tth = _pool->rent_c("bhb_tth", n_param, n_param);
    const auto jtj = _pool->rent("jtj", n_param, n_param);
    const auto jtf = _pool->rent("jtf", n_param, 1);
    tth->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, t, t, ZERO);
    bhb_tth->hadamard_product(bhb, tth);
    jtj->real(bhb_tth);
    jtf->col_sum_imag(bhb_tth);
  }
  double calc_fx(const std::string& param_name, const size_t n_param) {
    const auto x = _pool->rent(param_name, n_param, 1);
    const auto zero = _pool->rent("zero", n_param, 1);
    const auto t = _pool->rent_c("t", n_param, 1);
    t->make_complex(zero, x);
    t->exp();

    const auto bhb = _pool->rent_c("bhb", n_param, n_param);
    const auto tmp_vec_c = _pool->rent_c("tmp_vec_c", n_param, 1);
    tmp_vec_c->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, bhb, t, ZERO);
    return t->dot(tmp_vec_c).real();
  }

  double _eps;
  size_t _k_max;
  std::vector<double> _initial;
};

/**
 * @brief Gain to produce multiple focal points with Levenberg-Marquardt method.
 * Refer to K.Levenberg, “A method for the solution of certain non-linear problems in
 * least squares,” Quarterly of applied mathematics, vol.2, no.2, pp.164–168, 1944.
 * D.W.Marquardt, “An algorithm for least-squares estimation of non-linear parameters,” Journal of the society for Industrial and
 * AppliedMathematics, vol.11, no.2, pp.431–441, 1963.
 * K.Madsen, H.Nielsen, and O.Tingleff, “Methods for non-linear least squares problems (2nd ed.),” 2004.
 */
template <typename P>
class LM final : public NLS<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps_1 parameter
   * @param[in] eps_2 parameter
   * @param[in] tau parameter
   * @param[in] k_max parameter
   * @param[in] initial initial phase of transducers
   */
  static std::shared_ptr<LM> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1 = 1e-8,
                                    const double eps_2 = 1e-8, const double tau = 1e-3, const size_t k_max = 5,
                                    const std::vector<double>& initial = {}) {
    return std::make_shared<LM>(foci, amps, eps_1, eps_2, tau, k_max, initial);
  }

  void calc(const core::GeometryPtr& geometry) override {
    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());
    const size_t n_param = n + m;

    const auto amps = _pool->rent_c("amps", m, 1);
    amps->copy_from(_amps);

    make_bhb(_foci, geometry);
    auto bhb = _pool->rent_c("bhb", n_param, n_param);

    const auto x = _pool->rent("x", n_param, 1);
    x->fill(0.0);
    x->copy_from(_initial);

    auto nu = 2.0;

    const auto t = _pool->rent_c("T", n_param, 1);
    const auto zero = _pool->rent("zero", n_param, 1);
    zero->fill(0.0);
    make_t(n_param);

    const auto tth = _pool->rent_c("tth", n_param, n_param);
    const auto bhb_tth = _pool->rent_c("bhb_tth", n_param, n_param);
    const auto a = _pool->rent("jtj", n_param, n_param);
    const auto g = _pool->rent("jtf", n_param, 1);
    calc_jtj_jtf(n_param);

    const auto a_diag = _pool->rent("a_diag", n_param, 1);
    a_diag->get_diagonal(a);
    const double a_max = a_diag->max_element();

    auto mu = _tau * a_max;

    const auto tmp_vec_c = _pool->rent_c("tmp_vec_c", n_param, 1);
    const auto t_ = _pool->rent_c("t", n_param, 1);
    double fx = calc_fx("x", n_param);

    const auto identity = _pool->rent("identity", n_param, n_param);
    const auto one = _pool->rent("one", n_param, 1);
    one->fill(1.0);
    identity->create_diagonal(one);

    const auto tmp_vec = _pool->rent("tmp_vec", n_param, 1);
    const auto h_lm = _pool->rent("h_lm", n_param, 1);
    const auto x_new = _pool->rent("x_new", n_param, 1);
    const auto tmp_mat = _pool->rent("tmp_mat", n_param, n_param);
    for (size_t k = 0; k < _k_max; k++) {
      if (g->max_element() <= _eps) break;

      tmp_mat->copy_from(a);
      tmp_mat->add(mu, identity);
      h_lm->copy_from(g);
      tmp_mat->solve(h_lm);
      if (std::sqrt(h_lm->dot(h_lm)) <= _eps_2 * (std::sqrt(x->dot(x)) + _eps_2)) break;

      x_new->copy_from(x);
      x_new->add(-1.0, h_lm);

      const double fx_new = calc_fx("x_new", n_param);

      tmp_vec->copy_from(g);
      tmp_vec->add(mu, h_lm);
      const double l0_lhlm = h_lm->dot(tmp_vec) / 2;

      const auto rho = (fx - fx_new) / l0_lhlm;
      fx = fx_new;
      if (rho > 0) {
        x->copy_from(x_new);

        make_t(n_param);
        calc_jtj_jtf(n_param);

        mu *= (std::max)(1. / 3., std::pow(1 - (2 * rho - 1), 3.0));
        nu = 2;
      } else {
        mu *= nu;
        nu *= 2;
      }
    }

    x->set_from_arg(_data, n);

    _built = true;
  }

  LM(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1, const double eps_2, const double tau,
     const size_t k_max, std::vector<double> initial)
      : NLS(foci, amps, eps_1, k_max, std::move(initial)), _eps_2(eps_2), _tau(tau) {}

 private:
  double _eps_2;
  double _tau;
};

/**
 * @brief Gain to produce multiple focal points with Gauss-Newton method.
 */
template <typename P>
class GaussNewton final : public NLS<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps_1 parameter
   * @param[in] eps_2 parameter
   * @param[in] k_max parameter
   * @param[in] initial initial phase of transducers
   */
  static std::shared_ptr<GaussNewton> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1 = 1e-6,
                                             const double eps_2 = 1e-6, const size_t k_max = 500, const std::vector<double>& initial = {}) {
    return std::make_shared<GaussNewton>(foci, amps, eps_1, eps_2, k_max, initial);
  }

  void calc(const core::GeometryPtr& geometry) override {
    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());
    const size_t n_param = n + m;

    const auto amps = _pool->rent_c("amps", m, 1);
    amps->copy_from(_amps);

    make_bhb(_foci, geometry);
    auto bhb = _pool->rent_c("bhb", n_param, n_param);

    const auto x = _pool->rent("x", n_param, 1);
    x->fill(0.0);
    x->copy_from(_initial);

    const auto t = _pool->rent_c("T", n_param, 1);
    const auto zero = _pool->rent("zero", n_param, 1);
    zero->fill(0.0);
    make_t(n_param);

    const auto tth = _pool->rent_c("tth", n_param, n_param);
    const auto bhb_tth = _pool->rent_c("bhb_tth", n_param, n_param);
    const auto a = _pool->rent("jtj", n_param, n_param);
    const auto g = _pool->rent("jtf", n_param, 1);
    calc_jtj_jtf(n_param);

    const auto h_lm = _pool->rent("h_lm", n_param, 1);
    const auto pia = _pool->rent("pis", n_param, n_param);
    auto u = this->_pool->rent("u", n_param, n_param);
    auto s = this->_pool->rent("s", n_param, n_param);
    auto vt = this->_pool->rent("vt", n_param, n_param);
    auto buf = this->_pool->rent("buf", n_param, n_param);
    const auto atmp = _pool->rent("atmp", n_param, n_param);
    for (size_t k = 0; k < _k_max; k++) {
      if (g->max_element() <= _eps) break;

      //_backend->solve_g(a, g, h_lm);
      atmp->copy_from(a);
      pia->pseudo_inverse_svd(atmp, 1e-3, u, s, vt, buf);
      h_lm->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, 1.0, pia, g, 0.0);
      if (std::sqrt(h_lm->dot(h_lm)) <= _eps_2 * (std::sqrt(x->dot(x)) + _eps_2)) break;

      x->add(-1.0, h_lm);

      make_t(n_param);
      calc_jtj_jtf(n_param);
    }

    x->set_from_arg(_data, n);

    _built = true;
  }

  GaussNewton(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1, const double eps_2, const size_t k_max,
              std::vector<double> initial)
      : NLS(foci, amps, eps_1, k_max, std::move(initial)), _eps_2(eps_2) {}

 private:
  double _eps_2;
};

/**
 * @brief Gain to produce multiple focal points with GradientDescent method.
 */
template <typename P>
class GradientDescent final : public NLS<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps parameter
   * @param[in] step parameter
   * @param[in] k_max parameter
   * @param[in] initial initial phase of transducers
   */
  static std::shared_ptr<GradientDescent> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps = 1e-6,
                                                 const double step = 0.5, const size_t k_max = 2000, const std::vector<double>& initial = {}) {
    return std::make_shared<GradientDescent>(foci, amps, eps, step, k_max, initial);
  }

  void calc(const core::GeometryPtr& geometry) override {
    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());
    const size_t n_param = n + m;

    const auto amps = _pool->rent_c("amps", m, 1);
    amps->copy_from(_amps);

    make_bhb(_foci, geometry);
    auto bhb = _pool->rent_c("bhb", n_param, n_param);

    const auto x = _pool->rent("x", n_param, 1);
    x->fill(0.0);
    x->copy_from(_initial);

    const auto t = _pool->rent_c("T", n_param, 1);
    const auto zero = _pool->rent("zero", n_param, 1);
    zero->fill(0.0);

    const auto tth = _pool->rent_c("tth", n_param, n_param);
    const auto bhb_tth = _pool->rent_c("bhb_tth", n_param, n_param);
    const auto jtf = _pool->rent("jtf", n_param, 1);
    for (size_t k = 0; k < _k_max; k++) {
      make_t(n_param);
      calc_jtf(n_param);
      if (jtf->max_element() <= _eps) break;
      x->add(-_step, jtf);
    }
    x->set_from_arg(_data, n);

    _built = true;
  }

  GradientDescent(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps, const double step, const size_t k_max,
                  std::vector<double> initial)
      : NLS(foci, amps, eps, k_max, std::move(initial)), _step(step) {}

 private:
  double _step;
};

/**
 * @brief Gain to produce multiple focal points with Acoustic Power Optimization method.
 * Refer to Keisuke Hasegawa, Hiroyuki Shinoda, and Takaaki Nara. Volumetric acoustic holography and its application to self-positioning by single
 * channel measurement.Journal of Applied Physics,127(24):244904, 2020.7
 */
template <typename P>
class APO final : public Holo<P> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps parameter
   * @param[in] lambda parameter
   * @param[in] k_max parameter
   */
  static std::shared_ptr<APO> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps = 1e-8,
                                     const double lambda = 1.0, const size_t k_max = 200) {
    return std::make_shared<APO>(foci, amps, eps, lambda, k_max);
  }

  void calc(const core::GeometryPtr& geometry) override {
    auto make_ri = [](const std::unique_ptr<P>& pool, const size_t m, const size_t n, const size_t i) {
      const auto g = pool->rent_c("g", m, n);

      const auto di = pool->rent_c("di", m, m);
      di->fill(ZERO);
      di->set(i, i, ONE);

      auto ri = pool->rent_c("ri" + std::to_string(i), n, n);
      const auto tmp = pool->rent_c("tmp_ri", n, m);
      tmp->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, di, ZERO);
      ri->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, tmp, g, ZERO);
    };

    auto calc_nabla_j = [](const std::unique_ptr<P>& pool, const size_t m, const size_t n, const double lambda, const std::string& nabla_j_name) {
      const auto tmp = pool->rent_c("cnj_tmp", n, 1);
      const auto q = pool->rent_c("q", n, 1);
      const auto p2 = pool->rent_c("p2", m, 1);
      const auto nabla_j = pool->rent_c(nabla_j_name, n, 1);
      for (size_t i = 0; i < m; i++) {
        auto ri = pool->rent_c("ri" + std::to_string(i), n, n);
        tmp->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, ri, q, ZERO);
        const auto s = p2->at(i, 0) - q->dot(tmp);
        tmp->scale(s);
        nabla_j->add(ONE, tmp);
      }

      nabla_j->add(complex(lambda, 0), q);
    };

    auto calc_j = [](const std::unique_ptr<P>& pool, const size_t m, const size_t n, const double lambda) {
      const auto q = pool->rent_c("q", n, 1);
      const auto p2 = pool->rent_c("p2", m, 1);
      const auto tmp = pool->rent_c("cj_tmp", n, 1);
      auto j = 0.0;
      for (size_t i = 0; i < m; i++) {
        auto ri = pool->rent_c("ri" + std::to_string(i), n, n);
        tmp->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, ri, q, ZERO);
        const auto s = p2->at(i, 0) - q->dot(tmp);
        j += std::norm(s);
      }

      j += std::abs(q->dot(q)) * lambda;
      return j;
    };

    auto line_search = [&calc_j](const std::unique_ptr<P>& pool, const size_t m, const size_t n, const double lambda, const size_t line_search_max) {
      auto alpha = 0.0;
      auto min = (std::numeric_limits<double>::max)();
      for (size_t i = 0; i < line_search_max; i++) {
        const auto a = static_cast<double>(i) / static_cast<double>(line_search_max);
        if (const auto v = calc_j(pool, m, n, lambda); v < min) {
          alpha = a;
          min = v;
        }
      }
      return alpha;
    };

    const auto m = (_foci.size());
    const auto n = (geometry->num_transducers());

    const auto g = _pool->rent_c("g", m, n);
    generate_transfer_matrix(_foci, geometry, g);

    const auto p = _pool->rent_c("p", m, 1);
    p->copy_from(_amps);

    const auto p2 = _pool->rent_c("p2", m, 1);
    p2->hadamard_product(p, p);

    const auto one = _pool->rent_c("one", n, 1);
    one->fill(ONE);
    const auto h = _pool->rent_c("h", n, n);
    h->create_diagonal(one);

    const auto tmp = _pool->rent_c("tmp", n, n);
    tmp->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, g, ZERO);
    tmp->add(complex(_lambda, 0.0), h);

    const auto q = _pool->rent_c("q", n, 1);
    q->mul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, ONE, g, p, ZERO);
    tmp->solve(q);

    for (size_t i = 0; i < m; i++) make_ri(_pool, m, n, i);

    const auto nabla_j = _pool->rent_c("nabla_j", n, 1);
    calc_nabla_j(_pool, m, n, _lambda, "nabla_j");

    const auto d = _pool->rent_c("d", n, 1);
    const auto nabla_j_new = _pool->rent_c("nabla_j_new", n, 1);
    const auto s = _pool->rent_c("s", n, 1);
    const auto hs = _pool->rent_c("hs", n, 1);
    for (size_t k = 0; k < _k_max; k++) {
      d->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, -ONE, h, nabla_j, ZERO);

      // FIXME
      const auto alpha = line_search(_pool, m, n, _lambda, _line_search_max);

      d->scale(complex(alpha, 0));

      if (std::sqrt(d->dot(d).real()) < _eps) break;

      q->add(ONE, d);
      calc_nabla_j(_pool, m, n, _lambda, "nabla_j_new");

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
    q->set_from_complex_drive(_data, true, max_coefficient);

    _built = true;
  }

  APO(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps, const double lambda, const size_t k_max)
      : Holo(foci, amps), _eps(eps), _lambda(lambda), _k_max(k_max) {}

 private:
  double _eps;
  double _lambda;
  size_t _k_max;
  size_t _line_search_max = 100;
};

/**
 * @brief Gain to produce multiple focal points with Greedy algorithm.
 * Refer to Shun suzuki, et al. “Radiation Pressure Field Reconstruction for Ultrasound Midair Haptics by Greedy Algorithm with Brute-Force Search,”
 * in IEEE Transactions on Haptics, doi: 10.1109/TOH.2021.3076489
 * @details This method is computed on the CPU.
 */
class Greedy final : public Holo<Backend> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] phase_div resolution of the phase to be searched
   */
  static std::shared_ptr<Greedy> create(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t phase_div = 16) {
    return std::make_shared<Greedy>(foci, amps, phase_div);
  }

  void calc(const core::GeometryPtr& geometry) override {
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

  Greedy(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t phase_div) : Holo(foci, amps) {
    this->_phases.reserve(phase_div);
    for (size_t i = 0; i < phase_div; i++)
      this->_phases.emplace_back(std::exp(complex(0., 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(phase_div))));
  }

 private:
  std::vector<complex> _phases;
};

}  // namespace autd::gain::holo
