// File: holo_gain.hpp
// Project: include
// Created Date: 06/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gain.hpp"
#include "linalg.hpp"
#include "linalg_backend.hpp"
#include "utils.hpp"

namespace autd::gain::holo {

/**
 * @brief Gain to produce multiple focal points
 */
template <typename B>
class HoloGain : public Gain {
 public:
  HoloGain() = default;
  ~HoloGain() override = default;
  HoloGain(const HoloGain& v) noexcept = default;
  HoloGain& operator=(const HoloGain& obj) = default;
  HoloGain(HoloGain&& obj) = default;
  HoloGain& operator=(HoloGain&& obj) = default;

  std::vector<Vector3>& foci() { return this->_foci; }
  std::vector<Float>& amplitudes() { return this->_amps; }

 protected:
  std::vector<Vector3> _foci;
  std::vector<Float> _amps;
  B _backend;

  template <typename M>
  void MatrixMul(const M& a, const M& b, M* c) {
    this->_backend.MatMul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, std::complex<Float>(1, 0), a, b, std::complex<Float>(0, 0), c);
  }
  template <typename M, typename V>
  void MatrixVecMul(const M& a, const V& b, V* c) {
    this->_backend.MatVecMul(TRANSPOSE::NO_TRANS, std::complex<Float>(1, 0), a, b, std::complex<Float>(0, 0), c);
  }

  template <typename V>
  void SetFromComplexDrive(std::vector<AUTDDataArray>& data, const V& drive, const bool normalize, const Float max_coefficient) {
    const size_t n = drive.size();
    size_t dev_idx = 0;
    size_t trans_idx = 0;
    for (size_t j = 0; j < n; j++) {
      const auto f_amp = normalize ? Float{1} : abs(drive(j)) / max_coefficient;
      const auto f_phase = arg(drive(j)) / (2 * PI) + Float{0.5};
      const auto phase = static_cast<uint16_t>((1 - f_phase) * Float{255});
      const uint16_t duty = static_cast<uint16_t>(ToDuty(f_amp)) << 8 & 0xFF00;
      data[dev_idx][trans_idx++] = duty | phase;
      if (trans_idx == NUM_TRANS_IN_UNIT) {
        dev_idx++;
        trans_idx = 0;
      }
    }
  }

  [[nodiscard]] std::complex<Float> Transfer(const Vector3& trans_pos, const Vector3& trans_norm, const Vector3& target_pos, const Float wave_number,
                                             const Float attenuation = 0) const {
    const auto diff = target_pos - trans_pos;
    const auto dist = diff.norm();
    const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180 / PI;
    const auto directivity = utils::directivityT4010A1(theta);

    return directivity / dist * exp(std::complex<Float>(-dist * attenuation, -wave_number * dist));
  }

  template <typename M>
  M TransferMatrix() {
    const auto m = this->_foci.size();
    const auto n = this->_geometry->num_transducers();

    M g(m, n);

    const auto wave_number = 2 * PI / _geometry->wavelength();
    const auto attenuation = _geometry->attenuation_coeff();
    for (size_t i = 0; i < m; i++) {
      const auto& tp = _foci[i];
      for (size_t j = 0; j < n; j++) {
        const auto pos = _geometry->position(j);
        const auto dir = _geometry->direction(j / NUM_TRANS_IN_UNIT);
        g(i, j) = Transfer(pos, dir, tp, wave_number, attenuation);
      }
    }
    return g;
  }
};

/**
 * @brief Gain to produce multiple focal points with SDP method.
 * Refer to Inoue, Seki, Yasutoshi Makino, and Hiroyuki Shinoda. "Active touch
 * perception produced by airborne ultrasonic haptic hologram." 2015 IEEE
 * World Haptics Conference (WHC). IEEE, 2015.
 */
template <typename B>
class HoloGainSDP final : public HoloGain<B> {
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
  static std::shared_ptr<HoloGainSDP> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, Float alpha = Float{1e-3},
                                             Float lambda = Float{0.9}, size_t repeat = 100, bool normalize = true) {
    return std::make_shared<HoloGainSDP>(foci, amps, alpha, lambda, repeat, normalize);
  }

  Result<bool, std::string> Build() override {
    if (this->built()) return Ok(false);
    const auto geo = this->geometry();

    CheckAndInit(geo, &this->_data);

    if (!this->_backend.SupportsSvd() || !this->_backend.SupportsEVD()) return Err(std::string("This backend does not support this method."));

    const auto m = this->_foci.size();
    const auto n = this->_geometry->num_transducers();

    typename B::MatrixXc p = B::MatrixXc::Zero(m, m);
    for (size_t i = 0; i < m; i++) p(i, i) = std::complex<Float>(this->_amps[i], 0);

    auto b = this->template TransferMatrix<typename B::MatrixXc>();
    typename B::MatrixXc pseudo_inv_b(n, m);
    this->_backend.PseudoInverseSvd(&b, _alpha, &pseudo_inv_b);

    typename B::MatrixXc mm = B::MatrixXc::Identity(m, m);
    this->_backend.MatMul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, std::complex<Float>(1, 0), b, pseudo_inv_b, std::complex<Float>(-1, 0), &mm);
    typename B::MatrixXc tmp = B::MatrixXc::Zero(m, m);
    this->MatrixMul(p, mm, &tmp);
    this->MatrixMul(tmp, p, &mm);
    typename B::MatrixXc x_mat = B::MatrixXc::Identity(m, m);

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> range(0, 1);
    typename B::VectorXc zero = B::VectorXc::Zero(m);
    typename B::VectorXc x = B::VectorXc::Zero(m);
    for (size_t i = 0; i < _repeat; i++) {
      auto ii = static_cast<size_t>(static_cast<Float>(m) * range(mt));

      typename B::VectorXc mmc = mm.col(ii);
      mmc(ii) = 0;

      this->MatrixVecMul(x_mat, mmc, &x);
      if (std::complex<Float> gamma = this->_backend.DotC(x, mmc); gamma.real() > 0) {
        x = -x * sqrt(_lambda / gamma.real());
        SetBcdResult(x_mat, x, ii);
      } else {
        SetBcdResult(x_mat, zero, ii);
      }
    }

    typename B::VectorXc u = this->_backend.MaxEigenVector(&x_mat);

    typename B::VectorXc ut = B::VectorXc::Zero(m);
    this->MatrixVecMul(p, u, &ut);

    typename B::VectorXc q = B::VectorXc::Zero(n);
    this->MatrixVecMul(pseudo_inv_b, ut, &q);

    const auto max_coefficient = this->_backend.MaxCoeffC(q);
    this->SetFromComplexDrive(this->_data, q, _normalize, max_coefficient);

    this->_built = true;
    return Ok(true);
  }

  HoloGainSDP(const std::vector<Vector3>& foci, const std::vector<Float>& amps, const Float alpha, const Float lambda, const size_t repeat,
              const bool normalize)
      : HoloGain<B>(), _alpha(alpha), _lambda(lambda), _repeat(repeat), _normalize(normalize) {
    this->_foci = foci;
    this->_amps = amps;
  }

 private:
  template <typename M, typename V>
  void SetBcdResult(M& mat, const V& vec, size_t idx) {
    const size_t m = vec.size();
    for (size_t i = 0; i < idx; i++) mat(idx, i) = std::conj(vec(i));
    for (auto i = idx + 1; i < m; i++) mat(idx, i) = std::conj(vec(i));
    for (size_t i = 0; i < idx; i++) mat(i, idx) = vec(i);
    for (auto i = idx + 1; i < m; i++) mat(i, idx) = vec(i);
  }

  Float _alpha;
  Float _lambda;
  size_t _repeat;
  bool _normalize;
};

/**
 * @brief Gain to produce multiple focal points with EVD method.
 * Refer to Long, Benjamin, et al. "Rendering volumetric haptic shapes in mid-air
 * using ultrasound." ACM Transactions on Graphics (TOG) 33.6 (2014): 1-10.
 */
template <typename B>
class HoloGainEVD final : public HoloGain<B> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] gamma parameter
   * @param[in] normalize parameter
   */
  static std::shared_ptr<HoloGainEVD> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, Float gamma = 1,
                                             bool normalize = true) {
    return std::make_shared<HoloGainEVD>(foci, amps, gamma, normalize);
  }

  Result<bool, std::string> Build() override {
    if (this->built()) return Ok(false);
    const auto geo = this->geometry();

    CheckAndInit(geo, &this->_data);

    const auto m = this->_foci.size();
    const auto n = this->_geometry->num_transducers();

    const auto g = this->template TransferMatrix<typename B::MatrixXc>();

    typename B::VectorXc denominator(m);
    for (size_t i = 0; i < m; i++) {
      auto tmp = std::complex<Float>(0, 0);
      for (size_t j = 0; j < n; j++) {
        tmp += g(i, j);
      }
      denominator(i) = tmp;
    }

    typename B::MatrixXc x(n, m);
    for (size_t i = 0; i < m; i++) {
      auto c = std::complex<Float>(this->_amps[i], 0) / denominator(i);
      for (size_t j = 0; j < n; j++) {
        x(j, i) = c * std::conj(g(i, j));
      }
    }
    typename B::MatrixXc r = B::MatrixXc::Zero(m, m);
    this->MatrixMul(g, x, &r);
    typename B::VectorXc max_ev = this->_backend.MaxEigenVector(&r);

    typename B::MatrixXc sigma = B::MatrixXc::Zero(n, n);
    for (size_t j = 0; j < n; j++) {
      Float tmp = 0;
      for (size_t i = 0; i < m; i++) {
        tmp += abs(g(i, j)) * this->_amps[i];
      }
      sigma(j, j) = std::complex<Float>(pow(sqrt(tmp / static_cast<Float>(m)), _gamma), 0.0);
    }

    typename B::MatrixXc gr = this->_backend.ConcatRow(g, sigma);

    typename B::VectorXc f = B::VectorXc::Zero(m + n);
    for (size_t i = 0; i < m; i++) f(i) = this->_amps[i] * max_ev(i) / abs(max_ev(i));

    typename B::MatrixXc gtg = B::MatrixXc::Zero(n, n);
    this->_backend.MatMul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, std::complex<Float>(1, 0), gr, gr, std::complex<Float>(0, 0), &gtg);

    typename B::VectorXc gtf = B::VectorXc::Zero(n);
    this->_backend.MatVecMul(TRANSPOSE::CONJ_TRANS, std::complex<Float>(1, 0), gr, f, std::complex<Float>(0, 0), &gtf);

    this->_backend.SolveCh(&gtg, &gtf);

    const auto max_coefficient = this->_backend.MaxCoeffC(gtf);
    this->SetFromComplexDrive(this->_data, gtf, _normalize, max_coefficient);

    this->_built = true;
    return Ok(true);
  }

  HoloGainEVD(const std::vector<Vector3>& foci, const std::vector<Float>& amps, const Float gamma, const bool normalize)
      : HoloGain<B>(), _gamma(gamma), _normalize(normalize) {
    this->_foci = foci;
    this->_amps = amps;
  }

 private:
  Float _gamma;
  bool _normalize;
};

/**
 * @brief Gain to produce multiple focal points with naive method.
 */
template <typename B>
class HoloGainNaive final : public HoloGain<B> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   */
  static std::shared_ptr<HoloGainNaive> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps) {
    return std::make_shared<HoloGainNaive>(foci, amps);
  }

  Result<bool, std::string> Build() override {
    if (this->built()) return Ok(false);
    const auto geo = this->geometry();

    CheckAndInit(geo, &this->_data);

    const auto m = this->_foci.size();
    const auto n = this->_geometry->num_transducers();

    const auto g = this->template TransferMatrix<typename B::MatrixXc>();
    typename B::VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = std::complex<Float>(this->_amps[i], 0);

    typename B::VectorXc q = B::VectorXc::Zero(n);
    this->_backend.MatVecMul(TRANSPOSE::CONJ_TRANS, std::complex<Float>(1, 0), g, p, std::complex<Float>(0, 0), &q);

    this->SetFromComplexDrive(this->_data, q, true, 1.0);

    this->_built = true;
    return Ok(true);
  }

  HoloGainNaive(const std::vector<Vector3>& foci, const std::vector<Float>& amps) : HoloGain<B>() {
    this->_foci = foci;
    this->_amps = amps;
  }
};

/**
 * @brief Gain to produce multiple focal points with GS method.
 * Refer to Asier Marzo and Bruce W Drinkwater. Holographic acoustic
 * tweezers.Proceedings of theNational Academy of Sciences, 116(1):84–89, 2019.
 */
template <typename B>
class HoloGainGS final : public HoloGain<B> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<HoloGainGS> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, size_t repeat = 100) {
    return std::make_shared<HoloGainGS>(foci, amps, repeat);
  }

  Result<bool, std::string> Build() override {
    if (this->built()) return Ok(false);
    const auto geo = this->geometry();

    CheckAndInit(geo, &this->_data);

    const auto m = this->_foci.size();
    const auto n = this->_geometry->num_transducers();

    const auto g = this->template TransferMatrix<typename B::MatrixXc>();

    typename B::VectorXc q0 = B::VectorXc::Ones(n);

    typename B::VectorXc q(n);
    this->_backend.VecCpyC(q0, &q);

    typename B::VectorXc gamma = B::VectorXc::Zero(m);
    typename B::VectorXc p(m);
    typename B::VectorXc xi = B::VectorXc::Zero(n);
    for (size_t k = 0; k < _repeat; k++) {
      this->MatrixVecMul(g, q, &gamma);
      for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * this->_amps[i];
      this->_backend.MatVecMul(TRANSPOSE::CONJ_TRANS, std::complex<Float>(1, 0), g, p, std::complex<Float>(0, 0), &xi);
      for (size_t j = 0; j < n; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
    }

    this->SetFromComplexDrive(this->_data, q, true, 1.0);

    this->_built = true;
    return Ok(true);
  }

  HoloGainGS(const std::vector<Vector3>& foci, const std::vector<Float>& amps, const size_t repeat) : HoloGain<B>(), _repeat(repeat) {
    this->_foci = foci;
    this->_amps = amps;
  }

 private:
  size_t _repeat;
};

/**
 * @brief Gain to produce multiple focal points with GS-PAT method (not yet been implemented with GPU).
 * Refer to Diego Martinez Plasencia et al. "Gs-pat: high-speed multi-point
 * sound-fields for phased arrays of transducers," ACMTrans-actions on
 * Graphics (TOG), 39(4):138–1, 2020.
 */
template <typename B>
class HoloGainGSPAT final : public HoloGain<B> {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<HoloGainGSPAT> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, size_t repeat = 100) {
    return std::make_shared<HoloGainGSPAT>(foci, amps, repeat);
  }

  Result<bool, std::string> Build() override {
    if (this->built()) return Ok(false);
    const auto geo = this->geometry();

    CheckAndInit(geo, &this->_data);

    const auto m = this->_foci.size();
    const auto n = this->_geometry->num_transducers();

    const auto g = this->template TransferMatrix<typename B::MatrixXc>();

    typename B::VectorXc denominator(m);
    for (size_t i = 0; i < m; i++) {
      auto tmp = std::complex<Float>(0, 0);
      for (size_t j = 0; j < n; j++) tmp += abs(g(i, j));
      denominator(i) = tmp;
    }

    typename B::MatrixXc b(n, m);
    for (size_t i = 0; i < m; i++) {
      auto d = denominator(i) * denominator(i);
      for (size_t j = 0; j < n; j++) {
        b(j, i) = std::conj(g(i, j)) / d;
      }
    }

    typename B::MatrixXc r = B::MatrixXc::Zero(m, m);
    this->MatrixMul(g, b, &r);

    typename B::VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = std::complex<Float>(this->_amps[i], 0);

    typename B::VectorXc gamma = B::VectorXc::Zero(m);
    this->MatrixVecMul(r, p, &gamma);
    for (size_t k = 0; k < _repeat; k++) {
      for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * this->_amps[i];
      this->MatrixVecMul(r, p, &gamma);
    }

    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * this->_amps[i] * this->_amps[i];

    typename B::VectorXc q = B::VectorXc::Zero(n);
    this->MatrixVecMul(b, p, &q);

    this->SetFromComplexDrive(this->_data, q, true, 1.0);

    this->_built = true;
    return Ok(true);
  }

  HoloGainGSPAT(const std::vector<Vector3>& foci, const std::vector<Float>& amps, const size_t repeat) : HoloGain<B>(), _repeat(repeat) {
    this->_foci = foci;
    this->_amps = amps;
  }

 private:
  size_t _repeat;
};

/**
 * @brief Gain to produce multiple focal points with GS-PAT method.
 * Refer to K.Levenberg, “A method for the solution of certain non-linear problems in
 * least squares,” Quarterly of applied mathematics, vol.2, no.2, pp.164–168, 1944.
 * D.W.Marquardt, “An algorithm for least-squares estimation of non-linear parameters,” Journal of the society for Industrial and
 * AppliedMathematics, vol.11, no.2, pp.431–441, 1963.
 * K.Madsen, H.Nielsen, and O.Tingleff, “Methods for non-linear least squares problems (2nd ed.),” 2004.
 */
template <typename B>
class HoloGainLM final : public HoloGain<B> {
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
  static std::shared_ptr<HoloGainLM> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, Float eps_1 = Float{1e-8},
                                            Float eps_2 = Float{1e-8}, Float tau = Float{1e-3}, size_t k_max = 5,
                                            const std::vector<Float>& initial = {}) {
    return std::make_shared<HoloGainLM>(foci, amps, eps_1, eps_2, tau, k_max, initial);
  }

  Result<bool, std::string> Build() override {
    if (this->built()) return Ok(false);
    const auto geo = this->geometry();

    CheckAndInit(geo, &this->_data);

    if (!this->_backend.SupportsSolve()) return Err(std::string("This backend does not support this method."));

    const auto m = this->_foci.size();
    const auto n = this->_geometry->num_transducers();
    const auto n_param = n + m;

    typename B::MatrixXc bhb = B::MatrixXc::Zero(n_param, n_param);
    MakeBhB<typename B::MatrixXc>(&bhb);

    typename B::VectorX x = B::VectorX::Zero(n_param);
    for (size_t i = 0; i < _initial.size(); i++) x[i] = _initial[i];

    auto nu = Float{2};

    typename B::MatrixXc tth = B::MatrixXc::Zero(n_param, n_param);
    CalcTTh<typename B::MatrixXc, typename B::VectorX>(x, &tth);

    typename B::MatrixXc bhb_tth(n_param, n_param);
    this->_backend.HadamardProduct(bhb, tth, &bhb_tth);

    typename B::MatrixX a(n_param, n_param);
    this->_backend.Real(bhb_tth, &a);

    typename B::VectorX g(n_param);
    for (size_t i = 0; i < n_param; i++) {
      Float tmp = 0;
      for (size_t k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
      g(i) = tmp;
    }

    Float a_max = 0;
    for (size_t i = 0; i < n_param; i++) a_max = std::max(a_max, a(i, i));

    auto mu = _tau * a_max;

    auto is_found = this->_backend.MaxCoeff(g) <= _eps_1;

    typename B::VectorXc t(n_param);
    for (size_t i = 0; i < n_param; i++) t(i) = exp(std::complex<Float>(0, x(i)));

    typename B::VectorXc tmp_vec_c = B::VectorXc::Zero(n_param);
    this->MatrixVecMul(bhb, t, &tmp_vec_c);
    Float fx = this->_backend.DotC(t, tmp_vec_c).real();

    typename B::MatrixX identity = B::MatrixX::Identity(n_param, n_param);
    typename B::VectorX tmp_vec(n_param);
    typename B::VectorX h_lm(n_param);
    typename B::VectorX x_new(n_param);
    typename B::MatrixX tmp_mat(n_param, n_param);
    for (size_t k = 0; k < _k_max; k++) {
      if (is_found) break;

      this->_backend.MatCpy(a, &tmp_mat);
      this->_backend.MatAdd(mu, identity, Float{1.0}, &tmp_mat);
      this->_backend.Solveg(&tmp_mat, &g, &h_lm);
      if (h_lm.norm() <= _eps_2 * (x.norm() + _eps_2)) {
        is_found = true;
      } else {
        this->_backend.VecCpy(x, &x_new);
        this->_backend.VecAdd(Float{-1.0}, h_lm, Float{1.0}, &x_new);
        for (size_t i = 0; i < n_param; i++) t(i) = exp(std::complex<Float>(0, x_new(i)));

        this->MatrixVecMul(bhb, t, &tmp_vec_c);
        const Float fx_new = this->_backend.DotC(t, tmp_vec_c).real();

        this->_backend.VecCpy(g, &tmp_vec);
        this->_backend.VecAdd(mu, h_lm, Float{1.0}, &tmp_vec);
        const Float l0_lhlm = this->_backend.Dot(h_lm, tmp_vec) / 2;

        const auto rho = (fx - fx_new) / l0_lhlm;
        fx = fx_new;
        if (rho > 0) {
          this->_backend.VecCpy(x_new, &x);
          CalcTTh<typename B::MatrixXc, typename B::VectorX>(x, &tth);
          this->_backend.HadamardProduct(bhb, tth, &bhb_tth);
          this->_backend.Real(bhb_tth, &a);
          for (size_t i = 0; i < n_param; i++) {
            Float tmp = 0;
            for (size_t j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
            g(i) = tmp;
          }
          is_found = this->_backend.MaxCoeff(g) <= _eps_1;
          mu *= std::max(Float{1. / 3.}, std::pow(1 - (2 * rho - 1), Float{3}));
          nu = 2;
        } else {
          mu *= nu;
          nu *= 2;
        }
      }
    }

    size_t dev_idx = 0;
    size_t trans_idx = 0;
    for (size_t j = 0; j < n; j++) {
      const uint16_t duty = 0xFF00;
      const auto f_phase = fmod(x(j), 2 * PI) / (2 * PI);
      const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
      this->_data[dev_idx][trans_idx++] = duty | phase;
      if (trans_idx == NUM_TRANS_IN_UNIT) {
        dev_idx++;
        trans_idx = 0;
      }
    }

    this->_built = true;
    return Ok(true);
  }

  HoloGainLM(const std::vector<Vector3>& foci, const std::vector<Float>& amps, const Float eps_1, const Float eps_2, const Float tau,
             const size_t k_max, std::vector<Float> initial)
      : HoloGain<B>(), _eps_1(eps_1), _eps_2(eps_2), _tau(tau), _k_max(k_max), _initial(std::move(initial)) {
    this->_foci = foci;
    this->_amps = amps;
  }

 private:
  template <typename M>
  void MakeBhB(M* bhb) {
    const auto m = this->_foci.size();

    M p = M::Zero(m, m);
    for (size_t i = 0; i < m; i++) p(i, i) = -this->_amps[i];

    const auto g = this->template TransferMatrix<M>();

    M b = this->_backend.ConcatCol(g, p);
    this->_backend.MatMul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, std::complex<Float>(1, 0), b, b, std::complex<Float>(0, 0), bhb);
  }

  template <typename M, typename V>
  void CalcTTh(const V& x, M* tth) {
    const size_t len = x.size();
    M t(len, 1);
    for (size_t i = 0; i < len; i++) t(i, 0) = exp(std::complex<Float>(0, -x(i)));
    this->_backend.MatMul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, std::complex<Float>(1, 0), t, t, std::complex<Float>(0, 0), tth);
  }

  Float _eps_1;
  Float _eps_2;
  Float _tau;
  size_t _k_max;
  std::vector<Float> _initial;
};

}  // namespace autd::gain::holo
