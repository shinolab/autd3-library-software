// File: holo_gain.hpp
// Project: include
// Created Date: 06/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 01/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gain.hpp"
#include "linalg.hpp"
#include "linalg_backend.hpp"
#include "utils.hpp"

namespace autd::gain::holo {
/**
 * @brief Optimization method for generating multiple foci.
 */
enum class OPT_METHOD {
  //! Inoue, Seki, Yasutoshi Makino, and Hiroyuki Shinoda. "Active touch
  //! perception produced by airborne ultrasonic haptic hologram." 2015 IEEE
  //! World
  //! Haptics Conference (WHC). IEEE, 2015.
  SDP = 0,
  //! Long, Benjamin, et al. "Rendering volumetric haptic shapes in mid-air
  //! using ultrasound." ACM Transactions on Graphics (TOG) 33.6 (2014): 1-10.
  EVD = 1,
  //! Asier Marzo and Bruce W Drinkwater. Holographic acoustic
  //! tweezers.Proceedings of theNational Academy of Sciences, 116(1):84–89,
  //! 2019.
  GS = 2,
  //! Diego Martinez Plasencia et al. "Gs-pat: high-speed multi-point
  //! sound-fields for phased arrays of transducers," ACMTrans-actions on
  //! Graphics
  //! (TOG), 39(4):138–1, 2020.
  //! Not yet been implemented with GPU.
  GSPAT = 3,
  //! Naive linear synthesis method.
  NAIVE = 4,
  //! K.Levenberg, “A method for the solution of certain non-linear problems in
  //! least squares,” Quarterly of applied mathematics, vol.2, no.2,
  //! pp.164–168, 1944.
  //! D.W.Marquardt, “An algorithm for least-squares estimation of non-linear
  //! parameters,” Journal of the society for Industrial and
  //! AppliedMathematics, vol.11, no.2, pp.431–441, 1963.
  //! K.Madsen, H.Nielsen, and O.Tingleff, “Methods for non-linear least squares
  //! problems (2nd ed.),” 2004.
  LM = 5
};

struct SDPParams {
  Float regularization;
  int32_t repeat;
  Float lambda;
  bool normalize_amp;
};

struct EVDParams {
  Float regularization;
  bool normalize_amp;
};

struct NLSParams {
  Float eps_1;
  Float eps_2;
  int32_t k_max;
  Float tau;
  Float* initial;
};

/**
 * @brief Gain to produce multiple focal points
 */
template <typename B>
class HoloGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] method optimization method. see also @ref OPT_METHOD
   * @param[in] params pointer to optimization parameters
   */
  static std::shared_ptr<HoloGain> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, OPT_METHOD method = OPT_METHOD::SDP,
                                          void* params = nullptr) {
    std::shared_ptr<HoloGain> ptr = std::make_shared<HoloGain>(foci, amps, method, params);
    return ptr;
  }

  void Build() override {
    if (this->built()) return;
    const auto geo = this->geometry();

    CheckAndInit(geo, &this->_data);

    switch (this->_method) {
      case OPT_METHOD::SDP:
        SDP();
        break;
      case OPT_METHOD::EVD:
        EVD();
        break;
      case OPT_METHOD::NAIVE:
        NAIVE();
        break;
      case OPT_METHOD::GS:
        GS();
        break;
      case OPT_METHOD::GSPAT:
        GSPAT();
        break;
      case OPT_METHOD::LM:
        LM();
        break;
    }
    this->_built = true;
  }

  HoloGain(std::vector<Vector3> foci, std::vector<Float> amps, const OPT_METHOD method = OPT_METHOD::SDP, void* params = nullptr)
      : Gain(), _foci(std::move(foci)), _amps(std::move(amps)), _method(method), _params(params) {}
  ~HoloGain() override = default;
  HoloGain(const HoloGain& v) noexcept = default;
  HoloGain& operator=(const HoloGain& obj) = default;
  HoloGain(HoloGain&& obj) = default;
  HoloGain& operator=(HoloGain&& obj) = default;

  std::vector<Vector3>& foci() { return this->_foci; }
  std::vector<Float>& amplitudes() { return this->_amps; }
  void Rebuild() {
    this->_built = false;
    this->Build();
  }

 protected:
  std::vector<Vector3> _foci;
  std::vector<Float> _amps;
  OPT_METHOD _method = OPT_METHOD::SDP;
  void* _params = nullptr;
  B _backend;

 private:
  template <typename M>
  void matrixMul(const M& a, const M& b, M* c) {
    _backend.matMul(TRANSPOSE::NoTrans, TRANSPOSE::NoTrans, std::complex<Float>(1, 0), a, b, std::complex<Float>(0, 0), c);
  }
  template <typename M, typename V>
  void matrixVecMul(const M& a, const V& b, V* c) {
    _backend.matVecMul(TRANSPOSE::NoTrans, std::complex<Float>(1, 0), a, b, std::complex<Float>(0, 0), c);
  }
  template <typename M, typename V>
  void setBCDResult(M& mat, const V& vec, size_t idx) {
    const size_t m = vec.size();
    for (size_t i = 0; i < idx; i++) mat(idx, i) = std::conj(vec(i));
    for (auto i = idx + 1; i < m; i++) mat(idx, i) = std::conj(vec(i));
    for (size_t i = 0; i < idx; i++) mat(i, idx) = vec(i);
    for (auto i = idx + 1; i < m; i++) mat(i, idx) = vec(i);
  }

  template <typename V>
  void SetFromComplexDrive(std::vector<AUTDDataArray>& data, const V& drive, const bool normalize, const Float max_coeff) {
    const size_t n = drive.size();
    size_t dev_idx = 0;
    size_t trans_idx = 0;
    for (size_t j = 0; j < n; j++) {
      const auto f_amp = normalize ? Float{1} : abs(drive(j)) / max_coeff;
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

  std::complex<Float> transfer(const Vector3& trans_pos, const Vector3& trans_norm, const Vector3& target_pos, const Float wave_number,
                               const Float attenuation = 0) const {
    const auto diff = target_pos - trans_pos;
    const auto dist = diff.norm();
    const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180 / PI;
    const auto directivity = utils::directivityT4010A1(theta);

    return directivity / dist * exp(std::complex<Float>(-dist * attenuation, -wave_number * dist));
  }

  template <typename M>
  M transferMatrix() {
    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    M g(m, n);

    const auto wave_number = 2 * PI / _geometry->wavelength();
    const auto attenuation = _geometry->attenuation_coeff();
    for (size_t i = 0; i < m; i++) {
      const auto& tp = _foci[i];
      for (size_t j = 0; j < n; j++) {
        const auto pos = _geometry->position(j);
        const auto dir = _geometry->direction(j / NUM_TRANS_IN_UNIT);
        g(i, j) = transfer(pos, dir, tp, wave_number, attenuation);
      }
    }

    return g;
  }

  template <typename M>
  void makeBhB(M* bhb) {
    const auto m = _foci.size();

    M p = M::Zero(m, m);
    for (size_t i = 0; i < m; i++) p(i, i) = -_amps[i];

    const auto g = transferMatrix<M>();

    M b = _backend.concatCol(g, p);
    _backend.matMul(TRANSPOSE::ConjTrans, TRANSPOSE::NoTrans, std::complex<Float>(1, 0), b, b, std::complex<Float>(0, 0), bhb);
  }

  template <typename M, typename V>
  void calcTTh(const V& x, M* tth) {
    const size_t len = x.size();
    M t(len, 1);
    for (size_t i = 0; i < len; i++) t(i, 0) = exp(std::complex<Float>(0, -x(i)));
    _backend.matMul(TRANSPOSE::NoTrans, TRANSPOSE::ConjTrans, std::complex<Float>(1, 0), t, t, std::complex<Float>(0, 0), tth);
  }

  void SDP() {
    if (!_backend.supports_SVD() || !_backend.supports_EVD()) throw std::runtime_error("This backend does not support this method.");

    auto alpha = Float{1e-3};
    auto lambda = Float{0.9};
    auto repeat = 100;
    auto normalize = true;

    if (_params != nullptr) {
      auto* const sdp_params = static_cast<SDPParams*>(_params);
      alpha = sdp_params->regularization < 0 ? alpha : sdp_params->regularization;
      repeat = sdp_params->repeat < 0 ? repeat : sdp_params->repeat;
      lambda = sdp_params->lambda < 0 ? lambda : sdp_params->lambda;
      normalize = sdp_params->normalize_amp;
    }

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    typename B::MatrixXc P = B::MatrixXc::Zero(m, m);
    for (size_t i = 0; i < m; i++) P(i, i) = std::complex<Float>(_amps[i], 0);

    typename B::MatrixXc b = transferMatrix<typename B::MatrixXc>();
    typename B::MatrixXc pseudoInvB(n, m);
    _backend.pseudoInverseSVD(&b, alpha, &pseudoInvB);

    typename B::MatrixXc MM = B::MatrixXc::Identity(m, m);
    _backend.matMul(TRANSPOSE::NoTrans, TRANSPOSE::NoTrans, std::complex<Float>(1, 0), b, pseudoInvB, std::complex<Float>(-1, 0), &MM);
    typename B::MatrixXc tmp(m, m);
    matrixMul(P, MM, &tmp);
    matrixMul(tmp, P, &MM);
    typename B::MatrixXc X = B::MatrixXc::Identity(m, m);

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> range(0, 1);
    typename B::VectorXc zero = B::VectorXc::Zero(m);
    for (auto i = 0; i < repeat; i++) {
      auto ii = static_cast<size_t>(m * static_cast<double>(range(mt)));

      typename B::VectorXc mmc = MM.col(ii);
      mmc(ii) = 0;

      typename B::VectorXc x(m);
      matrixVecMul(X, mmc, &x);
      std::complex<Float> gamma = _backend.cdot(x, mmc);
      if (gamma.real() > 0) {
        x = -x * sqrt(lambda / gamma.real());
        setBCDResult(X, x, ii);
      } else {
        setBCDResult(X, zero, ii);
      }
    }

    typename B::VectorXc u = _backend.maxEigenVector(&X);

    typename B::VectorXc ut(m);
    matrixVecMul(P, u, &ut);

    typename B::VectorXc q(n);
    matrixVecMul(pseudoInvB, ut, &q);

    const auto max_coeff = _backend.cmaxCoeff(q);
    SetFromComplexDrive(_data, q, normalize, max_coeff);
  }

  void EVD() {
    if (!_backend.supports_EVD() || !_backend.supports_solve()) throw std::runtime_error("This backend does not support this method.");

    Float gamma = 1;
    auto normalize = true;

    if (_params != nullptr) {
      auto* const evd_params = static_cast<EVDParams*>(_params);
      gamma = evd_params->regularization < 0 ? gamma : evd_params->regularization;
      normalize = evd_params->normalize_amp;
    }

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    const typename B::MatrixXc g = transferMatrix<typename B::MatrixXc>();

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
      auto c = std::complex<Float>(_amps[i], 0) / denominator(i);
      for (size_t j = 0; j < n; j++) {
        x(j, i) = c * std::conj(g(i, j));
      }
    }
    typename B::MatrixXc r(m, m);
    matrixMul(g, x, &r);
    typename B::VectorXc max_ev = _backend.maxEigenVector(&r);

    typename B::MatrixXc sigma = B::MatrixXc::Zero(n, n);
    for (size_t j = 0; j < n; j++) {
      Float tmp = 0;
      for (size_t i = 0; i < m; i++) {
        tmp += abs(g(i, j)) * _amps[i];
      }
      sigma(j, j) = std::complex<Float>(pow(sqrt(tmp / static_cast<Float>(m)), gamma), 0.0);
    }

    typename B::MatrixXc gr = _backend.concatRow(g, sigma);

    typename B::VectorXc f = B::VectorXc::Zero(m + n);
    for (size_t i = 0; i < m; i++) f(i) = _amps[i] * max_ev(i) / abs(max_ev(i));

    typename B::MatrixXc gtg(n, n);
    _backend.matMul(TRANSPOSE::ConjTrans, TRANSPOSE::NoTrans, std::complex<Float>(1, 0), gr, gr, std::complex<Float>(0, 0), &gtg);

    typename B::VectorXc gtf(n);
    _backend.matVecMul(TRANSPOSE::ConjTrans, std::complex<Float>(1, 0), gr, f, std::complex<Float>(0, 0), &gtf);

    _backend.csolveh(&gtg, &gtf);

    const auto max_coeff = _backend.cmaxCoeff(gtf);
    SetFromComplexDrive(_data, gtf, normalize, max_coeff);
  }

  void NAIVE() {
    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    const typename B::MatrixXc g = transferMatrix<typename B::MatrixXc>();
    typename B::VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = std::complex<Float>(_amps[i], 0);

    typename B::VectorXc q(n);
    _backend.matVecMul(TRANSPOSE::ConjTrans, std::complex<Float>(1, 0), g, p, std::complex<Float>(0, 0), &q);

    SetFromComplexDrive(_data, q, true, 1.0);
  }

  void GS() {
    const int32_t repeat = _params == nullptr ? 100 : *static_cast<uint32_t*>(_params);

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    const typename B::MatrixXc g = transferMatrix<typename B::MatrixXc>();

    typename B::VectorXc q0 = B::VectorXc::Ones(n);

    typename B::VectorXc q = q0;
    typename B::VectorXc gamma(m);
    typename B::VectorXc p(m);
    typename B::VectorXc xi(n);
    for (auto k = 0; k < repeat; k++) {
      matrixVecMul(g, q, &gamma);
      for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * _amps[i];
      _backend.matVecMul(TRANSPOSE::ConjTrans, std::complex<Float>(1, 0), g, p, std::complex<Float>(0, 0), &xi);
      for (size_t j = 0; j < n; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
    }

    SetFromComplexDrive(_data, q, true, 1.0);
  }

  void GSPAT() {
    const int32_t repeat = _params == nullptr ? 100 : *static_cast<uint32_t*>(_params);

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    const typename B::MatrixXc g = transferMatrix<typename B::MatrixXc>();

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

    typename B::MatrixXc r(m, m);
    matrixMul(g, b, &r);

    typename B::VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = std::complex<Float>(_amps[i], 0);

    typename B::VectorXc gamma(m);
    matrixVecMul(r, p, &gamma);
    for (auto k = 0; k < repeat; k++) {
      for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * _amps[i];
      matrixVecMul(r, p, &gamma);
    }

    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * _amps[i] * _amps[i];

    typename B::VectorXc q(n);
    matrixVecMul(b, p, &q);

    SetFromComplexDrive(_data, q, true, 1.0);
  }

  void LM() {
    if (!_backend.supports_solve()) throw std::runtime_error("This backend does not support this method.");

    auto eps_1 = Float{1e-8};
    auto eps_2 = Float{1e-8};
    auto tau = Float{1e-3};
    auto k_max = 5;
    Float* initial = nullptr;

    if (_params != nullptr) {
      auto* const nlp_params = static_cast<NLSParams*>(_params);
      eps_1 = nlp_params->eps_1 < 0 ? eps_1 : nlp_params->eps_1;
      eps_2 = nlp_params->eps_2 < 0 ? eps_2 : nlp_params->eps_2;
      k_max = nlp_params->k_max < 0 ? k_max : nlp_params->k_max;
      tau = nlp_params->tau < 0 ? tau : nlp_params->tau;
      initial = nlp_params->initial;
    }

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();
    const auto n_param = n + m;

    typename B::MatrixXc bhb(n_param, n_param);
    makeBhB<typename B::MatrixXc>(&bhb);

    typename B::VectorX x(n_param);
    if (initial == nullptr) {
      std::memset(x.data(), 0, x.size() * sizeof(Float));
    } else {
      std::memcpy(x.data(), initial, x.size() * sizeof(Float));
    }

    auto nu = Float{2};

    typename B::MatrixXc tth(n_param, n_param);
    calcTTh<typename B::MatrixXc, typename B::VectorX>(x, &tth);

    typename B::MatrixXc bhb_tth(n_param, n_param);
    _backend.hadamardProduct(bhb, tth, &bhb_tth);

    typename B::MatrixX a(n_param, n_param);
    _backend.real(bhb_tth, &a);

    typename B::VectorX g(n_param);
    for (size_t i = 0; i < n_param; i++) {
      Float tmp = 0;
      for (size_t k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
      g(i) = tmp;
    }

    Float a_max = 0;
    for (size_t i = 0; i < n_param; i++) a_max = std::max(a_max, a(i, i));

    auto mu = tau * a_max;

    auto is_found = _backend.maxCoeff(g) <= eps_1;

    typename B::VectorXc t(n_param);
    for (size_t i = 0; i < n_param; i++) t(i) = exp(std::complex<Float>(0, x(i)));

    typename B::VectorXc tmp_vec_c(n_param);
    matrixVecMul(bhb, t, &tmp_vec_c);
    Float fx = _backend.cdot(t, tmp_vec_c).real();

    typename B::MatrixX identity = B::MatrixX::Identity(n_param, n_param);
    typename B::VectorX tmp_vec(n_param);
    typename B::VectorX h_lm(n_param);
    typename B::VectorX x_new(n_param);
    typename B::MatrixX tmp_mat(n_param, n_param);
    for (auto k = 0; k < k_max; k++) {
      if (is_found) break;

      _backend.matCpy(a, &tmp_mat);
      _backend.matAdd(mu, identity, Float{1.0}, &tmp_mat);
      _backend.solveg(&tmp_mat, &g, &h_lm);
      if (h_lm.norm() <= eps_2 * (x.norm() + eps_2)) {
        is_found = true;
      } else {
        _backend.vecCpy(x, &x_new);
        _backend.vecAdd(Float{-1.0}, h_lm, Float{1.0}, &x_new);
        for (size_t i = 0; i < n_param; i++) t(i) = exp(std::complex<Float>(0, x_new(i)));

        matrixVecMul(bhb, t, &tmp_vec_c);
        const Float fx_new = _backend.cdot(t, tmp_vec_c).real();

        _backend.vecCpy(g, &tmp_vec);
        _backend.vecAdd(mu, h_lm, Float{1.0}, &tmp_vec);
        const Float l0_lhlm = _backend.dot(h_lm, tmp_vec) / 2;

        const auto rho = (fx - fx_new) / l0_lhlm;
        fx = fx_new;
        if (rho > 0) {
          _backend.vecCpy(x_new, &x);
          calcTTh<typename B::MatrixXc, typename B::VectorX>(x, &tth);
          _backend.hadamardProduct(bhb, tth, &bhb_tth);
          _backend.real(bhb_tth, &a);
          for (size_t i = 0; i < n_param; i++) {
            Float tmp = 0;
            for (size_t j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
            g(i) = tmp;
          }
          is_found = _backend.maxCoeff(g) <= eps_1;
          mu *= std::max(Float{1. / 3.}, pow(1 - (2 * rho - 1), Float{3.}));
          nu = 2.0;
        } else {
          mu *= nu;
          nu *= 2;
        }
      }
    }

    const uint16_t duty = 0xFF00;
    size_t dev_idx = 0;
    size_t trans_idx = 0;
    for (size_t j = 0; j < n; j++) {
      const auto f_phase = fmod(x(j), 2 * PI) / (2 * PI);
      const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
      _data[dev_idx][trans_idx++] = duty | phase;
      if (trans_idx == NUM_TRANS_IN_UNIT) {
        dev_idx++;
        trans_idx = 0;
      }
    }
  }
};

#ifndef DISABLE_EIGEN
using HoloGainE = HoloGain<Eigen3Backend>;
#endif
#ifdef ENABLE_BLAS
using HoloGainB = HoloGain<BLASBackend>;
#endif

}  // namespace autd::gain::holo
