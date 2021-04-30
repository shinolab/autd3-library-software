// File: holo_gain.hpp
// Project: include
// Created Date: 06/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2021
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
                                          const void* params = nullptr) {
    std::shared_ptr<HoloGain> ptr = std::make_shared<HoloGain>(foci, amps, method, params);
    return ptr;
  }

  Result<bool, std::string> Build() override {
    if (this->built()) return Ok(false);
    const auto geo = this->geometry();

    CheckAndInit(geo, &this->_data);

    this->_built = true;
    switch (this->_method) {
      case OPT_METHOD::SDP:
        return SDP();
      case OPT_METHOD::EVD:
        return EVD();
      case OPT_METHOD::NAIVE:
        return Naive();
      case OPT_METHOD::GS:
        return GS();
      case OPT_METHOD::GSPAT:
        return GSPAT();
      case OPT_METHOD::LM:
        return LM();
    }
    return Ok(false);
  }

  HoloGain(std::vector<Vector3> foci, std::vector<Float> amps, const OPT_METHOD method = OPT_METHOD::SDP, const void* params = nullptr)
      : Gain(), _method(method), _params(params) {
    if constexpr (sizeof(Float) == sizeof(HoloFloat)) {
      _foci = std::move(foci);
      _amps = std::move(amps);
    } else {
      for (auto&& p : foci) _foci.emplace_back(HoloVector3(p.x(), p.y(), p.z()));
      for (auto&& a : amps) _amps.emplace_back(static_cast<HoloFloat>(a));
    }
  }
  ~HoloGain() override = default;
  HoloGain(const HoloGain& v) noexcept = default;
  HoloGain& operator=(const HoloGain& obj) = default;
  HoloGain(HoloGain&& obj) = default;
  HoloGain& operator=(HoloGain&& obj) = default;

  std::vector<HoloVector3>& foci() { return this->_foci; }
  std::vector<HoloFloat>& amplitudes() { return this->_amps; }
  void Rebuild() {
    this->_built = false;
    this->Build();
  }

 protected:
  std::vector<HoloVector3> _foci;
  std::vector<HoloFloat> _amps;
  OPT_METHOD _method = OPT_METHOD::SDP;
  const void* _params = nullptr;
  B _backend;

 private:
  template <typename M>
  void MatrixMul(const M& a, const M& b, M* c) {
    _backend.MatMul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, std::complex<HoloFloat>(1, 0), a, b, std::complex<HoloFloat>(0, 0), c);
  }
  template <typename M, typename V>
  void MatrixVecMul(const M& a, const V& b, V* c) {
    _backend.MatVecMul(TRANSPOSE::NO_TRANS, std::complex<HoloFloat>(1, 0), a, b, std::complex<HoloFloat>(0, 0), c);
  }
  template <typename M, typename V>
  void SetBcdResult(M& mat, const V& vec, size_t idx) {
    const size_t m = vec.size();
    for (size_t i = 0; i < idx; i++) mat(idx, i) = std::conj(vec(i));
    for (auto i = idx + 1; i < m; i++) mat(idx, i) = std::conj(vec(i));
    for (size_t i = 0; i < idx; i++) mat(i, idx) = vec(i);
    for (auto i = idx + 1; i < m; i++) mat(i, idx) = vec(i);
  }

  template <typename V>
  void SetFromComplexDrive(std::vector<AUTDDataArray>& data, const V& drive, const bool normalize, const HoloFloat max_coeff) {
    const size_t n = drive.size();
    size_t dev_idx = 0;
    size_t trans_idx = 0;
    for (size_t j = 0; j < n; j++) {
      const auto f_amp = normalize ? HoloFloat{1} : abs(drive(j)) / max_coeff;
      const auto f_phase = arg(drive(j)) / (2 * PI) + HoloFloat{0.5};
      const auto phase = static_cast<uint16_t>((1 - f_phase) * HoloFloat{255});
      const uint16_t duty = static_cast<uint16_t>(ToDuty(f_amp)) << 8 & 0xFF00;
      data[dev_idx][trans_idx++] = duty | phase;
      if (trans_idx == NUM_TRANS_IN_UNIT) {
        dev_idx++;
        trans_idx = 0;
      }
    }
  }

  [[nodiscard]] std::complex<HoloFloat> Transfer(const HoloVector3& trans_pos, const HoloVector3& trans_norm, const HoloVector3& target_pos,
                                                 const HoloFloat wave_number, const HoloFloat attenuation = 0) const {
    const auto diff = target_pos - trans_pos;
    const auto dist = diff.norm();
    const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180 / PI;
    const auto directivity = utils::directivityT4010A1(theta);

    return directivity / dist * exp(std::complex<HoloFloat>(-dist * attenuation, -wave_number * dist));
  }

  template <typename M>
  M TransferMatrix() {
    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    M g(m, n);

    const auto wave_number = 2 * PI / _geometry->wavelength();
    if constexpr (const auto attenuation = _geometry->attenuation_coeff(); sizeof(Float) == sizeof(HoloFloat)) {
      for (size_t i = 0; i < m; i++) {
        const auto& tp = _foci[i];
        for (size_t j = 0; j < n; j++) {
          const auto pos = _geometry->position(j);
          const auto dir = _geometry->direction(j / NUM_TRANS_IN_UNIT);
          g(i, j) = Transfer(pos, dir, tp, wave_number, attenuation);
        }
      }
    } else {
      for (size_t i = 0; i < m; i++) {
        const auto& tp = _foci[i];
        for (size_t j = 0; j < n; j++) {
          const auto p = _geometry->position(j);
          const auto pos = HoloVector3(p.x(), p.y(), p.z());
          const auto d = _geometry->direction(j / NUM_TRANS_IN_UNIT);
          const auto dir = HoloVector3(d.x(), d.y(), d.z());
          g(i, j) = Transfer(pos, dir, tp, wave_number, attenuation);
        }
      }
    }
    return g;
  }

  template <typename M>
  void MakeBhB(M* bhb) {
    const auto m = _foci.size();

    M p = M::Zero(m, m);
    for (size_t i = 0; i < m; i++) p(i, i) = -_amps[i];

    const auto g = TransferMatrix<M>();

    M b = _backend.ConcatCol(g, p);
    _backend.MatMul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, std::complex<HoloFloat>(1, 0), b, b, std::complex<HoloFloat>(0, 0), bhb);
  }

  template <typename M, typename V>
  void CalcTTh(const V& x, M* tth) {
    const size_t len = x.size();
    M t(len, 1);
    for (size_t i = 0; i < len; i++) t(i, 0) = exp(std::complex<HoloFloat>(0, -x(i)));
    _backend.MatMul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, std::complex<HoloFloat>(1, 0), t, t, std::complex<HoloFloat>(0, 0), tth);
  }

  Result<bool, std::string> SDP() {
    if (!_backend.SupportsSvd() || !_backend.SupportsEVD()) return Err(std::string("This backend does not support this method."));

    auto alpha = HoloFloat{1e-3};
    auto lambda = HoloFloat{0.9};
    auto repeat = 100;
    auto normalize = true;

    if (_params != nullptr) {
      auto* const sdp_params = static_cast<const SDPParams*>(_params);
      alpha = sdp_params->regularization < 0 ? alpha : sdp_params->regularization;
      repeat = sdp_params->repeat < 0 ? repeat : sdp_params->repeat;
      lambda = sdp_params->lambda < 0 ? lambda : sdp_params->lambda;
      normalize = sdp_params->normalize_amp;
    }

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    typename B::MatrixXc p = B::MatrixXc::Zero(m, m);
    for (size_t i = 0; i < m; i++) p(i, i) = std::complex<HoloFloat>(_amps[i], 0);

    typename B::MatrixXc b = TransferMatrix<typename B::MatrixXc>();
    typename B::MatrixXc pseudo_inv_b(n, m);
    _backend.PseudoInverseSvd(&b, alpha, &pseudo_inv_b);

    typename B::MatrixXc mm = B::MatrixXc::Identity(m, m);
    _backend.MatMul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, std::complex<HoloFloat>(1, 0), b, pseudo_inv_b, std::complex<HoloFloat>(-1, 0), &mm);
    typename B::MatrixXc tmp(m, m);
    MatrixMul(p, mm, &tmp);
    MatrixMul(tmp, p, &mm);
    typename B::MatrixXc x_mat = B::MatrixXc::Identity(m, m);

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> range(0, 1);
    typename B::VectorXc zero = B::VectorXc::Zero(m);
    for (auto i = 0; i < repeat; i++) {
      auto ii = static_cast<size_t>(m * range(mt));

      typename B::VectorXc mmc = mm.col(ii);
      mmc(ii) = 0;

      typename B::VectorXc x(m);
      MatrixVecMul(x_mat, mmc, &x);
      if (std::complex<HoloFloat> gamma = _backend.DotC(x, mmc); gamma.real() > 0) {
        x = -x * sqrt(lambda / gamma.real());
        SetBcdResult(x_mat, x, ii);
      } else {
        SetBcdResult(x_mat, zero, ii);
      }
    }

    typename B::VectorXc u = _backend.MaxEigenVector(&x_mat);

    typename B::VectorXc ut(m);
    MatrixVecMul(p, u, &ut);

    typename B::VectorXc q(n);
    MatrixVecMul(pseudo_inv_b, ut, &q);

    const auto max_coeff = _backend.MaxCoeffC(q);
    SetFromComplexDrive(_data, q, normalize, max_coeff);
    return Ok(true);
  }

  Result<bool, std::string> EVD() {
    if (!_backend.SupportsEVD() || !_backend.SupportsSolve()) return Err(std::string("This backend does not support this method."));

    HoloFloat gamma = 1;
    auto normalize = true;

    if (_params != nullptr) {
      auto* const evd_params = static_cast<const EVDParams*>(_params);
      gamma = evd_params->regularization < 0 ? gamma : evd_params->regularization;
      normalize = evd_params->normalize_amp;
    }

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    const typename B::MatrixXc g = TransferMatrix<typename B::MatrixXc>();

    typename B::VectorXc denominator(m);
    for (size_t i = 0; i < m; i++) {
      auto tmp = std::complex<HoloFloat>(0, 0);
      for (size_t j = 0; j < n; j++) {
        tmp += g(i, j);
      }
      denominator(i) = tmp;
    }

    typename B::MatrixXc x(n, m);
    for (size_t i = 0; i < m; i++) {
      auto c = std::complex<HoloFloat>(_amps[i], 0) / denominator(i);
      for (size_t j = 0; j < n; j++) {
        x(j, i) = c * std::conj(g(i, j));
      }
    }
    typename B::MatrixXc r(m, m);
    MatrixMul(g, x, &r);
    typename B::VectorXc max_ev = _backend.MaxEigenVector(&r);

    typename B::MatrixXc sigma = B::MatrixXc::Zero(n, n);
    for (size_t j = 0; j < n; j++) {
      HoloFloat tmp = 0;
      for (size_t i = 0; i < m; i++) {
        tmp += abs(g(i, j)) * _amps[i];
      }
      sigma(j, j) = std::complex<HoloFloat>(pow(sqrt(tmp / static_cast<HoloFloat>(m)), gamma), 0.0);
    }

    typename B::MatrixXc gr = _backend.ConcatRow(g, sigma);

    typename B::VectorXc f = B::VectorXc::Zero(m + n);
    for (size_t i = 0; i < m; i++) f(i) = _amps[i] * max_ev(i) / abs(max_ev(i));

    typename B::MatrixXc gtg(n, n);
    _backend.MatMul(TRANSPOSE::CONJ_TRANS, TRANSPOSE::NO_TRANS, std::complex<HoloFloat>(1, 0), gr, gr, std::complex<HoloFloat>(0, 0), &gtg);

    typename B::VectorXc gtf(n);
    _backend.MatVecMul(TRANSPOSE::CONJ_TRANS, std::complex<HoloFloat>(1, 0), gr, f, std::complex<HoloFloat>(0, 0), &gtf);

    _backend.SolveCh(&gtg, &gtf);

    const auto max_coeff = _backend.MaxCoeffC(gtf);
    SetFromComplexDrive(_data, gtf, normalize, max_coeff);
    return Ok(true);
  }

  Result<bool, std::string> Naive() {
    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    const typename B::MatrixXc g = TransferMatrix<typename B::MatrixXc>();
    typename B::VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = std::complex<HoloFloat>(_amps[i], 0);

    typename B::VectorXc q(n);
    _backend.MatVecMul(TRANSPOSE::CONJ_TRANS, std::complex<HoloFloat>(1, 0), g, p, std::complex<HoloFloat>(0, 0), &q);

    SetFromComplexDrive(_data, q, true, 1.0);
    return Ok(true);
  }

  Result<bool, std::string> GS() {
    const int32_t repeat = _params == nullptr ? 100 : *static_cast<const uint32_t*>(_params);

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    const typename B::MatrixXc g = TransferMatrix<typename B::MatrixXc>();

    typename B::VectorXc q0 = B::VectorXc::Ones(n);

    typename B::VectorXc q(n);
    _backend.VecCpyC(q0, &q);

    typename B::VectorXc gamma(m);
    typename B::VectorXc p(m);
    typename B::VectorXc xi(n);
    for (auto k = 0; k < repeat; k++) {
      MatrixVecMul(g, q, &gamma);
      for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * _amps[i];
      _backend.MatVecMul(TRANSPOSE::CONJ_TRANS, std::complex<HoloFloat>(1, 0), g, p, std::complex<HoloFloat>(0, 0), &xi);
      for (size_t j = 0; j < n; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
    }

    SetFromComplexDrive(_data, q, true, 1.0);
    return Ok(true);
  }

  Result<bool, std::string> GSPAT() {
    const int32_t repeat = _params == nullptr ? 100 : *static_cast<const uint32_t*>(_params);

    const auto m = _foci.size();
    const auto n = _geometry->num_transducers();

    const typename B::MatrixXc g = TransferMatrix<typename B::MatrixXc>();

    typename B::VectorXc denominator(m);
    for (size_t i = 0; i < m; i++) {
      auto tmp = std::complex<HoloFloat>(0, 0);
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
    MatrixMul(g, b, &r);

    typename B::VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = std::complex<HoloFloat>(_amps[i], 0);

    typename B::VectorXc gamma(m);
    MatrixVecMul(r, p, &gamma);
    for (auto k = 0; k < repeat; k++) {
      for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * _amps[i];
      MatrixVecMul(r, p, &gamma);
    }

    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * _amps[i] * _amps[i];

    typename B::VectorXc q(n);
    MatrixVecMul(b, p, &q);

    SetFromComplexDrive(_data, q, true, 1.0);
    return Ok(true);
  }

  Result<bool, std::string> LM() {
    if (!_backend.SupportsSolve()) return Err(std::string("This backend does not support this method."));

    auto eps_1 = HoloFloat{1e-8};
    auto eps_2 = HoloFloat{1e-8};
    auto tau = HoloFloat{1e-3};
    auto k_max = 5;
    Float* initial = nullptr;

    if (_params != nullptr) {
      auto* const nlp_params = static_cast<const NLSParams*>(_params);
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
    MakeBhB<typename B::MatrixXc>(&bhb);

    typename B::VectorX x(n_param);
    if (initial == nullptr) {
      std::memset(x.data(), 0, x.size() * sizeof(HoloFloat));
    } else {
      std::memcpy(x.data(), initial, x.size() * sizeof(HoloFloat));
    }

    auto nu = HoloFloat{2};

    typename B::MatrixXc tth(n_param, n_param);
    CalcTTh<typename B::MatrixXc, typename B::VectorX>(x, &tth);

    typename B::MatrixXc bhb_tth(n_param, n_param);
    _backend.HadamardProduct(bhb, tth, &bhb_tth);

    typename B::MatrixX a(n_param, n_param);
    _backend.Real(bhb_tth, &a);

    typename B::VectorX g(n_param);
    for (size_t i = 0; i < n_param; i++) {
      HoloFloat tmp = 0;
      for (size_t k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
      g(i) = tmp;
    }

    HoloFloat a_max = 0;
    for (size_t i = 0; i < n_param; i++) a_max = std::max(a_max, a(i, i));

    auto mu = tau * a_max;

    auto is_found = _backend.MaxCoeff(g) <= eps_1;

    typename B::VectorXc t(n_param);
    for (size_t i = 0; i < n_param; i++) t(i) = exp(std::complex<HoloFloat>(0, x(i)));

    typename B::VectorXc tmp_vec_c(n_param);
    MatrixVecMul(bhb, t, &tmp_vec_c);
    HoloFloat fx = _backend.DotC(t, tmp_vec_c).real();

    typename B::MatrixX identity = B::MatrixX::Identity(n_param, n_param);
    typename B::VectorX tmp_vec(n_param);
    typename B::VectorX h_lm(n_param);
    typename B::VectorX x_new(n_param);
    typename B::MatrixX tmp_mat(n_param, n_param);
    for (auto k = 0; k < k_max; k++) {
      if (is_found) break;

      _backend.MatCpy(a, &tmp_mat);
      _backend.MatAdd(mu, identity, HoloFloat{1.0}, &tmp_mat);
      _backend.Solveg(&tmp_mat, &g, &h_lm);
      if (h_lm.norm() <= eps_2 * (x.norm() + eps_2)) {
        is_found = true;
      } else {
        _backend.VecCpy(x, &x_new);
        _backend.VecAdd(HoloFloat{-1.0}, h_lm, HoloFloat{1.0}, &x_new);
        for (size_t i = 0; i < n_param; i++) t(i) = exp(std::complex<HoloFloat>(0, x_new(i)));

        MatrixVecMul(bhb, t, &tmp_vec_c);
        const HoloFloat fx_new = _backend.DotC(t, tmp_vec_c).real();

        _backend.VecCpy(g, &tmp_vec);
        _backend.VecAdd(mu, h_lm, HoloFloat{1.0}, &tmp_vec);
        const HoloFloat l0_lhlm = _backend.Dot(h_lm, tmp_vec) / 2;

        const auto rho = (fx - fx_new) / l0_lhlm;
        fx = fx_new;
        if (rho > 0) {
          _backend.VecCpy(x_new, &x);
          CalcTTh<typename B::MatrixXc, typename B::VectorX>(x, &tth);
          _backend.HadamardProduct(bhb, tth, &bhb_tth);
          _backend.Real(bhb_tth, &a);
          for (size_t i = 0; i < n_param; i++) {
            HoloFloat tmp = 0;
            for (size_t j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
            g(i) = tmp;
          }
          is_found = _backend.MaxCoeff(g) <= eps_1;
          mu *= std::max(HoloFloat{1. / 3.}, std::pow(1 - (2 * rho - 1), HoloFloat{3}));
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
      _data[dev_idx][trans_idx++] = duty | phase;
      if (trans_idx == NUM_TRANS_IN_UNIT) {
        dev_idx++;
        trans_idx = 0;
      }
    }
    return Ok(true);
  }
};

#ifndef DISABLE_EIGEN
using HoloGainE = HoloGain<Eigen3Backend>;
#endif
#ifdef ENABLE_BLAS
using HoloGainB = HoloGain<BLASBackend>;
#endif

}  // namespace autd::gain::holo
