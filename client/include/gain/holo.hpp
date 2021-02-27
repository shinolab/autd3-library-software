// File: holo_gain.hpp
// Project: include
// Created Date: 06/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 27/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gain.hpp"
#include "linalg.hpp"

namespace autd::gain {
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
  GS_PAT = 3,
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
};

template <typename MCx, typename VCx>
class B {
 public:
  using MatrixXc = MCx;
  using VectorXc = VCx;

  virtual bool is_support_SVD() = 0;
  virtual bool is_support_EVD() = 0;
  virtual MatrixXc pseudoInverseSVD(const MatrixXc& matrix, Float alpha) = 0;
  virtual VectorXc maxEigenVector(const MatrixXc& matrix) = 0;
  virtual MatrixXc transferMatrix(const GeometryPtr& geometry, const std::vector<Vector3>& foci) = 0;
  virtual void matmul(std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b, std::complex<Float> beta, MatrixXc* c) = 0;
  virtual void matvecmul(std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta, VectorXc* c) = 0;
  virtual std::complex<Float> dot(const VectorXc& a, const VectorXc& b) = 0;
  virtual Float maxCoeff(const VectorXc& v) = 0;
  virtual ~B() {}
};

class Eigen3Backend final : public B<Eigen::Matrix<std::complex<Float>, -1, -1>, Eigen::Matrix<std::complex<Float>, -1, 1>> {
 public:
  bool is_support_SVD() override { return true; }
  bool is_support_EVD() override { return true; }
  MatrixXc pseudoInverseSVD(const MatrixXc& matrix, Float alpha) override;
  VectorXc maxEigenVector(const MatrixXc& matrix) override;
  MatrixXc transferMatrix(const GeometryPtr& geometry, const std::vector<Vector3>& foci) override;
  void matmul(std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b, std::complex<Float> beta, MatrixXc* c) override;
  void matvecmul(std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta, VectorXc* c) override;
  std::complex<Float> dot(const VectorXc& a, const VectorXc& b) override;
  Float maxCoeff(const VectorXc& v) override;
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
        // case OPT_METHOD::EVD:
        //  hologainimpl::HoloGainImplEVD(_data, foci, amps, geo, _params);
        //  break;
        // case OPT_METHOD::NAIVE:
        //  hologainimpl::HoloGainImplNaive(_data, foci, amps, geo);
        //  break;
        // case OPT_METHOD::GS:
        //  hologainimpl::HoloGainImplGS(_data, foci, amps, geo, _params);
        //  break;
        // case OPT_METHOD::GS_PAT:
        //  hologainimpl::HoloGainImplGSPAT(_data, foci, amps, geo, _params);
        //  break;
        // case OPT_METHOD::LM:
        //  hologainimpl::HoloGainImplLM(_data, foci, amps, geo, _params);
        //  break;
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
  void Rebuild() { this->_built = false; }

 protected:
  std::vector<Vector3> _foci;
  std::vector<Float> _amps;
  OPT_METHOD _method = OPT_METHOD::SDP;
  void* _params = nullptr;
  B _backend;

 private:
  template <typename M>
  inline void matmul(const M& a, const M& b, M* c) {
    _backend.matmul(std::complex<Float>(1, 0), a, b, std::complex<Float>(0, 0), c);
  }
  template <typename M, typename V>
  inline void matvecmul(const M& a, const V& b, V* c) {
    _backend.matvecmul(std::complex<Float>(1, 0), a, b, std::complex<Float>(0, 0), c);
  }
  template <typename M, typename V>
  inline void setBCDResult(M& mat, const V& vec, size_t idx) {
    const size_t M = vec.size();
    for (size_t i = 0; i < idx; i++) mat(idx, i) = std::conj(vec(i));
    for (size_t i = idx + 1; i < M; i++) mat(idx, i) = std::conj(vec(i));
    for (size_t i = 0; i < idx; i++) mat(i, idx) = vec(i);
    for (size_t i = idx + 1; i < M; i++) mat(i, idx) = vec(i);
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

  void SDP() {
    auto alpha = Float{1e-3};
    auto lambda = Float{0.9};
    auto repeat = 100;
    auto normalize = true;

    if (_params != nullptr) {
      auto* const sdp_params = static_cast<autd::gain::SDPParams*>(_params);
      alpha = sdp_params->regularization < 0 ? alpha : sdp_params->regularization;
      repeat = sdp_params->repeat < 0 ? repeat : sdp_params->repeat;
      lambda = sdp_params->lambda < 0 ? lambda : sdp_params->lambda;
      normalize = sdp_params->normalize_amp;
    }

    const size_t M = _foci.size();
    const auto N = _geometry->num_transducers();

    B::MatrixXc P = B::MatrixXc::Zero(M, M);
    for (size_t i = 0; i < M; i++) P(i, i) = _amps[i];

    const B::MatrixXc B = _backend.transferMatrix(_geometry, _foci);
    const B::MatrixXc pinvB = _backend.pseudoInverseSVD(B, alpha);

    B::MatrixXc MM = B::MatrixXc::Identity(M, M);
    _backend.matmul(std::complex<Float>(1, 0), B, pinvB, std::complex<Float>(-1, 0), &MM);
    B::MatrixXc tmp(M, M);
    matmul(P, MM, &tmp);
    matmul(tmp, P, &MM);
    B::MatrixXc X = B::MatrixXc::Identity(M, M);

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> range(0, 1);
    B::VectorXc zero = B::VectorXc::Zero(M);
    for (auto i = 0; i < repeat; i++) {
      auto ii = static_cast<size_t>(M * static_cast<double>(range(mt)));

      B::VectorXc mmc = MM.col(ii);
      mmc(ii) = 0;

      B::VectorXc x(M);
      matvecmul(X, mmc, &x);
      std::complex<Float> gamma = _backend.dot(x.adjoint(), mmc);
      if (gamma.real() > 0) {
        x = -x * sqrt(lambda / gamma.real());
        setBCDResult(X, x, ii);
      } else {
        setBCDResult(X, zero, ii);
      }
    }

    B::VectorXc u = _backend.maxEigenVector(X);

    B::VectorXc ut(M);
    matvecmul(P, u, &ut);

    B::VectorXc q(N);
    matvecmul(pinvB, ut, &q);

    const auto max_coeff = _backend.maxCoeff(q);
    SetFromComplexDrive(_data, q, normalize, max_coeff);
  }
};
}  // namespace autd::gain
