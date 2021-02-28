// File: holo_gain.hpp
// Project: include
// Created Date: 06/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 28/02/2021
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

namespace {

using autd::Float;
using autd::PI;

#ifdef USE_DOUBLE_AUTD
constexpr double DIR_COEFF_A[] = {1.0, 1.0, 1.0, 0.891250938, 0.707945784, 0.501187234, 0.354813389, 0.251188643, 0.199526231};
constexpr double DIR_COEFF_B[] = {
    0., 0., -0.00459648054721, -0.0155520765675, -0.0208114779827, -0.0182211227016, -0.0122437497109, -0.00780345575475, -0.00312857467007};
constexpr double DIR_COEFF_C[] = {
    0., 0., -0.000787968093807, -0.000307591508224, -0.000218348633296, 0.00047738416141, 0.000120353137658, 0.000323676257958, 0.000143850511};
constexpr double DIR_COEFF_D[] = {
    0., 0., 1.60125528528e-05, 2.9747624976e-06, 2.31910931569e-05, -1.1901034125e-05, 6.77743734332e-06, -5.99548024824e-06, -4.79372835035e-06};
using complex = std::complex<double>;
#else
constexpr float DIR_COEFF_A[] = {1.0f, 1.0f, 1.0f, 0.891250938f, 0.707945784f, 0.501187234f, 0.354813389f, 0.251188643f, 0.199526231f};
constexpr float DIR_COEFF_B[] = {
    0.f, 0.f, -0.00459648054721f, -0.0155520765675f, -0.0208114779827f, -0.0182211227016f, -0.0122437497109f, -0.00780345575475f, -0.00312857467007f};
constexpr float DIR_COEFF_C[] = {0.f,
                                 0.f,
                                 -0.000787968093807f,
                                 -0.000307591508224f,
                                 -0.000218348633296f,
                                 0.00047738416141f,
                                 0.000120353137658f,
                                 0.000323676257958f,
                                 0.000143850511f};
constexpr float DIR_COEFF_D[] = {0.f,
                                 0.f,
                                 1.60125528528e-05f,
                                 2.9747624976e-06f,
                                 2.31910931569e-05f,
                                 -1.1901034125e-05f,
                                 6.77743734332e-06f,
                                 -5.99548024824e-06f,
                                 -4.79372835035e-06f};
using complex = std::complex<float>;
#endif

static constexpr Float ATTENUATION = Float{1.15e-4};

static Float directivityT4010A1(Float theta_deg) {
  theta_deg = abs(theta_deg);

  while (theta_deg > 90) {
    theta_deg = abs(180 - theta_deg);
  }

  const auto i = static_cast<size_t>(ceil(theta_deg / 10));

  if (i == 0) {
    return 1;
  }

  const auto a = DIR_COEFF_A[i - 1];
  const auto b = DIR_COEFF_B[i - 1];
  const auto c = DIR_COEFF_C[i - 1];
  const auto d = DIR_COEFF_D[i - 1];
  const auto x = theta_deg - static_cast<Float>(i - 1) * 10;
  return a + b * x + c * x * x + d * x * x * x;
}

complex transfer(const autd::Vector3& trans_pos, const autd::Vector3& trans_norm, const autd::Vector3& target_pos, const Float wave_number) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180 / PI;
  const auto directivity = directivityT4010A1(theta);

  return directivity / dist * exp(complex(-dist * ATTENUATION, -wave_number * dist));
}
}  // namespace

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
};

template <typename MCx, typename VCx, typename Mx, typename Vx>
class B {
 public:
  using MatrixXc = MCx;
  using VectorXc = VCx;
  using MatrixX = Mx;
  using VectorX = Vx;

  virtual bool supports_SVD() = 0;
  virtual bool supports_EVD() = 0;
  virtual bool supports_solve() = 0;
  virtual void hadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) = 0;
  virtual void real(const MatrixXc& a, MatrixX* b) = 0;
  virtual MatrixXc pseudoInverseSVD(const MatrixXc& matrix, Float alpha) = 0;
  virtual VectorXc maxEigenVector(const MatrixXc& matrix) = 0;
  virtual void matadd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) = 0;
  virtual void matmul(const char* transa, const char* transb, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b,
                      std::complex<Float> beta, MatrixXc* c) = 0;
  virtual void matvecmul(const char* transa, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta,
                         VectorXc* c) = 0;
  virtual void vecadd(Float alpha, const VectorX& a, Float beta, VectorX* b) = 0;
  virtual void csolve(const MatrixXc& a, const VectorXc& b, VectorXc* c) = 0;
  virtual void solve(const MatrixX& a, const VectorX& b, VectorX* c) = 0;
  virtual Float dot(const VectorX& a, const VectorX& b) = 0;
  virtual std::complex<Float> cdot(const VectorXc& a, const VectorXc& b) = 0;
  virtual Float maxCoeff(const VectorXc& v) = 0;
  virtual MatrixXc concat_in_row(const MatrixXc& a, const MatrixXc& b) = 0;
  virtual MatrixXc concat_in_col(const MatrixXc& a, const MatrixXc& b) = 0;

  virtual ~B() {}
};

class Eigen3Backend final : public B<Eigen::Matrix<std::complex<Float>, -1, -1>, Eigen::Matrix<std::complex<Float>, -1, 1>,
                                     Eigen::Matrix<Float, -1, 1>, Eigen::Matrix<Float, -1, 1>> {
 public:
  bool supports_SVD() override { return true; }
  bool supports_EVD() override { return true; }
  bool supports_solve() override { return true; }
  void hadamardProduct(const MatrixXc& a, const MatrixXc& b, MatrixXc* c);
  void real(const MatrixXc& a, MatrixX* b);
  MatrixXc pseudoInverseSVD(const MatrixXc& matrix, Float alpha) override;
  VectorXc maxEigenVector(const MatrixXc& matrix) override;
  void matadd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) override;
  void matmul(const char* transa, const char* transb, std::complex<Float> alpha, const MatrixXc& a, const MatrixXc& b, std::complex<Float> beta,
              MatrixXc* c) override;
  void matvecmul(const char* transa, std::complex<Float> alpha, const MatrixXc& a, const VectorXc& b, std::complex<Float> beta, VectorXc* c) override;
  void vecadd(Float alpha, const VectorX& a, Float beta, VectorX* b) override;
  void csolve(const MatrixXc& a, const VectorXc& b, VectorXc* c) override;
  void solve(const MatrixX& a, const VectorX& b, VectorX* c) override;
  Float dot(const VectorX& a, const VectorX& b) override;
  std::complex<Float> cdot(const VectorXc& a, const VectorXc& b) override;
  Float maxCoeff(const VectorXc& v) override;
  MatrixXc concat_in_row(const MatrixXc& a, const MatrixXc& b) override;
  MatrixXc concat_in_col(const MatrixXc& a, const MatrixXc& b) override;
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
    _backend.matmul("N", "N", std::complex<Float>(1, 0), a, b, std::complex<Float>(0, 0), c);
  }
  template <typename M, typename V>
  inline void matvecmul(const M& a, const V& b, V* c) {
    _backend.matvecmul("N", std::complex<Float>(1, 0), a, b, std::complex<Float>(0, 0), c);
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

  template <typename M>
  M transferMatrix() {
    const size_t m = _foci.size();
    const size_t n = _geometry->num_transducers();

    M g(m, n);

    const auto wave_number = 2 * PI / _geometry->wavelength();
    for (size_t i = 0; i < m; i++) {
      const auto& tp = _foci[i];
      for (size_t j = 0; j < n; j++) {
        const auto pos = _geometry->position(j);
        const auto dir = _geometry->direction(j / NUM_TRANS_IN_UNIT);
        g(i, j) = transfer(pos, dir, tp, wave_number);
      }
    }

    return g;
  }

  void SDP() {
    if (!_backend.supports_SVD() || !_backend.supports_EVD()) std::cerr << "This backend does not support this method.\n";

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

    const size_t m = _foci.size();
    const size_t n = _geometry->num_transducers();

    B::MatrixXc P = B::MatrixXc::Zero(m, m);
    for (size_t i = 0; i < m; i++) P(i, i) = _amps[i];

    const B::MatrixXc B = transferMatrix<B::MatrixXc>();
    const B::MatrixXc pinvB = _backend.pseudoInverseSVD(B, alpha);

    B::MatrixXc MM = B::MatrixXc::Identity(m, m);
    _backend.matmul("N", "N", std::complex<Float>(1, 0), B, pinvB, std::complex<Float>(-1, 0), &MM);
    B::MatrixXc tmp(m, m);
    matmul(P, MM, &tmp);
    matmul(tmp, P, &MM);
    B::MatrixXc X = B::MatrixXc::Identity(m, m);

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> range(0, 1);
    B::VectorXc zero = B::VectorXc::Zero(m);
    for (auto i = 0; i < repeat; i++) {
      auto ii = static_cast<size_t>(m * static_cast<double>(range(mt)));

      B::VectorXc mmc = MM.col(ii);
      mmc(ii) = 0;

      B::VectorXc x(m);
      matvecmul(X, mmc, &x);
      std::complex<Float> gamma = _backend.cdot(x, mmc);
      if (gamma.real() > 0) {
        x = -x * sqrt(lambda / gamma.real());
        setBCDResult(X, x, ii);
      } else {
        setBCDResult(X, zero, ii);
      }
    }

    B::VectorXc u = _backend.maxEigenVector(X);

    B::VectorXc ut(m);
    matvecmul(P, u, &ut);

    B::VectorXc q(n);
    matvecmul(pinvB, ut, &q);

    const auto max_coeff = _backend.maxCoeff(q);
    SetFromComplexDrive(_data, q, normalize, max_coeff);
  }

  void EVD() {
    if (!_backend.supports_EVD() || !_backend.supports_solve()) std::cerr << "This backend does not support this method.\n";

    Float gamma = 1;
    auto normalize = true;

    if (_params != nullptr) {
      auto* const evd_params = static_cast<autd::gain::EVDParams*>(_params);
      gamma = evd_params->regularization < 0 ? gamma : evd_params->regularization;
      normalize = evd_params->normalize_amp;
    }

    const size_t m = _foci.size();
    const size_t n = _geometry->num_transducers();

    const B::MatrixXc g = transferMatrix<B::MatrixXc>();

    B::VectorXc denominator(m);
    for (size_t i = 0; i < m; i++) {
      auto tmp = std::complex<Float>(0, 0);
      for (size_t j = 0; j < n; j++) {
        tmp += g(i, j);
      }
      denominator(i) = tmp;
    }

    B::MatrixXc x(n, m);
    for (size_t i = 0; i < m; i++) {
      auto c = std::complex<Float>(_amps[i], 0) / denominator(i);
      for (size_t j = 0; j < n; j++) {
        x(j, i) = c * std::conj(g(i, j));
      }
    }
    B::MatrixXc r(m, m);
    matmul(g, x, &r);
    B::VectorXc max_ev = _backend.maxEigenVector(r);

    B::MatrixXc sigma = B::MatrixXc::Zero(n, n);
    for (size_t j = 0; j < n; j++) {
      Float tmp = 0;
      for (size_t i = 0; i < m; i++) {
        tmp += abs(g(i, j)) * _amps[i];
      }
      sigma(j, j) = std::complex<Float>(pow(sqrt(tmp / static_cast<Float>(m)), gamma), 0.0);
    }

    B::MatrixXc gr = _backend.concat_in_row(g, sigma);

    B::VectorXc f = B::VectorXc::Zero(m + n);
    for (size_t i = 0; i < m; i++) f(i) = _amps[i] * max_ev(i) / abs(max_ev(i));

    B::MatrixXc gtg(n, n);
    _backend.matmul("C", "N", std::complex<Float>(1, 0), gr, gr, std::complex<Float>(0, 0), &gtg);

    B::VectorXc gtf(n);
    _backend.matvecmul("C", std::complex<Float>(1, 0), gr, f, std::complex<Float>(0, 0), &gtf);

    B::VectorXc q(n);
    _backend.csolve(gtg, gtf, &q);

    const auto max_coeff = _backend.maxCoeff(q);
    SetFromComplexDrive(_data, q, normalize, max_coeff);
  }

  void NAIVE() {
    const size_t m = _foci.size();
    const size_t n = _geometry->num_transducers();

    const B::MatrixXc g = transferMatrix<B::MatrixXc>();
    B::VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = std::complex<Float>(_amps[i], 0);

    B::VectorXc q(n);
    _backend.matvecmul("C", std::complex<Float>(1, 0), g, p, std::complex<Float>(0, 0), &q);

    SetFromComplexDrive(_data, q, true, 1.0);
  }

  void GS() {
    const int32_t repeat = _params == nullptr ? 100 : *static_cast<uint32_t*>(_params);

    const size_t m = _foci.size();
    const size_t n = _geometry->num_transducers();

    const B::MatrixXc g = transferMatrix<B::MatrixXc>();

    B::VectorXc q0 = B::VectorXc::Ones(n);

    B::VectorXc q = q0;
    B::VectorXc gamma(m);
    B::VectorXc p(m);
    B::VectorXc xi(n);
    for (auto k = 0; k < repeat; k++) {
      matvecmul(g, q, &gamma);
      for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * _amps[i];
      _backend.matvecmul("C", std::complex<Float>(1, 0), g, p, std::complex<Float>(0, 0), &xi);
      for (size_t j = 0; j < n; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
    }

    SetFromComplexDrive(_data, q, true, 1.0);
  }

  void GSPAT() {
    const int32_t repeat = _params == nullptr ? 100 : *static_cast<uint32_t*>(_params);

    const size_t m = _foci.size();
    const size_t n = _geometry->num_transducers();

    const B::MatrixXc g = transferMatrix<B::MatrixXc>();

    B::VectorXc denominator(m);
    for (size_t i = 0; i < m; i++) {
      auto tmp = std::complex<Float>(0, 0);
      for (size_t j = 0; j < n; j++) tmp += abs(g(i, j));
      denominator(i) = tmp;
    }

    B::MatrixXc b(n, m);
    for (size_t i = 0; i < m; i++) {
      auto d = denominator(i) * denominator(i);
      for (size_t j = 0; j < n; j++) {
        b(j, i) = std::conj(g(i, j)) / d;
      }
    }

    B::MatrixXc r(m, m);
    matmul(g, b, &r);

    B::VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = std::complex<Float>(_amps[i], 0);

    B::VectorXc gamma(m);
    matvecmul(r, p, &gamma);
    for (auto k = 0; k < repeat; k++) {
      for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * _amps[i];
      matvecmul(r, p, &gamma);
    }

    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * _amps[i] * _amps[i];

    B::VectorXc q(n);
    matvecmul(b, p, &q);

    SetFromComplexDrive(_data, q, true, 1.0);
  }

  template <typename M>
  inline void makeBhB(M* bhb) {
    const size_t m = _foci.size();
    const size_t n = _geometry->num_transducers();

    M p = M::Zero(m, m);
    for (size_t i = 0; i < m; i++) p(i, i) = -_amps[i];

    const auto g = transferMatrix<M>();

    M b = _backend.concat_in_col(g, p);
    _backend.matmul("C", "N", std::complex<Float>(1, 0), b, b, std::complex<Float>(0, 0), bhb);
  }

  template <typename M, typename V, typename Vc>
  inline void calcTTh(const V& x, M* tth) {
    const size_t len = x.size();
    Vc t(len);
    for (size_t i = 0; i < len; i++) t(i) = exp(std::complex<Float>(0, -x(i)));
    _backend.matmul("N", "C", std::complex<Float>(1, 0), t, t, std::complex<Float>(0, 0), tth);
  }

  void LM() {
    if (!_backend.supports_solve()) std::cerr << "This backend does not support this method.\n";

    auto eps_1 = Float{1e-8};
    auto eps_2 = Float{1e-8};
    auto tau = Float{1e-3};
    auto k_max = 5;

    if (_params != nullptr) {
      auto* const nlp_params = static_cast<autd::gain::NLSParams*>(_params);
      eps_1 = nlp_params->eps_1 < 0 ? eps_1 : nlp_params->eps_1;
      eps_2 = nlp_params->eps_2 < 0 ? eps_2 : nlp_params->eps_2;
      k_max = nlp_params->k_max < 0 ? k_max : nlp_params->k_max;
      tau = nlp_params->tau < 0 ? tau : nlp_params->tau;
    }

    const size_t m = _foci.size();
    const size_t n = _geometry->num_transducers();
    const size_t n_param = n + m;

    B::MatrixXc bhb(n_param, n_param);
    makeBhB<B::MatrixXc>(&bhb);

    B::VectorX x = B::VectorX::Zero(n_param);
    Float nu = Float{2};

    B::MatrixXc tth(n_param, n_param);
    calcTTh<B::MatrixXc, B::VectorX, B::VectorXc>(x, &tth);

    B::MatrixXc bhb_tth(n_param, n_param);
    _backend.hadamardProduct(bhb, tth, &bhb_tth);

    B::MatrixX a(n_param, n_param);
    _backend.real(bhb_tth, &a);

    B::VectorX g(n_param);
    for (size_t i = 0; i < n_param; i++) {
      Float tmp = 0;
      for (size_t k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
      g(i) = tmp;
    }

    Float a_max = 0;
    for (size_t i = 0; i < n_param; i++) a_max = std::max(a_max, a(i, i));

    auto mu = tau * a_max;

    auto is_found = g.maxCoeff() <= eps_1;

    B::VectorXc t(n_param);
    for (size_t i = 0; i < n_param; i++) t(i) = exp(std::complex<Float>(0, x(i)));

    B::VectorXc tmpc(n_param);
    matvecmul(bhb, t, &tmpc);
    Float fx = _backend.cdot(t, tmpc).real();

    B::MatrixX identity = B::MatrixX::Identity(n_param, n_param);
    B::VectorX tmp(n_param);
    B::VectorX h_lm(n_param);
    B::VectorX x_new(n_param);
    B::MatrixX tmpm(n_param, n_param);
    for (auto k = 0; k < k_max; k++) {
      if (is_found) break;

      _backend.matadd(Float{-1.0}, a, Float{0.0}, &tmpm);
      _backend.matadd(-mu, identity, Float{1.0}, &tmpm);
      _backend.solve(tmpm, g, &h_lm);

      if (h_lm.norm() <= eps_2 * (x.norm() + eps_2)) {
        is_found = true;
      } else {
        _backend.vecadd(Float{1.0}, x, Float{0.0}, &x_new);
        _backend.vecadd(Float{1.0}, h_lm, Float{1.0}, &x_new);
        for (size_t i = 0; i < n_param; i++) t(i) = exp(std::complex<Float>(0, x_new(i)));

        matvecmul(bhb, t, &tmpc);
        Float fx_new = _backend.cdot(t, tmpc).real();

        _backend.vecadd(mu, h_lm, Float{0.0}, &tmp);
        _backend.vecadd(Float{-1.0}, g, 1.0, &tmp);
        Float l0_lhlm = _backend.dot(h_lm, tmp) / 2;

        Float rho = (fx - fx_new) / l0_lhlm;
        fx = fx_new;
        if (rho > 0) {
          _backend.vecadd(Float{1.0}, x_new, Float{0.0}, &x);
          calcTTh<B::MatrixXc, B::VectorX, B::VectorXc>(x, &tth);
          _backend.hadamardProduct(bhb, tth, &bhb_tth);
          _backend.real(bhb_tth, &a);
          for (size_t i = 0; i < n_param; i++) {
            Float tmp = 0;
            for (size_t j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
            g(i) = tmp;
          }
          is_found = g.maxCoeff() <= eps_1;
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
}  // namespace autd::gain
