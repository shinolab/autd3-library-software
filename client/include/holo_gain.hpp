// File: holo_gain.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 02/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/gain.hpp"
#include "linalg_backend.hpp"

namespace autd::gain::holo {

/**
 * @brief Gain to produce multiple focal points
 */
class HoloGain : public core::Gain {
 public:
  HoloGain(BackendPtr backend, std::vector<core::Vector3>& foci, std::vector<double>& amps)
      : _backend(std::move(backend)), _foci(std::move(foci)), _amps(std::move(amps)) {}
  ~HoloGain() override = default;
  HoloGain(const HoloGain& v) noexcept = default;
  HoloGain& operator=(const HoloGain& obj) = default;
  HoloGain(HoloGain&& obj) = default;
  HoloGain& operator=(HoloGain&& obj) = default;

  BackendPtr& backend() { return this->_backend; }
  std::vector<core::Vector3>& foci() { return this->_foci; }
  std::vector<double>& amplitudes() { return this->_amps; }

 protected:
  BackendPtr _backend;
  std::vector<core::Vector3> _foci;
  std::vector<double> _amps;

  void matrix_mul(const Backend::MatrixXc& a, const Backend::MatrixXc& b, Backend::MatrixXc* c) const;

  void matrix_vec_mul(const Backend::MatrixXc& a, const Backend::VectorXc& b, Backend::VectorXc* c) const;
  static void set_from_complex_drive(std::vector<core::DataArray>& data, const Backend::VectorXc& drive, bool normalize, double max_coefficient);
  static std::complex<double> transfer(const core::Vector3& trans_pos, const core::Vector3& trans_norm, const core::Vector3& target_pos,
                                       double wave_number, double attenuation = 0);
  static Backend::MatrixXc transfer_matrix(const std::vector<core::Vector3>& foci, const core::GeometryPtr& geometry);
};

/**
 * @brief Gain to produce multiple focal points with SDP method.
 * Refer to Inoue, Seki, Yasutoshi Makino, and Hiroyuki Shinoda. "Active touch
 * perception produced by airborne ultrasonic haptic hologram." 2015 IEEE
 * World Haptics Conference (WHC). IEEE, 2015.
 */
class HoloGainSDP final : public HoloGain {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] alpha parameter
   * @param[in] lambda parameter
   * @param[in] repeat parameter
   * @param[in] normalize parameter
   */
  static std::shared_ptr<HoloGainSDP> create(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps,
                                             double alpha = 1e-3, double lambda = 0.9, size_t repeat = 100, bool normalize = true) {
    return std::make_shared<HoloGainSDP>(backend, foci, amps, alpha, lambda, repeat, normalize);
  }

  Error calc(const core::GeometryPtr& geometry) override;
  HoloGainSDP(BackendPtr backend, std::vector<core::Vector3>& foci, std::vector<double>& amps, const double alpha, const double lambda,
              const size_t repeat, const bool normalize)
      : HoloGain(std::move(backend), foci, amps), _alpha(alpha), _lambda(lambda), _repeat(repeat), _normalize(normalize) {}

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
class HoloGainEVD final : public HoloGain {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] gamma parameter
   * @param[in] normalize parameter
   */
  static std::shared_ptr<HoloGainEVD> create(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps, double gamma = 1,
                                             bool normalize = true) {
    return std::make_shared<HoloGainEVD>(backend, foci, amps, gamma, normalize);
  }

  Error calc(const core::GeometryPtr& geometry) override;
  HoloGainEVD(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps, const double gamma, const bool normalize)
      : HoloGain(backend, foci, amps), _gamma(gamma), _normalize(normalize) {}

 private:
  double _gamma;
  bool _normalize;
};

/**
 * @brief Gain to produce multiple focal points with naive method.
 */
class HoloGainNaive final : public HoloGain {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   */
  static std::shared_ptr<HoloGainNaive> create(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps) {
    return std::make_shared<HoloGainNaive>(backend, foci, amps);
  }

  Error calc(const core::GeometryPtr& geometry) override;

  HoloGainNaive(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps) : HoloGain(backend, foci, amps) {}
};

/**
 * @brief Gain to produce multiple focal points with GS method.
 * Refer to Asier Marzo and Bruce W Drinkwater, "Holographic acoustic
 * tweezers," Proceedings of theNational Academy of Sciences, 116(1):84–89, 2019.
 */
class HoloGainGS final : public HoloGain {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<HoloGainGS> create(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps,
                                            size_t repeat = 100) {
    return std::make_shared<HoloGainGS>(backend, foci, amps, repeat);
  }

  Error calc(const core::GeometryPtr& geometry) override;

  HoloGainGS(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps, const size_t repeat)
      : HoloGain(backend, foci, amps), _repeat(repeat) {}

 private:
  size_t _repeat;
};

/**
 * @brief Gain to produce multiple focal points with GS-PAT method (not yet been implemented with GPU).
 * Refer to Diego Martinez Plasencia et al. "Gs-pat: high-speed multi-point
 * sound-fields for phased arrays of transducers," ACMTrans-actions on
 * Graphics (TOG), 39(4):138–1, 2020.
 */
class HoloGainGSPAT final : public HoloGain {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<HoloGainGSPAT> create(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps,
                                               size_t repeat = 100) {
    return std::make_shared<HoloGainGSPAT>(backend, foci, amps, repeat);
  }
  Error calc(const core::GeometryPtr& geometry) override;

  HoloGainGSPAT(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps, const size_t repeat)
      : HoloGain(backend, foci, amps), _repeat(repeat) {}

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
class HoloGainLM final : public HoloGain {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps_1 parameter
   * @param[in] eps_2 parameter
   * @param[in] tau parameter
   * @param[in] k_max parameter
   * @param[in] initial initial phase of transducers
   */
  static std::shared_ptr<HoloGainLM> create(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps,
                                            double eps_1 = 1e-8, double eps_2 = 1e-8, double tau = 1e-3, size_t k_max = 5,
                                            const std::vector<double>& initial = {}) {
    return std::make_shared<HoloGainLM>(backend, foci, amps, eps_1, eps_2, tau, k_max, initial);
  }

  Error calc(const core::GeometryPtr& geometry) override;

  HoloGainLM(const BackendPtr& backend, std::vector<core::Vector3>& foci, std::vector<double>& amps, const double eps_1, const double eps_2,
             const double tau, const size_t k_max, std::vector<double> initial)
      : HoloGain(backend, foci, amps), _eps_1(eps_1), _eps_2(eps_2), _tau(tau), _k_max(k_max), _initial(std::move(initial)) {}

 private:
  double _eps_1;
  double _eps_2;
  double _tau;
  size_t _k_max;
  std::vector<double> _initial;
};

/**
 * @brief Gain to produce multiple focal points with Greedy algorithm.
 * Refer to Shun suzuki, et al. “Radiation Pressure Field Reconstruction for Ultrasound Midair Haptics by Greedy Algorithm with Brute-Force Search,”
 * in IEEE Transactions on Haptics, doi: 10.1109/TOH.2021.3076489
 */
class HoloGainGreedy final : public HoloGain {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] phase_div resolution of the phase to be searched
   */
  static std::shared_ptr<HoloGainGreedy> create(std::vector<core::Vector3>& foci, std::vector<double>& amps, const size_t phase_div = 16) {
    return std::make_shared<HoloGainGreedy>(foci, amps, phase_div);
  }

  Error calc(const core::GeometryPtr& geometry) override;
  HoloGainGreedy(std::vector<core::Vector3>& foci, std::vector<double>& amps, const size_t phase_div) : HoloGain(nullptr, foci, amps) {
    this->_phases.reserve(phase_div);
    for (size_t i = 0; i < phase_div; i++)
      this->_phases.emplace_back(std::exp(std::complex<double>(0., 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(phase_div))));
  }

 private:
  std::vector<std::complex<double>> _phases;
};

}  // namespace autd::gain::holo
