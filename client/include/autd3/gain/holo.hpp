// File: holo_gain.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "autd3/core/exception.hpp"
#include "autd3/core/gain.hpp"
#include "linalg_backend.hpp"

namespace autd::gain::holo {

/**
 * @brief Gain to produce multiple focal points
 */
class Holo : public core::Gain {
 public:
  Holo(BackendPtr backend, std::vector<core::Vector3> foci, const std::vector<double>& amps) : _backend(std::move(backend)), _foci(std::move(foci)) {
    if (_foci.size() != amps.size()) throw core::GainBuildError("The size of foci and amps are not the same");
    _amps.reserve(amps.size());
    for (const auto amp : amps) _amps.emplace_back(complex(amp, 0.0));
  }
  ~Holo() override = default;
  Holo(const Holo& v) noexcept = default;
  Holo& operator=(const Holo& obj) = default;
  Holo(Holo&& obj) = default;
  Holo& operator=(Holo&& obj) = default;

  BackendPtr& backend() { return this->_backend; }
  std::vector<core::Vector3>& foci() { return this->_foci; }
  std::vector<complex>& amplitudes() { return this->_amps; }

 protected:
  BackendPtr _backend;
  std::vector<core::Vector3> _foci;
  std::vector<complex> _amps;
};

/**
 * @brief Gain to produce multiple focal points with SDP method.
 * Refer to Inoue, Seki, Yasutoshi Makino, and Hiroyuki Shinoda. "Active touch
 * perception produced by airborne ultrasonic haptic hologram." 2015 IEEE
 * World Haptics Conference (WHC). IEEE, 2015.
 */
class SDP final : public Holo {
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
  static std::shared_ptr<SDP> create(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                     double alpha = 1e-3, double lambda = 0.9, size_t repeat = 100, bool normalize = true) {
    return std::make_shared<SDP>(backend, foci, amps, alpha, lambda, repeat, normalize);
  }

  void calc(const core::GeometryPtr& geometry) override;
  SDP(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double alpha, const double lambda,
      const size_t repeat, const bool normalize)
      : Holo(std::move(backend), foci, amps), _alpha(alpha), _lambda(lambda), _repeat(repeat), _normalize(normalize) {}

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
class EVD final : public Holo {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] gamma parameter
   * @param[in] normalize parameter
   */
  static std::shared_ptr<EVD> create(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                     double gamma = 1, bool normalize = true) {
    return std::make_shared<EVD>(backend, foci, amps, gamma, normalize);
  }

  void calc(const core::GeometryPtr& geometry) override;
  EVD(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double gamma, const bool normalize)
      : Holo(backend, foci, amps), _gamma(gamma), _normalize(normalize) {}

 private:
  double _gamma;
  bool _normalize;
};

/**
 * @brief Gain to produce multiple focal points with naive method.
 */
class Naive final : public Holo {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   */
  static std::shared_ptr<Naive> create(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps) {
    return std::make_shared<Naive>(backend, foci, amps);
  }

  void calc(const core::GeometryPtr& geometry) override;

  Naive(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps) : Holo(backend, foci, amps) {}
};

/**
 * @brief Gain to produce multiple focal points with GS method.
 * Refer to Asier Marzo and Bruce W Drinkwater, "Holographic acoustic
 * tweezers," Proceedings of theNational Academy of Sciences, 116(1):84–89, 2019.
 */
class GS final : public Holo {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<GS> create(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                    const size_t repeat = 100) {
    return std::make_shared<GS>(backend, foci, amps, repeat);
  }

  void calc(const core::GeometryPtr& geometry) override;

  GS(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat)
      : Holo(backend, foci, amps), _repeat(repeat) {}

 private:
  size_t _repeat;
};

/**
 * @brief Gain to produce multiple focal points with GS-PAT method (not yet been implemented with GPU).
 * Refer to Diego Martinez Plasencia et al. "Gs-pat: high-speed multi-point
 * sound-fields for phased arrays of transducers," ACMTrans-actions on
 * Graphics (TOG), 39(4):138–1, 2020.
 */
class GSPAT final : public Holo {
 public:
  /**
   * @brief Generate function
   * @param[in] backend linear algebra calculation backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<GSPAT> create(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                       const size_t repeat = 100) {
    return std::make_shared<GSPAT>(backend, foci, amps, repeat);
  }
  void calc(const core::GeometryPtr& geometry) override;

  GSPAT(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat)
      : Holo(backend, foci, amps), _repeat(repeat) {}

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
class LM final : public Holo {
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
  static std::shared_ptr<LM> create(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                    const double eps_1 = 1e-8, const double eps_2 = 1e-8, const double tau = 1e-3, const size_t k_max = 5,
                                    const std::vector<double>& initial = {}) {
    return std::make_shared<LM>(backend, foci, amps, eps_1, eps_2, tau, k_max, initial);
  }

  void calc(const core::GeometryPtr& geometry) override;

  LM(const BackendPtr& backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1, const double eps_2,
     const double tau, const size_t k_max, std::vector<double> initial)
      : Holo(backend, foci, amps), _eps_1(eps_1), _eps_2(eps_2), _tau(tau), _k_max(k_max), _initial(std::move(initial)) {}

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
 * @details This method is computed on the CPU regardless of the Backend.
 */
class Greedy final : public Holo {
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

  void calc(const core::GeometryPtr& geometry) override;
  Greedy(const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t phase_div) : Holo(nullptr, foci, amps) {
    this->_phases.reserve(phase_div);
    for (size_t i = 0; i < phase_div; i++)
      this->_phases.emplace_back(std::exp(complex(0., 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(phase_div))));
  }

 private:
  std::vector<complex> _phases;
};

}  // namespace autd::gain::holo
