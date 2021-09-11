// File: holo_gain.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/09/2021
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

#include "autd3/core/exception.hpp"
#include "autd3/core/gain.hpp"
#include "autd3/utils.hpp"
#include "backend.hpp"

namespace autd::gain::holo {
/**
 * @brief Gain to produce multiple focal points
 */
class Holo : public core::Gain {
 public:
  Holo(BackendPtr backend, std::vector<core::Vector3> foci, const std::vector<double>& amps) : _backend(std::move(backend)), _foci(std::move(foci)) {
    if (this->_foci.size() != amps.size()) throw core::exception::GainBuildError("The size of foci and amps are not the same");
    this->_amps.reserve(amps.size());
    for (const auto amp : amps) this->_amps.emplace_back(complex(amp, 0.0));
  }
  ~Holo() override = default;
  Holo(const Holo& v) noexcept = default;
  Holo& operator=(const Holo& obj) = default;
  Holo(Holo&& obj) = default;
  Holo& operator=(Holo&& obj) = default;

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
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] alpha parameter
   * @param[in] lambda parameter
   * @param[in] repeat parameter
   * @param[in] normalize parameter
   */
  static std::shared_ptr<SDP> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, double alpha = 1e-3,
                                     double lambda = 0.9, size_t repeat = 100, bool normalize = true) {
    return std::make_shared<SDP>(std::move(backend), foci, amps, alpha, lambda, repeat, normalize);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->sdp(geometry, this->_foci, this->_amps, _alpha, _lambda, _repeat, _normalize, this->_data);
    this->_built = true;
  }

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
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] gamma parameter
   * @param[in] normalize parameter
   */
  static std::shared_ptr<EVD> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, double gamma = 1,
                                     bool normalize = true) {
    return std::make_shared<EVD>(std::move(backend), foci, amps, gamma, normalize);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->evd(geometry, this->_foci, this->_amps, _gamma, _normalize, this->_data);
    this->_built = true;
  }

  EVD(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double gamma, const bool normalize)
      : Holo(std::move(backend), foci, amps), _gamma(gamma), _normalize(normalize) {}

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
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   */
  static std::shared_ptr<Naive> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps) {
    return std::make_shared<Naive>(std::move(backend), foci, amps);
  }
  void calc(const core::GeometryPtr& geometry) {
    this->_backend->naive(geometry, this->_foci, this->_amps, this->_data);
    this->_built = true;
  }

  Naive(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps) : Holo(std::move(backend), foci, amps) {}
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
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<GS> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                    const size_t repeat = 100) {
    return std::make_shared<GS>(std::move(backend), foci, amps, repeat);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->gs(geometry, this->_foci, this->_amps, _repeat, this->_data);
    this->_built = true;
  }

  GS(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat)
      : Holo(std::move(backend), foci, amps), _repeat(repeat) {}

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
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  static std::shared_ptr<GSPAT> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                       const size_t repeat = 100) {
    return std::make_shared<GSPAT>(std::move(backend), foci, amps, repeat);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->gspat(geometry, this->_foci, this->_amps, _repeat, this->_data);
    this->_built = true;
  }

  GSPAT(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat)
      : Holo(std::move(backend), foci, amps), _repeat(repeat) {}

 private:
  size_t _repeat;
};

/**
 * @brief Gain to produce multiple focal points with Levenberg-Marquardt method.
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
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps_1 parameter
   * @param[in] eps_2 parameter
   * @param[in] tau parameter
   * @param[in] k_max parameter
   * @param[in] initial initial phase of transducers
   */
  static std::shared_ptr<LM> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                    const double eps_1 = 1e-8, const double eps_2 = 1e-8, const double tau = 1e-3, const size_t k_max = 5,
                                    const std::vector<double>& initial = {}) {
    return std::make_shared<LM>(std::move(backend), foci, amps, eps_1, eps_2, tau, k_max, initial);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->lm(geometry, this->_foci, this->_amps, _eps_1, _eps_2, _tau, _k_max, _initial, this->_data);
    this->_built = true;
  }

  LM(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1, const double eps_2,
     const double tau, const size_t k_max, std::vector<double> initial)
      : Holo(std::move(backend), foci, amps), _eps_1(eps_1), _eps_2(eps_2), _tau(tau), _k_max(k_max), _initial(std::move(initial)) {}

 private:
  double _eps_1;
  double _eps_2;
  double _tau;
  size_t _k_max;
  std::vector<double> _initial;
};

/**
 * @brief Gain to produce multiple focal points with Gauss-Newton method.
 */
class GaussNewton final : public Holo {
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
  static std::shared_ptr<GaussNewton> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                             const double eps_1 = 1e-6, const double eps_2 = 1e-6, const size_t k_max = 500,
                                             const std::vector<double>& initial = {}) {
    return std::make_shared<GaussNewton>(std::move(backend), foci, amps, eps_1, eps_2, k_max, initial);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->gauss_newton(geometry, this->_foci, this->_amps, _eps_1, _eps_2, _k_max, _initial, this->_data);
    this->_built = true;
  }

  GaussNewton(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1, const double eps_2,
              const size_t k_max, std::vector<double> initial)
      : Holo(std::move(backend), foci, amps), _eps_1(eps_1), _eps_2(eps_2), _k_max(k_max), _initial(std::move(initial)) {}

 private:
  double _eps_1;
  double _eps_2;
  size_t _k_max;
  std::vector<double> _initial;
};

/**
 * @brief Gain to produce multiple focal points with GradientDescent method.
 */
class GradientDescent final : public Holo {
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
  static std::shared_ptr<GradientDescent> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                                 const double eps = 1e-6, const double step = 0.5, const size_t k_max = 2000,
                                                 const std::vector<double>& initial = {}) {
    return std::make_shared<GradientDescent>(std::move(backend), foci, amps, eps, step, k_max, initial);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->gradient_descent(geometry, this->_foci, this->_amps, _eps, _step, _k_max, _initial, this->_data);
    this->_built = true;
  }

  GradientDescent(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps, const double step,
                  const size_t k_max, std::vector<double> initial)
      : Holo(std::move(backend), foci, amps), _eps(eps), _k_max(k_max), _step(step), _initial(std::move(initial)) {}

 private:
  double _eps;
  size_t _k_max;
  double _step;
  std::vector<double> _initial;
};

/**
 * @brief Gain to produce multiple focal points with Acoustic Power Optimization method.
 * Refer to Keisuke Hasegawa, Hiroyuki Shinoda, and Takaaki Nara. Volumetric acoustic holography and its application to self-positioning by single
 * channel measurement.Journal of Applied Physics,127(24):244904, 2020.7
 */
class APO final : public Holo {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps parameter
   * @param[in] lambda parameter
   * @param[in] k_max parameter
   */
  static std::shared_ptr<APO> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                     const double eps = 1e-8, const double lambda = 1.0, const size_t k_max = 200) {
    return std::make_shared<APO>(std::move(backend), foci, amps, eps, lambda, k_max);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->apo(geometry, this->_foci, this->_amps, _eps, _lambda, _k_max, this->_data);
    this->_built = true;
  }

  APO(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps, const double lambda,
      const size_t k_max)
      : Holo(std::move(backend), foci, amps), _eps(eps), _lambda(lambda), _k_max(k_max) {}

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
class Greedy final : public Holo {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] phase_div resolution of the phase to be searched
   */
  static std::shared_ptr<Greedy> create(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps,
                                        const size_t phase_div = 16) {
    return std::make_shared<Greedy>(std::move(backend), foci, amps, phase_div);
  }

  void calc(const core::GeometryPtr& geometry) {
    this->_backend->greedy(geometry, this->_foci, this->_amps, _phase_div, this->_data);
    this->_built = true;
  }

  Greedy(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t phase_div)
      : Holo(std::move(backend), foci, amps), _phase_div(phase_div) {}

 private:
  size_t _phase_div;
};

}  // namespace autd::gain::holo
