// File: holo_gain.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/12/2021
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
  Holo(const Holo& v) noexcept = delete;
  Holo& operator=(const Holo& obj) = delete;
  Holo(Holo&& obj) = default;
  Holo& operator=(Holo&& obj) = default;

  [[nodiscard]] const std::vector<core::Vector3>& foci() const { return this->_foci; }
  [[nodiscard]] const std::vector<complex>& amplitudes() const { return this->_amps; }

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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] alpha parameter
   * @param[in] lambda parameter
   * @param[in] repeat parameter
   * @param[in] normalize parameter
   */
  explicit SDP(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double alpha = 1e-3,
               const double lambda = 0.9, const size_t repeat = 100, const bool normalize = true)
      : Holo(std::move(backend), foci, amps), _alpha(alpha), _lambda(lambda), _repeat(repeat), _normalize(normalize) {}

  void calc(const core::Geometry& geometry) override;

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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] gamma parameter
   * @param[in] normalize parameter
   */
  EVD(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double gamma = 1.0,
      const bool normalize = true)
      : Holo(std::move(backend), foci, amps), _gamma(gamma), _normalize(normalize) {}

  void calc(const core::Geometry& geometry) override;

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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   */
  Naive(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps) : Holo(std::move(backend), foci, amps) {}

  void calc(const core::Geometry& geometry) override;
};

/**
 * @brief Gain to produce multiple focal points with GS method.
 * Refer to Asier Marzo and Bruce W Drinkwater, "Holographic acoustic
 * tweezers," Proceedings of theNational Academy of Sciences, 116(1):84–89, 2019.
 */
class GS final : public Holo {
 public:
  /**
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  GS(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat = 100)
      : Holo(std::move(backend), foci, amps), _repeat(repeat) {}

  void calc(const core::Geometry& geometry) override;

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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] repeat parameter
   */
  GSPAT(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t repeat = 100)
      : Holo(std::move(backend), foci, amps), _repeat(repeat) {}

  void calc(const core::Geometry& geometry) override;

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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps_1 parameter
   * @param[in] eps_2 parameter
   * @param[in] tau parameter
   * @param[in] k_max parameter
   * @param[in] initial initial phase of transducers
   */
  LM(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1 = 1e-8,
     const double eps_2 = 1e-8, const double tau = 1e-3, const size_t k_max = 5, std::vector<double> initial = {})
      : Holo(std::move(backend), foci, amps), _eps_1(eps_1), _eps_2(eps_2), _tau(tau), _k_max(k_max), _initial(std::move(initial)) {}

  void calc(const core::Geometry& geometry) override;

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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps_1 parameter
   * @param[in] eps_2 parameter
   * @param[in] k_max parameter
   * @param[in] initial initial phase of transducers
   */
  GaussNewton(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps_1 = 1e-6,
              const double eps_2 = 1e-6, const size_t k_max = 500, std::vector<double> initial = {})
      : Holo(std::move(backend), foci, amps), _eps_1(eps_1), _eps_2(eps_2), _k_max(k_max), _initial(std::move(initial)) {}

  void calc(const core::Geometry& geometry) override;

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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps parameter
   * @param[in] step parameter
   * @param[in] k_max parameter
   * @param[in] initial initial phase of transducers
   */
  GradientDescent(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps = 1e-6,
                  const double step = 0.5, const size_t k_max = 2000, std::vector<double> initial = {})
      : Holo(std::move(backend), foci, amps), _eps(eps), _k_max(k_max), _step(step), _initial(std::move(initial)) {}

  void calc(const core::Geometry& geometry) override;

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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] eps parameter
   * @param[in] lambda parameter
   * @param[in] k_max parameter
   */
  APO(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const double eps = 1e-8, const double lambda = 1.0,
      const size_t k_max = 200)
      : Holo(std::move(backend), foci, amps), _eps(eps), _lambda(lambda), _k_max(k_max) {}

  void calc(const core::Geometry& geometry) override;

 private:
  double _eps;
  double _lambda;
  size_t _k_max;
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
   * @param[in] backend pointer to Backend
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] phase_div resolution of the phase to be searched
   */
  Greedy(BackendPtr backend, const std::vector<core::Vector3>& foci, const std::vector<double>& amps, const size_t phase_div = 16)
      : Holo(std::move(backend), foci, amps), _phase_div(phase_div) {}

  void calc(const core::Geometry& geometry) override;

 private:
  size_t _phase_div;
};

}  // namespace autd::gain::holo
