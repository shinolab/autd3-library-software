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
#include <string>
#include <utility>
#include <vector>

#include "gain.hpp"

namespace autd::gain {
/**
 * @brief Optimization method for generating multiple foci.
 */
enum class OPT_METHOD {
  //! Inoue, Seki, Yasutoshi Makino, and Hiroyuki Shinoda. "Active touch perception produced by airborne ultrasonic haptic hologram." 2015 IEEE World
  //! Haptics Conference (WHC). IEEE, 2015.
  SDP = 0,
  //! Long, Benjamin, et al. "Rendering volumetric haptic shapes in mid-air using ultrasound." ACM Transactions on Graphics (TOG) 33.6 (2014): 1-10.
  EVD = 1,
  //! Asier Marzo and Bruce W Drinkwater. Holographic acoustic tweezers.Proceedings of theNational Academy of Sciences, 116(1):84–89, 2019.
  GS = 2,
  //! Diego Martinez Plasencia et al. "Gs-pat: high-speed multi-point sound-fields for phased arrays of transducers," ACMTrans-actions on Graphics
  //! (TOG), 39(4):138–1, 2020.
  //! Not yet been implemented with GPU.
  GS_PAT = 3,
  //! Naive linear synthesis method.
  NAIVE = 4,
  //! K.Levenberg, “A method for the solution of certain non-linear problems in least squares,” Quarterly of applied mathematics, vol.2, no.2,
  //! pp.164–168, 1944.
  //! D.W.Marquardt, “An algorithm for least-squares estimation of non-linear parameters,” Journal of the society for Industrial and
  //! AppliedMathematics, vol.11, no.2, pp.431–441, 1963.
  //! K.Madsen, H.Nielsen, and O.Tingleff, “Methods for non-linear least squares problems (2nd ed.),” 2004.
  LM = 5
};

enum class BACKEND {
  Eigen = 0,
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

/**
 * @brief Gain to produce multiple focal points
 */
class HoloGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] backend backend of optimization. see also @ref BACKEND
   * @param[in] method optimization method. see also @ref OPT_METHOD
   * @param[in] params pointer to optimization parameters
   */
  static std::shared_ptr<HoloGain> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, BACKEND backend = BACKEND::Eigen,
                                          OPT_METHOD method = OPT_METHOD::SDP, void* params = nullptr);
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] method optimization method. see also @ref OptMethod
   * @param[in] params pointer to optimization parameters
   */
  static std::shared_ptr<HoloGain> Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, OPT_METHOD method = OPT_METHOD::SDP,
                                          void* params = nullptr);

  /**
   * @brief Generate function
   * @param[in] backend backend of optimization. see also @ref BACKEND
   * @param[in] method optimization method. see also @ref OptMethod
   * @param[in] params pointer to optimization parameters
   */
  static std::shared_ptr<HoloGain> Create(BACKEND backend = BACKEND::Eigen, OPT_METHOD method = OPT_METHOD::SDP, void* params = nullptr);

  void Build() override;
  HoloGain(std::vector<Vector3> foci, std::vector<Float> amps, const BACKEND backend = BACKEND::Eigen, const OPT_METHOD method = OPT_METHOD::SDP,
           void* params = nullptr)
      : Gain(), _foci(std::move(foci)), _amps(std::move(amps)), _backend(backend), _method(method), _params(params) {}
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
  BACKEND _backend = BACKEND::Eigen;
  OPT_METHOD _method = OPT_METHOD::SDP;
  void* _params = nullptr;
};
}  // namespace autd::gain
