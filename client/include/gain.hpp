// File: gain.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 24/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "consts.hpp"
#include "core.hpp"
#include "geometry.hpp"
#include "vector3.hpp"

namespace autd {
namespace gain {

inline uint8_t AdjustAmp(double amp) noexcept {
  auto d = asin(amp) / M_PI;  //  duty (0 ~ 0.5)
  return static_cast<uint8_t>(511 * d);
}

inline void CheckAndInit(GeometryPtr geometry, std::vector<std::vector<uint16_t>> *data) {
  assert(geometry != nullptr);

  data->clear();

  const auto ndevice = geometry->numDevices();
  data->resize(ndevice);
  for (size_t i = 0; i < ndevice; i++) {
    data->at(i).resize(NUM_TRANS_IN_UNIT);
  }
}

/**
 * @brief Gain controls the amplitude and phase of each transducer in the AUTD
 */
class Gain {
 public:
  /**
   * @brief Generate empty gain
   */
  static GainPtr Create();
  /**
   * @brief Calcurate amplitude and phase of each transducer
   */
  virtual void Build();
  /**
   * @brief Set AUTD Geometry which is required to build gain
   */
  void SetGeometry(const GeometryPtr &geometry) noexcept;
  /**
   * @brief Get AUTD Geometry
   */
  GeometryPtr geometry() noexcept;
  /**
   * @brief Getter function for the data of amplitude and phase of each transducers
   * @details Each data is 16 bit unsigned integer, where MSB represents amplitude and LSB represents phase
   */
  std::vector<std::vector<uint16_t>> &data();

  Gain() noexcept;

 protected:
  bool _built;
  GeometryPtr _geometry;
  std::vector<std::vector<uint16_t>> _data;
  bool built() noexcept;
};

/**
 * @brief Gain to create plane wave
 */
class PlaneWaveGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(Vector3 direction, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(Vector3 direction, double amp);
  void Build() override;

 private:
  Vector3 _direction = Vector3::unit_z();
  uint8_t _duty = 0xFF;
};

/**
 * @brief Gain to produce single focal point
 */
class FocalPointGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(Vector3 point, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(Vector3 point, double amp);
  void Build() override;

 private:
  Vector3 _point = Vector3::zero();
  uint8_t _duty = 0xff;
};

/**
 * @brief Gain to produce Bessel Beam
 */
class BesselBeamGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(Vector3 point, Vector3 vec_n, double theta_z, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(Vector3 point, Vector3 vec_n, double theta_z, double amp);
  void Build() override;

 private:
  Vector3 _point = Vector3::zero();
  Vector3 _vec_n = Vector3::unit_z();
  double _theta_z = 0;
  uint8_t _duty = 0xff;
};

/**
 * @brief Gain that can set the phase and amplitude freely
 */
class CustomGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] data pointer to data of amplitude and phase of each transducer
   * @param[in] data_length length of the data
   * @details The data length should be the same as the number of transducers you use. The data is 16 bit unsigned integer, where MSB represents
   * amplitude and LSB represents phase
   */
  static GainPtr Create(uint16_t *data, size_t data_length);
  void Build() override;

 private:
  std::vector<uint16_t> _rawdata;
};

/**
 * @brief Gain to group some gains
 */
class GroupedGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] gainmap ｍap from group ID to gain
   * @details group ID must be specified in Geometry::AddDevice() in advance
   */
  static GainPtr Create(std::map<size_t, GainPtr> gainmap);
  void Build() override;

 private:
  std::map<size_t, GainPtr> _gainmap;
};

/**
 * @brief Optimization method for generating multiple foci.
 */
enum class OptMethod {
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

struct SDPParams {
  double regularization;
  int32_t repeat;
  double lambda;
  bool normalize_amp;
};

struct EVDParams {
  double regularization;
  bool normalize_amp;
};

struct NLSParams {
  double eps_1;
  double eps_2;
  int32_t k_max;
  double tau;
};

/**
 * @brief Gain to produce multiple focal points
 */
class HoloGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] method optimization method. see also @ref OptMethod
   * @param[in] params pointer to optimization parameters
   */
  static GainPtr Create(std::vector<Vector3> foci, std::vector<double> amps, OptMethod method = OptMethod::SDP, void *params = nullptr);
  void Build() override;

 protected:
  std::vector<Vector3> _foci;
  std::vector<double> _amps;
  OptMethod _method = OptMethod::SDP;
  void *_params = nullptr;
};

/**
 * @brief Gain created from Matlab mat file
 */
class MatlabGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] filename mat file path
   * @param[in] varname variable name in mat file
   */
  static GainPtr Create(std::string filename, std::string varname);
  void Build() override;

 protected:
  std::string _filename, _varname;
};

/**
 * @brief Gain to test a transducer
 */
class TransducerTestGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] transducer_index index of the transducer
   * @param[in] duty duty ratio of driving signal
   * @param[in] phase phase of the phase
   */
  static GainPtr Create(size_t transducer_index, uint8_t duty, uint8_t phase);
  void Build() override;

 protected:
  size_t _xdcr_idx = 0;
  uint8_t _duty = 0;
  uint8_t _phase = 0;
};
}  // namespace gain
}  // namespace autd
