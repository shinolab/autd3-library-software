// File: gain.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 26/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#define _USE_MATH_DEFINES  // NOLINT
#include <math.h>

#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "core.hpp"
#include "geometry.hpp"
#include "vector3.hpp"

namespace autd {
namespace gain {

inline uint8_t AdjustAmp(const double amp) noexcept {
  const auto d = asin(amp) / M_PI;  //  duty (0 ~ 0.5)
  return static_cast<uint8_t>(511 * d);
}

inline void CheckAndInit(const GeometryPtr& geometry, std::vector<AUTDDataArray>* data) {
  assert(geometry != nullptr);

  data->clear();

  const auto num_device = geometry->num_devices();
  data->resize(num_device);
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
   * @brief Calculate amplitude and phase of each transducer
   */
  virtual void Build();
  /**
   * @brief Set AUTD Geometry which is required to build gain
   */
  void SetGeometry(const GeometryPtr& geometry) noexcept;
  /**
   * @brief Get AUTD Geometry
   */
  [[nodiscard]] GeometryPtr geometry() const noexcept;
  /**
   * @brief Getter function for the data of amplitude and phase of each transducers
   * @details Each data is 16 bit unsigned integer, where MSB represents amplitude and LSB represents phase
   */
  std::vector<AUTDDataArray>& data();

  Gain() noexcept;
  virtual ~Gain() = default;
  Gain(const Gain& v) noexcept = default;
  Gain& operator=(const Gain& obj) = default;
  Gain(Gain&& obj) = default;
  Gain& operator=(Gain&& obj) = default;

 protected:
  bool _built;
  GeometryPtr _geometry;
  std::vector<AUTDDataArray> _data;
  [[nodiscard]] bool built() const noexcept;
};

/**
 * @brief Gain to create plane wave
 */
class PlaneWaveGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(const utils::Vector3& direction, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const utils::Vector3& direction, double amp);
#ifdef USE_EIGEN_AUTD
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(const Vector3& direction, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const Vector3& direction, double amp);
#endif

  void Build() override;
  explicit PlaneWaveGain(Vector3 direction, const uint8_t duty) : Gain(), _direction(std::move(direction)), _duty(duty) {}
  ~PlaneWaveGain() override = default;
  PlaneWaveGain(const PlaneWaveGain& v) noexcept = default;
  PlaneWaveGain& operator=(const PlaneWaveGain& obj) = default;
  PlaneWaveGain(PlaneWaveGain&& obj) = default;
  PlaneWaveGain& operator=(PlaneWaveGain&& obj) = default;

 private:
  Vector3 _direction = Vector3::UnitZ();
  uint8_t _duty = 0xFF;
};

/**
 * @brief Gain to produce single focal point
 */
class FocalPointGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(const utils::Vector3& point, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const utils::Vector3& point, double amp);

#ifdef USE_EIGEN_AUTD
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(const Vector3& point, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const Vector3& point, double amp);
#endif

  void Build() override;
  explicit FocalPointGain(Vector3 point, const uint8_t duty) : Gain(), _point(std::move(point)), _duty(duty) {}
  ~FocalPointGain() override = default;
  FocalPointGain(const FocalPointGain& v) noexcept = default;
  FocalPointGain& operator=(const FocalPointGain& obj) = default;
  FocalPointGain(FocalPointGain&& obj) = default;
  FocalPointGain& operator=(FocalPointGain&& obj) = default;

 private:
  Vector3 _point = Vector3::Zero();
  uint8_t _duty = 0xff;
};

/**
 * @brief Gain to produce Bessel Beam
 */
class BesselBeamGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(const utils::Vector3& point, const utils::Vector3& vec_n, double theta_z, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const utils::Vector3& point, const utils::Vector3& vec_n, double theta_z, double amp);

#ifdef USE_EIGEN_AUTD
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr Create(const Vector3& point, const Vector3& vec_n, double theta_z, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const Vector3& point, const Vector3& vec_n, double theta_z, double amp);
#endif

  void Build() override;
  explicit BesselBeamGain(Vector3 point, Vector3 vec_n, const double theta_z, const uint8_t duty)
      : Gain(), _point(std::move(point)), _vec_n(std::move(vec_n)), _theta_z(theta_z), _duty(duty) {}
  ~BesselBeamGain() override = default;
  BesselBeamGain(const BesselBeamGain& v) noexcept = default;
  BesselBeamGain& operator=(const BesselBeamGain& obj) = default;
  BesselBeamGain(BesselBeamGain&& obj) = default;
  BesselBeamGain& operator=(BesselBeamGain&& obj) = default;

 private:
  Vector3 _point = Vector3::Zero();
  Vector3 _vec_n = Vector3::UnitZ();
  double _theta_z = 0;
  uint8_t _duty = 0xff;
};

/**
 * @brief Gain that can set the phase and amplitude freely
 */
class CustomGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] data pointer to data of amplitude and phase of each transducer
   * @param[in] data_length length of the data
   * @details The data length should be the same as the number of transducers you use. The data is 16 bit unsigned integer, where MSB represents
   * amplitude and LSB represents phase
   */
  static GainPtr Create(const uint16_t* data, size_t data_length);
  void Build() override;
  explicit CustomGain(std::vector<uint16_t> raw_data) : Gain(), _raw_data(std::move(raw_data)) {}
  ~CustomGain() override = default;
  CustomGain(const CustomGain& v) noexcept = default;
  CustomGain& operator=(const CustomGain& obj) = default;
  CustomGain(CustomGain&& obj) = default;
  CustomGain& operator=(CustomGain&& obj) = default;

 private:
  std::vector<uint16_t> _raw_data;
};

/**
 * @brief Gain to group some gains
 */
class GroupedGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] gain_map ｍap from group ID to gain
   * @details group ID must be specified in Geometry::AddDevice() in advance
   */
  static GainPtr Create(const std::map<size_t, GainPtr>& gain_map);
  void Build() override;
  explicit GroupedGain(std::map<size_t, GainPtr> gain_map) : Gain(), _gain_map(std::move(gain_map)) {}
  ~GroupedGain() override = default;
  GroupedGain(const GroupedGain& v) noexcept = default;
  GroupedGain& operator=(const GroupedGain& obj) = default;
  GroupedGain(GroupedGain&& obj) = default;
  GroupedGain& operator=(GroupedGain&& obj) = default;

 private:
  std::map<size_t, GainPtr> _gain_map;
};

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
class HoloGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] method optimization method. see also @ref OptMethod
   * @param[in] params pointer to optimization parameters
   */
  static GainPtr Create(const std::vector<utils::Vector3>& foci, const std::vector<double>& amps, OPT_METHOD method = OPT_METHOD::SDP,
                        void* params = nullptr);
#ifdef USE_EIGEN_AUTD
  /**
   * @brief Generate function
   * @param[in] foci focal points
   * @param[in] amps amplitudes of the foci
   * @param[in] method optimization method. see also @ref OptMethod
   * @param[in] params pointer to optimization parameters
   */
  static GainPtr Create(const std::vector<Vector3>& foci, const std::vector<double>& amps, OPT_METHOD method = OPT_METHOD::SDP,
                        void* params = nullptr);
#endif

  void Build() override;
  HoloGain(std::vector<Vector3> foci, std::vector<double> amps, const OPT_METHOD method = OPT_METHOD::SDP, void* params = nullptr)
      : Gain(), _foci(std::move(foci)), _amps(std::move(amps)), _method(method), _params(params) {}
  ~HoloGain() override = default;
  HoloGain(const HoloGain& v) noexcept = default;
  HoloGain& operator=(const HoloGain& obj) = default;
  HoloGain(HoloGain&& obj) = default;
  HoloGain& operator=(HoloGain&& obj) = default;

 protected:
  std::vector<Vector3> _foci;
  std::vector<double> _amps;
  OPT_METHOD _method = OPT_METHOD::SDP;
  void* _params = nullptr;
};

/**
 * @brief Gain created from Matlab mat file
 */
class MatlabGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] filename mat file path
   * @param[in] var_name variable name in mat file
   */
  static GainPtr Create(const std::string& filename, const std::string& var_name);
  void Build() override;
  MatlabGain(std::string filename, std::string var_name) : Gain(), _filename(std::move(filename)), _var_name(std::move(var_name)) {}
  ~MatlabGain() override = default;
  MatlabGain(const MatlabGain& v) noexcept = default;
  MatlabGain& operator=(const MatlabGain& obj) = default;
  MatlabGain(MatlabGain&& obj) = default;
  MatlabGain& operator=(MatlabGain&& obj) = default;

 protected:
  std::string _filename, _var_name;
};

/**
 * @brief Gain to test a transducer
 */
class TransducerTestGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] transducer_index index of the transducer
   * @param[in] duty duty ratio of driving signal
   * @param[in] phase phase of the phase
   */
  static GainPtr Create(size_t transducer_index, uint8_t duty, uint8_t phase);
  void Build() override;
  TransducerTestGain(const size_t transducer_index, const uint8_t duty, const uint8_t phase)
      : Gain(), _transducer_idx(transducer_index), _duty(duty), _phase(phase) {}
  ~TransducerTestGain() override = default;
  TransducerTestGain(const TransducerTestGain& v) noexcept = default;
  TransducerTestGain& operator=(const TransducerTestGain& obj) = default;
  TransducerTestGain(TransducerTestGain&& obj) = default;
  TransducerTestGain& operator=(TransducerTestGain&& obj) = default;

 protected:
  size_t _transducer_idx = 0;
  uint8_t _duty = 0;
  uint8_t _phase = 0;
};
}  // namespace gain
}  // namespace autd
