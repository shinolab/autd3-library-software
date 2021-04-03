// File: gain.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "autd_types.hpp"
#include "consts.hpp"
#include "geometry.hpp"
#include "linalg.hpp"
#include "result.hpp"

namespace autd {
namespace gain {
class Gain;
}
using GainPtr = std::shared_ptr<gain::Gain>;
}  // namespace autd

namespace autd::gain {

inline Float PosMod(const Float a, const Float b) { return a - floor(a / b) * b; }

inline uint8_t ToDuty(const Float amp) noexcept {
  const auto d = asin(amp) / PI;  //  duty (0 ~ 0.5)
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
  virtual Result<bool, std::string> Build();
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
  explicit Gain(std::vector<AUTDDataArray> data) noexcept;
  bool _built;
  GeometryPtr _geometry;
  std::vector<AUTDDataArray> _data;
  [[nodiscard]] bool built() const noexcept;
};

using NullGain = Gain;

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
  Result<bool, std::string> Build() override;
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
 * @brief Gain to create plane wave
 */
class PlaneWaveGain final : public Gain {
 public:
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
  static GainPtr Create(const Vector3& direction, Float amp);

  Result<bool, std::string> Build() override;
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
  static GainPtr Create(const Vector3& point, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const Vector3& point, Float amp);

  Result<bool, std::string> Build() override;
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
  static GainPtr Create(const Vector3& point, const Vector3& vec_n, Float theta_z, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const Vector3& point, const Vector3& vec_n, Float theta_z, Float amp);

  Result<bool, std::string> Build() override;
  explicit BesselBeamGain(Vector3 point, Vector3 vec_n, const Float theta_z, const uint8_t duty)
      : Gain(), _point(std::move(point)), _vec_n(std::move(vec_n)), _theta_z(theta_z), _duty(duty) {}
  ~BesselBeamGain() override = default;
  BesselBeamGain(const BesselBeamGain& v) noexcept = default;
  BesselBeamGain& operator=(const BesselBeamGain& obj) = default;
  BesselBeamGain(BesselBeamGain&& obj) = default;
  BesselBeamGain& operator=(BesselBeamGain&& obj) = default;

 private:
  Vector3 _point = Vector3::Zero();
  Vector3 _vec_n = Vector3::UnitZ();
  Float _theta_z = 0;
  uint8_t _duty = 0xff;
};

/**
 * @brief Gain that can set the phase and amplitude freely
 */
class CustomGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] data data of amplitude and phase of each transducer
   * @details The data size should be the same as the number of devices you use. The data is 16 bit unsigned integer, where MSB represents
   * amplitude and LSB represents phase
   */
  static GainPtr Create(const std::vector<AUTDDataArray>& data);
  /**
   * @brief Generate function
   * @param[in] data pointer to data of amplitude and phase of each transducer
   * @param[in] data_length length of the data
   * @details The data length should be the same as the number of transducers you use. The data is 16 bit unsigned integer, where MSB represents
   * amplitude and LSB represents phase
   */
  static GainPtr Create(const uint16_t* data, size_t data_length);
  Result<bool, std::string> Build() override;
  explicit CustomGain(std::vector<AUTDDataArray> data) : Gain(std::move(data)) {}
  ~CustomGain() override = default;
  CustomGain(const CustomGain& v) noexcept = default;
  CustomGain& operator=(const CustomGain& obj) = default;
  CustomGain(CustomGain&& obj) = default;
  CustomGain& operator=(CustomGain&& obj) = default;
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
  Result<bool, std::string> Build() override;
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
}  // namespace autd::gain
