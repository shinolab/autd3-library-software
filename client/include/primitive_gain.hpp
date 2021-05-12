// File: primitive_gain.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/gain.hpp"
#include "core/hardware_defined.hpp"
#include "core/result.hpp"

namespace autd::gain {

using core::Gain;
using core::GainPtr;
using NullGain = Gain;

using core::AUTDDataArray;
using core::Vector3;

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

  Result<bool, std::string> Calc(core::GeometryPtr geometry) override;
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
  static GainPtr Create(const Vector3& direction, double amp);

  Result<bool, std::string> Calc(core::GeometryPtr geometry) override;
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
  static GainPtr Create(const Vector3& point, double amp);

  Result<bool, std::string> Calc(core::GeometryPtr geometry) override;
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
  static GainPtr Create(const Vector3& point, const Vector3& vec_n, double theta_z, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr Create(const Vector3& point, const Vector3& vec_n, double theta_z, double amp);

  Result<bool, std::string> Calc(core::GeometryPtr geometry) override;
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
  Result<bool, std::string> Calc(core::GeometryPtr geometry) override;
  explicit CustomGain(std::vector<AUTDDataArray> data) : Gain() { this->_data = std::move(data); }
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
  Result<bool, std::string> Calc(core::GeometryPtr geometry) override;
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
