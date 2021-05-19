// File: primitive_gain.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "core/gain.hpp"
#include "core/hardware_defined.hpp"
#include "core/result.hpp"

namespace autd::gain {

using core::Gain;
using core::GainPtr;
using NullGain = Gain;

using core::DataArray;
using core::Vector3;

/**
 * @brief Gain to group some gains
 */
class Grouped final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] gain_map ｍap from group ID to gain
   * @details group ID must be specified in Geometry::AddDevice() in advance
   */
  static GainPtr create(const std::map<size_t, GainPtr>& gain_map);

  Error calc(const core::GeometryPtr& geometry) override;
  explicit Grouped(std::map<size_t, GainPtr> gain_map) : Gain(), _gain_map(std::move(gain_map)) {}
  ~Grouped() override = default;
  Grouped(const Grouped& v) noexcept = default;
  Grouped& operator=(const Grouped& obj) = default;
  Grouped(Grouped&& obj) = default;
  Grouped& operator=(Grouped&& obj) = default;

 private:
  std::map<size_t, GainPtr> _gain_map;
};

/**
 * @brief Gain to create plane wave
 */
class PlaneWave final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr create(const Vector3& direction, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr create(const Vector3& direction, double amp);

  Error calc(const core::GeometryPtr& geometry) override;
  explicit PlaneWave(Vector3 direction, const uint8_t duty) : Gain(), _direction(std::move(direction)), _duty(duty) {}
  ~PlaneWave() override = default;
  PlaneWave(const PlaneWave& v) noexcept = default;
  PlaneWave& operator=(const PlaneWave& obj) = default;
  PlaneWave(PlaneWave&& obj) = default;
  PlaneWave& operator=(PlaneWave&& obj) = default;

 private:
  Vector3 _direction = Vector3::UnitZ();
  uint8_t _duty = 0xFF;
};

/**
 * @brief Gain to produce single focal point
 */
class FocalPoint final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr create(const Vector3& point, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr create(const Vector3& point, double amp);

  Error calc(const core::GeometryPtr& geometry) override;
  explicit FocalPoint(Vector3 point, const uint8_t duty) : Gain(), _point(std::move(point)), _duty(duty) {}
  ~FocalPoint() override = default;
  FocalPoint(const FocalPoint& v) noexcept = default;
  FocalPoint& operator=(const FocalPoint& obj) = default;
  FocalPoint(FocalPoint&& obj) = default;
  FocalPoint& operator=(FocalPoint&& obj) = default;

 private:
  Vector3 _point = Vector3::Zero();
  uint8_t _duty = 0xff;
};

/**
 * @brief Gain to produce Bessel Beam
 */
class BesselBeam final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr create(const Vector3& point, const Vector3& vec_n, double theta_z, uint8_t duty = 0xff);
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr create(const Vector3& point, const Vector3& vec_n, double theta_z, double amp);

  Error calc(const core::GeometryPtr& geometry) override;
  explicit BesselBeam(Vector3 point, Vector3 vec_n, const double theta_z, const uint8_t duty)
      : Gain(), _point(std::move(point)), _vec_n(std::move(vec_n)), _theta_z(theta_z), _duty(duty) {}
  ~BesselBeam() override = default;
  BesselBeam(const BesselBeam& v) noexcept = default;
  BesselBeam& operator=(const BesselBeam& obj) = default;
  BesselBeam(BesselBeam&& obj) = default;
  BesselBeam& operator=(BesselBeam&& obj) = default;

 private:
  Vector3 _point = Vector3::Zero();
  Vector3 _vec_n = Vector3::UnitZ();
  double _theta_z = 0;
  uint8_t _duty = 0xff;
};

/**
 * @brief Gain that can set the phase and amplitude freely
 */
class Custom final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] data data of amplitude and phase of each transducer
   * @details The data size should be the same as the number of devices you use. The data is 16 bit unsigned integer, where MSB represents
   * amplitude and LSB represents phase
   */
  static GainPtr create(const std::vector<DataArray>& data);
  /**
   * @brief Generate function
   * @param[in] data pointer to data of amplitude and phase of each transducer
   * @param[in] data_length length of the data
   * @details The data length should be the same as the number of transducers you use. The data is 16 bit unsigned integer, where MSB represents
   * amplitude and LSB represents phase
   */
  static GainPtr create(const uint16_t* data, size_t data_length);
  Error calc(const core::GeometryPtr& geometry) override;
  explicit Custom(std::vector<DataArray> data) : Gain() { this->_data = std::move(data); }
  ~Custom() override = default;
  Custom(const Custom& v) noexcept = default;
  Custom& operator=(const Custom& obj) = default;
  Custom(Custom&& obj) = default;
  Custom& operator=(Custom&& obj) = default;
};

/**
 * @brief Gain to test a transducer
 */
class TransducerTest final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] transducer_index index of the transducer
   * @param[in] duty duty ratio of driving signal
   * @param[in] phase phase of the phase
   */
  static GainPtr create(size_t transducer_index, uint8_t duty, uint8_t phase);
  Error calc(const core::GeometryPtr& geometry) override;
  TransducerTest(const size_t transducer_index, const uint8_t duty, const uint8_t phase)
      : Gain(), _transducer_idx(transducer_index), _duty(duty), _phase(phase) {}
  ~TransducerTest() override = default;
  TransducerTest(const TransducerTest& v) noexcept = default;
  TransducerTest& operator=(const TransducerTest& obj) = default;
  TransducerTest(TransducerTest&& obj) = default;
  TransducerTest& operator=(TransducerTest&& obj) = default;

 protected:
  size_t _transducer_idx = 0;
  uint8_t _duty = 0;
  uint8_t _phase = 0;
};
}  // namespace autd::gain
