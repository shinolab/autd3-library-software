// File: primitive_gain.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <map>
#include <memory>
#include <utility>

#include "autd3/core/gain.hpp"
#include "autd3/core/hardware_defined.hpp"

namespace autd::gain {

using core::Gain;
using core::GainPtr;
using Null = Gain;

using core::DataArray;
using core::Vector3;

/**
 * @brief Gain to group some gains
 */
class Grouped final : public Gain {
 public:
  /**
   * @brief Generate function
   */
  static std::shared_ptr<Grouped> create();

  /**
   * \brief Decide which device outputs which Gain
   * \param device_id device id
   * \param gain gain
   */
  void add(size_t device_id, const GainPtr& gain);

  void calc(const core::Geometry& geometry) override;
  Grouped() : Gain() {}
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

  void calc(const core::Geometry& geometry) override;
  explicit PlaneWave(Vector3 direction, const uint8_t duty) : Gain(), _direction(std::move(direction)), _duty(duty) {}
  ~PlaneWave() override = default;
  PlaneWave(const PlaneWave& v) noexcept = default;
  PlaneWave& operator=(const PlaneWave& obj) = default;
  PlaneWave(PlaneWave&& obj) = default;
  PlaneWave& operator=(PlaneWave&& obj) = default;

 private:
  Vector3 _direction;
  uint8_t _duty;
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

  void calc(const core::Geometry& geometry) override;
  explicit FocalPoint(Vector3 point, const uint8_t duty) : Gain(), _point(std::move(point)), _duty(duty) {}
  ~FocalPoint() override = default;
  FocalPoint(const FocalPoint& v) noexcept = default;
  FocalPoint& operator=(const FocalPoint& obj) = default;
  FocalPoint(FocalPoint&& obj) = default;
  FocalPoint& operator=(FocalPoint&& obj) = default;

 private:
  Vector3 _point;
  uint8_t _duty;
};

/**
 * @brief Gain to produce Bessel Beam
 */
class BesselBeam final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] apex apex of the conical wavefront of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the side of the cone and the plane perpendicular to direction of the beam
   * @param[in] duty duty ratio of driving signal
   */
  static GainPtr create(const Vector3& apex, const Vector3& vec_n, double theta_z, uint8_t duty = 0xff);

  /**
   * @brief Generate function
   * @param[in] apex apex of the conical wavefront of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  static GainPtr create(const Vector3& apex, const Vector3& vec_n, double theta_z, double amp);

  void calc(const core::Geometry& geometry) override;
  explicit BesselBeam(Vector3 apex, Vector3 vec_n, const double theta_z, const uint8_t duty)
      : Gain(), _apex(std::move(apex)), _vec_n(std::move(vec_n)), _theta_z(theta_z), _duty(duty) {}
  ~BesselBeam() override = default;
  BesselBeam(const BesselBeam& v) noexcept = default;
  BesselBeam& operator=(const BesselBeam& obj) = default;
  BesselBeam(BesselBeam&& obj) = default;
  BesselBeam& operator=(BesselBeam&& obj) = default;

 private:
  Vector3 _apex;
  Vector3 _vec_n;
  double _theta_z;
  uint8_t _duty;
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
  void calc(const core::Geometry& geometry) override;
  TransducerTest(const size_t transducer_index, const uint8_t duty, const uint8_t phase)
      : Gain(), _transducer_idx(transducer_index), _duty(duty), _phase(phase) {}
  ~TransducerTest() override = default;
  TransducerTest(const TransducerTest& v) noexcept = default;
  TransducerTest& operator=(const TransducerTest& obj) = default;
  TransducerTest(TransducerTest&& obj) = default;
  TransducerTest& operator=(TransducerTest&& obj) = default;

 protected:
  size_t _transducer_idx;
  uint8_t _duty;
  uint8_t _phase;
};
}  // namespace autd::gain
