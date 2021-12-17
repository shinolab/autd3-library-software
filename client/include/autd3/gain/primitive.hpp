// File: primitive_gain.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 15/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "autd3/core/gain.hpp"
#include "autd3/core/type_traits.hpp"
#include "autd3/core/utils.hpp"

namespace autd::gain {

using core::Gain;
using core::Vector3;

/**
 * @brief Gain to produce nothing
 */
class Null final : public Gain {
 public:
  void calc(const core::Geometry& geometry) override {
    for (const auto& device : geometry)
      for (const auto& transducer : device) this->_data[transducer.id()] = core::Drive(0x00, 0x00);
  }

  Null() : Gain() {}
  ~Null() override = default;
  Null(const Null& v) noexcept = delete;
  Null& operator=(const Null& obj) = delete;
  Null(Null&& obj) = default;
  Null& operator=(Null&& obj) = default;
};

/**
 * @brief Gain to group some gains
 */
class Grouped final : public Gain {
 public:
  /**
   * \brief Decide which device outputs which Gain
   * \param device_id device id
   * \param gain gain
   */
  template <class T>
  std::enable_if_t<core::type_traits::is_gain_v<T>> add(const size_t device_id, T&& gain) {
    Gain& g = core::type_traits::to_gain(gain);

    g.build(_geometry);

    if (device_id < _geometry.num_devices())
      std::memcpy(&this->_tmp_data[device_id * core::NUM_TRANS_IN_UNIT], &g.data()[device_id * core::NUM_TRANS_IN_UNIT],
                  core::NUM_TRANS_IN_UNIT * sizeof(core::Drive));
  }

  void calc(const core::Geometry& geometry) override {
    std::memcpy(this->_data.data(), this->_tmp_data.data(), this->_data.size() * sizeof(core::Drive));
  }

  explicit Grouped(const core::Geometry& geometry) : Gain(), _geometry(geometry) { this->_tmp_data.resize(_geometry.num_transducers()); }
  ~Grouped() override = default;
  Grouped(const Grouped& v) noexcept = delete;
  Grouped& operator=(const Grouped& obj) = delete;
  Grouped(Grouped&& obj) = default;
  Grouped& operator=(Grouped&& obj) = delete;

 private:
  const core::Geometry& _geometry;
  std::vector<core::Drive> _tmp_data;
};

/**
 * @brief Gain to create plane wave
 */
class PlaneWave final : public Gain {
 public:
  /**
   * @param[in] direction wave direction
   * @param[in] duty duty ratio of driving signal
   */
  explicit PlaneWave(Vector3 direction, const uint8_t duty = 0xFF) : Gain(), _direction(std::move(direction)), _duty(duty) {}

  /**
   * @param[in] direction wave direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  explicit PlaneWave(Vector3 direction, const double amp) : PlaneWave(std::move(direction), core::utils::to_duty(amp)) {}

  void calc(const core::Geometry& geometry) override;

  ~PlaneWave() override = default;
  PlaneWave(const PlaneWave& v) noexcept = delete;
  PlaneWave& operator=(const PlaneWave& obj) = delete;
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
   * @param[in] point focal point
   * @param[in] duty duty ratio of driving signal
   */
  explicit FocalPoint(Vector3 point, const uint8_t duty = 0xFF) : Gain(), _point(std::move(point)), _duty(duty) {}
  /**
   * @param[in] point focal point
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  explicit FocalPoint(Vector3 point, const double amp) : FocalPoint(std::move(point), core::utils::to_duty(amp)) {}

  void calc(const core::Geometry& geometry) override;

  ~FocalPoint() override = default;
  FocalPoint(const FocalPoint& v) noexcept = delete;
  FocalPoint& operator=(const FocalPoint& obj) = delete;
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
   * @param[in] apex apex of the conical wavefront of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the side of the cone and the plane perpendicular to direction of the beam
   * @param[in] duty duty ratio of driving signal
   */
  explicit BesselBeam(Vector3 apex, Vector3 vec_n, const double theta_z, const uint8_t duty = 0xFF)
      : Gain(), _apex(std::move(apex)), _vec_n(std::move(vec_n)), _theta_z(theta_z), _duty(duty) {}

  /**
   * @param[in] apex apex of the conical wavefront of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the wave (from 0.0 to 1.0)
   */
  explicit BesselBeam(Vector3 apex, Vector3 vec_n, const double theta_z, const double amp)
      : BesselBeam(std::move(apex), std::move(vec_n), theta_z, core::utils::to_duty(amp)) {}

  void calc(const core::Geometry& geometry) override;

  ~BesselBeam() override = default;
  BesselBeam(const BesselBeam& v) noexcept = delete;
  BesselBeam& operator=(const BesselBeam& obj) = delete;
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
   * @param[in] transducer_index index of the transducer
   * @param[in] duty duty ratio of driving signal
   * @param[in] phase phase of the phase
   */
  TransducerTest(const size_t transducer_index, const uint8_t duty, const uint8_t phase)
      : Gain(), _transducer_idx(transducer_index), _duty(duty), _phase(phase) {}

  void calc(const core::Geometry& geometry) override;

  ~TransducerTest() override = default;
  TransducerTest(const TransducerTest& v) noexcept = delete;
  TransducerTest& operator=(const TransducerTest& obj) = delete;
  TransducerTest(TransducerTest&& obj) = default;
  TransducerTest& operator=(TransducerTest&& obj) = default;

 protected:
  size_t _transducer_idx;
  uint8_t _duty;
  uint8_t _phase;
};
}  // namespace autd::gain
