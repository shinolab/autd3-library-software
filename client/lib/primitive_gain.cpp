// File: primitive_gain.cpp
// Project: lib
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 21/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "primitive_gain.hpp"

#include <memory>
#include <vector>

namespace autd::gain {

using core::DataArray;
using core::NUM_TRANS_IN_UNIT;
using core::Vector3;

inline double PosMod(const double a, const double b) { return a - floor(a / b) * b; }

GainPtr Grouped::create() {
  GainPtr gain = std::make_shared<Grouped>();
  return gain;
}

void Grouped::add(const size_t id, const GainPtr& gain) { this->_gain_map[id] = gain; }

Error Grouped::calc(const core::GeometryPtr& geometry) {
  for (const auto& [_, g] : this->_gain_map)
    if (auto res = g->build(geometry); res.is_err()) return res;

  for (size_t i = 0; i < geometry->num_devices(); i++) {
    auto group_id = geometry->group_id_for_device_idx(i);
    this->_data[i] = _gain_map.count(group_id) ? _gain_map[group_id]->data()[i] : DataArray{0x0000};
  }
  this->_built = true;
  return Ok(true);
}

GainPtr PlaneWave::create(const Vector3& direction, const double amp) { return create(direction, to_duty(amp)); }

GainPtr PlaneWave::create(const Vector3& direction, uint8_t duty) { return std::make_shared<PlaneWave>(direction, duty); }

Error PlaneWave::calc(const core::GeometryPtr& geometry) {
  const auto dir = this->_direction.normalized();

  const auto ultrasound_wavelength = geometry->wavelength();
  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto trp = geometry->position(dev, i);
      const auto dist = trp.dot(dir);
      const auto f_phase = PosMod(dist, ultrasound_wavelength) / ultrasound_wavelength;
      const auto phase = static_cast<uint16_t>(std::round(255.0 * (1.0 - f_phase)));
      this->_data[dev][i] = duty | phase;
    }

  this->_built = true;
  return Ok(true);
}

GainPtr FocalPoint::create(const Vector3& point, const double amp) { return create(point, to_duty(amp)); }
GainPtr FocalPoint::create(const Vector3& point, uint8_t duty) { return std::make_shared<FocalPoint>(point, duty); }

Error FocalPoint::calc(const core::GeometryPtr& geometry) {
  const auto ultrasound_wavelength = geometry->wavelength();
  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto trp = geometry->position(dev, i);
      const auto dist = (trp - this->_point).norm();
      const auto f_phase = fmod(dist, ultrasound_wavelength) / ultrasound_wavelength;
      const auto phase = static_cast<uint16_t>(std::round(255.0 * (1.0 - f_phase)));
      this->_data[dev][i] = duty | phase;
    }

  this->_built = true;
  return Ok(true);
}

GainPtr BesselBeam::create(const Vector3& point, const Vector3& vec_n, const double theta_z, const double amp) {
  return create(point, vec_n, theta_z, to_duty(amp));
}

GainPtr BesselBeam::create(const Vector3& point, const Vector3& vec_n, double theta_z, uint8_t duty) {
  return std::make_shared<BesselBeam>(point, vec_n, theta_z, duty);
}

Error BesselBeam::calc(const core::GeometryPtr& geometry) {
  if (_vec_n.norm() > 0) _vec_n = _vec_n.normalized();
  const Vector3 v(_vec_n.y(), -_vec_n.x(), 0.);

  const auto theta_w = std::asin(v.norm());

  const auto ultrasound_wavelength = geometry->wavelength();
  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto trp = geometry->position(dev, i);
      const auto r = trp - this->_point;
      const auto v_x_r = r.cross(v);
      const auto rr = std::cos(theta_w) * r + sin(theta_w) * v_x_r + v.dot(r) * (1.0 - std::cos(theta_w)) * v;
      const auto f_phase =
          std::fmod(std::sin(_theta_z) * std::sqrt(rr.x() * rr.x() + rr.y() * rr.y()) - std::cos(_theta_z) * rr.z(), ultrasound_wavelength) /
          ultrasound_wavelength;
      const auto phase = static_cast<uint16_t>(std::round(255.0 * (1.0 - f_phase)));
      this->_data[dev][i] = duty | phase;
    }
  this->_built = true;
  return Ok(true);
}

GainPtr Custom::create(const uint16_t* data, const size_t data_length) {
  const auto dev_num = data_length / NUM_TRANS_IN_UNIT;

  std::vector<DataArray> raw_data(dev_num);
  size_t dev_idx = 0;
  size_t tran_idx = 0;
  for (size_t i = 0; i < data_length; i++) {
    raw_data[dev_idx][tran_idx++] = data[i];
    if (tran_idx == NUM_TRANS_IN_UNIT) {
      dev_idx++;
      tran_idx = 0;
    }
  }
  return create(raw_data);
}

GainPtr Custom::create(const std::vector<DataArray>& data) { return std::make_shared<Custom>(data); }

Error Custom::calc(const core::GeometryPtr& geometry) {
  this->_data = std::move(this->_raw_data);
  this->_built = true;
  return Ok(true);
}

GainPtr TransducerTest::create(const size_t transducer_index, const uint8_t duty, const uint8_t phase) {
  return std::make_shared<TransducerTest>(transducer_index, duty, phase);
}

Error TransducerTest::calc(const core::GeometryPtr& geometry) {
  const uint16_t d = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  const uint16_t s = static_cast<uint16_t>(this->_phase) & 0x00FF;
  this->_data[geometry->device_idx_for_trans_idx(_transducer_idx)][_transducer_idx % NUM_TRANS_IN_UNIT] = d | s;

  this->_built = true;
  return Ok(true);
}
}  // namespace autd::gain
