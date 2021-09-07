// File: primitive.cpp
// Project: gain
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 03/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/primitive.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "autd3/core/utils.hpp"

namespace autd::gain {

using core::DataArray;
using core::NUM_TRANS_IN_UNIT;
using core::Vector3;

std::shared_ptr<Grouped> Grouped::create() { return std::make_shared<Grouped>(); }

void Grouped::add(const size_t id, const GainPtr& gain) { this->_gain_map[id] = gain; }

void Grouped::calc(const core::GeometryPtr& geometry) {
  for (const auto& [_, g] : this->_gain_map) g->build(geometry);

  for (size_t i = 0; i < geometry->num_devices(); i++) {
    auto group_id = geometry->group_id_for_device_idx(i);
    this->_data[i] = _gain_map.count(group_id) ? _gain_map[group_id]->data()[i] : DataArray{0x0000};
  }
  this->_built = true;
}

GainPtr PlaneWave::create(const Vector3& direction, const double amp) { return create(direction, core::Utilities::to_duty(amp)); }

GainPtr PlaneWave::create(const Vector3& direction, uint8_t duty) { return std::make_shared<PlaneWave>(direction, duty); }

void PlaneWave::calc(const core::GeometryPtr& geometry) {
  const auto dir = this->_direction.normalized();

  const auto ultrasound_wavelength = geometry->wavelength();
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto dist = geometry->position(dev, i).dot(dir);
      const auto phase = core::Utilities::to_phase(dist / ultrasound_wavelength);
      this->_data[dev][i] = core::Utilities::pack_to_u16(this->_duty, phase);
    }

  this->_built = true;
}

GainPtr FocalPoint::create(const Vector3& point, const double amp) { return create(point, core::Utilities::to_duty(amp)); }
GainPtr FocalPoint::create(const Vector3& point, uint8_t duty) { return std::make_shared<FocalPoint>(point, duty); }

void FocalPoint::calc(const core::GeometryPtr& geometry) {
  const auto ultrasound_wavelength = geometry->wavelength();
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto dist = (geometry->position(dev, i) - this->_point).norm();
      const auto phase = core::Utilities::to_phase(dist / ultrasound_wavelength);
      this->_data[dev][i] = core::Utilities::pack_to_u16(this->_duty, phase);
    }

  this->_built = true;
}

GainPtr BesselBeam::create(const Vector3& point, const Vector3& vec_n, const double theta_z, const double amp) {
  return create(point, vec_n, theta_z, core::Utilities::to_duty(amp));
}

GainPtr BesselBeam::create(const Vector3& point, const Vector3& vec_n, double theta_z, uint8_t duty) {
  return std::make_shared<BesselBeam>(point, vec_n, theta_z, duty);
}

void BesselBeam::calc(const core::GeometryPtr& geometry) {
  if (_vec_n.norm() > 0) _vec_n = _vec_n.normalized();
  const Vector3 v(_vec_n.y(), -_vec_n.x(), 0.);

  const auto theta_w = std::asin(v.norm());

  const auto ultrasound_wavelength = geometry->wavelength();
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto r = geometry->position(dev, i) - this->_point;
      const auto v_x_r = r.cross(v);
      const auto rr = std::cos(theta_w) * r + std::sin(theta_w) * v_x_r + v.dot(r) * (1.0 - std::cos(theta_w)) * v;
      const auto d = std::sin(_theta_z) * std::sqrt(rr.x() * rr.x() + rr.y() * rr.y()) - std::cos(_theta_z) * rr.z();
      const auto phase = core::Utilities::to_phase(d / ultrasound_wavelength);
      this->_data[dev][i] = core::Utilities::pack_to_u16(this->_duty, phase);
    }
  this->_built = true;
}

GainPtr Custom::create(const uint16_t* data, const size_t data_length) {
  const auto dev_num = (data_length + NUM_TRANS_IN_UNIT - 1) / NUM_TRANS_IN_UNIT;
  std::vector<DataArray> raw_data(dev_num);
  for (size_t i = 0; i < dev_num; i++) {
    const auto rem = std::clamp(data_length - i * NUM_TRANS_IN_UNIT, size_t{0}, NUM_TRANS_IN_UNIT);
    std::memcpy(&raw_data[i][0], data + i * NUM_TRANS_IN_UNIT, rem);
  }
  return create(raw_data);
}

GainPtr Custom::create(const std::vector<DataArray>& data) { return std::make_shared<Custom>(data); }

void Custom::calc(const core::GeometryPtr&) {
  this->_data = std::move(this->_raw_data);
  this->_built = true;
}

GainPtr TransducerTest::create(const size_t transducer_index, const uint8_t duty, const uint8_t phase) {
  return std::make_shared<TransducerTest>(transducer_index, duty, phase);
}

void TransducerTest::calc(const core::GeometryPtr& geometry) {
  this->_data[geometry->device_idx_for_trans_idx(_transducer_idx)][_transducer_idx % NUM_TRANS_IN_UNIT] =
      core::Utilities::pack_to_u16(this->_duty, this->_phase);

  this->_built = true;
}
}  // namespace autd::gain
