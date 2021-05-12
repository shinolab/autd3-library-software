// File: primitive_gain.cpp
// Project: lib
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "primitive_gain.hpp"

#include <memory>
#include <vector>

namespace autd::gain {

using core::AUTDDataArray;
using core::NUM_TRANS_IN_UNIT;
using core::Vector3;

inline double PosMod(const double a, const double b) { return a - floor(a / b) * b; }

template <typename T>
uint8_t ToDuty(const T amp) noexcept {
  const auto d = std::asin(amp) / static_cast<T>(M_PI);  //  duty (0 ~ 0.5)
  return static_cast<uint8_t>(511 * d);
}

GainPtr GroupedGain::Create(const std::map<size_t, GainPtr>& gain_map) {
  GainPtr gain = std::make_shared<GroupedGain>(gain_map);
  return gain;
}

Result<bool, std::string> GroupedGain::Calc(core::GeometryPtr geometry) {
  for (const auto& [fst, g] : this->_gain_map) {
    if (auto res = g->Build(geometry); res.is_err()) return res;
  }

  for (size_t i = 0; i < geometry->num_devices(); i++) {
    if (auto group_id = geometry->group_id_for_device_idx(i); _gain_map.count(group_id)) {
      auto& data = _gain_map[group_id]->data();
      this->_data[i] = data[i];
    } else {
      this->_data[i] = AUTDDataArray{0x0000};
    }
  }

  this->_built = true;
  return Ok(true);
}

GainPtr PlaneWaveGain::Create(const Vector3& direction, const double amp) {
  const auto d = ToDuty(amp);
  return Create(direction, d);
}

GainPtr PlaneWaveGain::Create(const Vector3& direction, uint8_t duty) {
  GainPtr ptr = std::make_shared<PlaneWaveGain>(direction, duty);
  return ptr;
}

Result<bool, std::string> PlaneWaveGain::Calc(core::GeometryPtr geometry) {
  const auto dir = this->_direction.normalized();

  const auto ultrasound_wavelength = geometry->wavelength();
  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto trp = geometry->position(dev, i);
      const auto dist = trp.dot(dir);
      const auto f_phase = PosMod(dist, ultrasound_wavelength) / ultrasound_wavelength;
      const auto phase = static_cast<uint16_t>(round(255 * (1 - f_phase)));
      this->_data[dev][i] = duty | phase;
    }

  this->_built = true;
  return Ok(true);
}

GainPtr FocalPointGain::Create(const Vector3& point, const double amp) {
  const auto d = ToDuty(amp);
  return Create(point, d);
}

GainPtr FocalPointGain::Create(const Vector3& point, uint8_t duty) {
  GainPtr gain = std::make_shared<FocalPointGain>(point, duty);
  return gain;
}

Result<bool, std::string> FocalPointGain::Calc(core::GeometryPtr geometry) {
  const auto ultrasound_wavelength = geometry->wavelength();
  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto trp = geometry->position(dev, i);
      const auto dist = (trp - this->_point).norm();
      const auto f_phase = fmod(dist, ultrasound_wavelength) / ultrasound_wavelength;
      const auto phase = static_cast<uint16_t>(round(255 * (1 - f_phase)));
      this->_data[dev][i] = duty | phase;
    }

  this->_built = true;
  return Ok(true);
}

GainPtr BesselBeamGain::Create(const Vector3& point, const Vector3& vec_n, const double theta_z, const double amp) {
  const auto duty = ToDuty(amp);
  return Create(point, vec_n, theta_z, duty);
}

GainPtr BesselBeamGain::Create(const Vector3& point, const Vector3& vec_n, double theta_z, uint8_t duty) {
  GainPtr gain = std::make_shared<BesselBeamGain>(point, vec_n, theta_z, duty);
  return gain;
}

Result<bool, std::string> BesselBeamGain::Calc(core::GeometryPtr geometry) {
  if (_vec_n.norm() > 0) _vec_n = _vec_n.normalized();
  const Vector3 v(_vec_n.y(), -_vec_n.x(), 0.);

  const auto theta_w = asin(v.norm());

  const auto ultrasound_wavelength = geometry->wavelength();
  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t dev = 0; dev < geometry->num_devices(); dev++)
    for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
      const auto trp = geometry->position(dev, i);
      const auto r = trp - this->_point;
      const auto v_x_r = r.cross(v);
      const auto rr = cos(theta_w) * r + sin(theta_w) * v_x_r + v.dot(r) * (1 - cos(theta_w)) * v;
      const auto f_phase =
          fmod(sin(_theta_z) * sqrt(rr.x() * rr.x() + rr.y() * rr.y()) - cos(_theta_z) * rr.z(), ultrasound_wavelength) / ultrasound_wavelength;
      const auto phase = static_cast<uint16_t>(round(255 * (1 - f_phase)));
      this->_data[dev][i] = duty | phase;
    }
  this->_built = true;
  return Ok(true);
}

GainPtr CustomGain::Create(const uint16_t* data, const size_t data_length) {
  const auto dev_num = data_length / NUM_TRANS_IN_UNIT;

  std::vector<AUTDDataArray> raw_data(dev_num);
  size_t dev_idx = 0;
  size_t tran_idx = 0;
  for (size_t i = 0; i < data_length; i++) {
    raw_data[dev_idx][tran_idx++] = data[i];
    if (tran_idx == NUM_TRANS_IN_UNIT) {
      dev_idx++;
      tran_idx = 0;
    }
  }
  GainPtr gain = std::make_shared<CustomGain>(raw_data);
  return gain;
}

GainPtr CustomGain::Create(const std::vector<AUTDDataArray>& data) {
  GainPtr gain = std::make_shared<CustomGain>(data);
  return gain;
}

Result<bool, std::string> CustomGain::Calc(core::GeometryPtr geometry) {
  this->_built = true;
  return Ok(true);
}

GainPtr TransducerTestGain::Create(const size_t transducer_index, const uint8_t duty, const uint8_t phase) {
  GainPtr gain = std::make_shared<TransducerTestGain>(transducer_index, duty, phase);
  return gain;
}

Result<bool, std::string> TransducerTestGain::Calc(core::GeometryPtr geometry) {
  const uint16_t d = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  const uint16_t s = static_cast<uint16_t>(this->_phase) & 0x00FF;
  this->_data[geometry->device_idx_for_trans_idx(_transducer_idx)][_transducer_idx % NUM_TRANS_IN_UNIT] = d | s;

  this->_built = true;
  return Ok(true);
}
}  // namespace autd::gain
