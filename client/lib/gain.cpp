// File: gain.cpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 26/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "gain.hpp"

#include <iostream>
#include <vector>

#include "consts.hpp"
#include "convert.hpp"
#include "core.hpp"
#include "vector3.hpp"

namespace autd::gain {

inline double PosMod(const double a, const double b) { return a - floor(a / b) * b; }

GainPtr Gain::Create() { return std::make_shared<Gain>(); }

Gain::Gain() noexcept {
  this->_built = false;
  this->_geometry = nullptr;
}

void Gain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  for (size_t i = 0; i < geometry->num_devices(); i++) this->_data[i].fill(0x0000);

  this->_built = true;
}

bool Gain::built() const noexcept { return this->_built; }

GeometryPtr Gain::geometry() const noexcept { return this->_geometry; }

void Gain::SetGeometry(const GeometryPtr& geometry) noexcept { this->_geometry = geometry; }

std::vector<AUTDDataArray>& Gain::data() { return this->_data; }

GainPtr PlaneWaveGain::Create(const utils::Vector3& direction, const double amp) {
  const auto d = AdjustAmp(amp);
  return Create(direction, d);
}

GainPtr PlaneWaveGain::Create(const utils::Vector3& direction, uint8_t duty) {
  GainPtr ptr = std::make_shared<PlaneWaveGain>(Convert(direction), duty);
  return ptr;
}

#ifdef USE_EIGEN_AUTD
GainPtr PlaneWaveGain::Create(const Vector3& direction, const double amp) {
  const auto d = AdjustAmp(amp);
  return Create(direction, d);
}

GainPtr PlaneWaveGain::Create(const Vector3& direction, uint8_t duty) {
  GainPtr ptr = std::make_shared<PlaneWaveGain>(direction, duty);
  return ptr;
}
#endif

void PlaneWaveGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const auto num_trans = geometry->num_transducers();
  const auto dir = this->_direction.normalized();

  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t i = 0; i < num_trans; i++) {
    const auto trp = geometry->position(i);
    const auto dist = trp.dot(dir);
    const auto f_phase = PosMod(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint16_t>(round(255.0 * (1.0 - f_phase)));
    this->_data[geometry->device_idx_for_trans_idx(i)][i % NUM_TRANS_IN_UNIT] = duty | phase;
  }

  this->_built = true;
}

GainPtr FocalPointGain::Create(const utils::Vector3& point, const double amp) {
  const auto d = AdjustAmp(amp);
  return Create(point, d);
}

GainPtr FocalPointGain::Create(const utils::Vector3& point, uint8_t duty) {
  GainPtr gain = std::make_shared<FocalPointGain>(Convert(point), duty);
  return gain;
}

#ifdef USE_EIGEN_AUTD
GainPtr FocalPointGain::Create(const Vector3& point, const double amp) {
  const auto d = AdjustAmp(amp);
  return Create(point, d);
}

GainPtr FocalPointGain::Create(const Vector3& point, uint8_t duty) {
  GainPtr gain = std::make_shared<FocalPointGain>(point, duty);
  return gain;
}
#endif

void FocalPointGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t i = 0; i < geometry->num_transducers(); i++) {
    const auto trp = geometry->position(i);
    const auto dist = (trp - this->_point).norm();
    const auto f_phase = fmod(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint16_t>(round(255.0 * (1.0 - f_phase)));
    this->_data[geometry->device_idx_for_trans_idx(i)][i % NUM_TRANS_IN_UNIT] = duty | phase;
  }

  this->_built = true;
}

GainPtr BesselBeamGain::Create(const utils::Vector3& point, const utils::Vector3& vec_n, const double theta_z, const double amp) {
  const auto duty = AdjustAmp(amp);
  return Create(point, vec_n, theta_z, duty);
}

GainPtr BesselBeamGain::Create(const utils::Vector3& point, const utils::Vector3& vec_n, double theta_z, uint8_t duty) {
  GainPtr gain = std::make_shared<BesselBeamGain>(Convert(point), Convert(vec_n), theta_z, duty);
  return gain;
}

#ifdef USE_EIGEN_AUTD
GainPtr BesselBeamGain::Create(const Vector3& point, const Vector3& vec_n, const double theta_z, const double amp) {
  const auto duty = AdjustAmp(amp);
  return Create(point, vec_n, theta_z, duty);
}

GainPtr BesselBeamGain::Create(const Vector3& point, const Vector3& vec_n, double theta_z, uint8_t duty) {
  GainPtr gain = std::make_shared<BesselBeamGain>(point, vec_n, theta_z, duty);
  return gain;
}
#endif

void BesselBeamGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  if (_vec_n.norm() > 0) _vec_n = _vec_n.normalized();
  const Vector3 v(_vec_n.y(), -_vec_n.x(), 0.);

  const auto theta_w = asin(v.norm());

  const uint16_t duty = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  for (size_t i = 0; i < geometry->num_transducers(); i++) {
    const auto trp = geometry->position(i);
    const auto r = trp - this->_point;
    const auto v_x_r = r.cross(v);
    const auto rr = cos(theta_w) * r + sin(theta_w) * v_x_r + v.dot(r) * (1.0 - cos(theta_w)) * v;
    const auto f_phase =
        fmod(sin(_theta_z) * sqrt(rr.x() * rr.x() + rr.y() * rr.y()) - cos(_theta_z) * rr.z(), ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint16_t>(round(255.0 * (1.0 - f_phase)));
    this->_data[geometry->device_idx_for_trans_idx(i)][i % NUM_TRANS_IN_UNIT] = duty | phase;
  }

  this->_built = true;
}

GainPtr CustomGain::Create(const uint16_t* data, const size_t data_length) {
  std::vector<uint16_t> raw_data(data_length);
  for (size_t i = 0; i < data_length; i++) raw_data.at(i) = data[i];
  GainPtr gain = std::make_shared<CustomGain>(raw_data);
  return gain;
}

void CustomGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const auto num_trans = geometry->num_transducers();

  for (size_t i = 0; i < num_trans; i++) {
    const auto data = this->_raw_data[i];
    this->_data[geometry->device_idx_for_trans_idx(i)][i % NUM_TRANS_IN_UNIT] = data;
  }

  this->_built = true;
}

GainPtr TransducerTestGain::Create(const size_t transducer_index, const uint8_t duty, const uint8_t phase) {
  GainPtr gain = std::make_shared<TransducerTestGain>(transducer_index, duty, phase);
  return gain;
}

void TransducerTestGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const uint16_t d = static_cast<uint16_t>(this->_duty) << 8 & 0xFF00;
  const uint16_t s = static_cast<uint16_t>(this->_phase) & 0x00FF;
  this->_data[geometry->device_idx_for_trans_idx(_transducer_idx)][_transducer_idx % NUM_TRANS_IN_UNIT] = d | s;

  this->_built = true;
}
}  // namespace autd::gain
