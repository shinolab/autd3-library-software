// File: gain.cpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "gain.hpp"

#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#include "consts.hpp"
#include "core.hpp"
#include "vector3.hpp"

namespace autd::gain {

inline double pos_mod(const double a, const double b) { return a - floor(a / b) * b; }

GainPtr Gain::Create() { return std::make_shared<Gain>(); }

Gain::Gain() noexcept {
  this->_built = false;
  this->_geometry = nullptr;
}

void Gain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();
  assert(geometry != nullptr);

  this->_data.resize(geometry->numDevices());
  for (size_t i = 0; i < geometry->numDevices(); i++) {
    this->_data[i] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
  }

  this->_built = true;
}

bool Gain::built() const noexcept { return this->_built; }

GeometryPtr Gain::geometry() const noexcept { return this->_geometry; }

void Gain::SetGeometry(const GeometryPtr& geometry) noexcept { this->_geometry = geometry; }

std::vector<std::vector<uint16_t>>& Gain::data() { return this->_data; }

GainPtr PlaneWaveGain::Create(const Vector3 direction, const double amp) {
  const auto d = AdjustAmp(amp);
  return Create(direction, d);
}

GainPtr PlaneWaveGain::Create(const Vector3 direction, const uint8_t duty) {
  GainPtr ptr = std::make_shared<PlaneWaveGain>(direction, duty);
  return ptr;
}

void PlaneWaveGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const auto num_trans = geometry->numTransducers();
  const auto dir = this->_direction.normalized();

  const uint16_t duty = (static_cast<uint16_t>(this->_duty) << 8) & 0xFF00;
  for (size_t i = 0; i < num_trans; i++) {
    const auto trp = geometry->position(i);
    const auto dist = trp.dot(dir);
    const auto f_phase = pos_mod(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint16_t>(round(255.0 * (1.0 - f_phase)));
    this->_data[geometry->deviceIdxForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = duty | phase;
  }

  this->_built = true;
}

GainPtr FocalPointGain::Create(const Vector3 point, const double amp) {
  const auto d = AdjustAmp(amp);
  return Create(point, d);
}

GainPtr FocalPointGain::Create(Vector3 point, uint8_t duty) {
  GainPtr gain = std::make_shared<FocalPointGain>(point, duty);
  return gain;
}

void FocalPointGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const uint16_t duty = (static_cast<uint16_t>(this->_duty) << 8) & 0xFF00;
  for (size_t i = 0; i < geometry->numTransducers(); i++) {
    const auto trp = geometry->position(i);
    const auto dist = (trp - this->_point).l2_norm();
    const auto f_phase = fmod(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint16_t>(round(255.0 * (1.0 - f_phase)));
    this->_data[geometry->deviceIdxForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = duty | phase;
  }

  this->_built = true;
}

GainPtr BesselBeamGain::Create(const Vector3 point, const Vector3 vec_n, const double theta_z, const double amp) {
  const auto D = AdjustAmp(amp);
  return Create(point, vec_n, theta_z, D);
}

GainPtr BesselBeamGain::Create(Vector3 point, Vector3 vec_n, double theta_z, uint8_t duty) {
  GainPtr gain = std::make_shared<BesselBeamGain>(point, vec_n, theta_z, duty);
  return gain;
}

void BesselBeamGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  if (_vec_n.l2_norm() > 0) _vec_n = _vec_n.normalized();
  const Vector3 _v(_vec_n.y(), -_vec_n.x(), 0.);

  const auto _theta_w = asin(_v.l2_norm());

  const uint16_t duty = (static_cast<uint16_t>(this->_duty) << 8) & 0xFF00;
  for (size_t i = 0; i < geometry->numTransducers(); i++) {
    const auto trp = geometry->position(i);
    const auto _r = trp - this->_point;
    const auto _v_x_r = _r.cross(_v);
    const auto _R = cos(_theta_w) * _r + sin(_theta_w) * _v_x_r + _v.dot(_r) * (1.0 - cos(_theta_w)) * _v;
    const auto f_phase =
        fmod(sin(_theta_z) * sqrt(_R.x() * _R.x() + _R.y() * _R.y()) - cos(_theta_z) * _R.z(), ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint16_t>(round(255.0 * (1.0 - f_phase)));
    this->_data[geometry->deviceIdxForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = duty | phase;
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

  const auto num_trans = geometry->numTransducers();

  for (size_t i = 0; i < num_trans; i++) {
    const auto data = this->_raw_data[i];
    this->_data[geometry->deviceIdxForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = data;
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

  const uint16_t d = (static_cast<uint16_t>(this->_duty) << 8) & 0xFF00;
  const uint16_t s = static_cast<uint16_t>(this->_phase) & 0x00FF;
  this->_data[geometry->deviceIdxForTransIdx(_transducer_idx)].at(_transducer_idx % NUM_TRANS_IN_UNIT) = d | s;

  this->_built = true;
}
}  // namespace autd::gain
