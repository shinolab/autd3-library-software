﻿// File: gain.cpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "gain.hpp"

#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "consts.hpp"
#include "core.hpp"
#include "privdef.hpp"
#include "vector3.hpp"

namespace autd::gain {

inline double pos_mod(double a, double b) { return a - floor(a / b) * b; }

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

bool Gain::built() noexcept { return this->_built; }

GeometryPtr Gain::geometry() noexcept { return this->_geometry; }

void Gain::SetGeometry(const GeometryPtr& geometry) noexcept { this->_geometry = geometry; }

std::vector<std::vector<uint16_t>>& Gain::data() { return this->_data; }

GainPtr PlaneWaveGain::Create(Vector3 direction, double amp) {
  uint8_t D = AdjustAmp(amp);
  return PlaneWaveGain::Create(direction, D);
}

GainPtr PlaneWaveGain::Create(Vector3 direction, uint8_t duty) {
  auto ptr = std::make_shared<PlaneWaveGain>();
  ptr->_direction = direction;
  ptr->_duty = duty;
  return ptr;
}

void PlaneWaveGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const auto ntrans = geometry->numTransducers();
  const auto dir = this->_direction.normalized();

  const uint8_t duty = this->_duty;
  for (size_t i = 0; i < ntrans; i++) {
    const auto trp = geometry->position(i);
    const auto dist = trp.dot(dir);
    const auto fphase = pos_mod(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const uint8_t phase = static_cast<uint8_t>(round(255.0 * (1.0 - fphase)));
    this->_data[geometry->deviceIdxForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(duty) << 8) + phase;
  }

  this->_built = true;
}

GainPtr FocalPointGain::Create(Vector3 direction, double amp) {
  uint8_t D = AdjustAmp(amp);
  return FocalPointGain::Create(direction, D);
}

GainPtr FocalPointGain::Create(Vector3 point, uint8_t duty) {
  auto gain = std::make_shared<FocalPointGain>();
  gain->_point = point;
  gain->_geometry = nullptr;
  gain->_duty = duty;
  return gain;
}

void FocalPointGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const uint8_t duty = this->_duty;
  const auto ntrans = geometry->numTransducers();
  for (size_t i = 0; i < ntrans; i++) {
    const auto trp = geometry->position(i);
    const auto dist = (trp - this->_point).l2_norm();
    const auto fphase = fmod(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const uint8_t phase = static_cast<uint8_t>(round(255.0 * (1.0 - fphase)));
    this->_data[geometry->deviceIdxForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(duty) << 8) + phase;
  }

  this->_built = true;
}

GainPtr BesselBeamGain::Create(Vector3 point, Vector3 vec_n, double theta_z, double amp) {
  uint8_t D = AdjustAmp(amp);

  return BesselBeamGain::Create(point, vec_n, theta_z, D);
}

GainPtr BesselBeamGain::Create(Vector3 point, Vector3 vec_n, double theta_z, uint8_t duty) {
  auto gain = std::make_shared<BesselBeamGain>();
  gain->_point = point;
  gain->_vec_n = vec_n;
  gain->_theta_z = theta_z;
  gain->_geometry = nullptr;
  gain->_duty = duty;
  return gain;
}

void BesselBeamGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const auto ntrans = geometry->numTransducers();

  const Vector3 _ez(0., 0., 1.0);

  if (_vec_n.l2_norm() > 0) _vec_n = _vec_n.normalized();
  const Vector3 _v(_vec_n.y(), -_vec_n.x(), 0.);

  auto _theta_w = asin(_v.l2_norm());

  const uint8_t duty = this->_duty;
  for (size_t i = 0; i < ntrans; i++) {
    const auto trp = geometry->position(i);
    const auto _r = trp - this->_point;
    const Vector3 _v_x_r = _r.cross(_v);
    const Vector3 _R = cos(_theta_w) * _r + sin(_theta_w) * _v_x_r + _v.dot(_r) * (1.0 - cos(_theta_w)) * _v;
    const auto fphase =
        fmod(sin(_theta_z) * sqrt(_R.x() * _R.x() + _R.y() * _R.y()) - cos(_theta_z) * _R.z(), ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const uint8_t phase = static_cast<uint8_t>(round(255.0 * (1.0 - fphase)));
    this->_data[geometry->deviceIdxForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(duty) << 8) + phase;
  }

  this->_built = true;
}

GainPtr CustomGain::Create(uint16_t* data, size_t data_length) {
  auto gain = std::make_shared<CustomGain>();
  gain->_rawdata.resize(data_length);
  for (size_t i = 0; i < data_length; i++) gain->_rawdata.at(i) = data[i];
  gain->_geometry = nullptr;
  return gain;
}

void CustomGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  const auto ntrans = geometry->numTransducers();

  for (size_t i = 0; i < ntrans; i++) {
    const auto data = this->_rawdata[i];
    this->_data[geometry->deviceIdxForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = data;
  }

  this->_built = true;
}

GainPtr TransducerTestGain::Create(size_t idx, uint8_t duty, uint8_t phase) {
  auto gain = std::make_shared<TransducerTestGain>();
  gain->_xdcr_idx = idx;
  gain->_duty = duty;
  gain->_phase = phase;
  return gain;
}

void TransducerTestGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  uint16_t d = (static_cast<uint16_t>(this->_duty) << 8) & 0xFF00;
  uint16_t s = static_cast<uint16_t>(this->_phase) & 0x00FF;
  this->_data[geometry->deviceIdxForTransIdx(_xdcr_idx)].at(_xdcr_idx % NUM_TRANS_IN_UNIT) = d | s;

  this->_built = true;
}
}  // namespace autd::gain
