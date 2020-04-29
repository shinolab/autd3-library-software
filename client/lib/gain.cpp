// File: gain.cpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 30/04/2020
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

namespace autd {

inline double pos_mod(double a, double b) { return a - floor(a / b) * b; }

GainPtr Gain::Create() { return CreateHelper<Gain>(); }

Gain::Gain() noexcept {
  this->_built = false;
  this->_geometry = nullptr;
}

void Gain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();
  assert(geometry != nullptr);

  for (int i = 0; i < geometry->numDevices(); i++) {
    this->_data[geometry->deviceIdForDeviceIdx(i)] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
  }

  this->_built = true;
}

bool Gain::built() noexcept { return this->_built; }

GeometryPtr Gain::geometry() noexcept { return this->_geometry; }

void Gain::SetGeometry(const GeometryPtr& geometry) noexcept { this->_geometry = geometry; }

std::map<int, std::vector<uint16_t>> Gain::data() { return this->_data; }

GainPtr PlaneWaveGain::Create(Vector3 direction) { return PlaneWaveGain::Create(direction, 0xFF); }

GainPtr PlaneWaveGain::Create(Vector3 direction, uint8_t amp) {
  auto ptr = CreateHelper<PlaneWaveGain>();
  ptr->_direction = direction;
  ptr->_amp = amp;
  return ptr;
}

void PlaneWaveGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();
  assert(geometry != nullptr);

  this->_data.clear();
  const auto ndevice = geometry->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }

  const auto ntrans = geometry->numTransducers();
  const auto dir = this->_direction.normalized();

  for (int i = 0; i < ntrans; i++) {
    const auto trp = geometry->position(i);
    const auto dist = trp.dot(dir);
    const auto fphase = pos_mod(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint8_t>(round(255.0 * (1.0 - fphase)));
    uint8_t D, S;
    SignalDesign(this->_amp, phase, &D, &S);
    this->_data[geometry->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }

  this->_built = true;
}

GainPtr FocalPointGain::Create(Vector3 point) { return FocalPointGain::Create(point, 255); }

GainPtr FocalPointGain::Create(Vector3 point, uint8_t amp) {
  auto gain = CreateHelper<FocalPointGain>();
  gain->_point = point;
  gain->_geometry = nullptr;
  gain->_amp = amp;
  return gain;
}

void FocalPointGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();
  assert(geometry != nullptr);

  this->_data.clear();

  const auto ndevice = geometry->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }

  const auto ntrans = geometry->numTransducers();
  for (int i = 0; i < ntrans; i++) {
    const auto trp = geometry->position(i);
    const auto dist = (trp - this->_point).l2_norm();
    const auto fphase = fmod(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint8_t>(round(255.0 * (1.0 - fphase)));
    uint8_t D, S;
    SignalDesign(this->_amp, phase, &D, &S);
    this->_data[geometry->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }

  this->_built = true;
}

GainPtr BesselBeamGain::Create(Vector3 point, Vector3 vec_n, double theta_z) { return BesselBeamGain::Create(point, vec_n, theta_z, 255); }

GainPtr BesselBeamGain::Create(Vector3 point, Vector3 vec_n, double theta_z, uint8_t amp) {
  auto gain = CreateHelper<BesselBeamGain>();
  gain->_point = point;
  gain->_vec_n = vec_n;
  gain->_theta_z = theta_z;
  gain->_geometry = nullptr;
  gain->_amp = amp;
  return gain;
}

void BesselBeamGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();
  assert(geometry != nullptr);

  this->_data.clear();
  const auto ndevice = geometry->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }
  const auto ntrans = geometry->numTransducers();

  const Vector3 _ez(0., 0., 1.0);

  if (_vec_n.l2_norm() > 0) _vec_n = _vec_n.normalized();
  const Vector3 _v(_vec_n.y(), -_vec_n.x(), 0.);

  auto _theta_w = asin(_v.l2_norm());

  for (int i = 0; i < ntrans; i++) {
    const auto trp = geometry->position(i);
    const auto _r = trp - this->_point;
    const Vector3 _v_x_r = _r.cross(_v);
    const Vector3 _R = cos(_theta_w) * _r + sin(_theta_w) * _v_x_r + _v.dot(_r) * (1.0 - cos(_theta_w)) * _v;
    const auto fphase =
        fmod(sin(_theta_z) * sqrt(_R.x() * _R.x() + _R.y() * _R.y()) - cos(_theta_z) * _R.z(), ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
    const auto phase = static_cast<uint8_t>(round(255.0 * (1.0 - fphase)));
    uint8_t D, S;
    SignalDesign(this->_amp, phase, &D, &S);
    this->_data[geometry->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }

  this->_built = true;
}

GainPtr CustomGain::Create(uint16_t* data, int data_length) {
  auto gain = CreateHelper<CustomGain>();
  gain->_rawdata.resize(data_length);
  for (int i = 0; i < data_length; i++) gain->_rawdata.at(i) = data[i];
  gain->_geometry = nullptr;
  return gain;
}

void CustomGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();
  assert(geometry != nullptr);

  this->_data.clear();
  const auto ndevice = geometry->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }
  const auto ntrans = geometry->numTransducers();

  for (int i = 0; i < ntrans; i++) {
    const auto data = this->_rawdata[i];
    const auto amp = static_cast<uint8_t>(data >> 8);
    const auto phase = static_cast<uint8_t>(data);
    uint8_t D, S;
    SignalDesign(amp, phase, &D, &S);
    this->_data[geometry->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }

  this->_built = true;
}

GainPtr TransducerTestGain::Create(int idx, int amp, int phase) {
  auto gain = CreateHelper<TransducerTestGain>();
  gain->_xdcr_idx = idx;
  gain->_amp = amp;
  gain->_phase = phase;
  return gain;
}

void TransducerTestGain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();
  assert(geometry != nullptr);

  this->_data.clear();
  const auto ndevice = geometry->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }

  uint8_t D, S;
  SignalDesign(this->_amp, this->_phase, &D, &S);
  this->_data[geometry->deviceIdForTransIdx(_xdcr_idx)].at(_xdcr_idx % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;

  this->_built = true;
}
}  // namespace autd
